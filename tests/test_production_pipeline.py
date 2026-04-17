"""End-to-end integration test for the production pipeline.

Trains a minimal XGBoost on synthetic data, saves & reloads via joblib,
and checks that predictions round-trip correctly.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import build_production_pipeline


def test_production_pipeline_fits_and_predicts(synthetic_car_df):
    X = synthetic_car_df.drop("price_PLN", axis=1)
    y_log = np.log1p(synthetic_car_df["price_PLN"])

    pipe = build_production_pipeline(
        xgb_params={
            "n_estimators": 50, "max_depth": 4,
            "learning_rate": 0.1, "tree_method": "hist", "random_state": 0,
        }
    )
    pipe.fit(X, y_log)

    preds_log = pipe.predict(X.iloc[:10])
    assert preds_log.shape == (10,)
    assert not np.isnan(preds_log).any()

    preds_pln = np.expm1(preds_log)
    assert (preds_pln > 0).all()


def test_production_pipeline_roundtrip_through_joblib(synthetic_car_df, tmp_path):
    X = synthetic_car_df.drop("price_PLN", axis=1)
    y_log = np.log1p(synthetic_car_df["price_PLN"])

    pipe = build_production_pipeline(
        xgb_params={"n_estimators": 30, "max_depth": 3, "tree_method": "hist", "random_state": 0}
    )
    pipe.fit(X, y_log)

    fp = tmp_path / "prod.joblib"
    joblib.dump({"pipeline": pipe, "trained_at": "2026-01-01"}, fp)
    loaded = joblib.load(fp)
    reloaded_pipe = loaded["pipeline"]

    original = pipe.predict(X.iloc[:15])
    reloaded = reloaded_pipe.predict(X.iloc[:15])
    np.testing.assert_allclose(original, reloaded, rtol=1e-6)


def test_pipeline_accepts_raw_single_row(synthetic_car_df):
    """Regression test: the pipeline must accept a 1-row DataFrame with
    just the RAW columns (no pre-engineered 41-feature layout)."""
    X = synthetic_car_df.drop("price_PLN", axis=1)
    y_log = np.log1p(synthetic_car_df["price_PLN"])

    pipe = build_production_pipeline(
        xgb_params={"n_estimators": 30, "max_depth": 3, "tree_method": "hist", "random_state": 0}
    )
    pipe.fit(X, y_log)

    single_row = pd.DataFrame([{
        "Condition":        "used",
        "Vehicle_brand":    "audi",
        "Vehicle_model":    "a4",
        "Production_year":  2018,
        "Mileage_km":       120_000,
        "Power_HP":         190,
        "Displacement_cm3": 1968,
        "Fuel_type":        "diesel",
        "Drive":            "front wheels",
        "Transmission":     "automatic",
        "Type":             "sedan",
        "Doors_number":     4,
        "Colour":           "black",
        "Origin_country":   "germany",
        "First_owner":      1,
        "Offer_location":   "Warszawa",
        "Features":         "airbags, abs, navigation",
    }])

    prediction = pipe.predict(single_row)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])
    assert np.expm1(prediction[0]) > 1000  # sanity check
