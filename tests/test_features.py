"""Tests for feature-engineering logic (src.features).

These tests specifically target the train/inference-consistency bug that
motivated the FeatureEngineeringTransformer refactor: features produced
by the transformer at inference time must match those produced at
training time for the same raw row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureEngineeringTransformer,
    apply_advanced_transformations,
    engineer_base_features,
)


def test_engineer_base_features_creates_expected_columns(synthetic_car_df):
    df_f = engineer_base_features(synthetic_car_df)
    for col in (
        "Vehicle_age", "Age_category", "Is_new_car", "Is_old_car",
        "Is_collector", "Mileage_per_year", "Usage_intensity",
        "HP_per_liter", "Performance_category",
        "Is_premium", "Is_supercar", "Num_features",
    ):
        assert col in df_f.columns, f"Missing engineered column: {col}"


def test_fit_transform_is_idempotent_shape(synthetic_car_df):
    X = synthetic_car_df.drop("price_PLN", axis=1)
    t = FeatureEngineeringTransformer()
    X1 = t.fit_transform(X)
    X2 = t.transform(X)
    assert X1.shape == X2.shape
    # All columns must match.
    assert list(X1.columns) == list(X2.columns)


def test_transformer_learns_brand_frequencies(synthetic_car_df):
    X = synthetic_car_df.drop("price_PLN", axis=1)
    t = FeatureEngineeringTransformer().fit(X)
    # Brands present in training should be in the fitted map.
    for brand in ("volkswagen", "toyota", "bmw"):
        assert brand in t.brand_freq_
    assert t.max_brand_freq_ >= 1


def test_inference_consistency_train_vs_test(synthetic_car_df):
    """Regression test for the original C1 bug.

    A row identical in train and test must receive the SAME brand-level
    features whether we see it during fit or only during transform.
    """
    X = synthetic_car_df.drop("price_PLN", axis=1)
    t = FeatureEngineeringTransformer().fit(X)

    # Build a single-row "inference" DataFrame picked from the training data.
    probe = X.iloc[[5]].copy()
    train_features = t.transform(X).iloc[[5]][["Brand_frequency", "Brand_tier",
                                                "BrandModel_frequency", "Rarity_index"]]
    inference_features = t.transform(probe)[["Brand_frequency", "Brand_tier",
                                              "BrandModel_frequency", "Rarity_index"]]

    pd.testing.assert_frame_equal(
        train_features.reset_index(drop=True),
        inference_features.reset_index(drop=True),
        check_dtype=False,
    )


def test_unknown_brand_gets_freq_one(synthetic_car_df):
    X = synthetic_car_df.drop("price_PLN", axis=1)
    t = FeatureEngineeringTransformer().fit(X)

    unseen = X.iloc[[0]].copy()
    unseen["Vehicle_brand"] = "UFO-motors"
    unseen["Vehicle_model"] = "zorg"
    out = t.transform(unseen)
    assert int(out["Brand_frequency"].iloc[0]) == 1
    assert int(out["BrandModel_frequency"].iloc[0]) == 1
    assert out["Brand_tier"].iloc[0] == "Niche"


def test_transformer_is_picklable(synthetic_car_df, tmp_path):
    import joblib
    X = synthetic_car_df.drop("price_PLN", axis=1)
    t = FeatureEngineeringTransformer().fit(X)
    fp = tmp_path / "transformer.pkl"
    joblib.dump(t, fp)
    loaded = joblib.load(fp)
    pd.testing.assert_frame_equal(
        t.transform(X.iloc[:10]).reset_index(drop=True),
        loaded.transform(X.iloc[:10]).reset_index(drop=True),
        check_dtype=False,
    )


def test_legacy_api_still_works(synthetic_car_df):
    """apply_advanced_transformations must still function (notebook compat)."""
    df_f = engineer_base_features(synthetic_car_df.drop("price_PLN", axis=1))
    split = len(df_f) // 2
    X_train, X_test = df_f.iloc[:split], df_f.iloc[split:]
    tf_train, tf_test = apply_advanced_transformations(X_train, X_test)
    assert "Brand_frequency" in tf_train.columns
    assert "Brand_frequency" in tf_test.columns
    assert tf_train.shape[1] == tf_test.shape[1]


def test_no_nan_in_critical_columns_after_transform(synthetic_car_df):
    X = synthetic_car_df.drop("price_PLN", axis=1)
    # Inject NaNs to exercise the imputer.
    X_with_nans = X.copy()
    X_with_nans.loc[0:5, "Mileage_km"] = np.nan
    X_with_nans.loc[3:9, "Power_HP"] = np.nan

    t = FeatureEngineeringTransformer().fit(X)
    out = t.transform(X_with_nans)

    for col in ("Mileage_km", "Power_HP", "Displacement_cm3"):
        assert not out[col].isna().any(), f"{col} still contains NaN after transform"
