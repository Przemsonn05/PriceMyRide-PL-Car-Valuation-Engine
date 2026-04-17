"""src/features.py — Feature engineering.

Two APIs are exposed:

1. **Legacy functions** ``engineer_base_features`` and
   ``apply_advanced_transformations`` — kept for backward compatibility
   with the existing main.py pipeline and the notebook.

2. **``FeatureEngineeringTransformer``** — a scikit-learn–compatible
   transformer that encapsulates *both* steps above.  It is picklable
   and fits brand-frequency maps on the training data, eliminating the
   inference-time feature mismatch that existed between the Streamlit
   app and the trained model.

All brand/tier constants come from :mod:`src.config` — this file never
redefines them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    PowerTransformer,
    StandardScaler,
)

from .config import (
    BRAND_FREQUENCY_FALLBACK,
    IS_PREMIUM_BRANDS,
    get_age_category,
    get_brand_popularity,
    get_brand_tier,
    get_performance_category,
    get_usage_category,
)

# ---------------------------------------------------------------------------
# Stateless feature creation (pure function of a single row).
# ---------------------------------------------------------------------------


def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create row-wise features that do not require fitting.

    Covers age / usage / performance / premium / collector / num_features /
    listing_year — everything that depends only on the current row.
    """
    df = df.copy()

    if "Vehicle_age" not in df.columns and "Production_year" in df.columns:
        df["Vehicle_age"] = datetime.now().year - df["Production_year"]

    if "Vehicle_age" in df.columns:
        df["Age_category"] = df["Vehicle_age"].apply(get_age_category)
        df["Is_new_car"] = (df["Vehicle_age"] < 3).astype("Int64")
        df["Is_old_car"] = (df["Vehicle_age"] > 16).astype("Int64")
        df["Is_collector"] = (df["Vehicle_age"] > 25).astype("Int64")

    if "Mileage_km" in df.columns and "Vehicle_age" in df.columns:
        df["Mileage_per_year"] = df["Mileage_km"] / df["Vehicle_age"].replace(0, 1)
        df["Usage_intensity"] = df["Mileage_per_year"].apply(get_usage_category)

    if "Power_HP" in df.columns and "Displacement_cm3" in df.columns:
        displacement_safe = df["Displacement_cm3"].replace(0, 100)
        df["HP_per_liter"] = df["Power_HP"] / (displacement_safe / 1000)
        df["HP_per_liter"] = df["HP_per_liter"].replace([np.inf, -np.inf], np.nan)
        df["Performance_category"] = df["HP_per_liter"].apply(get_performance_category)

    if "Vehicle_brand" in df.columns:
        df["Is_premium"] = (
            df["Vehicle_brand"].astype(str).str.lower().str.strip()
            .isin(IS_PREMIUM_BRANDS).astype("Int64")
        )

    if "Power_HP" in df.columns and "Is_premium" in df.columns:
        df["Is_supercar"] = (
            (df["Power_HP"] > 500) & (df["Is_premium"] == 1)
        ).astype("Int64")

    if "Features" in df.columns:
        df["Num_features"] = (
            df["Features"].fillna("").apply(
                lambda x: len([f for f in str(x).split(",") if f.strip()])
            )
        )

    if "Offer_publication_date" in df.columns:
        df["Listing_year"] = (
            pd.to_datetime(df["Offer_publication_date"], errors="coerce")
            .dt.year.fillna(datetime.now().year).astype(int)
        )

    cols_to_drop = [
        "Vehicle_generation", "Production_year", "Index", "Offer_publication_date",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


# ---------------------------------------------------------------------------
# Legacy function — kept for notebook / existing main.py pipeline.
# New code should use ``FeatureEngineeringTransformer`` instead.
# ---------------------------------------------------------------------------


def apply_advanced_transformations(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit-on-train, apply-to-both advanced transformations.

    **Deprecated for new pipelines** — use
    :class:`FeatureEngineeringTransformer` which bundles the same logic
    into a picklable sklearn transformer.  Kept here so existing notebook
    / main.py code keeps working.
    """
    transformer = FeatureEngineeringTransformer(run_base_features=False)
    X_train_tf = transformer.fit_transform(X_train)
    X_test_tf = transformer.transform(X_test)
    return X_train_tf, X_test_tf


# ---------------------------------------------------------------------------
# sklearn-compatible transformer — picklable, self-contained.
# ---------------------------------------------------------------------------


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """End-to-end feature engineering as a single sklearn transformer.

    On ``fit``:
        * computes ``brand_freq_`` and ``brand_model_freq_`` maps from the
          training data,
        * stores medians / modes of numeric and categorical columns for
          imputation at transform time.

    On ``transform``:
        * applies stateless feature creation (age / usage / performance / …)
          via :func:`engineer_base_features`,
        * imputes numeric / categorical columns using the fitted statistics,
        * creates log, squared, and interaction features,
        * creates brand-level features (``Brand_frequency``,
          ``Brand_tier``, ``Rarity_index``, ``BrandModel_frequency``,
          ``Brand_popularity``) using the fitted maps.

    Because all state is stored as ``self._x`` attributes, the transformer
    is fully picklable — no external constants are required at inference.

    Parameters
    ----------
    run_base_features : bool, default True
        If True, run :func:`engineer_base_features` as the first step.
        Set to False when the caller has already called that function
        (used by the legacy :func:`apply_advanced_transformations` wrapper).
    """

    _NUMERIC_IMPUTE_COLS = (
        "Mileage_km", "Power_HP", "Displacement_cm3", "Doors_number", "Vehicle_age",
    )
    _CATEGORICAL_IMPUTE_COLS = ("Drive", "Type", "Transmission")
    _LOG_COLS = ("Mileage_km", "Power_HP", "Displacement_cm3")

    def __init__(self, run_base_features: bool = True):
        self.run_base_features = run_base_features

    # -- sklearn API --------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if self.run_base_features:
            X = engineer_base_features(X)

        self.numeric_fill_: dict[str, float] = {}
        for col in self._NUMERIC_IMPUTE_COLS:
            if col in X.columns:
                if col == "Doors_number":
                    mode_result = X[col].mode()
                    fill = float(mode_result.iloc[0]) if not mode_result.empty else 4.0
                else:
                    fill = X[col].median()
                if pd.isna(fill):
                    fill = 0.0
                self.numeric_fill_[col] = float(fill)

        self.categorical_fill_: dict[str, str] = {}
        for col in self._CATEGORICAL_IMPUTE_COLS:
            if col in X.columns:
                mode_result = X[col].mode()
                self.categorical_fill_[col] = (
                    str(mode_result.iloc[0]) if not mode_result.empty else "Unknown"
                )

        if "Vehicle_brand" in X.columns:
            brand_col = X["Vehicle_brand"].astype(str).str.lower().str.strip()
            self.brand_freq_: dict[str, int] = brand_col.value_counts().to_dict()
            self.max_brand_freq_: int = max(self.brand_freq_.values()) if self.brand_freq_ else 1

            if "Vehicle_model" in X.columns:
                model_col = (
                    brand_col + "_"
                    + X["Vehicle_model"].astype(str).str.lower().str.strip()
                )
                self.brand_model_freq_: dict[str, int] = model_col.value_counts().to_dict()
            else:
                self.brand_model_freq_ = {}
        else:
            self.brand_freq_ = {}
            self.max_brand_freq_ = 1
            self.brand_model_freq_ = {}

        self.feature_names_out_: list[str] | None = None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.run_base_features:
            X = engineer_base_features(X)

        # -- Impute numeric cols --------------------------------------------
        for col, fill in self.numeric_fill_.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill)

        # -- Recompute features that depend on imputed values ---------------
        if "Power_HP" in X.columns and "Displacement_cm3" in X.columns:
            displacement_safe = X["Displacement_cm3"].replace(0, 100)
            X["HP_per_liter"] = X["Power_HP"] / (displacement_safe / 1000)
            X["HP_per_liter"] = X["HP_per_liter"].replace([np.inf, -np.inf], np.nan)

        if "Mileage_km" in X.columns and "Vehicle_age" in X.columns:
            X["Mileage_per_year"] = X["Mileage_km"] / X["Vehicle_age"].replace(0, 1)

        if "Is_premium" in X.columns and "Power_HP" in X.columns:
            X["Is_supercar"] = (
                (X["Power_HP"] > 500) & (X["Is_premium"] == 1)
            ).astype("Int64")

        # -- Log transforms -------------------------------------------------
        for col in self._LOG_COLS:
            if col in X.columns:
                X[f"{col}_log"] = np.log1p(X[col].clip(lower=0))

        # -- Impute categorical cols ----------------------------------------
        for col, fill in self.categorical_fill_.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill)

        # -- Polynomial features --------------------------------------------
        for col, new_name in (
            ("Vehicle_age", "Vehicle_age_squared"),
            ("Power_HP", "Power_HP_squared"),
            ("Mileage_km", "Mileage_km_squared"),
        ):
            if col in X.columns:
                X[new_name] = X[col].fillna(0) ** 2

        # -- Interaction terms ----------------------------------------------
        if "Vehicle_age" in X.columns and "Mileage_km" in X.columns:
            X["Age_Mileage_interaction"] = (
                X["Vehicle_age"].fillna(0) * X["Mileage_km"].fillna(0)
            )
        if "Power_HP" in X.columns and "Vehicle_age" in X.columns:
            X["Power_Age_interaction"] = (
                X["Power_HP"].fillna(0) * X["Vehicle_age"].fillna(0)
            )
        if "Mileage_per_year" in X.columns and "Vehicle_age" in X.columns:
            X["Mileage_per_year_Age"] = (
                X["Mileage_per_year"].fillna(0) * X["Vehicle_age"].fillna(0)
            )

        # -- Brand-level market features ------------------------------------
        if "Vehicle_brand" in X.columns:
            brand_lower = X["Vehicle_brand"].astype(str).str.lower().str.strip()
            X["Brand_tier"] = brand_lower.apply(get_brand_tier)
            X["Brand_frequency"] = (
                brand_lower.map(self.brand_freq_).fillna(1).astype(int)
            )

            raw_rarity = np.log1p(self.max_brand_freq_ / X["Brand_frequency"].clip(lower=1))
            max_rarity = np.log1p(self.max_brand_freq_)
            X["Rarity_index"] = (raw_rarity / max_rarity).clip(upper=1.0).round(4)

            if "Vehicle_model" in X.columns:
                model_col = (
                    brand_lower + "_"
                    + X["Vehicle_model"].astype(str).str.lower().str.strip()
                )
                X["BrandModel_frequency"] = (
                    model_col.map(self.brand_model_freq_).fillna(1).astype(int)
                )
            else:
                X["BrandModel_frequency"] = 1

            X["Brand_popularity"] = X["Brand_frequency"].apply(get_brand_popularity)

        if self.feature_names_out_ is None:
            self.feature_names_out_ = list(X.columns)

        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_ or [])


# ---------------------------------------------------------------------------
# Preprocessors (column-transformers) used by different models.
# ---------------------------------------------------------------------------


def get_preprocessor_tree() -> ColumnTransformer:
    """Preprocessor for tree-based models: median impute + ordinal encode."""
    num_pipeline = SimpleImputer(strategy="median")
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, make_column_selector(dtype_include="number")),
        ("cat", cat_pipeline, make_column_selector(dtype_include=["object", "category"])),
    ])


def get_preprocessor_mastered(smoothing: int = 200) -> ColumnTransformer:
    """Preprocessor for linear models: yeo-johnson + poly + target encoding."""
    num_cols = ["Mileage_km", "Power_HP", "Displacement_cm3", "Vehicle_age"]
    cat_cols_to_encode = ["Vehicle_brand", "Vehicle_model"]
    cat_cols_simple = ["Fuel_type", "Transmission", "Drive", "Type"]

    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("yeo", PowerTransformer(method="yeo-johnson")),
        ("poly", PolynomialFeatures(degree=2, interaction_only=True)),
    ])

    return ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("target", TargetEncoder(smoothing=smoothing), cat_cols_to_encode),
        ("cat_simple", OneHotEncoder(handle_unknown="ignore"), cat_cols_simple),
    ])


def get_preprocessor_v2(smoothing: int = 300) -> ColumnTransformer:
    """Preprocessor for the tuned XGBoost with brand-level numeric features."""
    num_cols_v2 = [
        "Mileage_km", "Power_HP", "Displacement_cm3", "Vehicle_age",
        "Brand_frequency", "Rarity_index", "BrandModel_frequency",
    ]
    cat_cols_encode = ["Vehicle_brand", "Vehicle_model"]
    cat_cols_ohe = [
        "Fuel_type", "Transmission", "Drive", "Type",
        "Brand_tier", "Brand_popularity",
    ]

    return ColumnTransformer([
        ("num", StandardScaler(), num_cols_v2),
        ("target", TargetEncoder(smoothing=smoothing), cat_cols_encode),
        ("cat_simple", OneHotEncoder(handle_unknown="ignore"), cat_cols_ohe),
    ])
