"""src/models.py — Model training and optimization.

Includes:
* Ridge regression with GridSearchCV
* Random Forest with RandomizedSearchCV
* XGBoost with Optuna (Bayesian) — both un-weighted and sample-weighted
* A helper :func:`build_production_pipeline` that bundles feature engineering,
  preprocessing, and the XGBoost estimator into a single picklable Pipeline —
  this is what gets saved as ``final_car_price_model.joblib``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .config import (
    RANDOM_STATE,
    TIER_WEIGHT_MAP,
    get_brand_tier,
    print_if_verbose,
)
from .features import (
    FeatureEngineeringTransformer,
    get_preprocessor_mastered,
    get_preprocessor_tree,
    get_preprocessor_v2,
)


def get_log_transformed_target(
    y_train: pd.Series, y_test: pd.Series,
) -> Dict[str, pd.Series]:
    """Apply ``log1p`` to train/test targets to stabilise variance."""
    return {
        "y_train_log": np.log1p(y_train),
        "y_test_log": np.log1p(y_test),
    }


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """Inverse of ``log1p`` — map log-scale predictions back to PLN."""
    return np.expm1(y_log)


# ---------------------------------------------------------------------------
# Ridge / Random Forest / XGBoost training functions.
# ---------------------------------------------------------------------------


def train_ridge_grid_search(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    smoothing: int = 200,
    n_folds: int = 5,
) -> Pipeline:
    """Train Ridge with log target and grid-search CV."""
    print_if_verbose("\n" + "=" * 80)
    print_if_verbose("TRAINING RIDGE REGRESSION WITH GRIDSEARCH")
    print_if_verbose("=" * 80)

    preprocessor = get_preprocessor_mastered(smoothing=smoothing)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", Ridge())])

    param_grid = {
        "model__alpha": [0.1, 1.0, 5.0, 10.0, 100.0],
        "model__solver": ["auto", "cholesky", "lsqr"],
        "model__fit_intercept": [False, True],
    }
    scoring = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "neg_mse": "neg_mean_squared_error",
    }
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring=scoring,
        refit="neg_rmse", cv=kfold, n_jobs=-1, verbose=1,
    )
    print_if_verbose("Starting GridSearch...")
    grid.fit(X_train, y_train_log)
    print_if_verbose(f"\n[OK] Best parameters: {grid.best_params_}")
    print_if_verbose(f"[OK] Best CV RMSE: {-grid.best_score_:,.2f}")
    return grid.best_estimator_


def train_random_forest_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 12,
    n_folds: int = 3,
) -> Pipeline:
    """Train Random Forest with randomised search CV on raw-scale target."""
    print_if_verbose("\n" + "=" * 80)
    print_if_verbose("TRAINING RANDOM FOREST WITH RANDOMIZEDSEARCH")
    print_if_verbose("=" * 80)

    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    preprocessor = get_preprocessor_tree()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
    ])

    param_dist = {
        "model__n_estimators": [200, 500],
        "model__max_depth": [10, 15, 20],
        "model__min_samples_leaf": [5, 10, 20],
        "model__min_samples_split": [10, 20],
        "model__max_features": ["sqrt", 0.5],
        "model__max_samples": [0.7, 0.8],
        "model__bootstrap": [True],
    }
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipeline, param_distributions=param_dist, n_iter=n_iter,
        scoring="neg_mean_absolute_error", cv=kfold, n_jobs=-1, verbose=2,
        random_state=RANDOM_STATE,
    )
    print_if_verbose("Starting RandomizedSearch...")
    search.fit(X_train_clean, y_train)
    print_if_verbose(f"\n[OK] Best parameters: {search.best_params_}")
    print_if_verbose(f"[OK] Best CV MAE: {-search.best_score_:,.2f} PLN")
    return search.best_estimator_


def train_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    n_trials: int = 50,
    n_folds: int = 3,
) -> Pipeline:
    """Train XGBoost with Optuna (Bayesian) hyper-parameter search."""
    print_if_verbose("\n" + "=" * 80)
    print_if_verbose("TRAINING XGBOOST WITH OPTUNA OPTIMIZATION")
    print_if_verbose("=" * 80)

    preprocessor = get_preprocessor_tree()

    def objective(trial):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        params = {
            "n_estimators": 1000,
            "early_stopping_rounds": 30,
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        }

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]

            X_tr_proc = preprocessor.fit_transform(X_tr, y_tr)
            X_val_proc = preprocessor.transform(X_val)

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr_proc, y_tr, eval_set=[(X_val_proc, y_val)], verbose=False)
            preds = model.predict(X_val_proc)
            scores.append(np.sqrt(mean_squared_error(y_val, preds)))

            trial.report(scores[-1], step=len(scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(scores))

    print_if_verbose(f"Starting Optuna optimization with {n_trials} trials...")
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print_if_verbose(f"\n[OK] Best trial RMSE: {study.best_value:.4f}")
    print_if_verbose(f"[OK] Best parameters: {study.best_params}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb.XGBRegressor(
            **study.best_params, n_estimators=1000, random_state=RANDOM_STATE,
        )),
    ])
    pipeline.fit(X_train, y_train_log)
    return pipeline


# ---------------------------------------------------------------------------
# Sample weighting (brand-tier driven).
# ---------------------------------------------------------------------------


def calculate_sample_weights(X_train: pd.DataFrame) -> np.ndarray:
    """Assign higher weights to rare / luxury brand tiers."""
    brands = X_train["Vehicle_brand"].astype(str)
    tiers = brands.apply(get_brand_tier)
    weights = tiers.map(TIER_WEIGHT_MAP).fillna(2.0).values

    print_if_verbose("\n[OK] Sample weights calculated:")
    for tier, w in TIER_WEIGHT_MAP.items():
        n = (tiers == tier).sum()
        print_if_verbose(f"  {tier:<15}: weight={w:.1f}  n={n:,}")
    return weights


def train_xgboost_weighted(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    n_trials: int = 50,
    n_folds: int = 3,
) -> Pipeline:
    """Train XGBoost with brand-tier sample weights + Optuna CV.

    **Changed from the original single-split implementation:** this now uses
    ``KFold`` cross-validation inside the Optuna objective, matching the
    un-weighted ``train_xgboost_optuna`` and avoiding the noise bias caused
    by tuning against a single fixed validation split.
    """
    print_if_verbose("\n" + "=" * 80)
    print_if_verbose("TRAINING WEIGHTED XGBOOST (KFold-CV OPTUNA)")
    print_if_verbose("=" * 80)

    weights = calculate_sample_weights(X_train)
    preprocessor = get_preprocessor_v2(smoothing=300)

    def objective(trial):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        scores: list[float] = []
        params = {
            "n_estimators": 2000,
            "early_stopping_rounds": 50,
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.7),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 10),
            "gamma": trial.suggest_float("gamma", 0.1, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1.0, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        }

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
            w_tr, w_val = weights[train_idx], weights[val_idx]

            X_tr_tf = preprocessor.fit_transform(X_tr, y_tr)
            X_val_tf = preprocessor.transform(X_val)

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr_tf, y_tr,
                eval_set=[(X_val_tf, y_val)],
                sample_weight=w_tr,
                verbose=False,
            )
            preds = model.predict(X_val_tf)
            scores.append(
                np.sqrt(mean_squared_error(y_val, preds, sample_weight=w_val))
            )

            trial.report(scores[-1], step=len(scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(scores))

    print_if_verbose(f"Starting weighted Optuna optimization ({n_trials} trials, {n_folds}-fold CV)...")
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print_if_verbose(f"\n[OK] Best trial weighted RMSE: {study.best_value:.4f}")
    print_if_verbose(f"[OK] Best parameters: {study.best_params}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb.XGBRegressor(
            **study.best_params, n_estimators=2000, random_state=RANDOM_STATE,
        )),
    ])
    pipeline.fit(X_train, y_train_log, model__sample_weight=weights)
    return pipeline


# ---------------------------------------------------------------------------
# Production pipeline — bundles everything into a picklable artifact.
# ---------------------------------------------------------------------------


def build_production_pipeline(xgb_params: dict | None = None) -> Pipeline:
    """Return an unfitted production pipeline:
    ``FeatureEngineeringTransformer → tree-preprocessor → XGBRegressor``.

    The caller is responsible for calling ``.fit(X_raw, y_log)`` on the
    returned pipeline, then pickling it as ``final_car_price_model.joblib``.
    Because the FeatureEngineeringTransformer is fitted on the raw training
    data, the resulting artifact can be passed raw 14-column input rows
    directly — no duplicated feature-engineering logic on the serving side.
    """
    xgb_params = xgb_params or {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
    }
    return Pipeline([
        ("features", FeatureEngineeringTransformer(run_base_features=True)),
        ("preprocessor", get_preprocessor_tree()),
        ("model", xgb.XGBRegressor(**xgb_params)),
    ])


def get_predictions(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    log_transformed: bool = True,
) -> Dict[str, np.ndarray]:
    """Return train/test predictions on the original PLN scale."""
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan)

    y_train_pred = model.predict(X_train_clean)
    y_test_pred = model.predict(X_test_clean)

    if log_transformed:
        y_train_pred = inverse_log_transform(y_train_pred)
        y_test_pred = inverse_log_transform(y_test_pred)

    return {"y_train_pred": y_train_pred, "y_test_pred": y_test_pred}
