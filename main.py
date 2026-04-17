"""Main pipeline for Car Price Prediction project.

This script runs the complete ML pipeline:
    1. Data loading and preprocessing
    2. Exploratory Data Analysis (EDA)
    3. Feature engineering + train/test split
    4. Model training (Ridge, Random Forest, XGBoost, Weighted XGBoost)
    5. Model evaluation + cross-validation
    6. Evaluation visualizations
    7. Saving individual experiment models
    8. Training + saving the self-contained production pipeline
       (``final_car_price_model.joblib`` — this is the artifact the
       Streamlit app loads)

Usage::

    python main.py --mode full                  # end-to-end
    python main.py --mode test                  # smoke tests
    python main.py --mode train                 # training only
    python main.py --mode visualize             # EDA plots only
    python main.py --mode evaluate              # re-score saved models
    python main.py --mode export                # upload to HuggingFace
    python main.py --mode production            # build final_car_price_model.joblib
    python main.py --mode update                # incremental data fetch
    python main.py --mode collect               # stratified fresh dataset
"""

import sys
from pathlib import Path

# Ensure src/ is importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import warnings
from datetime import datetime
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_raw_data, save_processed_data
from src.evaluation import (
    calculate_metrics,
    create_model_comparison_plot,
    cross_validate_model,
    format_cv_summary,
    plot_learning_curves,
    plot_mape_by_brand,
    plot_regression_diagnostics,
    plot_residuals_vs_age,
    plot_ridge_coefficients,
    plot_tree_feature_importance,
)
from src.features import (
    apply_advanced_transformations,
    engineer_base_features,
    get_preprocessor_mastered,
    get_preprocessor_tree,
)
from src.models import (
    build_production_pipeline,
    get_log_transformed_target,
    get_predictions,
    train_random_forest_search,
    train_ridge_grid_search,
    train_xgboost_optuna,
    train_xgboost_weighted,
)
from src.preprocessing import clean_car_data
from src.utils import upload_models_to_hf
from src.visualization import (
    plot_correlation_heatmap,
    plot_depreciation_analysis,
    plot_fuel_type_trends,
    plot_mileage_vs_price_by_age,
    plot_numerical_relationships,
    plot_price_distribution,
)

# Limit warning-silencing to noisy third-party categories only, instead of
# the previous global `warnings.filterwarnings('ignore')` which hid real bugs.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class Config:
    """Project configuration."""

    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_PATH = DATA_DIR / "Car_sale_ads.csv"
    BALANCED_DATA_PATH = DATA_DIR / "Car_sale_ads_balanced.csv"
    SAMPLE_DATA_PATH = DATA_DIR / "sample" / "Car_sale_ads_balanced_sample.csv"
    PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cars_cleaned.csv"
    MODELS_DIR = PROJECT_ROOT / "models"
    IMAGES_DIR = PROJECT_ROOT / "images"
    REPORTS_DIR = PROJECT_ROOT / "reports"

    #: Name expected by app.py / Hugging Face.
    PRODUCTION_MODEL_NAME = "final_car_price_model.joblib"

    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    RIDGE_CV_FOLDS = 5
    RF_N_ITER = 12
    RF_CV_FOLDS = 3
    XGB_N_TRIALS = 50
    XGB_CV_FOLDS = 3

    HF_REPO_ID = "Przemsonn/poland-car-price-model"

    def __init__(self):
        for directory in [
            self.DATA_DIR / "processed",
            self.MODELS_DIR,
            self.IMAGES_DIR,
            self.REPORTS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


config = Config()


def print_header(text: str, char: str = "=") -> None:
    print("\n" + char * 80)
    print(text.center(80))
    print(char * 80 + "\n")


def print_step(step_num: int, total_steps: int, description: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'=' * 80}\n")


def save_metrics_report(
    metrics_dict: dict,
    model_comparison: dict,
    save_path: Path,
    cv_summary: Optional[dict] = None,
    best_model_name: Optional[str] = None,
) -> None:
    """Write the evaluation summary report to ``save_path``."""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CAR PRICE PREDICTION - MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for model_name, metrics in metrics_dict.items():
            f.write(f"\n{'-' * 80}\n{model_name}\n{'-' * 80}\n")

            for label, m in (("Training Set", metrics["train"]), ("Test Set", metrics["test"])):
                f.write(f"\n{label}:\n")
                f.write(f"  R^2 Score : {m['R2']:.4f}\n")
                f.write(f"  RMSE      : {m['RMSE']:,.2f} PLN\n")
                f.write(f"  MAE       : {m['MAE']:,.2f} PLN\n")
                f.write(f"  MAPE      : {m['MAPE'] * 100:.2f}%\n")
                f.write(f"  MdAPE     : {m['MdAPE'] * 100:.2f}%\n")

        f.write(f"\n{'=' * 80}\nMODEL COMPARISON (Test Set)\n{'=' * 80}\n\n")
        comparison_df = pd.DataFrame(model_comparison).T
        f.write(comparison_df.to_string())
        f.write("\n")

        if best_model_name:
            f.write(f"\nBest model (by test R^2): {best_model_name}\n")

        if cv_summary is not None:
            f.write(f"\n{'=' * 80}\n{format_cv_summary(cv_summary)}\n{'=' * 80}\n")

    print(f"Metrics report saved: {save_path}")


# ---------------------------------------------------------------------------
# Pipeline steps.
# ---------------------------------------------------------------------------


def step_1_load_and_clean_data() -> pd.DataFrame:
    """Step 1: Load and clean the training dataset.

    The scraped balanced dataset is the primary source.  Pre-2022 legacy
    data is never merged with current listings — prices are not comparable
    across the 2022–2024 market shift.
    """
    print_step(1, 8, "DATA LOADING & PREPROCESSING")

    if config.BALANCED_DATA_PATH.exists():
        print(f"Loading scraped dataset from {config.BALANCED_DATA_PATH.name} ...")
        df_clean = pd.read_csv(config.BALANCED_DATA_PATH)
        print(f"Loaded {len(df_clean):,} rows, {len(df_clean.columns)} columns")
    elif config.SAMPLE_DATA_PATH.exists():
        print(
            f"Full dataset not found — loading the committed 1,000-row sample\n"
            f"  from {config.SAMPLE_DATA_PATH.relative_to(config.PROJECT_ROOT)}.\n"
            "  (This is the dataset used in CI / quick smoke runs. To train a"
            " production-quality model, run `python main.py --mode collect`.)"
        )
        df_clean = pd.read_csv(config.SAMPLE_DATA_PATH)
    else:
        print(f"Scraped dataset not found. Falling back to {config.RAW_DATA_PATH.name} ...")
        print("  TIP: python main.py --mode collect --target-rows 200000")
        df_raw = load_raw_data(config.RAW_DATA_PATH)
        print(f"Loaded {len(df_raw):,} rows")
        print("\nCleaning data...")
        df_clean = clean_car_data(df_raw)
        print(f"After cleaning: {len(df_clean):,} rows ({len(df_raw) - len(df_clean):,} removed)")

    save_processed_data(df_clean, config.PROCESSED_DATA_PATH)

    print("\nData Summary:")
    print(f"Price range:    {df_clean['price_PLN'].min():,.0f} - {df_clean['price_PLN'].max():,.0f} PLN")
    print(f"Median price:   {df_clean['price_PLN'].median():,.0f} PLN")
    if "Production_year" in df_clean.columns:
        print(f"Year range:     {int(df_clean['Production_year'].min())} - {int(df_clean['Production_year'].max())}")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")
    return df_clean


def step_2_exploratory_analysis(df: pd.DataFrame) -> None:
    """Step 2: EDA plots."""
    print_step(2, 8, "EXPLORATORY DATA ANALYSIS")

    if "Production_year" in df.columns and "Vehicle_age" not in df.columns:
        df = df.copy()
        df["Vehicle_age"] = datetime.now().year - df["Production_year"]

    plots = [
        ("1/6 Price distribution",     plot_price_distribution,       "eda_price_distribution.png"),
        ("2/6 Depreciation analysis",  plot_depreciation_analysis,    "eda_depreciation_analysis.png"),
        ("3/6 Feature relationships",  plot_numerical_relationships,  "eda_numerical_relationships.png"),
        ("4/6 Mileage analysis",       plot_mileage_vs_price_by_age,  "eda_mileage_by_age.png"),
        ("5/6 Fuel type trends",       plot_fuel_type_trends,         "eda_fuel_trends.png"),
        ("6/6 Correlation heatmap",    plot_correlation_heatmap,      "eda_correlation_heatmap.png"),
    ]
    for label, fn, filename in plots:
        print(label)
        fig = fn(df, save_path=config.IMAGES_DIR / filename)
        plt.close(fig)

    print(f"\nEDA plots saved to: {config.IMAGES_DIR}")


def _report_nans(df: pd.DataFrame, stage: str = "") -> None:
    """Log how many NaNs remain in each critical column."""
    key_cols = ["Vehicle_age", "Mileage_km", "Power_HP", "Displacement_cm3", "Vehicle_brand", "price_PLN"]
    print(f"\nDEBUG: NaN check ({stage})")
    for col in key_cols:
        if col in df.columns:
            n = df[col].isnull().sum()
            if n > 0:
                print(f"  {col:<20}: {n:,}  ({n / len(df) * 100:.2f}%)")


def step_3_feature_engineering(df: pd.DataFrame) -> tuple:
    """Step 3: Feature engineering + train/test split.

    Important: the advanced transformations are fit on training data only;
    the exact same imputations are then applied to the test set — this
    prevents the data-leakage pattern that was present in earlier versions
    of this pipeline (which filled test NaNs with post-hoc medians).
    """
    print_step(3, 8, "FEATURE ENGINEERING")

    _report_nans(df, "before feature engineering")

    print("\nCreating base features...")
    df_features = engineer_base_features(df)
    print(f"Created {len(df_features.columns) - len(df.columns)} new features")

    if df_features["price_PLN"].isnull().any():
        before = len(df_features)
        df_features = df_features[df_features["price_PLN"].notna()]
        print(f"Dropped {before - len(df_features)} rows with missing target")

    from sklearn.model_selection import train_test_split

    X = df_features.drop("price_PLN", axis=1)
    y = df_features["price_PLN"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
    )
    print(f"\nTrain set: {len(X_train):,}  |  Test set: {len(X_test):,}")

    print("\nApplying advanced transformations (fit-on-train)...")
    X_train, X_test = apply_advanced_transformations(X_train, X_test)

    remaining_train = X_train.isnull().sum().sum()
    remaining_test = X_test.isnull().sum().sum()
    print(f"Final feature count: {X_train.shape[1]}")
    print(f"Remaining NaNs — Train: {remaining_train}  Test: {remaining_test}")

    y_log = get_log_transformed_target(y_train, y_test)
    return X_train, X_test, y_train, y_test, y_log["y_train_log"], y_log["y_test_log"]


def step_4_train_models(X_train, y_train, y_train_log) -> dict:
    """Step 4: Train all four model variants."""
    print_step(4, 8, "MODEL TRAINING")
    models = {}

    print_header("Model 1: Ridge Regression", "-")
    models["Ridge"] = train_ridge_grid_search(X_train, y_train_log, n_folds=config.RIDGE_CV_FOLDS)

    print_header("Model 2: Random Forest", "-")
    models["RandomForest"] = train_random_forest_search(
        X_train, y_train, n_iter=config.RF_N_ITER, n_folds=config.RF_CV_FOLDS,
    )

    print_header("Model 3: XGBoost (Optuna)", "-")
    models["XGBoost_Base"] = train_xgboost_optuna(
        X_train, y_train_log, n_trials=config.XGB_N_TRIALS, n_folds=config.XGB_CV_FOLDS,
    )

    print_header("Model 4: XGBoost (Weighted, KFold-Optuna)", "-")
    models["XGBoost_Weighted"] = train_xgboost_weighted(
        X_train, y_train_log, n_trials=config.XGB_N_TRIALS, n_folds=config.XGB_CV_FOLDS,
    )
    return models


def step_5_evaluate_models(models, X_train, X_test, y_train, y_test) -> tuple:
    """Step 5: Evaluate all trained models."""
    print_step(5, 8, "MODEL EVALUATION")

    metrics_dict, model_comparison = {}, {}
    model_configs = [
        ("Ridge",             True),
        ("RandomForest",      False),
        ("XGBoost_Base",      True),
        ("XGBoost_Weighted",  True),
    ]
    for model_name, log_transformed in model_configs:
        print(f"\nEvaluating {model_name}...")
        preds = get_predictions(models[model_name], X_train, X_test, log_transformed)
        train_m = calculate_metrics(y_train, preds["y_train_pred"])
        test_m = calculate_metrics(y_test, preds["y_test_pred"])
        metrics_dict[model_name] = {"train": train_m, "test": test_m, "predictions": preds}
        model_comparison[model_name] = test_m
        print(
            f"  Test  R^2={test_m['R2']:.4f}  "
            f"MAPE={test_m['MAPE'] * 100:.2f}%  MdAPE={test_m['MdAPE'] * 100:.2f}%  "
            f"MAE={test_m['MAE']:,.0f} PLN"
        )

    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Test Set)")
    print("=" * 80)
    comparison_df = pd.DataFrame(model_comparison).T
    print(comparison_df.to_string())

    best_model = comparison_df["R2"].idxmax()
    print(f"\nBest model: {best_model} (R^2 = {comparison_df.loc[best_model, 'R2']:.4f})")
    return metrics_dict, model_comparison, best_model


def step_6_generate_visualizations(
    models, metrics_dict, model_comparison,
    X_train, X_test, y_train, y_test, y_train_log,
) -> None:
    """Step 6: Evaluation plots."""
    print_step(6, 8, "GENERATING EVALUATION VISUALIZATIONS")

    print("1/8 Model comparison...")
    fig = create_model_comparison_plot(model_comparison, save_path=config.IMAGES_DIR / "eval_model_comparison.png")
    plt.close(fig)

    print("2/8 Ridge diagnostics...")
    fig = plot_regression_diagnostics(
        y_test, metrics_dict["Ridge"]["predictions"]["y_test_pred"],
        "Ridge Regression", save_path=config.IMAGES_DIR / "eval_ridge_diagnostics.png",
    )
    plt.close(fig)

    print("3/8 Ridge coefficients...")
    fig = plot_ridge_coefficients(models["Ridge"], top_n=15, save_path=config.IMAGES_DIR / "eval_ridge_coefficients.png")
    plt.close(fig)

    print("4/8 Random Forest feature importance...")
    fig = plot_tree_feature_importance(models["RandomForest"], top_n=20, save_path=config.IMAGES_DIR / "eval_rf_feature_importance.png")
    plt.close(fig)

    print("5/8 XGBoost diagnostics...")
    xgb_preds = metrics_dict["XGBoost_Weighted"]["predictions"]["y_test_pred"]
    fig = plot_regression_diagnostics(
        y_test, xgb_preds, "XGBoost (Weighted)",
        save_path=config.IMAGES_DIR / "eval_xgb_diagnostics.png",
    )
    plt.close(fig)

    print("6/8 XGBoost feature importance...")
    fig = plot_tree_feature_importance(models["XGBoost_Weighted"], top_n=20, save_path=config.IMAGES_DIR / "eval_xgb_feature_importance.png")
    plt.close(fig)

    print("7/8 Learning curves...")
    fig = plot_learning_curves(
        models["XGBoost_Weighted"], X_train, y_train_log, cv=3,
        title="Learning Curves - XGBoost (Weighted)",
        save_path=config.IMAGES_DIR / "eval_learning_curves.png",
    )
    plt.close(fig)

    print("8/8 Residuals vs age...")
    fig = plot_residuals_vs_age(X_test, y_test, xgb_preds, save_path=config.IMAGES_DIR / "eval_residuals_vs_age.png")
    plt.close(fig)

    print(f"\nEvaluation plots saved to: {config.IMAGES_DIR}")


def step_7_save_models(models: dict, upload: Optional[bool] = None) -> None:
    """Step 7: Save experiment models as individual .pkl files."""
    print_step(7, 8, "SAVING EXPERIMENT MODELS")

    for name, model in models.items():
        fp = config.MODELS_DIR / f"{name.lower()}_model.pkl"
        joblib.dump(model, fp)
        print(f"Saved: {fp.name}")

    print(f"\nExperiment models saved to: {config.MODELS_DIR}")

    do_upload = upload
    if do_upload is None:
        try:
            do_upload = input("\nUpload experiment models to Hugging Face? (y/n): ").lower() == "y"
        except EOFError:
            do_upload = False
    if not do_upload:
        return

    try:
        print("\nUploading to Hugging Face...")
        upload_models_to_hf(
            {f"{name.lower()}_model.pkl": model for name, model in models.items()},
            config.HF_REPO_ID,
        )
        print("Models uploaded.")
    except Exception as e:
        print(f"Upload failed: {e}")


def step_8_train_production_model(df_clean: pd.DataFrame, upload: Optional[bool] = None) -> Path:
    """Step 8: Train and save the self-contained production pipeline.

    The production pipeline is ``FeatureEngineering → preprocessor → XGBoost``
    wrapped in a single ``sklearn.pipeline.Pipeline`` so it can be loaded
    and called on **raw** feature rows (no duplicated feature code in the
    serving layer).  Saved as ``models/final_car_price_model.joblib`` and
    optionally uploaded to Hugging Face.
    """
    print_step(8, 8, "BUILDING PRODUCTION MODEL")

    from sklearn.model_selection import train_test_split

    df = df_clean.copy()
    df = df[df["price_PLN"].notna()]
    X = df.drop("price_PLN", axis=1)
    y = df["price_PLN"]
    y_log = np.log1p(y)

    X_train, X_test, y_log_train, y_log_test = train_test_split(
        X, y_log, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
    )
    _, _, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
    )

    pipeline = build_production_pipeline()
    print("Fitting production pipeline (features → preprocessor → XGBoost)...")
    pipeline.fit(X_train, y_log_train)

    preds_pln = np.expm1(pipeline.predict(X_test))
    test_metrics = calculate_metrics(y_test, preds_pln)
    print(
        f"Holdout  R^2={test_metrics['R2']:.4f}  "
        f"MAPE={test_metrics['MAPE'] * 100:.2f}%  "
        f"MdAPE={test_metrics['MdAPE'] * 100:.2f}%  "
        f"MAE={test_metrics['MAE']:,.0f} PLN"
    )

    print("\nRunning 5-fold CV for 95% confidence interval...")
    cv_summary = cross_validate_model(
        build_production_pipeline(),
        X, y_log, n_folds=config.CV_FOLDS, log_transformed=True,
    )

    output_path = config.MODELS_DIR / config.PRODUCTION_MODEL_NAME
    artifact = {
        "pipeline": pipeline,
        "test_metrics": test_metrics,
        "cv_summary": cv_summary,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "schema": list(X.columns),
    }
    joblib.dump(artifact, output_path)
    print(f"\nSaved production model: {output_path}")

    do_upload = upload
    if do_upload is None:
        try:
            do_upload = input("\nUpload final_car_price_model.joblib to Hugging Face? (y/n): ").lower() == "y"
        except EOFError:
            do_upload = False
    if do_upload:
        try:
            upload_models_to_hf({config.PRODUCTION_MODEL_NAME: artifact}, config.HF_REPO_ID)
            print("Uploaded production model.")
        except Exception as e:
            print(f"Upload failed: {e}")

    return output_path


# ---------------------------------------------------------------------------
# Smoke tests.
# ---------------------------------------------------------------------------


def run_tests() -> bool:
    print_header("RUNNING PROJECT SMOKE TESTS")
    ok = True

    test_path = (
        config.PROCESSED_DATA_PATH if config.PROCESSED_DATA_PATH.exists()
        else config.BALANCED_DATA_PATH if config.BALANCED_DATA_PATH.exists()
        else config.SAMPLE_DATA_PATH if config.SAMPLE_DATA_PATH.exists()
        else config.RAW_DATA_PATH
    )
    scraped = test_path != config.RAW_DATA_PATH
    print(f"Using dataset: {test_path.name}\n")

    def run(name: str, fn):
        nonlocal ok
        try:
            fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            ok = False

    def _t1():
        df = pd.read_csv(test_path) if scraped else load_raw_data(test_path)
        assert len(df) > 0 and "price_PLN" in df.columns
    run("Data loading", _t1)

    def _t2():
        df = pd.read_csv(test_path) if scraped else clean_car_data(load_raw_data(test_path))
        assert "price_PLN" in df.columns and len(df) > 0
    run("Data preprocessing", _t2)

    def _t3():
        df = pd.read_csv(test_path) if scraped else clean_car_data(load_raw_data(test_path))
        df_f = engineer_base_features(df)
        for feat in ("Age_category", "Is_new_car", "Is_old_car", "Mileage_per_year", "HP_per_liter", "Is_premium"):
            assert feat in df_f.columns
    run("Feature engineering", _t3)

    def _t4():
        assert get_preprocessor_tree() is not None
        assert get_preprocessor_mastered() is not None
    run("Preprocessors", _t4)

    def _t5():
        df = pd.read_csv(test_path) if scraped else clean_car_data(load_raw_data(test_path))
        if "Vehicle_age" not in df.columns and "Production_year" in df.columns:
            df["Vehicle_age"] = datetime.now().year - df["Production_year"]
        fig = plot_price_distribution(df); plt.close(fig)
        fig = plot_depreciation_analysis(df); plt.close(fig)
    run("Visualization functions", _t5)

    def _t6():
        from src.features import FeatureEngineeringTransformer
        assert FeatureEngineeringTransformer is not None
    run("Production pipeline import", _t6)

    print("\n" + ("ALL TESTS PASSED" if ok else "SOME TESTS FAILED"))
    return ok


def run_full_pipeline() -> None:
    print_header("CAR PRICE PREDICTION — COMPLETE PIPELINE")
    start = datetime.now()

    df_clean = step_1_load_and_clean_data()
    step_2_exploratory_analysis(df_clean)
    X_train, X_test, y_train, y_test, y_train_log, _ = step_3_feature_engineering(df_clean)
    models = step_4_train_models(X_train, y_train, y_train_log)

    metrics_dict, model_comparison, best_model = step_5_evaluate_models(
        models, X_train, X_test, y_train, y_test,
    )
    step_6_generate_visualizations(
        models, metrics_dict, model_comparison,
        X_train, X_test, y_train, y_test, y_train_log,
    )
    step_7_save_models(models, upload=False)
    step_8_train_production_model(df_clean, upload=False)

    # Persist a CV report for the production pipeline alongside per-model metrics.
    cv_summary = cross_validate_model(
        build_production_pipeline(),
        df_clean.drop("price_PLN", axis=1),
        np.log1p(df_clean["price_PLN"]),
        n_folds=config.CV_FOLDS, log_transformed=True,
    )
    save_metrics_report(
        metrics_dict, model_comparison,
        config.REPORTS_DIR / "model_evaluation_report.txt",
        cv_summary=cv_summary,
        best_model_name=best_model,
    )

    duration = (datetime.now() - start).total_seconds()
    print_header("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total execution time: {duration / 60:.1f} minutes")
    print(f"  Models:   {config.MODELS_DIR}")
    print(f"  Images:   {config.IMAGES_DIR}")
    print(f"  Reports:  {config.REPORTS_DIR}")


def run_data_collect(target_rows: int = 200_000, detail_mode: bool = False, mock: bool = False, output_path: Optional[Path] = None) -> None:
    from src.data_cleaning import apply_stratified_sampling, clean_data, validate_schema
    from src.data_fetcher import fetch_balanced_dataset

    if output_path is None:
        output_path = config.DATA_DIR / "Car_sale_ads_balanced.csv"

    print_header("STRATIFIED DATA COLLECTION")
    print(f"Target rows:   {target_rows:,}")
    print(f"Detail mode:   {'Yes' if detail_mode else 'No'}")
    print(f"Mock mode:     {'Yes' if mock else 'No'}")
    print(f"Output path:   {output_path}\n")

    start = datetime.now()
    print("Step 1/3  Fetching stratified listings from Otomoto...")
    df_raw = fetch_balanced_dataset(target_rows=target_rows, detail_mode=detail_mode, mock=mock)
    print(f"  Raw rows fetched: {len(df_raw):,}")
    if "_category" in df_raw.columns:
        print(f"  Category breakdown:\n{df_raw['_category'].value_counts().to_string()}")

    print("\nStep 2/3  Cleaning and normalizing to project schema...")
    df_raw_no_cat = df_raw.drop(columns=["_category"], errors="ignore")
    df_clean = clean_data(df_raw_no_cat)
    print(f"  Rows after cleaning: {len(df_clean):,}")

    if "_category" in df_raw.columns:
        df_clean = df_clean.copy()
        df_clean["_category"] = df_raw["_category"].values[: len(df_clean)]

    print("\nStep 3/3  Applying stratified sampling...")
    df_balanced = apply_stratified_sampling(df_clean, target_rows=target_rows)
    df_balanced = df_balanced.drop(columns=["_category", "offer_id"], errors="ignore")
    print(f"  Final balanced rows: {len(df_balanced):,}")

    validation = validate_schema(df_balanced)
    if not validation["valid"]:
        print(f"  [WARN] Schema issues: {validation['issues']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(output_path, index=False)

    duration = (datetime.now() - start).total_seconds()
    print_header("COLLECTION COMPLETE")
    print(f"Balanced rows saved: {len(df_balanced):,}")
    print(f"Output file:         {output_path}")
    print(f"Duration:            {duration:.1f}s  ({duration / 60:.1f} min)")


def run_data_update(pages: int = 10, detail_mode: bool = False, mock: bool = False) -> None:
    from src.data_fetcher import fetch_incremental

    print_header("DATA UPDATE: FETCHING NEW LISTINGS FROM OTOMOTO")
    print(f"Pages to fetch: {pages}  (~{pages * 32} listings)")
    print(f"Detail mode:    {'Yes' if detail_mode else 'No'}")
    print(f"Mock mode:      {'Yes' if mock else 'No'}\n")

    start = datetime.now()
    new_rows = fetch_incremental(
        data_path=config.BALANCED_DATA_PATH,
        pages=pages, detail_mode=detail_mode, mock=mock,
    )

    duration = (datetime.now() - start).total_seconds()
    print_header("UPDATE COMPLETE")
    print(f"New rows appended: {len(new_rows):,}")
    print(f"Dataset path:      {config.BALANCED_DATA_PATH}")
    print(f"Duration:          {duration:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Car Price Prediction ML Pipeline")
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "test", "train", "visualize", "evaluate", "export", "production", "update", "collect"],
    )
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--target-rows", type=int, default=200_000)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--upload", action="store_true", help="Skip prompt, upload to HF after save")
    args = parser.parse_args()

    if args.mode == "collect":
        run_data_collect(target_rows=args.target_rows, detail_mode=args.detail, mock=args.mock)

    elif args.mode == "update":
        run_data_update(pages=args.pages, detail_mode=args.detail, mock=args.mock)

    elif args.mode == "test":
        sys.exit(0 if run_tests() else 1)

    elif args.mode == "full":
        run_full_pipeline()

    elif args.mode == "train":
        print_header("TRAINING MODELS ONLY")
        df_clean = step_1_load_and_clean_data()
        X_train, X_test, y_train, y_test, y_train_log, _ = step_3_feature_engineering(df_clean)
        models = step_4_train_models(X_train, y_train, y_train_log)
        step_7_save_models(models, upload=args.upload)

    elif args.mode == "visualize":
        print_header("GENERATING VISUALIZATIONS ONLY")
        vis_path = (
            config.PROCESSED_DATA_PATH if config.PROCESSED_DATA_PATH.exists()
            else config.BALANCED_DATA_PATH
        )
        step_2_exploratory_analysis(pd.read_csv(vis_path))

    elif args.mode == "evaluate":
        print_header("EVALUATION ONLY")
        df_clean = pd.read_csv(config.PROCESSED_DATA_PATH)
        X_train, X_test, y_train, y_test, y_train_log, _ = step_3_feature_engineering(df_clean)

        from src.utils import load_local_model
        models = {
            "Ridge":            load_local_model(config.MODELS_DIR / "ridge_model.pkl"),
            "RandomForest":     load_local_model(config.MODELS_DIR / "randomforest_model.pkl"),
            "XGBoost_Base":     load_local_model(config.MODELS_DIR / "xgboost_base_model.pkl"),
            "XGBoost_Weighted": load_local_model(config.MODELS_DIR / "xgboost_weighted_model.pkl"),
        }
        metrics_dict, model_comparison, best_model = step_5_evaluate_models(
            models, X_train, X_test, y_train, y_test,
        )
        step_6_generate_visualizations(
            models, metrics_dict, model_comparison,
            X_train, X_test, y_train, y_test, y_train_log,
        )
        save_metrics_report(
            metrics_dict, model_comparison,
            config.REPORTS_DIR / "model_evaluation_report.txt",
            best_model_name=best_model,
        )

    elif args.mode == "production":
        print_header("BUILD PRODUCTION MODEL ONLY")
        df_clean = step_1_load_and_clean_data()
        step_8_train_production_model(df_clean, upload=args.upload)

    elif args.mode == "export":
        print_header("EXPORT / UPLOAD EXISTING MODELS")
        from src.utils import load_local_model
        try:
            models = {
                "Ridge":            load_local_model(config.MODELS_DIR / "ridge_model.pkl"),
                "RandomForest":     load_local_model(config.MODELS_DIR / "randomforest_model.pkl"),
                "XGBoost_Base":     load_local_model(config.MODELS_DIR / "xgboost_base_model.pkl"),
                "XGBoost_Weighted": load_local_model(config.MODELS_DIR / "xgboost_weighted_model.pkl"),
            }
            step_7_save_models(models, upload=True)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
