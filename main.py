"""
Main pipeline for Car Price Prediction project.

This script runs the complete ML pipeline:
1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature engineering
4. Model training (Ridge, Random Forest, XGBoost)
5. Model evaluation and comparison
6. Visualization generation
7. Model saving and export

Usage:
    python main.py --mode full              # Run complete pipeline
    python main.py --mode test              # Run tests only
    python main.py --mode train             # Train models only
    python main.py --mode visualize         # Generate visualizations only
    python main.py --mode evaluate          # Checkpoint between modes
    python main.py --mode export            # Load models to HuggingFace
    python main.py --mode update            # Fetch new listings (incremental)
    python main.py --mode collect           # Collect balanced stratified dataset
    python main.py --mode collect --target-rows 200000 --mock   # Mock test
"""

import sys
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data import load_raw_data, save_processed_data
from src.preprocessing import clean_car_data
from src.features import (
    engineer_base_features,
    apply_advanced_transformations,
    get_preprocessor_tree,
    get_preprocessor_mastered
)
from src.models import (
    get_log_transformed_target,
    train_ridge_grid_search,
    train_random_forest_search,
    train_xgboost_optuna,
    train_xgboost_weighted,
    get_predictions
)
from src.evaluation import (
    calculate_metrics,
    create_metrics_table,
    plot_regression_diagnostics,
    plot_learning_curves,
    plot_tree_feature_importance,
    plot_ridge_coefficients,
    plot_residuals_vs_age,
    plot_mape_by_brand,
    create_model_comparison_plot
)
from src.visualization import (
    plot_price_distribution,
    plot_depreciation_analysis,
    plot_numerical_relationships,
    plot_mileage_vs_price_by_age,
    plot_fuel_type_trends,
    plot_correlation_heatmap
)
from src.utils import upload_models_to_hf

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

class Config:
    """Project configuration"""
    
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_PATH = DATA_DIR / "Car_sale_ads.csv"          # legacy fallback only
    BALANCED_DATA_PATH = DATA_DIR / "Car_sale_ads_balanced.csv"  # primary scraped dataset
    PROCESSED_DATA_PATH = DATA_DIR / "processed" / "cars_cleaned.csv"
    MODELS_DIR = PROJECT_ROOT / "models"
    IMAGES_DIR = PROJECT_ROOT / "images"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    RIDGE_CV_FOLDS = 5
    RF_N_ITER = 12
    RF_CV_FOLDS = 3
    XGB_N_TRIALS = 50
    XGB_CV_FOLDS = 3
    
    HF_REPO_ID = "Przemsonn/poland-car-price-model"
    
    def __init__(self):
        """Create directories if they don't exist"""
        for directory in [
            self.DATA_DIR / "processed",
            self.MODELS_DIR,
            self.IMAGES_DIR,
            self.REPORTS_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)


config = Config()

def print_header(text: str, char: str = "=") -> None:
    """Print formatted section header"""
    print("\n" + char * 80)
    print(text.center(80))
    print(char * 80 + "\n")


def print_step(step_num: int, total_steps: int, description: str) -> None:
    """Print step progress"""
    print(f"\n{'='*80}")
    print(f"STEP {step_num}/{total_steps}: {description}")
    print(f"{'='*80}\n")


def save_metrics_report(
    metrics_dict: dict,
    model_comparison: dict,
    save_path: Path
) -> None:
    """Save metrics to text report"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CAR PRICE PREDICTION - MODEL EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for model_name, metrics in metrics_dict.items():
            f.write(f"\n{'-'*80}\n")
            f.write(f"{model_name}\n")
            f.write(f"{'-'*80}\n")
            
            train_metrics = metrics['train']
            test_metrics = metrics['test']
            
            f.write(f"\nTraining Set:\n")
            f.write(f"  R^2 Score:  {train_metrics['R2']:.4f}\n")
            f.write(f"  RMSE:      {train_metrics['RMSE']:,.2f} PLN\n")
            f.write(f"  MAE:       {train_metrics['MAE']:,.2f} PLN\n")
            f.write(f"  MAPE:      {train_metrics['MAPE']*100:.2f}%\n")
            
            f.write(f"\nTest Set:\n")
            f.write(f"  R^2 Score:  {test_metrics['R2']:.4f}\n")
            f.write(f"  RMSE:      {test_metrics['RMSE']:,.2f} PLN\n")
            f.write(f"  MAE:       {test_metrics['MAE']:,.2f} PLN\n")
            f.write(f"  MAPE:      {test_metrics['MAPE']*100:.2f}%\n")
            
        f.write(f"\n{'='*80}\n")
        f.write(f"MODEL COMPARISON (Test Set)\n")
        f.write(f"{'='*80}\n\n")
        
        comparison_df = pd.DataFrame(model_comparison).T
        f.write(comparison_df.to_string())
    
    print(f"Metrics report saved: {save_path}")

def step_1_load_and_clean_data() -> pd.DataFrame:
    """Step 1: Load and clean data.

    Uses the scraped balanced dataset (Car_sale_ads_balanced.csv) when available
    — this is the primary, up-to-date source (2024–2026 prices).
    Falls back to the legacy raw dataset only when the balanced file does not exist.
    The two datasets are NEVER merged: pre-2022 prices are incompatible with
    current market prices and would confuse the models.
    """
    print_step(1, 7, "DATA LOADING & PREPROCESSING")

    balanced_path = config.DATA_DIR / "Car_sale_ads_balanced.csv"

    if balanced_path.exists():
        print(f"Loading scraped dataset from {balanced_path.name} ...")
        df_clean = pd.read_csv(balanced_path)
        print(f"Loaded {len(df_clean):,} rows, {len(df_clean.columns)} columns")
        print("(Using current-price scraping data — legacy dataset ignored)")
    else:
        print(f"Scraped dataset not found. Loading legacy data from {config.RAW_DATA_PATH.name} ...")
        print("  TIP: run  python main.py --mode collect --target-rows 200000  to build the up-to-date dataset.")
        df_raw = load_raw_data(config.RAW_DATA_PATH)
        print(f"Loaded {len(df_raw):,} rows, {len(df_raw.columns)} columns")
        print("\nCleaning data...")
        df_clean = clean_car_data(df_raw)
        print(f"After cleaning: {len(df_clean):,} rows")
        print(f"Duplicates removed: {len(df_raw) - len(df_clean):,}")

    save_processed_data(df_clean, config.PROCESSED_DATA_PATH)

    print("\nData Summary:")
    print(f"Price range:    {df_clean['price_PLN'].min():,.0f} - {df_clean['price_PLN'].max():,.0f} PLN")
    print(f"Median price:   {df_clean['price_PLN'].median():,.0f} PLN")
    print(f"Year range:     {int(df_clean['Production_year'].min())} - {int(df_clean['Production_year'].max())}")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")

    return df_clean


def step_2_exploratory_analysis(df: pd.DataFrame) -> None:
    """Step 2: Exploratory Data Analysis"""
    print_step(2, 7, "EXPLORATORY DATA ANALYSIS")
    
    if 'Production_year' in df.columns and 'Vehicle_age' not in df.columns:
        df['Vehicle_age'] = datetime.now().year - df['Production_year']
    
    print("Generating EDA visualizations...")
    
    print("1/6 Price distribution...")
    fig = plot_price_distribution(
        df, 
        save_path=config.IMAGES_DIR / "eda_price_distribution.png"
    )
    plt.close(fig)
    
    print("2/6 Depreciation analysis...")
    fig = plot_depreciation_analysis(
        df,
        save_path=config.IMAGES_DIR / "eda_depreciation_analysis.png"
    )
    plt.close(fig)
 
    print("3/6 Feature relationships...")
    fig = plot_numerical_relationships(
        df,
        save_path=config.IMAGES_DIR / "eda_numerical_relationships.png"
    )
    plt.close(fig)

    print("4/6 Mileage analysis...")
    fig = plot_mileage_vs_price_by_age(
        df,
        save_path=config.IMAGES_DIR / "eda_mileage_by_age.png"
    )
    plt.close(fig)
    
    print("5/6 Fuel type trends...")
    fig = plot_fuel_type_trends(
        df,
        save_path=config.IMAGES_DIR / "eda_fuel_trends.png"
    )
    plt.close(fig)
    
    print("6/6 Correlation heatmap...")
    fig = plot_correlation_heatmap(
        df,
        save_path=config.IMAGES_DIR / "eda_correlation_heatmap.png"
    )
    plt.close(fig)
    
    print(f"\nAll EDA plots saved to: {config.IMAGES_DIR}")

def debug_nan_issues(df: pd.DataFrame, stage: str = "") -> None:
    """Debug helper to identify NaN issues"""
    print(f"\n{'='*80}")
    print(f"DEBUG: NaN Analysis {stage}")
    print(f"{'='*80}")
    
    key_cols = [
        'Vehicle_age', 'Mileage_km', 'Power_HP', 
        'Displacement_cm3', 'Vehicle_brand', 'price_PLN'
    ]
    
    for col in key_cols:
        if col in df.columns:
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                print(f"{col}: {nan_count:,} NaN values ({nan_count/len(df)*100:.2f}%)")
            else:
                print(f"{col}: No NaN values")
    
    print(f"{'='*80}\n")

def step_3_feature_engineering(df: pd.DataFrame) -> tuple:
    """Step 3: Feature Engineering"""
    print_step(3, 7, "FEATURE ENGINEERING")
    
    debug_nan_issues(df, "BEFORE feature engineering")
    
    print("Creating base features...")
    df_features = engineer_base_features(df)
    print(f"Created {len(df_features.columns) - len(df.columns)} new features")
    
    if df_features['price_PLN'].isnull().any():
        before_len = len(df_features)
        df_features = df_features[df_features['price_PLN'].notna()]
        print(f"Dropped {before_len - len(df_features)} rows with missing target")
    
    print("\nSplitting data...")
    from sklearn.model_selection import train_test_split
    
    X = df_features.drop('price_PLN', axis=1)
    y = df_features['price_PLN']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    print(f"Train set: {len(X_train):,} samples")
    print(f"Test set:  {len(X_test):,} samples")

    print("\nApplying advanced transformations...")
    X_train, X_test = apply_advanced_transformations(X_train, X_test)
    
    print("\nChecking for NaN in X_train...")
    nan_cols_train = X_train.columns[X_train.isnull().any()].tolist()
    if nan_cols_train:
        print(f"Columns with NaN in X_train: {nan_cols_train}")
    else:
        print("No NaN in X_train")

    debug_nan_issues(X_train, "AFTER feature engineering")
    
    print(f"Final feature count: {X_train.shape[1]}")
    print(f"Missing values: Train={X_train.isnull().sum().sum()}, Test={X_test.isnull().sum().sum()}")
    
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X_train[col].isnull().any():
            print(f"Filling remaining NaN in {col} with median...")
            fill_value = X_train[col].median()
            X_train[col] = X_train[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)
  
    print("\nLog-transforming target variable...")
    y_log = get_log_transformed_target(y_train, y_test)
    
    return X_train, X_test, y_train, y_test, y_log['y_train_log'], y_log['y_test_log']


def step_4_train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    y_train_log: pd.Series,
) -> dict:
    """Step 4: Model Training"""
    print_step(4, 7, "MODEL TRAINING")
    
    models = {}
 
    print_header("Training Model 1: Ridge Regression", "-")
    ridge_model = train_ridge_grid_search(
        X_train, 
        y_train_log,
        n_folds=config.RIDGE_CV_FOLDS
    )
    models['Ridge'] = ridge_model
    print("Ridge model trained\n")
  
    print_header("Training Model 2: Random Forest", "-")
    rf_model = train_random_forest_search(
        X_train,
        y_train,
        n_iter=config.RF_N_ITER,
        n_folds=config.RF_CV_FOLDS
    )
    models['RandomForest'] = rf_model
    print("Random Forest model trained\n")
   
    print_header("Training Model 3: XGBoost (Optuna)", "-")
    xgb_model = train_xgboost_optuna(
        X_train,
        y_train_log,
        n_trials=config.XGB_N_TRIALS,
        n_folds=config.XGB_CV_FOLDS
    )
    models['XGBoost_Base'] = xgb_model
    print("XGBoost base model trained\n")
 
    print_header("Training Model 4: XGBoost (Weighted)", "-")
    xgb_weighted = train_xgboost_weighted(
        X_train,
        y_train_log,
        n_trials=config.XGB_N_TRIALS
    )
    models['XGBoost_Weighted'] = xgb_weighted
    print("XGBoost weighted model trained\n")
    
    return models


def step_5_evaluate_models(
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> tuple:
    """Step 5: Model Evaluation"""
    print_step(5, 7, "MODEL EVALUATION")
    
    metrics_dict = {}
    model_comparison = {}
    
    model_configs = [
        ('Ridge', True),
        ('RandomForest', False),
        ('XGBoost_Base', True),
        ('XGBoost_Weighted', True)
    ]
    
    for model_name, log_transformed in model_configs:
        print(f"\nEvaluating {model_name}...")
        
        model = models[model_name]
        
        preds = get_predictions(model, X_train, X_test, log_transformed)
        
        train_metrics = calculate_metrics(y_train, preds['y_train_pred'])
        test_metrics = calculate_metrics(y_test, preds['y_test_pred'])
        
        metrics_dict[model_name] = {
            'train': train_metrics,
            'test': test_metrics,
            'predictions': preds
        }
        
        model_comparison[model_name] = test_metrics
        
        print(f"  Test R^2:  {test_metrics['R2']:.4f}")
        print(f"  Test MAPE: {test_metrics['MAPE']*100:.2f}%")
        print(f"  Test MAE:  {test_metrics['MAE']:,.0f} PLN")
    
    print("\n" + "="*80)
    print("MODEL COMPARISON (Test Set)")
    print("="*80)
    comparison_df = pd.DataFrame(model_comparison).T
    print(comparison_df.to_string())
    
    best_model = comparison_df['R2'].idxmax()
    print(f"\nBest Model: {best_model} (R^2 = {comparison_df.loc[best_model, 'R2']:.4f})")
    
    return metrics_dict, model_comparison


def step_6_generate_visualizations(
    models: dict,
    metrics_dict: dict,
    model_comparison: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_log: pd.Series
) -> None:
    """Step 6: Generate Evaluation Visualizations"""
    print_step(6, 7, "GENERATING EVALUATION VISUALIZATIONS")
    
    print("Generating evaluation plots...")
    
    print("1/8 Model comparison...")
    fig = create_model_comparison_plot(
        model_comparison,
        save_path=config.IMAGES_DIR / "eval_model_comparison.png"
    )
    plt.close(fig)
    
    print("2/8 Ridge diagnostics...")
    ridge_preds = metrics_dict['Ridge']['predictions']['y_test_pred']
    fig = plot_regression_diagnostics(
        y_test,
        ridge_preds,
        "Ridge Regression",
        save_path=config.IMAGES_DIR / "eval_ridge_diagnostics.png"
    )
    plt.close(fig)
    
    print("3/8 Ridge coefficients...")
    fig = plot_ridge_coefficients(
        models['Ridge'],
        top_n=15,
        save_path=config.IMAGES_DIR / "eval_ridge_coefficients.png"
    )
    plt.close(fig)
    
    print("4/8 Random Forest feature importance...")
    fig = plot_tree_feature_importance(
        models['RandomForest'],
        top_n=20,
        save_path=config.IMAGES_DIR / "eval_rf_feature_importance.png"
    )
    plt.close(fig)
    
    print("5/8 XGBoost diagnostics...")
    xgb_preds = metrics_dict['XGBoost_Weighted']['predictions']['y_test_pred']
    fig = plot_regression_diagnostics(
        y_test,
        xgb_preds,
        "XGBoost (Weighted)",
        save_path=config.IMAGES_DIR / "eval_xgb_diagnostics.png"
    )
    plt.close(fig)
    
    print("6/8 XGBoost feature importance...")
    fig = plot_tree_feature_importance(
        models['XGBoost_Weighted'],
        top_n=20,
        save_path=config.IMAGES_DIR / "eval_xgb_feature_importance.png"
    )
    plt.close(fig)
    
    print("7/8 Learning curves...")
    fig = plot_learning_curves(
        models['XGBoost_Weighted'],
        X_train,
        y_train_log,
        cv=3,
        title="Learning Curves - XGBoost (Weighted)",
        save_path=config.IMAGES_DIR / "eval_learning_curves.png"
    )
    plt.close(fig)
    
    print("8/8 Error analysis by age...")
    fig = plot_residuals_vs_age(
        X_test,
        y_test,
        xgb_preds,
        save_path=config.IMAGES_DIR / "eval_residuals_vs_age.png"
    )
    plt.close(fig)
    
    print(f"\nAll evaluation plots saved to: {config.IMAGES_DIR}")


def step_7_save_models(models: dict) -> None:
    """Step 7: Save Models"""
    print_step(7, 7, "SAVING MODELS")
    
    print("Saving models locally...")
    
    import joblib
    
    for model_name, model in models.items():
        filename = f"{model_name.lower()}_model.pkl"
        filepath = config.MODELS_DIR / filename
        
        joblib.dump(model, filepath)
        print(f"Saved: {filename}")
    
    print(f"\nAll models saved to: {config.MODELS_DIR}")
    
    upload_to_hf = input("\nUpload models to Hugging Face? (y/n): ").lower() == 'y'
    
    if upload_to_hf:
        try:
            print("\nUploading to Hugging Face...")
            models_dict = {
                f"{name.lower()}_model.pkl": model 
                for name, model in models.items()
            }
            upload_models_to_hf(models_dict, config.HF_REPO_ID)
            print("Models uploaded to Hugging Face")
        except Exception as e:
            print(f"Upload failed: {e}")
            print("   Models are still saved locally.")

def run_tests() -> bool:
    """Run comprehensive tests"""
    print_header("RUNNING PROJECT TESTS")
    
    all_passed = True

    # Determine which data source to use for tests
    _test_data_path = (
        config.PROCESSED_DATA_PATH if config.PROCESSED_DATA_PATH.exists()
        else config.BALANCED_DATA_PATH if config.BALANCED_DATA_PATH.exists()
        else config.RAW_DATA_PATH
    )
    _using_scraped = _test_data_path != config.RAW_DATA_PATH
    print(f"Using dataset: {_test_data_path.name}\n")

    print("Test 1: Data Loading...")
    try:
        df = pd.read_csv(_test_data_path) if _using_scraped else load_raw_data(_test_data_path)
        assert len(df) > 0, "DataFrame is empty"
        assert 'price_PLN' in df.columns, "price_PLN column missing"
        print(" PASSED\n")
    except Exception as e:
        print(f"FAILED: {e}\n")
        all_passed = False

    print("Test 2: Data Preprocessing...")
    try:
        df_clean = pd.read_csv(_test_data_path) if _using_scraped else clean_car_data(load_raw_data(_test_data_path))
        assert 'price_PLN' in df_clean.columns, "price_PLN not present"
        assert len(df_clean) > 0, "All data removed"
        print("PASSED\n")
    except Exception as e:
        print(f"FAILED: {e}\n")
        all_passed = False

    print("Test 3: Feature Engineering...")
    try:
        df_clean = pd.read_csv(_test_data_path) if _using_scraped else clean_car_data(load_raw_data(_test_data_path))
        df_features = engineer_base_features(df_clean)

        required_features = [
            'Age_category', 'Is_new_car', 'Is_old_car',
            'Mileage_per_year', 'HP_per_liter', 'Is_premium'
        ]

        for feat in required_features:
            assert feat in df_features.columns, f"Feature {feat} not created"

        print("PASSED\n")
    except Exception as e:
        print(f"FAILED: {e}\n")
        all_passed = False

    print("Test 4: Preprocessors...")
    try:
        preprocessor_tree = get_preprocessor_tree()
        preprocessor_mastered = get_preprocessor_mastered()

        assert preprocessor_tree is not None, "Tree preprocessor is None"
        assert preprocessor_mastered is not None, "Mastered preprocessor is None"

        print("PASSED\n")
    except Exception as e:
        print(f"FAILED: {e}\n")
        all_passed = False

    print("Test 5: Visualization Functions...")
    try:
        df_clean = pd.read_csv(_test_data_path) if _using_scraped else clean_car_data(load_raw_data(_test_data_path))

        if 'Vehicle_age' not in df_clean.columns and 'Production_year' in df_clean.columns:
            df_clean['Vehicle_age'] = datetime.now().year - df_clean['Production_year']

        fig = plot_price_distribution(df_clean)
        plt.close(fig)

        fig = plot_depreciation_analysis(df_clean)
        plt.close(fig)

        print("PASSED\n")
    except Exception as e:
        print(f"FAILED: {e}\n")
        all_passed = False
  
    print("Test 6: Model Training Functions...")
    try:
        from src.models import (
            train_ridge_grid_search,
            train_random_forest_search,
            train_xgboost_optuna
        )
        print("PASSED\n")
    except Exception as e:
        print(f"FAILED: {e}\n")
        all_passed = False

    print("="*80)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*80)
    
    return all_passed

def run_full_pipeline() -> None:
    """Run the complete ML pipeline"""
    print_header("CAR PRICE PREDICTION - COMPLETE PIPELINE")
    
    start_time = datetime.now()
    
    try:
        df_clean = step_1_load_and_clean_data()
        
        step_2_exploratory_analysis(df_clean)
        
        X_train, X_test, y_train, y_test, y_train_log, y_test_log = step_3_feature_engineering(df_clean)
        
        models = step_4_train_models(X_train, y_train, y_train_log)
        
        step_7_save_models(models)

        metrics_dict, model_comparison = step_5_evaluate_models(
            models, X_train, X_test, y_train, y_test
        )
        
        step_6_generate_visualizations(
            models, metrics_dict, model_comparison,
            X_train, X_test, y_train, y_test, y_train_log
        )
        
        save_metrics_report(
            metrics_dict,
            model_comparison,
            config.REPORTS_DIR / "model_evaluation_report.txt"
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Total execution time: {duration/60:.1f} minutes")
        print(f"\nOutputs saved to:")
        print(f"  - Models: {config.MODELS_DIR}")
        print(f"  - Images: {config.IMAGES_DIR}")
        print(f"  - Reports: {config.REPORTS_DIR}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Pipeline failed")
        print(f"{'='*80}")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_data_collect(
    target_rows: int = 200_000,
    detail_mode: bool = False,
    mock: bool = False,
    output_path: Optional[Path] = None,
) -> None:
    """
    Collect a stratified, balanced dataset from Otomoto and save it to CSV.

    Fetches listings across all segments defined in STRATIFIED_CONFIG
    (popular brands, luxury brands, electric vehicles), then applies
    stratified sampling to produce a balanced dataset.

    The result is saved to ``output_path`` (default: data/Car_sale_ads_balanced.csv).
    This file replaces the old Car_sale_ads.csv as the primary training source —
    the two datasets are never merged (pre-2022 prices are not compatible with
    current market prices after the 2022–2024 car price boom).

    Parameters
    ----------
    target_rows : int
        Target total rows for the balanced output (default 200 000).
    detail_mode : bool
        Fetch individual listing pages for full field coverage.
    mock : bool
        Use synthetic data — no network requests (testing only).
    output_path : Path, optional
        Where to save the balanced CSV.  Defaults to
        ``data/Car_sale_ads_balanced.csv``.
    """
    from src.data_fetcher import fetch_balanced_dataset
    from src.data_cleaning import clean_data, apply_stratified_sampling, validate_schema

    if output_path is None:
        output_path = config.DATA_DIR / "Car_sale_ads_balanced.csv"

    print_header("STRATIFIED DATA COLLECTION")
    print(f"Target rows:   {target_rows:,}")
    print(f"Detail mode:   {'Yes' if detail_mode else 'No'}")
    print(f"Mock mode:     {'Yes' if mock else 'No'}")
    print(f"Output path:   {output_path}\n")

    start_time = datetime.now()

    # 1. Fetch raw stratified listings
    print("Step 1/3  Fetching stratified listings from Otomoto...")
    df_raw = fetch_balanced_dataset(
        target_rows=target_rows,
        detail_mode=detail_mode,
        mock=mock,
    )
    print(f"  Raw rows fetched: {len(df_raw):,}")
    if "_category" in df_raw.columns:
        print(f"  Category breakdown:\n{df_raw['_category'].value_counts().to_string()}")

    # 2. Normalize + clean to schema
    print("\nStep 2/3  Cleaning and normalizing to project schema...")
    df_raw_no_cat = df_raw.drop(columns=["_category"], errors="ignore")
    df_clean = clean_data(df_raw_no_cat)
    print(f"  Rows after cleaning: {len(df_clean):,}")

    # Re-attach category for sampling
    if "_category" in df_raw.columns:
        df_clean = df_clean.copy()
        df_clean["_category"] = df_raw["_category"].values[: len(df_clean)]

    # 3. Apply stratified sampling
    print("\nStep 3/3  Applying stratified sampling...")
    df_balanced = apply_stratified_sampling(df_clean, target_rows=target_rows)
    df_balanced = df_balanced.drop(columns=["_category", "offer_id"], errors="ignore")
    print(f"  Final balanced rows: {len(df_balanced):,}")

    # Schema validation
    validation = validate_schema(df_balanced)
    if not validation["valid"]:
        print(f"  [WARN] Schema issues: {validation['issues']}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_balanced.to_csv(output_path, index=False)

    duration = (datetime.now() - start_time).total_seconds()
    print_header("COLLECTION COMPLETE")
    print(f"Balanced rows saved: {len(df_balanced):,}")
    print(f"Output file:         {output_path}")
    print(f"Duration:            {duration:.1f}s  ({duration / 60:.1f} min)")


def run_data_update(pages: int = 10, detail_mode: bool = False, mock: bool = False) -> None:
    """
    Fetch new car listings from Otomoto and append them to the raw dataset.

    This function implements incremental data collection:
    1. Scrapes current Otomoto listings (search results, newest first)
    2. Normalizes fields to match the project schema
    3. Deduplicates against the existing dataset by offer ID
    4. Appends only genuinely new rows to data/Car_sale_ads.csv

    Parameters
    ----------
    pages : int
        Number of Otomoto search pages to fetch (32 listings each).
    detail_mode : bool
        If True, fetch individual listing pages for full field coverage
        (Drive, Colour, body type, condition, doors). Slower but more complete.
    mock : bool
        Use synthetic data instead of live scraping (for testing).

    Note
    ----
    Historical listings (2021–2024) are not available via scraping —
    Otomoto only exposes active offers. This mode collects future listings
    for continuous dataset enrichment.
    """
    from src.data_fetcher import fetch_incremental

    print_header("DATA UPDATE: FETCHING NEW LISTINGS FROM OTOMOTO")
    print(f"Pages to fetch: {pages}  (~{pages * 32} listings)")
    print(f"Detail mode:    {'Yes (fetches individual pages)' if detail_mode else 'No (search results only)'}")
    print(f"Mock mode:      {'Yes' if mock else 'No'}\n")

    start_time = datetime.now()
    new_rows = fetch_incremental(
        data_path=config.BALANCED_DATA_PATH,
        pages=pages,
        detail_mode=detail_mode,
        mock=mock,
    )

    duration = (datetime.now() - start_time).total_seconds()
    print_header("UPDATE COMPLETE")
    print(f"New rows appended: {len(new_rows):,}")
    print(f"Dataset path:      {config.BALANCED_DATA_PATH}")
    print(f"Duration:          {duration:.1f}s")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Car Price Prediction ML Pipeline"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'test', 'train', 'visualize', 'evaluate', 'export', 'update', 'collect'],
        help='Execution mode (default: full)'
    )
    parser.add_argument(
        '--pages',
        type=int,
        default=10,
        help='Number of Otomoto search pages to fetch in update mode (default: 10)'
    )
    parser.add_argument(
        '--target-rows',
        type=int,
        default=200_000,
        help='Target balanced row count for collect mode (default: 200000)'
    )
    parser.add_argument(
        '--detail',
        action='store_true',
        help='Fetch full listing details in update/collect mode (slower, more complete)'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Use mock/synthetic data instead of live scraping (testing only)'
    )

    args = parser.parse_args()
    
    if args.mode == 'collect':
        run_data_collect(
            target_rows=args.target_rows,
            detail_mode=args.detail,
            mock=args.mock,
        )

    elif args.mode == 'update':
        run_data_update(
            pages=args.pages,
            detail_mode=args.detail,
            mock=args.mock,
        )

    elif args.mode == 'test':
        success = run_tests()
        sys.exit(0 if success else 1)

    elif args.mode == 'full':
        run_full_pipeline()
        
    elif args.mode == 'train':
        print_header("TRAINING MODELS ONLY")
        df_clean = step_1_load_and_clean_data()
        X_train, X_test, y_train, y_test, y_train_log, y_test_log = step_3_feature_engineering(df_clean)
        models = step_4_train_models(X_train, y_train, y_train_log)
        step_7_save_models(models)
        
    elif args.mode == 'visualize':
        print_header("GENERATING VISUALIZATIONS ONLY")
        _vis_path = (
            config.PROCESSED_DATA_PATH if config.PROCESSED_DATA_PATH.exists()
            else config.BALANCED_DATA_PATH
        )
        df_clean = pd.read_csv(_vis_path)
        step_2_exploratory_analysis(df_clean)

    elif args.mode == 'evaluate':
        print_header("RESUMING FROM STEP 5: EVALUATION & VISUALIZATION")
        
        df_clean = pd.read_csv(config.PROCESSED_DATA_PATH)
        X_train, X_test, y_train, y_test, y_train_log, y_test_log = step_3_feature_engineering(df_clean)
        
        from src.utils import load_local_model
        print("Loading saved models from disk...")
        models = {
            'Ridge': load_local_model(config.MODELS_DIR / "ridge_model.pkl"),
            'RandomForest': load_local_model(config.MODELS_DIR / "randomforest_model.pkl"),
            'XGBoost_Base': load_local_model(config.MODELS_DIR / "xgboost_base_model.pkl"),
            'XGBoost_Weighted': load_local_model(config.MODELS_DIR / "xgboost_weighted_model.pkl")
        }
        
        metrics_dict, model_comparison = step_5_evaluate_models(
            models, X_train, X_test, y_train, y_test
        )
        
        step_6_generate_visualizations(
            models, metrics_dict, model_comparison,
            X_train, X_test, y_train, y_test, y_train_log
        )
        
        save_metrics_report(
            metrics_dict, model_comparison,
            config.REPORTS_DIR / "model_evaluation_report.txt"
        )

    elif args.mode == 'export':
        print_header("RESUMING FROM STEP 7: EXPORT / UPLOAD")
        
        try:
            from src.utils import load_local_model
            models = {
                'Ridge': load_local_model(config.MODELS_DIR / "ridge_model.pkl"),
                'RandomForest': load_local_model(config.MODELS_DIR / "randomforest_model.pkl"),
                'XGBoost': load_local_model(config.MODELS_DIR / "xgboost_base_model.pkl"),
                'XGBoost_Weighted': load_local_model(config.MODELS_DIR / "xgboost_weighted_model.pkl")
            }
            step_7_save_models(models)
        except FileNotFoundError as e:
            print(f"ERROR: File not found error: {e}")
        except Exception as e:
            print(f"Other error: {e}")

if __name__ == "__main__":
    main()