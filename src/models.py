# src/models.py

"""
Model training and optimization module.

Contains functions for training different model types:
- Ridge Regression with GridSearchCV
- Random Forest with RandomizedSearchCV
- XGBoost with Optuna optimization
- XGBoost with sample weighting for imbalanced segments
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .features import get_preprocessor_mastered, get_preprocessor_tree
from .config import RANDOM_STATE, print_if_verbose

def get_log_transformed_target(
    y_train: pd.Series, 
    y_test: pd.Series
) -> Dict[str, pd.Series]:
    """
    Apply log transformation to target variable.
    
    Log transformation helps:
    - Reduce skewness (from 2.3 to 0.4)
    - Stabilize variance
    - Meet normality assumptions for some models
    
    Parameters
    ----------
    y_train : pd.Series
        Training target variable
    y_test : pd.Series
        Test target variable
        
    Returns
    -------
    dict
        Dictionary with log-transformed targets:
        - 'y_train_log': log(y_train + 1)
        - 'y_test_log': log(y_test + 1)
        
    Examples
    --------
    >>> y_log = get_log_transformed_target(y_train, y_test)
    >>> y_train_log = y_log['y_train_log']
    """
    return {
        'y_train_log': np.log1p(y_train),
        'y_test_log': np.log1p(y_test)
    }


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """
    Inverse log transformation to get predictions in original scale.
    
    Parameters
    ----------
    y_log : np.ndarray
        Log-transformed predictions
        
    Returns
    -------
    np.ndarray
        Predictions in original PLN scale
    """
    return np.expm1(y_log)

def train_ridge_grid_search(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    smoothing: int = 200,
    n_folds: int = 5
) -> Pipeline:
    """
    Train Ridge Regression with GridSearchCV.
    
    Ridge regression with log-transformed target works well as:
    - Simple baseline for comparison
    - Interpretable coefficients
    - Fast training time
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train_log : pd.Series
        Log-transformed training target
    smoothing : int, default=200
        Smoothing parameter for TargetEncoder
    n_folds : int, default=5
        Number of cross-validation folds
        
    Returns
    -------
    Pipeline
        Fitted Ridge regression pipeline
        
    Examples
    --------
    >>> model = train_ridge_grid_search(X_train, y_train_log)
    >>> predictions = model.predict(X_test)
    """
    print_if_verbose("\n" + "="*80)
    print_if_verbose("TRAINING RIDGE REGRESSION WITH GRIDSEARCH")
    print_if_verbose("="*80)
    
    preprocessor = get_preprocessor_mastered(smoothing=smoothing)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge())
    ])
    
    param_grid = {
        'model__alpha': [0.1, 1.0, 5.0, 10.0, 100.0],
        'model__solver': ['auto', 'cholesky', 'lsqr'],
        'model__fit_intercept': [False, True]
    }
    
    scoring = {
        'r2': 'r2',
        'neg_rmse': 'neg_root_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_mse': 'neg_mean_squared_error'
    }
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit='neg_rmse', 
        cv=kfold,
        n_jobs=-1,
        verbose=1
    )
    
    print_if_verbose("Starting GridSearch...")
    grid.fit(X_train, y_train_log)
    
    print_if_verbose(f"\n✓ Best parameters: {grid.best_params_}")
    print_if_verbose(f"✓ Best CV RMSE: {-grid.best_score_:,.2f}")
    print_if_verbose("="*80)
    
    return grid.best_estimator_

def train_random_forest_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 12,
    n_folds: int = 3
) -> Pipeline:
    """
    Train Random Forest with RandomizedSearchCV.
    
    Random Forest advantages:
    - Captures non-linear relationships
    - Feature importance for interpretability
    - Handles mixed data types naturally
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target (original scale, not log-transformed)
    n_iter : int, default=12
        Number of parameter settings sampled
    n_folds : int, default=3
        Number of CV folds
        
    Returns
    -------
    Pipeline
        Fitted Random Forest pipeline
        
    Notes
    -----
    Uses OrdinalEncoder instead of TargetEncoder for tree models
    to avoid overfitting and reduce training time.
    """
    print_if_verbose("\n" + "="*80)
    print_if_verbose("TRAINING RANDOM FOREST WITH RANDOMIZEDSEARCH")
    print_if_verbose("="*80)
    
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    
    preprocessor = get_preprocessor_tree()
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=RANDOM_STATE))
    ])
    
    param_dist = {
        'model__n_estimators': [200, 500],
        'model__max_depth': [10, 15, 20],
        'model__min_samples_leaf': [5, 10, 20],
        'model__min_samples_split': [10, 20],
        'model__max_features': ['sqrt', 0.5],
        'model__max_samples': [0.7, 0.8],
        'model__bootstrap': [True]
    }
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=kfold,
        n_jobs=-1,
        verbose=2,
        random_state=RANDOM_STATE
    )
    
    print_if_verbose("Starting RandomizedSearch...")
    random_search.fit(X_train_clean, y_train)
    
    print_if_verbose(f"\n✓ Best parameters: {random_search.best_params_}")
    print_if_verbose(f"✓ Best CV MAE: {-random_search.best_score_:,.2f} PLN")
    print_if_verbose("="*80)
    
    return random_search.best_estimator_

def train_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    n_trials: int = 50,
    n_folds: int = 3
) -> Pipeline:
    """
    Train XGBoost with Bayesian optimization using Optuna.
    
    Optuna advantages over GridSearch:
    - Smarter search (Bayesian, not brute-force)
    - Early stopping of unpromising trials
    - Typically finds better hyperparameters faster
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train_log : pd.Series
        Log-transformed training target
    n_trials : int, default=50
        Number of Optuna trials
    n_folds : int, default=3
        Number of CV folds
        
    Returns
    -------
    Pipeline
        Fitted XGBoost pipeline with optimized hyperparameters
        
    Notes
    -----
    Optimization objective: Minimize RMSE on validation set.
    Uses early stopping to prevent overfitting.
    """
    print_if_verbose("\n" + "="*80)
    print_if_verbose("TRAINING XGBOOST WITH OPTUNA OPTIMIZATION")
    print_if_verbose("="*80)
    
    preprocessor = get_preprocessor_tree()
    
    def objective(trial):
        """Optuna objective function"""
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = []
        
        params = {
            'n_estimators': 1000,
            'early_stopping_rounds': 30,
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'tree_method': 'hist',
            'random_state': RANDOM_STATE
        }
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
            
            X_tr_proc = preprocessor.fit_transform(X_tr, y_tr)
            X_val_proc = preprocessor.transform(X_val)
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr_proc, y_tr,
                eval_set=[(X_val_proc, y_val)],
                verbose=False
            )
            
            preds = model.predict(X_val_proc)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            cv_scores.append(rmse)
            
            trial.report(rmse, step=len(cv_scores))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return np.mean(cv_scores)
    
    print_if_verbose(f"Starting Optuna optimization with {n_trials} trials...")
    
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    
    print_if_verbose(f"\n✓ Best trial RMSE: {study.best_value:.4f}")
    print_if_verbose(f"✓ Best parameters: {best_params}")
    
    print_if_verbose("\nTraining final model on full training set...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(
            **best_params,
            n_estimators=1000,
            random_state=RANDOM_STATE
        ))
    ])
    
    pipeline.fit(X_train, y_train_log)
    
    print_if_verbose("="*80)
    
    return pipeline

def get_brand_reliability_category(brand: str) -> str:
    """
    Categorize car brands by market segment.
    
    Categories help identify segments where model struggles:
    - Vintage: Old Polish/Soviet cars (rare, unpredictable prices)
    - Luxury: Exotic supercars (few samples, high variance)
    - American: US brands (different depreciation patterns)
    - Budget: Economy brands (stable, well-represented)
    - Standard: Mainstream brands (most data)
    
    Parameters
    ----------
    brand : str
        Car brand name
        
    Returns
    -------
    str
        Category: Vintage/Luxury/American/Budget/Standard
    """
    brand_lower = str(brand).lower()
    
    luxury = [
        'ferrari', 'lamborghini', 'rolls-royce', 'bentley', 
        'aston martin', 'porsche', 'maserati', 'mclaren', 'lotus'
    ]
    american = ['ram', 'dodge', 'chevrolet', 'cadillac', 'hummer', 'jeep']
    vintage = [
        'syrena', 'nysa', 'warszawa', 'polonez', 'żuk', 
        'gaz', 'moskwicz', 'lada', 'trabant', 'wartburg'
    ]
    budget = ['dacia', 'fiat', 'daewoo', 'lancia', 'tata']
    
    if brand_lower in luxury:
        return 'Luxury'
    if brand_lower in american:
        return 'American'
    if brand_lower in vintage:
        return 'Vintage'
    if brand_lower in budget:
        return 'Budget'
    
    return 'Standard'


def calculate_sample_weights(X_train: pd.DataFrame) -> np.ndarray:
    """
    Calculate sample weights to handle imbalanced market segments.
    
    Strategy:
    - Upweight rare/difficult segments (vintage, luxury)
    - Downweight very rare brands (< 20 listings)
    - Keep standard brands at weight 1.0
    
    This helps model learn better on underrepresented segments
    without being dominated by common mass-market cars.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features with 'Vehicle_brand' column
        
    Returns
    -------
    np.ndarray
        Sample weights (same length as X_train)
        
    Examples
    --------
    >>> weights = calculate_sample_weights(X_train)
    >>> # Use in model training:
    >>> model.fit(X_train, y_train, sample_weight=weights)
    """
    X_train = X_train.copy()
    
    X_train['Brand_category'] = X_train['Vehicle_brand'].apply(
        get_brand_reliability_category
    )
    
    brand_freq = X_train['Vehicle_brand'].value_counts().to_dict()
    X_train['Brand_frequency'] = X_train['Vehicle_brand'].map(brand_freq)
    
    weight_map = {
        'Vintage': 1.5,   
        'Luxury': 1.2,    
        'American': 1.2, 
        'Budget': 1.1,   
        'Standard': 1.0   
    }
    
    weights = np.ones(len(X_train))
    
    for category, weight in weight_map.items():
        mask = X_train['Brand_category'] == category
        weights[mask] = weight
    
    rare_mask = X_train['Brand_frequency'] < 20
    weights[rare_mask] = np.minimum(weights[rare_mask] * 0.5, 0.5)
    
    print_if_verbose(f"\n✓ Sample weights calculated:")
    print_if_verbose(f"  Vintage cars: {(X_train['Brand_category'] == 'Vintage').sum()}")
    print_if_verbose(f"  Luxury cars: {(X_train['Brand_category'] == 'Luxury').sum()}")
    print_if_verbose(f"  Rare brands (<20): {rare_mask.sum()}")
    
    return weights


def train_xgboost_weighted(
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    n_trials: int = 50
) -> Pipeline:
    """
    Train XGBoost with sample weighting and Optuna optimization.
    
    This is the final production model that:
    - Uses Bayesian optimization (Optuna)
    - Applies sample weights for rare segments
    - Includes strong regularization
    - Uses early stopping
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train_log : pd.Series
        Log-transformed training target
    n_trials : int, default=50
        Number of Optuna trials
        
    Returns
    -------
    Pipeline
        Production-ready XGBoost pipeline
        
    Notes
    -----
    This model achieves:
    - R² = 0.926
    - MAPE = 17.2%
    - Good balance between accuracy and generalization
    """
    print_if_verbose("\n" + "="*80)
    print_if_verbose("TRAINING FINAL XGBOOST WITH SAMPLE WEIGHTING")
    print_if_verbose("="*80)
    
    weights = calculate_sample_weights(X_train)
    
    preprocessor = get_preprocessor_tree()
    
    def objective(trial):
        """Optuna objective with sample weights"""
        
        X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
            X_train, y_train_log, weights,
            test_size=0.2,
            random_state=RANDOM_STATE
        )
        
        params = {
            'n_estimators': 2000,
            'early_stopping_rounds': 50,
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.7),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
            'gamma': trial.suggest_float('gamma', 0.1, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 100.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 100.0, log=True),
            'tree_method': 'hist',
            'random_state': RANDOM_STATE
        }
        
        X_tr_tf = preprocessor.fit_transform(X_tr, y_tr)
        X_val_tf = preprocessor.transform(X_val)
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr_tf, y_tr,
            eval_set=[(X_val_tf, y_val)],
            sample_weight=w_tr,
            verbose=False
        )
        
        preds = model.predict(X_val_tf)
        rmse = np.sqrt(mean_squared_error(y_val, preds, sample_weight=w_val))
        
        return rmse
    
    print_if_verbose(f"Starting weighted Optuna optimization with {n_trials} trials...")
    
    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    
    print_if_verbose(f"\n✓ Best trial weighted RMSE: {study.best_value:.4f}")
    print_if_verbose(f"✓ Best parameters: {best_params}")
    
    print_if_verbose("\nTraining final weighted model on full training set...")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(
            **best_params,
            n_estimators=2000,
            random_state=RANDOM_STATE
        ))
    ])
    
    pipeline.fit(X_train, y_train_log, model__sample_weight=weights)
    
    print_if_verbose("="*80)
    
    return pipeline

def get_predictions(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    log_transformed: bool = True
) -> Dict[str, np.ndarray]:
    """
    Generate predictions from fitted model.
    
    Parameters
    ----------
    model : Pipeline
        Fitted model pipeline
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    log_transformed : bool, default=True
        Whether model was trained on log-transformed target
        
    Returns
    -------
    dict
        Dictionary with predictions:
        - 'y_train_pred': Training predictions
        - 'y_test_pred': Test predictions
    """
    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
    X_test_clean = X_test.replace([np.inf, -np.inf], np.nan)
    
    y_train_pred = model.predict(X_train_clean)
    y_test_pred = model.predict(X_test_clean)
    
    if log_transformed:
        y_train_pred = inverse_log_transform(y_train_pred)
        y_test_pred = inverse_log_transform(y_test_pred)
    
    return {
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }