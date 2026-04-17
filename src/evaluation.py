# src/evaluation.py

"""
Model evaluation and visualization module.

Contains functions for:
- Calculating regression metrics
- Generating diagnostic plots
- Analyzing errors by segment
- Creating learning curves
- Feature importance visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from typing import Dict, Optional, Tuple
from pathlib import Path

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, print_if_verbose

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics.

    Returns a dictionary with:

    * ``R2``   — coefficient of determination
    * ``RMSE`` — root mean squared error (PLN)
    * ``MAE``  — mean absolute error (PLN)
    * ``MAPE`` — mean absolute percentage error (0-1 scale)
    * ``MdAPE`` — *median* absolute percentage error (0-1 scale).
      Robust to outliers, less biased than MAPE for heavy-tailed
      price distributions like this one.

    A small epsilon is added to the denominator when computing percentage
    errors to protect against zero-priced rows.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    eps = 1e-9
    abs_pct_err = np.abs(y_true_arr - y_pred_arr) / np.maximum(np.abs(y_true_arr), eps)

    return {
        'R2':    r2_score(y_true_arr, y_pred_arr),
        'RMSE':  np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)),
        'MAE':   mean_absolute_error(y_true_arr, y_pred_arr),
        'MAPE':  mean_absolute_percentage_error(y_true_arr, y_pred_arr),
        'MdAPE': float(np.median(abs_pct_err)),
    }


def cross_validate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    n_folds: int = 5,
    log_transformed: bool = True,
) -> Dict[str, float]:
    """Run k-fold CV and return mean / std / 95% CI for RMSE, MAE, R2, MAPE.

    Parameters
    ----------
    model : Pipeline
        Unfitted pipeline (will be cloned internally by cross_val_score).
    X : pd.DataFrame
        Features (raw — pipeline handles feature engineering).
    y : np.ndarray
        Target **on the scale the pipeline expects to predict** (log or raw).
    n_folds : int, default 5
    log_transformed : bool, default True
        If True, predictions are inverse-log-transformed before scoring,
        so reported RMSE/MAE/MAPE are in PLN.

    Returns
    -------
    dict with keys ``R2_mean``, ``R2_std``, ``R2_ci95``, ``RMSE_*``, etc.
    """
    from sklearn.base import clone

    print_if_verbose(f"\nRunning {n_folds}-fold cross-validation...")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics: Dict[str, list] = {"R2": [], "RMSE": [], "MAE": [], "MAPE": [], "MdAPE": []}

    y_arr = np.asarray(y)
    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)

        if log_transformed:
            preds_pln = np.expm1(preds)
            y_val_pln = np.expm1(y_val)
        else:
            preds_pln, y_val_pln = preds, y_val

        metrics = calculate_metrics(y_val_pln, preds_pln)
        for k, v in metrics.items():
            fold_metrics[k].append(v)
        print_if_verbose(
            f"  fold {fold_i}/{n_folds}: R2={metrics['R2']:.4f}  "
            f"MAE={metrics['MAE']:,.0f}  MAPE={metrics['MAPE']*100:.2f}%"
        )

    summary: Dict[str, float] = {}
    for metric, values in fold_metrics.items():
        arr = np.asarray(values, dtype=float)
        summary[f"{metric}_mean"] = float(arr.mean())
        summary[f"{metric}_std"] = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        # Approximate 95% CI (Gaussian) for the fold-mean statistic.
        summary[f"{metric}_ci95"] = 1.96 * summary[f"{metric}_std"] / np.sqrt(max(len(arr), 1))

    print_if_verbose(
        f"[OK] CV summary — R² = {summary['R2_mean']:.4f} ± {summary['R2_ci95']:.4f}, "
        f"MAPE = {summary['MAPE_mean']*100:.2f}% ± {summary['MAPE_ci95']*100:.2f}pp"
    )
    return summary


def format_cv_summary(summary: Dict[str, float]) -> str:
    """Pretty-print a ``cross_validate_model`` summary for the report."""
    lines = ["Cross-validated (5-fold) test metrics — mean ± 95% CI:"]
    lines.append(
        f"  R^2   : {summary['R2_mean']:.4f} ± {summary['R2_ci95']:.4f}"
    )
    lines.append(
        f"  RMSE  : {summary['RMSE_mean']:,.0f} ± {summary['RMSE_ci95']:,.0f} PLN"
    )
    lines.append(
        f"  MAE   : {summary['MAE_mean']:,.0f} ± {summary['MAE_ci95']:,.0f} PLN"
    )
    lines.append(
        f"  MAPE  : {summary['MAPE_mean']*100:.2f}% ± {summary['MAPE_ci95']*100:.2f} pp"
    )
    lines.append(
        f"  MdAPE : {summary['MdAPE_mean']*100:.2f}% ± {summary['MdAPE_ci95']*100:.2f} pp"
    )
    return "\n".join(lines)


def create_metrics_table(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_pred: np.ndarray
) -> pd.DataFrame:
    """
    Create formatted metrics comparison table.
    
    Parameters
    ----------
    y_train : np.ndarray
        True training values
    y_test : np.ndarray
        True test values
    y_train_pred : np.ndarray
        Predicted training values
    y_test_pred : np.ndarray
        Predicted test values
        
    Returns
    -------
    pd.DataFrame
        Metrics table with Train and Test columns
    """
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    metrics_df = pd.DataFrame({
        'Train': train_metrics,
        'Test': test_metrics
    })
    
    return metrics_df

def plot_regression_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate regression diagnostic plots.
    
    Creates:
    1. Residuals distribution histogram
    2. Actual vs Predicted scatter plot
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    model_name : str, default="Model"
        Model name for plot title
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(residuals, bins=50, kde=True, ax=axes[0], color="#38B338")
    axes[0].axvline(0, color='crimson', linestyle='--', linewidth=2)
    axes[0].set_title(f'Residuals Distribution ({model_name})', fontsize=14, weight='bold')
    axes[0].set_xlabel('Residual (PLN)')
    axes[0].set_ylabel('Frequency')
    axes[0].xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
    )
    axes[0].grid(alpha=0.3)
    
    axes[1].scatter(y_true, y_pred, alpha=0.3, color="#38B338")
    axes[1].plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        'r--', linewidth=2, label='Perfect prediction'
    )
    axes[1].set_title('Actual vs Predicted Prices', fontsize=14, weight='bold')
    axes[1].set_xlabel('Actual Price (PLN)')
    axes[1].set_ylabel('Predicted Price (PLN)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    for ax in axes:
        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
        )
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_if_verbose(f"[OK] Saved plot: {save_path}")
    
    return fig


def plot_residuals_vs_age(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    age_threshold: int = 30,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Analyze prediction errors by vehicle age.
    
    Helps identify if model struggles with:
    - Old/vintage cars (>30 years)
    - New cars (<3 years)
    
    Parameters
    ----------
    X_test : pd.DataFrame
        Test features with 'Vehicle_age' column
    y_test : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    age_threshold : int, default=30
        Age cutoff for highlighting old cars
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    residuals = y_test - y_pred
    ages = X_test['Vehicle_age'].values
    old_mask = ages > age_threshold
    
    rmse_old = np.sqrt(mean_squared_error(y_test[old_mask], y_pred[old_mask]))
    rmse_new = np.sqrt(mean_squared_error(y_test[~old_mask], y_pred[~old_mask]))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(
        ages[~old_mask], residuals[~old_mask],
        alpha=0.5, label=f'≤ {age_threshold} years (RMSE: {rmse_new:,.0f} PLN)',
        color='#2E7D32'
    )
    ax.scatter(
        ages[old_mask], residuals[old_mask],
        color='#D32F2F', alpha=0.6,
        label=f'> {age_threshold} years (RMSE: {rmse_old:,.0f} PLN)'
    )
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_title('Residuals vs Vehicle Age', fontsize=16, weight='bold')
    ax.set_xlabel('Vehicle Age (years)', fontsize=12)
    ax.set_ylabel('Residual (PLN)', fontsize=12)
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_if_verbose(f"[OK] Saved plot: {save_path}")
    
    return fig

def plot_mape_by_brand(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    min_listings: int = 30,
    max_listings: Optional[int] = None,
    top_n: int = 20,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Analyze MAPE by car brand to identify problem segments.
    
    Parameters
    ----------
    X_test : pd.DataFrame
        Test features with 'Vehicle_brand' column
    y_test : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    min_listings : int, default=30
        Minimum number of listings to include brand
    max_listings : int, optional
        Maximum number of listings (for niche brands)
    top_n : int, default=20
        Number of brands to show
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    test_df = X_test.copy()
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred
    test_df['error_pct'] = (
        np.abs(test_df['actual'] - test_df['predicted']) / test_df['actual'] * 100
    )
    
    brand_stats = test_df.groupby('Vehicle_brand')['error_pct'].agg(['mean', 'count'])
    
    if max_listings:
        filtered_stats = brand_stats[brand_stats['count'] < max_listings]
        title = f'MAPE by Brand (Niche: < {max_listings} listings)'
    else:
        filtered_stats = brand_stats[brand_stats['count'] >= min_listings]
        title = f'MAPE by Brand (Popular: ≥ {min_listings} listings)'
    
    filtered_stats = filtered_stats.sort_values('mean', ascending=False).head(top_n)
    filtered_stats = filtered_stats.reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.barplot(
        data=filtered_stats,
        x='Vehicle_brand',
        y='mean',
        palette='mako',
        ax=ax
    )
    
    plt.xticks(rotation=75, ha='right')
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel('Vehicle Brand', fontsize=12)
    ax.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_if_verbose(f"[OK] Saved plot: {save_path}")
    
    return fig

def plot_ridge_coefficients(
    model: Pipeline,
    top_n: int = 15,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot Ridge regression coefficients.
    
    Parameters
    ----------
    model : Pipeline
        Fitted Ridge pipeline
    top_n : int, default=15
        Number of top features to show
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    coefficients = model.named_steps['model'].coef_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2E7D32' if c > 0 else '#D32F2F' for c in importance_df['Coefficient']]
    
    ax.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors, alpha=0.7)
    ax.invert_yaxis()
    ax.set_title(f'Top {top_n} Ridge Coefficients', fontsize=16, weight='bold')
    ax.set_xlabel('Coefficient Value', fontsize=12)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_if_verbose(f"[OK] Saved plot: {save_path}")
    
    return fig


def plot_tree_feature_importance(
    model: Pipeline,
    top_n: int = 20,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot feature importance for tree-based models.
    
    Works for Random Forest and XGBoost.
    
    Parameters
    ----------
    model : Pipeline
        Fitted tree model pipeline
    top_n : int, default=20
        Number of top features to show
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    importances = model.named_steps['model'].feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    importance_df['Importance_pct'] = (
        importance_df['Importance'] / importance_df['Importance'].sum() * 100
    )
    importance_df = importance_df.sort_values('Importance_pct', ascending=True)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.barplot(
        data=importance_df,
        y='Feature',
        x='Importance_pct',
        palette='Blues_d',
        legend=False,
        ax=ax
    )
    
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=16, weight='bold')
    ax.set_xlabel('Relative Importance (%)', fontsize=12)
    ax.set_ylabel('')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_if_verbose(f"[OK] Saved plot: {save_path}")
    
    return fig

def plot_learning_curves(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int = 5,
    title: str = "Learning Curves",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate learning curves to diagnose overfitting/underfitting.
    
    Parameters
    ----------
    model : Pipeline
        Model to evaluate
    X : pd.DataFrame
        Features
    y : np.ndarray
        Target variable
    cv : int, default=5
        Number of CV folds
    title : str, default="Learning Curves"
        Plot title
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
        
    Notes
    -----
    - Converging curves: Good generalization
    - Large gap: Overfitting
    - High training error: Underfitting
    """
    print_if_verbose(f"\nGenerating learning curves (CV={cv})...")
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error',
        random_state=42
    )
    
    train_mean = -train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = -val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_sizes, train_mean, 'o-', color='#1976D2', label='Training MSE', linewidth=2)
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
        color='#1976D2'
    )
    
    ax.plot(train_sizes, val_mean, 'o-', color='#D32F2F', label='Validation MSE', linewidth=2)
    ax.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.2,
        color='#D32F2F'
    )
    
    ax.set_title(title, fontsize=16, weight='bold')
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    
    ax.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_if_verbose(f"[OK] Saved plot: {save_path}")
    
    return fig

def create_model_comparison_plot(
    models_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create comprehensive model comparison visualization.
    
    Parameters
    ----------
    models_metrics : dict
        Dictionary of {model_name: {metric: value}}
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
        
    Examples
    --------
    >>> metrics = {
    ...     'Ridge': {'R2': 0.724, 'RMSE': 69707, 'MAE': 19355, 'MAPE': 0.285},
    ...     'RF': {'R2': 0.922, 'RMSE': 37185, 'MAE': 13097, 'MAPE': 0.228},
    ...     'XGB': {'R2': 0.930, 'RMSE': 35170, 'MAE': 11900, 'MAPE': 0.186}
    ... }
    >>> fig = create_model_comparison_plot(metrics)
    """
    df = pd.DataFrame(models_metrics).T
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics_config = [
        ('R2', 'R² Score', True),
        ('RMSE', 'RMSE (PLN)', False),
        ('MAE', 'MAE (PLN)', False),
        ('MAPE', 'MAPE (%)', False)
    ]
    
    for idx, (metric, title, ascending) in enumerate(metrics_config):
        ax = axes[idx]
        
        data = df[metric].sort_values(ascending=ascending)
        
        colors = ['#2E7D32' if i == len(data)-1 else '#1976D2' for i in range(len(data))]
        ax.barh(data.index, data.values, color=colors, alpha=0.8)
        
        for i, (idx_name, value) in enumerate(data.items()):
            if metric == 'MAPE':
                label = f'{value*100:.1f}%'
            elif metric in ['RMSE', 'MAE']:
                label = f'{value:,.0f}'
            else:
                label = f'{value:.3f}'
            
            ax.text(value, i, f'  {label}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel(metric if metric == 'R2' else 'Value', fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        
        if metric in ['RMSE', 'MAE']:
            ax.xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
            )
    
    plt.suptitle('Model Performance Comparison', fontsize=18, weight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_if_verbose(f"[OK] Saved plot: {save_path}")
    
    return fig