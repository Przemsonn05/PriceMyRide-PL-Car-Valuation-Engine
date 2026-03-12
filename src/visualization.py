# src/visualization.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Optional, Tuple
from pathlib import Path


def plot_price_distribution(
    df: pd.DataFrame,
    market_limit: int = 500000,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generate linear and logarithmic price distribution plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'price_PLN' column
    market_limit : int, default=500000
        Maximum price for linear plot
    figsize : tuple, default=(12, 6)
        Figure size (width, height)
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    sns.histplot(
        df[df['price_PLN'] <= market_limit]['price_PLN'], 
        kde=True, ax=axes[0], color='#2a9d8f', bins=50
    )
    axes[0].set_title(
        f'Distribution of Prices (up to {int(market_limit/1000)}k PLN)', 
        fontsize=14, fontweight='bold'
    )
    axes[0].set_xlabel('Price (PLN)')
    axes[0].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
    )

    sns.histplot(
        df['price_PLN'], 
        kde=True, ax=axes[1], color='#e76f51', log_scale=True
    )
    axes[1].set_title(
        'Price Distribution (Log Scale - All Data)', 
        fontsize=14, fontweight='bold'
    )
    axes[1].set_xlabel('Price (PLN) - Log Scale')
    axes[1].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
    )

    median_price = df['price_PLN'].median()
    mean_price = df['price_PLN'].mean()
    axes[0].axvline(
        median_price, color='red', linestyle='--', 
        label=f'Median: {int(median_price):,} PLN'
    )
    axes[0].axvline(
        mean_price, color='blue', linestyle='--', 
        label=f'Mean: {int(mean_price):,} PLN'
    )
    axes[0].legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_depreciation_analysis(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Analyze depreciation curve and annual depreciation rates.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'Vehicle_age' and 'price_PLN' columns
    figsize : tuple, default=(14, 7)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    age_price = df.groupby('Vehicle_age')['price_PLN'].agg(['median', 'mean', 'count'])
    age_price = age_price[age_price['count'] >= 10]

    axes[0].plot(
        age_price.index, age_price['median'], 
        'o-', linewidth=2.5, markersize=6, 
        color='#e74c3c', label='Median Price', alpha=0.8
    )
    axes[0].fill_between(
        age_price.index, age_price['median'], 
        alpha=0.2, color='#e74c3c'
    )
    axes[0].set_title('Depreciation Curve', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Vehicle Age (years)')
    axes[0].set_ylabel('Median Price (PLN)')
    axes[0].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    depreciation_rates = []
    valid_ages = age_price.index.values
    
    for i in range(1, min(22, int(valid_ages.max()))):
        if i in age_price.index and i-1 in age_price.index:
            price_now = age_price.loc[i, 'median']
            price_prev = age_price.loc[i-1, 'median']
            rate = ((price_prev - price_now) / price_prev) * 100
            depreciation_rates.append({'age': i, 'rate': rate})

    if depreciation_rates:
        dep_df = pd.DataFrame(depreciation_rates)
        
        colors = [
            '#e74c3c' if r > 15 else '#f39c12' if r > 10 else '#2ecc71' 
            for r in dep_df['rate']
        ]
        
        axes[1].bar(
            dep_df['age'], dep_df['rate'], 
            color=colors, alpha=0.7, edgecolor='black'
        )
        axes[1].axhline(
            y=dep_df['rate'].mean(), 
            color='blue', linestyle='--', 
            label=f'Avg: {dep_df["rate"].mean():.1f}%'
        )
        axes[1].set_title(
            'Annual Depreciation Rate', 
            fontsize=16, fontweight='bold'
        )
        axes[1].set_xlabel('Vehicle Age (years)')
        axes[1].set_ylabel('Depreciation Rate (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_numerical_relationships(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create scatter plots for 4 key numerical features vs price.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numerical features
    figsize : tuple, default=(16, 12)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    point_color = "#43a196"
    
    plot_df = df[
        (df['Mileage_km'] <= 1000000) & 
        (df['price_PLN'] <= 1000000) & 
        (df['Power_HP'] <= 1000)
    ].copy()

    fig, ax = plt.subplots(2, 2, figsize=figsize)
    
    configs = [
        ('Production_year', 'Price vs Production Year', 'Production Year'),
        ('Mileage_km', 'Price vs Mileage', 'Mileage [km]'),
        ('Power_HP', 'Price vs Engine Power', 'Power [HP]'),
        ('Displacement_cm3', 'Price vs Engine Displacement', 'Displacement [cm³]')
    ]

    for i, (col, title, xlabel) in enumerate(configs):
        curr_ax = ax[i//2, i%2]
        
        curr_ax.scatter(
            plot_df[col], plot_df['price_PLN'], 
            alpha=0.2, s=10, c=point_color
        )
        curr_ax.set_title(title, weight='bold', fontsize=14)
        curr_ax.set_xlabel(xlabel)
        curr_ax.set_ylabel('Price (PLN)')
        curr_ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
        )
        
        if col in ['Mileage_km', 'Displacement_cm3']:
            curr_ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
            )
        
        curr_ax.grid(alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_mileage_vs_price_by_age(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Scatter plot of mileage vs price grouped by age categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with mileage, price, and age columns
    figsize : tuple, default=(14, 8)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    plot_df = df[
        (df['Mileage_km'] < 800000) & 
        (df['price_PLN'] < 2000000)
    ].copy()
    
    fig, ax = plt.subplots(figsize=figsize)

    age_groups = [
        (0, 2, 'New (0-2y)', '#264653'),      
        (3, 8, 'Recent (3-8y)', '#2a9d8f'),   
        (9, 16, 'Used (9-16y)', '#f4a261'),  
        (17, 100, 'Old (>16y)', '#e76f51')    
    ]

    for min_age, max_age, label, color in age_groups:
        subset = plot_df[
            (plot_df['Vehicle_age'] >= min_age) & 
            (plot_df['Vehicle_age'] <= max_age)
        ]
        ax.scatter(
            subset['Mileage_km'], subset['price_PLN'], 
            c=color, label=label, alpha=0.7, s=12
        )

    ax.set_title(
        'Price vs Mileage - Colored by Vehicle Age', 
        fontsize=16, fontweight='bold'
    )
    ax.set_xlabel('Mileage (km)')
    ax.set_ylabel('Price (PLN)')
    ax.legend(title="Vehicle Age")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_fuel_type_trends(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 7),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot average price trends over years by fuel type.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price, production year, and fuel type
    figsize : tuple, default=(12, 7)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    plot_df = df[df['price_PLN'] <= 1000000].copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.lineplot(
        data=plot_df, 
        x='Production_year', 
        y='price_PLN', 
        hue='Fuel_type', 
        errorbar=None, 
        lw=2,
        ax=ax
    )
    
    ax.axhline(
        df['price_PLN'].mean(), 
        color='crimson', linestyle='--', 
        alpha=0.5, label='Market average'
    )
    ax.set_title(
        "Average Car Price Over the Years by Fuel Type", 
        weight='bold', fontsize=16
    )
    ax.set_xlabel('Production Year')
    ax.set_ylabel('Average Price (PLN)')
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, p: f'{int(x/1000)}k')
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create correlation heatmap of numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numerical features
    figsize : tuple, default=(10, 8)
        Figure size
    save_path : Path, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    corr = df.select_dtypes(include=[np.number]).corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr, 
        mask=mask, 
        annot=True, 
        vmin=-1, vmax=1, 
        cmap='coolwarm', 
        fmt='.2f', 
        square=True, 
        ax=ax
    )
    ax.set_title(
        "Correlation Heatmap of Numerical Features", 
        fontsize=16, fontweight="bold", pad=20
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig