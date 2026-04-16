# src/features.py

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OrdinalEncoder, 
    OneHotEncoder, 
    StandardScaler, 
    PowerTransformer, 
    PolynomialFeatures
)
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder

PREMIUM_BRANDS = [
    'alfa romeo', 'alpine', 'aston martin', 'audi', 'bentley', 'bmw',
    'cadillac', 'cupra', 'ds automobiles', 'ferrari', 'genesis',
    'infiniti', 'jaguar', 'lamborghini', 'land rover', 'lexus',
    'lincoln', 'lotus', 'maserati', 'maybach', 'mclaren',
    'mercedes-benz', 'mini', 'porsche', 'rolls-royce', 'tesla', 'volvo'
]

def get_age_category(age: float) -> str:
    """Categorize vehicle by age."""
    if age < 3:
        return 'New'
    elif age < 9:
        return 'Recent'
    elif age < 17:
        return 'Used'
    else:
        return 'Old'


def get_usage_category(mileage_per_year: float) -> str:
    """Categorize vehicle by usage intensity."""
    if pd.isna(mileage_per_year):
        return 'Unknown'
    elif mileage_per_year < 10000:
        return 'Low'
    elif mileage_per_year < 20000:
        return 'Average'
    elif mileage_per_year < 30000:
        return 'High'
    else:
        return 'Very_High'


def get_performance_category(hp_per_liter: float) -> str:
    """Categorize engine by performance tier."""
    if pd.isna(hp_per_liter):
        return 'Unknown'
    elif hp_per_liter < 60:
        return 'Economy'
    elif hp_per_liter < 100:
        return 'Standard'
    elif hp_per_liter < 150:
        return 'Performance'
    else:
        return 'High_Performance'

def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create base features that don't require fitting on training data.
    
    Creates:
    - Age features (category, flags)
    - Usage features (mileage per year, intensity)
    - Performance features (HP per liter, category)
    - Market segment features (premium, supercar, collector)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with vehicle data
        
    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features
    """
    df = df.copy()

    if 'Vehicle_age' not in df.columns and 'Production_year' in df.columns:
        from datetime import datetime
        df['Vehicle_age'] = datetime.now().year - df['Production_year']
    
    if 'Vehicle_age' in df.columns:
        df['Age_category'] = df['Vehicle_age'].apply(get_age_category)
        df['Is_new_car'] = (df['Vehicle_age'] < 3).astype('Int64')
        df['Is_old_car'] = (df['Vehicle_age'] > 16).astype('Int64')
    
    if 'Mileage_km' in df.columns and 'Vehicle_age' in df.columns:
        df['Mileage_per_year'] = df['Mileage_km'] / df['Vehicle_age'].replace(0, 1)
        df['Usage_intensity'] = df['Mileage_per_year'].apply(get_usage_category)
    
    if 'Power_HP' in df.columns and 'Displacement_cm3' in df.columns:
        displacement_safe = df['Displacement_cm3'].replace(0, 100)  # 100cc minimum
        df['HP_per_liter'] = df['Power_HP'] / (displacement_safe / 1000)
        df['HP_per_liter'] = df['HP_per_liter'].replace([np.inf, -np.inf], np.nan)
        df['Performance_category'] = df['HP_per_liter'].apply(get_performance_category)
    
    if 'Vehicle_brand' in df.columns:
        df['Is_premium'] = (
            df['Vehicle_brand'].str.lower().isin(PREMIUM_BRANDS).astype('Int64')
        )
    
    if 'Power_HP' in df.columns and 'Is_premium' in df.columns:
        df['Is_supercar'] = (
            (df['Power_HP'] > 500) & (df['Is_premium'] == 1)
        ).astype('Int64')
    
    if 'Vehicle_age' in df.columns:
        df['Is_collector'] = (df['Vehicle_age'] > 25).astype('Int64')

    if 'Features' in df.columns:
        df['Num_features'] = (
            df['Features']
            .fillna('')
            .apply(lambda x: len([f for f in str(x).split(',') if f.strip()]))
        )

    # Extract listing year before dropping Offer_publication_date so the model
    # can distinguish 2019 prices from 2024 prices for the same car specs.
    if 'Offer_publication_date' in df.columns:
        from datetime import datetime
        df['Listing_year'] = (
            pd.to_datetime(df['Offer_publication_date'], errors='coerce')
            .dt.year
            .fillna(datetime.now().year)
            .astype(int)
        )

    cols_to_drop = [
        'Vehicle_generation',
        'Production_year',
        'Index',
        'Offer_publication_date',
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df


def _get_brand_tier(brand: str) -> str:
    """Classify a brand into one of five market tiers."""
    b = str(brand).strip().lower()

    _ultra_luxury = {
        'ferrari', 'lamborghini', 'rolls-royce', 'bentley', 'mclaren',
        'bugatti', 'koenigsegg', 'pagani', 'aston martin', 'maybach',
    }
    _luxury = {
        'mercedes-benz', 'bmw', 'audi', 'porsche', 'lexus', 'jaguar',
        'maserati', 'tesla', 'land rover', 'infiniti', 'lincoln',
        'genesis', 'cadillac', 'volvo',
    }
    _premium = {
        'alfa romeo', 'mini', 'saab', 'ds automobiles', 'cupra',
        'alpine', 'lotus', 'subaru', 'acura', 'baic', 'ssangyong',
    }
    _mass_market = {
        'volkswagen', 'toyota', 'ford', 'hyundai', 'kia', 'honda',
        'opel', 'chevrolet', 'peugeot', 'renault', 'seat', 'skoda',
        'fiat', 'nissan', 'mazda', 'mitsubishi', 'suzuki', 'dacia',
        'citroen', 'citroën', 'dodge', 'ram', 'jeep', 'chrysler',
        'lancia', 'daewoo', 'lada',
    }

    if b in _ultra_luxury:
        return 'Ultra_Luxury'
    if b in _luxury:
        return 'Luxury'
    if b in _premium:
        return 'Premium'
    if b in _mass_market:
        return 'Mass_Market'
    return 'Niche'


def apply_advanced_transformations(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply transformations that require fitting on training data.
    
    Includes:
    - Missing value imputation
    - Log transformations
    - Polynomial features
    - Interaction terms
    - Brand-level market features (tier, frequency, rarity, popularity)
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
        
    Returns
    -------
    tuple
        (X_train_transformed, X_test_transformed)
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    
    # ========================================================================
    # IMPUTE MISSING VALUES
    # ========================================================================
    
    numeric_cols_to_fill = ['Mileage_km', 'Power_HP', 'Displacement_cm3', 'Doors_number']

    for col in numeric_cols_to_fill:
        if col in X_train.columns:
            if col == 'Doors_number':
                mode_result = X_train[col].mode()
                fill_value = mode_result.iloc[0] if not mode_result.empty else 4
            else:
                fill_value = X_train[col].median()
            # If median is NaN (all values missing), fall back to a safe default
            if pd.isna(fill_value):
                fill_value = 0

            X_train[col] = X_train[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)
        
    if 'Vehicle_age' in X_train.columns:
        age_median = X_train['Vehicle_age'].median()
        X_train['Vehicle_age'] = X_train['Vehicle_age'].fillna(age_median)
        X_test['Vehicle_age'] = X_test['Vehicle_age'].fillna(age_median)

    for df in [X_train, X_test]:
        if 'Power_HP' in df.columns and 'Displacement_cm3' in df.columns:
            displacement_safe = df['Displacement_cm3'].replace(0, 100)
            df['HP_per_liter'] = df['Power_HP'] / (displacement_safe / 1000)

        if 'Mileage_km' in df.columns and 'Vehicle_age' in df.columns:
            df['Mileage_per_year'] = df['Mileage_km'] / df['Vehicle_age'].replace(0, 1)

        if 'Is_premium' in df.columns and 'Power_HP' in df.columns:
            df['Is_supercar'] = ((df['Power_HP'] > 500) & (df['Is_premium'] == 1)).astype('Int64')
    
    # ========================================================================
    # LOG TRANSFORMATIONS (protect against zeros and negatives)
    # ========================================================================
    for df in [X_train, X_test]:
        if 'Mileage_km' in df.columns:
            df['Mileage_km_log'] = np.log1p(df['Mileage_km'].clip(lower=0))
        
        if 'Power_HP' in df.columns:
            df['Power_HP_log'] = np.log1p(df['Power_HP'].clip(lower=0))
        
        if 'Displacement_cm3' in df.columns:
            df['Displacement_cm3_log'] = np.log1p(df['Displacement_cm3'].clip(lower=0))
    
    if 'Drive' in X_train.columns:
        mode_result = X_train['Drive'].mode()
        train_mode = mode_result.iloc[0] if not mode_result.empty else 'Unknown'

        X_train['Drive'] = X_train['Drive'].fillna(train_mode)
        X_test['Drive'] = X_test['Drive'].fillna(train_mode)

    if 'Type' in X_train.columns:
        mode_result = X_train['Type'].mode()
        if not mode_result.empty:
            train_mode = mode_result.iloc[0]
            X_train['Type'] = X_train['Type'].fillna(train_mode)
            X_test['Type'] = X_test['Type'].fillna(train_mode)

    if 'Transmission' in X_train.columns:
        mode_result = X_train['Transmission'].mode()
        train_mode = mode_result.iloc[0] if not mode_result.empty else 'Unknown'

        X_train['Transmission'] = X_train['Transmission'].fillna(train_mode)
        X_test['Transmission'] = X_test['Transmission'].fillna(train_mode)

    # ========================================================================
    # POLYNOMIAL FEATURES (protect against NaN)
    # ========================================================================
    for df in [X_train, X_test]:
        if 'Vehicle_age' in df.columns:
            age_safe = df['Vehicle_age'].fillna(0)
            df['Vehicle_age_squared'] = age_safe ** 2
        
        if 'Power_HP' in df.columns:
            power_safe = df['Power_HP'].fillna(0)
            df['Power_HP_squared'] = power_safe ** 2
        
        if 'Mileage_km' in df.columns:
            mileage_safe = df['Mileage_km'].fillna(0)
            df['Mileage_km_squared'] = mileage_safe ** 2
    
    # ========================================================================
    # INTERACTION TERMS (protect against NaN)
    # ========================================================================
    for df in [X_train, X_test]:
        if 'Vehicle_age' in df.columns and 'Mileage_km' in df.columns:
            age_safe = df['Vehicle_age'].fillna(0)
            mileage_safe = df['Mileage_km'].fillna(0)
            df['Age_Mileage_interaction'] = age_safe * mileage_safe
        
        if 'Power_HP' in df.columns and 'Vehicle_age' in df.columns:
            power_safe = df['Power_HP'].fillna(0)
            age_safe = df['Vehicle_age'].fillna(0)
            df['Power_Age_interaction'] = power_safe * age_safe
        
        if 'Mileage_per_year' in df.columns and 'Vehicle_age' in df.columns:
            mpy_safe = df['Mileage_per_year'].fillna(0)
            age_safe = df['Vehicle_age'].fillna(0)
            df['Mileage_per_year_Age'] = mpy_safe * age_safe

    # ========================================================================
    # BRAND-LEVEL MARKET FEATURES
    # ========================================================================
    if 'Vehicle_brand' in X_train.columns:
        brand_col = X_train['Vehicle_brand'].str.lower().str.strip()
        brand_freq = brand_col.value_counts().to_dict()

        for df, is_train in [(X_train, True), (X_test, False)]:
            brand_lower = df['Vehicle_brand'].str.lower().str.strip()
            df['Brand_tier'] = brand_lower.apply(_get_brand_tier)
            df['Brand_frequency'] = brand_lower.map(brand_freq).fillna(1).astype(int)

            max_freq = max(brand_freq.values()) if brand_freq else 1
            raw_rarity = np.log1p(max_freq / df['Brand_frequency'].clip(lower=1))
            max_rarity = np.log1p(max_freq)
            df['Rarity_index'] = (raw_rarity / max_rarity).clip(upper=1.0).round(4)

            model_col = (brand_lower + '_' + df['Vehicle_model'].astype(str).str.lower().str.strip())
            if is_train:
                bm_freq_map = model_col.value_counts().to_dict()
            df['BrandModel_frequency'] = model_col.map(bm_freq_map).fillna(1).astype(int)

            df['Brand_popularity'] = pd.cut(
                df['Brand_frequency'],
                bins=[0, 5, 20, 100, 500, np.inf],
                labels=['Ultra_Rare', 'Rare', 'Uncommon', 'Common', 'Popular'],
                right=True,
            ).astype(str)

    return X_train, X_test

def get_preprocessor_tree() -> ColumnTransformer:
    """
    Get preprocessor for tree-based models.
    
    Returns
    -------
    ColumnTransformer
        Preprocessor with median imputation and ordinal encoding
    """
    num_pipeline_tree = SimpleImputer(strategy='median')

    cat_pipeline_tree = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value', 
            unknown_value=-1
        ))
    ])

    preprocessor_tree = ColumnTransformer([
        ('num', num_pipeline_tree, make_column_selector(dtype_include='number')),
        ('cat', cat_pipeline_tree, make_column_selector(dtype_include=['object', 'category']))
    ])

    return preprocessor_tree


def get_preprocessor_mastered(smoothing: int = 200) -> ColumnTransformer:
    """
    Get advanced preprocessor for linear models.
    
    Parameters
    ----------
    smoothing : int, default=200
        TargetEncoder smoothing parameter
        
    Returns
    -------
    ColumnTransformer
        Preprocessor with transformations and encodings
    """
    num_cols = ['Mileage_km', 'Power_HP', 'Displacement_cm3', 'Vehicle_age']
    cat_cols_to_encode = ['Vehicle_brand', 'Vehicle_model']
    cat_cols_simple = ['Fuel_type', 'Transmission', 'Drive', 'Type']

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('yeo', PowerTransformer(method='yeo-johnson')),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True))
    ])

    preprocessor_mastered = ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('target', TargetEncoder(smoothing=smoothing), cat_cols_to_encode),
        ('cat_simple', OneHotEncoder(handle_unknown='ignore'), cat_cols_simple)
    ])

    return preprocessor_mastered


def get_preprocessor_v2(smoothing: int = 300) -> ColumnTransformer:
    """
    Get preprocessor for the tuned XGBoost model with brand-level features.
    
    Uses StandardScaler for numeric features (including brand frequency metrics),
    TargetEncoder for high-cardinality categoricals, and OneHotEncoder for
    low-cardinality categoricals including brand tier and popularity.
    
    Parameters
    ----------
    smoothing : int, default=300
        TargetEncoder smoothing parameter
        
    Returns
    -------
    ColumnTransformer
        Preprocessor matching the notebook's preprocessor_v2
    """
    num_cols_v2 = [
        'Mileage_km', 'Power_HP', 'Displacement_cm3', 'Vehicle_age',
        'Brand_frequency', 'Rarity_index', 'BrandModel_frequency',
    ]
    cat_cols_encode = ['Vehicle_brand', 'Vehicle_model']
    cat_cols_ohe = ['Fuel_type', 'Transmission', 'Drive', 'Type', 'Brand_tier', 'Brand_popularity']

    preprocessor_v2 = ColumnTransformer([
        ('num', StandardScaler(), num_cols_v2),
        ('target', TargetEncoder(smoothing=smoothing), cat_cols_encode),
        ('cat_simple', OneHotEncoder(handle_unknown='ignore'), cat_cols_ohe),
    ])

    return preprocessor_v2