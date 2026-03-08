# src/features.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, PowerTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder

# --- STAŁE ---
PREMIUM_BRANDS = [
    'alfa romeo', 'alpine', 'aston martin', 'audi', 'bentley', 'bmw',
    'cadillac', 'cupra', 'ds automobiles', 'ferrari', 'infiniti',
    'jaguar', 'lamborghini', 'land rover', 'lexus', 'lincoln',
    'lotus', 'maserati', 'maybach', 'mclaren', 'mercedes-benz',
    'mini', 'porsche', 'rolls-royce', 'tesla'
]


def get_age_category(age):
    if age < 3: return 'New'
    elif age < 9: return 'Recent'
    elif age < 17: return 'Used'
    else: return 'Old'

def get_usage_category(mileage_per_year):
    if mileage_per_year < 10000: return 'Low'
    elif mileage_per_year < 20000: return 'Average'
    elif mileage_per_year < 30000: return 'High'
    else: return 'Very_High'

def get_performance_category(hp_per_liter):
    if pd.isna(hp_per_liter): return 'Unknown'
    elif hp_per_liter < 60: return 'Economy'
    elif hp_per_liter < 100: return 'Standard'
    elif hp_per_liter < 150: return 'Performance'
    else: return 'High_Performance'


def engineer_base_features(df):
    """Tworzy podstawowe cechy, które nie wymagają statystyk z X_train."""
    df = df.copy()
    
    df['Age_category'] = df['Vehicle_age'].apply(get_age_category)
    df['Is_new_car'] = (df['Vehicle_age'] < 3).astype(int)
    df['Is_old_car'] = (df['Vehicle_age'] > 16).astype(int)
    
    df['Mileage_per_year'] = df['Mileage_km'] / df['Vehicle_age'].replace(0, 1)
    df['Usage_intensity'] = df['Mileage_per_year'].apply(get_usage_category)
    
    df['HP_per_liter'] = df['Power_HP'] / (df['Displacement_cm3'] / 1000 + 0.1) # +0.1 zapobiega div/0
    df['Performance_category'] = df['HP_per_liter'].apply(get_performance_category)
    
    df['Is_premium'] = df['Vehicle_brand'].str.lower().isin(PREMIUM_BRANDS).astype(int)
    df['Is_supercar'] = ((df['Power_HP'] > 500) & (df['Is_premium'] == 1)).astype(int)
    df['is_collector'] = (df['Vehicle_age'] > 25).astype(int)
    
    cols_to_drop = ['Vehicle_generation', 'Production_year', 'Index', 'Offer_publication_date']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df

def apply_advanced_transformations(X_train, X_test):
    """Wykonuje transformacje wymagające dopasowania (fit) do X_train."""
    X_train, X_test = X_train.copy(), X_test.copy()
    
    numeric_cols = ['Mileage_km', 'Power_HP', 'Displacement_cm3', 'Doors_number']
    imputer_num = SimpleImputer(strategy='median')
    
    door_fill = X_train['Doors_number'].mode()[0] if not X_train['Doors_number'].mode().empty else 5
    X_train['Doors_number'] = X_train['Doors_number'].fillna(door_fill)
    X_test['Doors_number'] = X_test['Doors_number'].fillna(door_fill)
    
    cols_to_impute = ['Mileage_km', 'Power_HP', 'Displacement_cm3']
    X_train[cols_to_impute] = imputer_num.fit_transform(X_train[cols_to_impute])
    X_test[cols_to_impute] = imputer_num.transform(X_test[cols_to_impute])
    
    for df in [X_train, X_test]:
        df['Mileage_km_log'] = np.log1p(df['Mileage_km'])
        df['Power_HP_log'] = np.log1p(df['Power_HP'])
        df['Displacement_cm3_log'] = np.log1p(df['Displacement_cm3'])
        
        df['Vehicle_age_squared'] = df['Vehicle_age'] ** 2
        df['Power_HP_squared'] = df['Power_HP'] ** 2
        df['Mileage_km_squared'] = df['Mileage_km'] ** 2
        
        df['Age_Mileage_interaction'] = df['Vehicle_age'] * df['Mileage_km']
        df['Power_Age_interaction'] = df['Power_HP'] * df['Vehicle_age']
        df['Mileage_per_year_Age'] = (df['Mileage_km'] / df['Vehicle_age'].replace(0, 1)) * df['Vehicle_age']

    return X_train, X_test

def get_preprocessor_mastered():
    """Returns compiled ColumnTransformer ready to use in a model."""
    num_cols = ['Mileage_km', 'Power_HP', 'Displacement_cm3', 'Vehicle_age']
    cat_cols_to_encode = ['Vehicle_brand', 'Vehicle_model']
    cat_cols_simple = ['Fuel_type', 'Transmission', 'Drive', 'Type']

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('yeo', PowerTransformer(method='yeo-johnson')),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True))
    ])

    return ColumnTransformer([
        ('num', num_transformer, num_cols),
        ('target', TargetEncoder(smoothing=200), cat_cols_to_encode),
        ('cat_simple', OneHotEncoder(handle_unknown='ignore'), cat_cols_simple)
    ])