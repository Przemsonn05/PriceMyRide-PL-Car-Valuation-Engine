# src/preprocessing.py

import pandas as pd
import numpy as np

from .utils import get_current_eur_pln_rate

#: Mean EUR/PLN rate in 2021 (NBP archive).  Used as a safe historical
#: fallback for the legacy Car_sale_ads.csv dataset.
HISTORICAL_EUR_PLN_2021 = 4.565


def _historical_eur_pln_rate(df: pd.DataFrame) -> float:
    """Pick the most appropriate EUR/PLN rate for *df*.

    If the dataset looks historical (median offer publication year <= 2022),
    use a fixed historical rate to avoid applying 2025+ FX rates to 2021
    listings.  Otherwise fetch the current rate from NBP.
    """
    if 'Offer_publication_date' in df.columns:
        dates = pd.to_datetime(df['Offer_publication_date'], errors='coerce', format='%d/%m/%Y')
        if not dates.isna().all() and int(dates.dt.year.median()) <= 2022:
            return HISTORICAL_EUR_PLN_2021
    try:
        return get_current_eur_pln_rate()
    except Exception:
        return HISTORICAL_EUR_PLN_2021


def clean_car_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess raw car data.
    
    Performs:
    - Currency conversion (EUR → PLN)
    - Duplicate removal
    - Missing value handling
    - Type conversions
    - Text normalization
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw car data
        
    Returns
    -------
    pd.DataFrame
        Cleaned data
    """
    df = df.copy()

    # Convert EUR listings to PLN. The legacy CSV is from 2021, so we use a
    # historical EUR/PLN rate (~4.55, year-2021 mean from NBP archives) rather
    # than today's rate — using the current rate would distort historical
    # prices. For the scraped balanced dataset this branch is a no-op
    # because Otomoto publishes PLN prices.
    eur_rate = _historical_eur_pln_rate(df)
    if {'Price', 'Currency'}.issubset(df.columns):
        df['price_PLN'] = df.apply(
            lambda row: row['Price'] * eur_rate if row['Currency'] == 'EUR' else row['Price'],
            axis=1,
        )
        df = df.drop(columns=['Currency', 'Price'])

    cols_to_drop = ['CO2_emissions', 'First_registration_date', 'Vehicle_version']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    cols_to_check = df.columns.difference(['Index'])
    df = df.drop_duplicates(subset=cols_to_check, keep='first')

    df['First_owner'] = (df['First_owner'] == 'Yes').astype(int)
    
    df['Origin_country'] = df['Origin_country'].fillna('unknown')

    cols_to_int = ['Doors_number', 'Mileage_km', 'Power_HP', 'Displacement_cm3']
    for col in cols_to_int:
        if col in df.columns:
            df[col] = df[col].astype('Int64')

    cat_cols = [
        'Condition', 'Fuel_type', 'Transmission', 
        'Drive', 'Type', 'Colour', 'Vehicle_brand'
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    if 'Offer_publication_date' in df.columns:
        df['Offer_publication_date'] = pd.to_datetime(
            df['Offer_publication_date'], 
            format='%d/%m/%Y',
            errors='coerce'
        )
    
    text_cols = [
        'Vehicle_brand', 'Vehicle_model', 'Fuel_type', 
        'Transmission', 'Drive', 'Type', 'Colour', 
        'Origin_country', 'Condition'
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()

    if 'Features' in df.columns:
        df['Features'] = df['Features'].str.replace(r"[\[\]']", "", regex=True).str.lower()

    return df