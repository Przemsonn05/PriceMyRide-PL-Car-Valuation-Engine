# src/preprocessing.py
import pandas as pd
import numpy as np
from src.utils import get_current_eur_pln_rate

def clean_car_data(df):
    """Function cleaning raw data; preparation for EDA section and feature engineering."""
    df = df.copy()

    eur_rate = get_current_eur_pln_rate()
    df['price_PLN'] = df.apply(
        lambda row: row['Price'] * eur_rate if row['Currency'] == 'EUR' else row['Price'],
        axis=1
    )
    df = df.drop(columns=['Currency', 'Price'])

    cols_to_drop = ['CO2_emissions', 'First_registration_date', 'Vehicle_version']
    df = df.drop(columns=cols_to_drop)
    
    cols_to_check = df.columns.difference(['Index'])
    df = df.drop_duplicates(subset=cols_to_check, keep='first')

    df['First_owner'] = (df['First_owner'] == 'Yes').astype(int)
    df['Origin_country'] = df['Origin_country'].fillna('unknown')

    cols_to_int = ['Doors_number', 'Mileage_km', 'Power_HP', 'Displacement_cm3']
    for col in cols_to_int:
        df[col] = df[col].astype('Int64')

    cat_cols = ['Condition', 'Fuel_type', 'Transmission', 'Drive', 'Type', 'Colour', 'Vehicle_brand']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    df['Offer_publication_date'] = pd.to_datetime(df['Offer_publication_date'], format='%d/%m/%Y')
    
    text_cols = ['Vehicle_brand', 'Vehicle_model', 'Fuel_type', 'Transmission', 'Drive', 'Type', 'Colour', 'Origin_country', 'Condition']
    for col in text_cols:
        df[col] = df[col].str.strip().str.lower()

    df['Features'] = df['Features'].str.replace(r"[\[\]']", "", regex=True).str.lower()

    return df