"""
src/data_cleaning.py

Cleaning and normalization pipeline for Otomoto-scraped car listing data.

Pipeline
--------
1. normalize_columns(df_raw)  — map raw Otomoto field names/values → project schema
2. clean_data(df_raw)         — full pipeline: normalize → clean_car_data()
3. deduplicate(df_new, df_existing) — remove rows whose offer_id already exists

The expected schema columns (must not change):
    Index, Condition, Vehicle_brand, Vehicle_model, Vehicle_generation,
    Production_year, Mileage_km, Power_HP, Displacement_cm3, Fuel_type,
    Drive, Transmission, Type, Doors_number, Colour, Origin_country,
    First_owner, Offer_publication_date, Offer_location, Features, price_PLN
"""

from __future__ import annotations

import re
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Translation maps  (Polish → English schema values)
# ---------------------------------------------------------------------------

FUEL_TYPE_MAP: dict[str, str] = {
    "benzyna": "Gasoline",
    "olej napędowy": "Diesel",
    "elektryczny": "Electric",
    "hybryda": "Hybrid",
    "hybryda plug-in": "Hybrid",
    "benzyna + lpg": "Gasoline + LPG",
    "benzyna+lpg": "Gasoline + LPG",
    "benzyna + cng": "Gasoline + CNG",
    "benzyna+cng": "Gasoline + CNG",
    "wodór": "Hydrogen",
    "hydrogen": "Hydrogen",
    "gasoline": "Gasoline",
    "diesel": "Diesel",
    "electric": "Electric",
    "hybrid": "Hybrid",
}

TRANSMISSION_MAP: dict[str, str] = {
    "manualna": "Manual",
    "automatyczna": "Automatic",
    "manual": "Manual",
    "automatic": "Automatic",
    "półautomatyczna": "Automatic",
}

DRIVE_MAP: dict[str, str] = {
    "przednie": "Front wheels",
    "tylne": "Rear wheels",
    "4x4 (dołączany automatycznie)": "4x4 (attached automatically)",
    "4x4 (stały)": "4x4 (permanent)",
    "4x4 (dołączany ręcznie)": "4x4 (attached manually)",
    "front wheels": "Front wheels",
    "rear wheels": "Rear wheels",
    "4x4 (attached automatically)": "4x4 (attached automatically)",
    "4x4 (permanent)": "4x4 (permanent)",
    "4x4 (attached manually)": "4x4 (attached manually)",
}

BODY_TYPE_MAP: dict[str, str] = {
    "sedan": "sedan",
    "kombi": "station_wagon",
    "station wagon": "station_wagon",
    "hatchback": "small_cars",
    "suv": "SUV",
    "minivan": "minivan",
    "van": "minivan",
    "coupe": "coupe",
    "kabriolet": "convertible",
    "convertible": "convertible",
    "kompakt": "compact",
    "compact": "compact",
    "miejski": "city_cars",
    "city car": "city_cars",
    "pick-up": "SUV",
    "pickup": "SUV",
}

COLOUR_MAP: dict[str, str] = {
    "biały": "White",
    "czarny": "Black",
    "szary": "Grey",
    "srebrny": "Silver",
    "niebieski": "Blue",
    "czerwony": "Red",
    "żółty": "Yellow",
    "zielony": "Green",
    "brązowy": "Other",
    "beżowy": "Other",
    "pomarańczowy": "Other",
    "fioletowy": "Other",
    "złoty": "Other",
    "biała": "White",
    "czarna": "Black",
    "szara": "Grey",
    "srebrna": "Silver",
    "niebieska": "Blue",
    "czerwona": "Red",
    "white": "White",
    "black": "Black",
    "grey": "Grey",
    "silver": "Silver",
    "blue": "Blue",
    "red": "Red",
    "yellow": "Yellow",
    "green": "Green",
}

CONDITION_MAP: dict[str, str] = {
    "używany": "Used",
    "nowy": "New",
    "used": "Used",
    "new": "New",
}

COUNTRY_TRANSLATE: dict[str, str] = {
    "polska": "Poland",
    "niemcy": "Germany",
    "francja": "France",
    "włochy": "Italy",
    "japonia": "Japan",
    "stany zjednoczone": "USA",
    "usa": "USA",
    "wielka brytania": "United Kingdom",
    "czechy": "Czech Republic",
    "austria": "Austria",
    "szwajcaria": "Switzerland",
    "belgia": "Belgium",
    "holandia": "Netherlands",
    "szwecja": "Sweden",
    "hisznania": "Spain",
    "dania": "Denmark",
    "finlandia": "Finland",
    "norwegia": "Norway",
    "portugalia": "Portugal",
    "korea": "South Korea",
    "chiny": "China",
    "rumunia": "Romania",
    "węgry": "Hungary",
    "słowacja": "Slovakia",
    "spain": "Spain",
    "hungary": "Hungary",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_numeric(value: str, unit_pattern: str = r"[^\d]") -> Optional[float]:
    """Extract numeric value from a string like '106 665 km' or '1 984 cm3'."""
    if not value or pd.isna(value):
        return None
    cleaned = re.sub(unit_pattern, "", str(value).replace("\xa0", " ").replace(" ", ""))
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_mileage(raw: str) -> Optional[float]:
    """Parse '55 005 km' → 55005.0"""
    return _parse_numeric(raw, r"[^\d]")


def _parse_displacement(raw: str) -> Optional[float]:
    """Parse '1 984 cm3' → 1984.0"""
    return _parse_numeric(raw, r"[^\d]")


def _parse_power(raw: str) -> Optional[float]:
    """Parse '272 KM' → 272.0"""
    return _parse_numeric(raw, r"[^\d]")


def _parse_date(raw: str) -> Optional[str]:
    """Parse ISO datetime string → dd/mm/yyyy string (existing format)."""
    if not raw or pd.isna(raw):
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            return dt.strftime("%d/%m/%Y")
        except ValueError:
            continue
    return None


def _translate(value: str, mapping: dict[str, str], default: str = "") -> str:
    if not value or pd.isna(value):
        return default
    return mapping.get(str(value).strip().lower(), default or str(value).strip())


def _first_owner_to_int(raw: str) -> int:
    """'Tak' / 'Yes' / 1 → 1, everything else → 0."""
    if not raw or pd.isna(raw):
        return 0
    normalized = str(raw).strip().lower()
    return 1 if normalized in ("tak", "yes", "1", "true") else 0


# ---------------------------------------------------------------------------
# Main normalization function
# ---------------------------------------------------------------------------

def normalize_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw Otomoto-scraped fields to the project schema columns.

    Input columns (from data_fetcher.RAW_COLUMNS):
        offer_id, offer_url, scraped_at, make, model, version, generation,
        year, mileage_raw, engine_capacity_raw, engine_power_raw,
        fuel_type_raw, gearbox_raw, transmission_raw, body_type_raw,
        door_count_raw, colour_raw, condition_raw, country_origin_raw,
        original_owner_raw, price_value, price_currency,
        city, region, features_raw, created_at

    Output columns (project schema):
        Index, Condition, Vehicle_brand, Vehicle_model, Vehicle_generation,
        Production_year, Mileage_km, Power_HP, Displacement_cm3, Fuel_type,
        Drive, Transmission, Type, Doors_number, Colour, Origin_country,
        First_owner, Offer_publication_date, Offer_location, Features,
        Price, Currency, offer_id
        (Price + Currency are kept so that clean_car_data() can convert them)

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw DataFrame returned by fetch_data().

    Returns
    -------
    pd.DataFrame
        DataFrame with schema-aligned columns.
    """
    df = df_raw.copy()

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------
    df["Index"] = range(len(df))

    # ------------------------------------------------------------------
    # Vehicle info
    # ------------------------------------------------------------------
    df["Vehicle_brand"] = df["make"].str.strip()
    df["Vehicle_model"] = df["model"].str.strip()
    df["Vehicle_generation"] = df.apply(
        lambda r: r.get("version", "") or r.get("generation", ""),
        axis=1
    ).str.strip()

    df["Production_year"] = pd.to_numeric(df["year"], errors="coerce")

    # ------------------------------------------------------------------
    # Numeric specs
    # ------------------------------------------------------------------
    df["Mileage_km"] = df["mileage_raw"].apply(_parse_mileage)
    df["Displacement_cm3"] = df["engine_capacity_raw"].apply(_parse_displacement)
    df["Power_HP"] = df["engine_power_raw"].apply(_parse_power)

    # ------------------------------------------------------------------
    # Categorical translations
    # ------------------------------------------------------------------
    df["Fuel_type"] = df["fuel_type_raw"].apply(
        lambda v: _translate(v, FUEL_TYPE_MAP)
    )
    df["Transmission"] = df["gearbox_raw"].apply(
        lambda v: _translate(v, TRANSMISSION_MAP)
    )
    df["Drive"] = df["transmission_raw"].apply(
        lambda v: _translate(v, DRIVE_MAP)
    )
    df["Type"] = df["body_type_raw"].apply(
        lambda v: _translate(v, BODY_TYPE_MAP)
    )
    df["Colour"] = df["colour_raw"].apply(
        lambda v: _translate(v, COLOUR_MAP, default="Other")
    )
    df["Condition"] = df["condition_raw"].apply(
        lambda v: _translate(v, CONDITION_MAP, default="Used")
    )
    df["Origin_country"] = df["country_origin_raw"].apply(
        lambda v: _translate(v, COUNTRY_TRANSLATE, default="unknown")
    )

    # ------------------------------------------------------------------
    # First owner (int flag)
    # ------------------------------------------------------------------
    df["First_owner"] = df["original_owner_raw"].apply(_first_owner_to_int)

    # ------------------------------------------------------------------
    # Doors
    # ------------------------------------------------------------------
    df["Doors_number"] = pd.to_numeric(df["door_count_raw"], errors="coerce")

    # ------------------------------------------------------------------
    # Price and currency (kept as-is for clean_car_data() to handle)
    # ------------------------------------------------------------------
    df["Price"] = pd.to_numeric(df["price_value"], errors="coerce")
    df["Currency"] = df["price_currency"].fillna("PLN").str.upper()

    # ------------------------------------------------------------------
    # Date and location
    # ------------------------------------------------------------------
    df["Offer_publication_date"] = df["created_at"].apply(_parse_date)
    df["Offer_location"] = df["city"].str.strip()

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    df["Features"] = df["features_raw"].fillna("")

    # ------------------------------------------------------------------
    # Keep offer_id for deduplication tracking
    # ------------------------------------------------------------------
    # (will be dropped before final merge with existing data if needed)

    schema_cols = [
        "Index", "Condition", "Vehicle_brand", "Vehicle_model",
        "Vehicle_generation", "Production_year", "Mileage_km",
        "Power_HP", "Displacement_cm3", "Fuel_type", "Drive",
        "Transmission", "Type", "Doors_number", "Colour",
        "Origin_country", "First_owner", "Offer_publication_date",
        "Offer_location", "Features", "Price", "Currency", "offer_id",
    ]
    for col in schema_cols:
        if col not in df.columns:
            df[col] = None

    return df[schema_cols]


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for newly scraped data.

    Steps:
    1. normalize_columns()  — map Otomoto fields → schema
    2. clean_car_data()     — existing cleaning logic (currency conversion,
                               duplicate removal, type coercion, etc.)

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw DataFrame from fetch_data().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame matching the project schema (with price_PLN column).
    """
    from src.preprocessing import clean_car_data  # avoid circular at module level

    df_normalized = normalize_columns(df_raw)

    # Drop dedup helper before passing to clean_car_data
    df_for_clean = df_normalized.drop(columns=["offer_id"], errors="ignore")

    df_clean = clean_car_data(df_for_clean)

    # Re-attach offer_id for downstream deduplication
    if "offer_id" in df_normalized.columns:
        df_clean = df_clean.copy()
        df_clean["offer_id"] = df_normalized["offer_id"].values[: len(df_clean)]

    logger.info(
        "clean_data: %d raw rows → %d cleaned rows",
        len(df_raw),
        len(df_clean),
    )
    return df_clean


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

#: Columns that must be present and non-empty for the dataset to be usable
REQUIRED_SCHEMA_COLS: list[str] = [
    "Condition", "Vehicle_brand", "Vehicle_model", "Production_year",
    "Mileage_km", "Power_HP", "Displacement_cm3", "Fuel_type",
    "Transmission", "Type", "price_PLN",
]


def validate_schema(df: pd.DataFrame) -> dict:
    """
    Validate that *df* conforms to the expected project schema.

    Checks for required column presence, fully-NaN columns, negative prices,
    and out-of-range production years.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of clean_data or the existing dataset).

    Returns
    -------
    dict
        Keys:
        - ``valid`` (bool)        — True when no issues found
        - ``missing_columns``     — list of required columns not in df
        - ``issues``              — list of human-readable problem descriptions
        - ``rows``                — total row count
        - ``columns``             — list of column names present in df

    Example
    -------
    >>> result = validate_schema(df_clean)
    >>> if not result["valid"]:
    ...     print(result["issues"])
    """
    issues: list[str] = []
    missing_cols = [c for c in REQUIRED_SCHEMA_COLS if c not in df.columns]

    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    for col in REQUIRED_SCHEMA_COLS:
        if col in df.columns and df[col].isna().all():
            issues.append(f"Column '{col}' is entirely NaN")

    if "price_PLN" in df.columns:
        n_bad = int((df["price_PLN"] <= 0).sum())
        if n_bad:
            issues.append(f"{n_bad} rows with price_PLN <= 0")

    if "Production_year" in df.columns:
        current_year = datetime.now().year
        out_of_range = int(
            ((df["Production_year"] < 1900) | (df["Production_year"] > current_year)).sum()
        )
        if out_of_range:
            issues.append(
                f"{out_of_range} rows with Production_year outside [1900, {current_year}]"
            )

    result: dict = {
        "valid": len(issues) == 0,
        "missing_columns": missing_cols,
        "issues": issues,
        "rows": len(df),
        "columns": list(df.columns),
    }

    if result["valid"]:
        logger.info("validate_schema: OK — %d rows, %d columns", len(df), len(df.columns))
    else:
        for issue in issues:
            logger.warning("validate_schema: %s", issue)

    return result


# ---------------------------------------------------------------------------
# Stratified sampling for balanced datasets
# ---------------------------------------------------------------------------

#: Luxury brands recognised by apply_stratified_sampling (lowercase).
_LUXURY_BRANDS_SAMPLING: frozenset[str] = frozenset({
    "porsche", "ferrari", "lamborghini", "bentley", "rolls-royce",
    "maserati", "aston martin", "aston-martin", "mclaren",
    "bugatti", "koenigsegg", "pagani",
})


def apply_stratified_sampling(
    df: pd.DataFrame,
    target_rows: int = 120_000,
    electric_frac: float = 0.20,
    luxury_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Resample *df* to achieve a target size and category distribution.

    Categories are derived from the existing ``Fuel_type`` and
    ``Vehicle_brand`` columns — no external annotation required.

    Rare categories (electric, luxury) are preserved in full when the
    available rows are fewer than the target allocation.  Popular brand rows
    are downsampled to fill the remainder up to ``target_rows``.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with at least ``Vehicle_brand`` and ``Fuel_type``
        columns.  Typically the output of ``clean_data()`` or the combined
        result of ``fetch_balanced_dataset()`` after cleaning.
    target_rows : int
        Desired total rows in the output.
    electric_frac : float
        Target fraction for Electric fuel-type rows (0–1).
    luxury_frac : float
        Target fraction for luxury brand rows (0–1).
    seed : int
        Random state for reproducibility.

    Returns
    -------
    pd.DataFrame
        Resampled and shuffled DataFrame with at most ``target_rows`` rows.

    Notes
    -----
    Because Otomoto has only ~4 000 EV listings and a few hundred listings
    for ultra-rare brands (Ferrari, Lamborghini), the electric and luxury
    quotas are automatically capped at their available count.  The
    summary log shows actual counts so you can adjust fractions if needed.

    Example
    -------
    >>> df_balanced = apply_stratified_sampling(df_combined, target_rows=120_000)
    >>> print(df_balanced["Fuel_type"].value_counts(normalize=True))
    """
    brand_lower = df.get("Vehicle_brand", pd.Series(dtype=str)).str.lower().fillna("")
    fuel_lower  = df.get("Fuel_type",     pd.Series(dtype=str)).str.lower().fillna("")

    is_electric = fuel_lower == "electric"
    is_luxury   = brand_lower.isin(_LUXURY_BRANDS_SAMPLING) & ~is_electric
    is_popular  = ~is_electric & ~is_luxury

    df_elec = df[is_electric]
    df_lux  = df[is_luxury]
    df_pop  = df[is_popular]

    n_electric = int(target_rows * electric_frac)
    n_luxury   = int(target_rows * luxury_frac)

    parts: list[pd.DataFrame] = []

    # Electric ---------------------------------------------------------------
    actual_elec = min(n_electric, len(df_elec))
    if actual_elec > 0:
        parts.append(df_elec.sample(n=actual_elec, random_state=seed))
    if len(df_elec) < n_electric:
        logger.info(
            "Electric: %d available < %d target — using all",
            len(df_elec), n_electric,
        )

    # Luxury -----------------------------------------------------------------
    actual_lux = min(n_luxury, len(df_lux))
    if actual_lux > 0:
        parts.append(df_lux.sample(n=actual_lux, random_state=seed))
    if len(df_lux) < n_luxury:
        logger.info(
            "Luxury: %d available < %d target — using all",
            len(df_lux), n_luxury,
        )

    # Popular — fill remainder -----------------------------------------------
    n_pop_actual = target_rows - actual_elec - actual_lux
    actual_pop = min(n_pop_actual, len(df_pop))
    if actual_pop > 0:
        parts.append(df_pop.sample(n=actual_pop, random_state=seed))

    if not parts:
        logger.warning("apply_stratified_sampling: no data to sample from")
        return df

    result = (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    logger.info(
        "apply_stratified_sampling: %d total rows  "
        "(electric=%d / %.1f%%, luxury=%d / %.1f%%, popular=%d / %.1f%%)",
        len(result),
        actual_elec, actual_elec / len(result) * 100 if result is not None and len(result) else 0,
        actual_lux,  actual_lux  / len(result) * 100 if result is not None and len(result) else 0,
        actual_pop,  actual_pop  / len(result) * 100 if result is not None and len(result) else 0,
    )
    return result


def deduplicate(df_new: pd.DataFrame, df_existing: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows from df_new whose offer_id already appears in df_existing.

    Falls back to content-based hashing if offer_id is absent in either frame.

    Parameters
    ----------
    df_new : pd.DataFrame
        Freshly fetched (and normalized) rows.
    df_existing : pd.DataFrame
        Existing dataset to check against.

    Returns
    -------
    pd.DataFrame
        Subset of df_new that contains only genuinely new rows.
    """
    if "offer_id" in df_new.columns and "offer_id" in df_existing.columns:
        existing_ids = set(df_existing["offer_id"].dropna().astype(str))
        mask = ~df_new["offer_id"].astype(str).isin(existing_ids)
        n_removed = (~mask).sum()
        logger.info("deduplicate: removed %d duplicates by offer_id", n_removed)
        return df_new[mask].reset_index(drop=True)

    # Fallback: hash key columns
    key_cols = ["Vehicle_brand", "Vehicle_model", "Production_year", "Mileage_km", "Price"]
    key_cols_existing = [c for c in key_cols if c in df_existing.columns]
    key_cols_new = [c for c in key_cols if c in df_new.columns]

    if key_cols_existing and key_cols_new:
        existing_hashes = set(
            df_existing[key_cols_existing].astype(str).agg("|".join, axis=1)
        )
        new_hashes = df_new[key_cols_new].astype(str).agg("|".join, axis=1)
        mask = ~new_hashes.isin(existing_hashes)
        logger.info(
            "deduplicate (hash fallback): removed %d duplicates", (~mask).sum()
        )
        return df_new[mask].reset_index(drop=True)

    logger.warning("deduplicate: no usable key — returning all rows unfiltered")
    return df_new
