"""src/config.py — Single source of truth for project constants.

All brand classification, categorical thresholds, and helper functions used
across src/features.py, src/models.py, and app.py are defined here.  This
eliminates the drift that previously existed between the three modules
(e.g. the "mini" brand was "Premium" in one file and "Luxury" in another).
"""

from __future__ import annotations

RANDOM_STATE: int = 42
VERBOSE: bool = True


def print_if_verbose(msg: str) -> None:
    """Print ``msg`` to stdout iff the module-level VERBOSE flag is True."""
    if VERBOSE:
        print(msg)


# ---------------------------------------------------------------------------
# Brand-tier membership.
#
# These five sets partition the set of recognised car brands.  Every brand
# not in any of the four named sets is classified as ``Niche``.  Lower-case,
# canonical spelling only.
# ---------------------------------------------------------------------------

ULTRA_LUXURY_BRANDS: frozenset[str] = frozenset({
    "ferrari", "lamborghini", "rolls-royce", "bentley", "mclaren",
    "bugatti", "koenigsegg", "pagani", "aston martin", "maybach",
})

LUXURY_BRANDS: frozenset[str] = frozenset({
    "mercedes-benz", "bmw", "audi", "porsche", "lexus", "jaguar",
    "maserati", "tesla", "land rover", "infiniti", "lincoln",
    "genesis", "cadillac", "volvo",
})

PREMIUM_BRANDS: frozenset[str] = frozenset({
    "alfa romeo", "mini", "saab", "ds automobiles", "cupra",
    "alpine", "lotus", "subaru", "acura", "baic", "ssangyong",
})

MASS_MARKET_BRANDS: frozenset[str] = frozenset({
    "volkswagen", "toyota", "ford", "hyundai", "kia", "honda",
    "opel", "chevrolet", "peugeot", "renault", "seat", "skoda",
    "fiat", "nissan", "mazda", "mitsubishi", "suzuki", "dacia",
    "citroen", "citroën", "dodge", "ram", "jeep", "chrysler",
    "lancia", "daewoo", "lada",
})

#: Union of all brands considered "premium or above" for the
#: ``Is_premium`` binary flag.  Must stay consistent with features.py.
IS_PREMIUM_BRANDS: frozenset[str] = (
    ULTRA_LUXURY_BRANDS | LUXURY_BRANDS | PREMIUM_BRANDS
)


def get_brand_tier(brand: str | None) -> str:
    """Return the tier of *brand*: one of
    ``Ultra_Luxury``, ``Luxury``, ``Premium``, ``Mass_Market``, ``Niche``.
    """
    if not brand:
        return "Niche"
    b = str(brand).strip().lower()
    if b in ULTRA_LUXURY_BRANDS:
        return "Ultra_Luxury"
    if b in LUXURY_BRANDS:
        return "Luxury"
    if b in PREMIUM_BRANDS:
        return "Premium"
    if b in MASS_MARKET_BRANDS:
        return "Mass_Market"
    return "Niche"


# ---------------------------------------------------------------------------
# Fallback brand-frequency map.
#
# Used ONLY when a fitted ``BrandFeatureTransformer`` is not available
# (e.g. the legacy deployed model).  New training runs build the map from
# actual training-set value counts — see src/features.py.
# ---------------------------------------------------------------------------

BRAND_FREQUENCY_FALLBACK: dict[str, int] = {
    "volkswagen": 22000, "toyota": 15000, "audi": 14000, "bmw": 13500,
    "mercedes-benz": 13000, "opel": 11000, "ford": 10500, "kia": 10000,
    "hyundai": 9500, "renault": 9000, "peugeot": 8000, "skoda": 8000,
    "seat": 7500, "honda": 6500, "volvo": 6000, "mazda": 5500,
    "nissan": 5000, "mitsubishi": 4500, "fiat": 4000, "citroen": 3800,
    "suzuki": 3500, "subaru": 3000, "land rover": 2500, "jeep": 2200,
    "mini": 2000, "alfa romeo": 1800, "lexus": 1500, "infiniti": 800,
    "dacia": 3000, "tesla": 1200, "porsche": 1000, "jaguar": 700,
    "chevrolet": 600, "dodge": 300, "cadillac": 200, "bentley": 80,
    "ferrari": 60, "lamborghini": 50, "rolls-royce": 40, "maserati": 90,
    "mclaren": 30, "aston martin": 35, "lotus": 45, "maybach": 25,
    "lada": 400, "trabant": 150, "polonez": 120, "syrena": 80,
    "warszawa": 60, "wartburg": 70, "gaz": 55, "moskwicz": 65,
}


# ---------------------------------------------------------------------------
# Categorical-binning helpers.
# ---------------------------------------------------------------------------

def get_age_category(age: float | int | None) -> str:
    """Bin vehicle age into ``New`` / ``Recent`` / ``Used`` / ``Old``."""
    try:
        a = float(age)
    except (TypeError, ValueError):
        return "Used"
    if a < 3:
        return "New"
    if a < 9:
        return "Recent"
    if a < 17:
        return "Used"
    return "Old"


def get_usage_category(mileage_per_year: float | None) -> str:
    """Bin annual mileage into ``Low`` / ``Average`` / ``High`` / ``Very_High``."""
    import pandas as pd
    if mileage_per_year is None or pd.isna(mileage_per_year):
        return "Unknown"
    if mileage_per_year < 10_000:
        return "Low"
    if mileage_per_year < 20_000:
        return "Average"
    if mileage_per_year < 30_000:
        return "High"
    return "Very_High"


def get_performance_category(hp_per_liter: float | None) -> str:
    """Bin specific-power into ``Economy`` / ``Standard`` / ``Performance`` /
    ``High_Performance``."""
    import pandas as pd
    if hp_per_liter is None or pd.isna(hp_per_liter):
        return "Unknown"
    if hp_per_liter < 60:
        return "Economy"
    if hp_per_liter < 100:
        return "Standard"
    if hp_per_liter < 150:
        return "Performance"
    return "High_Performance"


def get_brand_popularity(brand_frequency: int) -> str:
    """Bin brand frequency into ``Ultra_Rare`` … ``Popular``."""
    if brand_frequency <= 5:
        return "Ultra_Rare"
    if brand_frequency <= 20:
        return "Rare"
    if brand_frequency <= 100:
        return "Uncommon"
    if brand_frequency <= 500:
        return "Common"
    return "Popular"


# ---------------------------------------------------------------------------
# Sample-weighting schedule (used by train_xgboost_weighted).
# ---------------------------------------------------------------------------

TIER_WEIGHT_MAP: dict[str, float] = {
    "Ultra_Luxury": 4.0,
    "Luxury":       3.0,
    "Niche":        3.5,
    "Premium":      1.5,
    "Mass_Market":  1.0,
}
