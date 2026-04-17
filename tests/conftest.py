"""Pytest fixtures: a tiny synthetic used-car dataset.

The real scraped CSV is not committed to the repo (size / licensing), so
these fixtures build an in-memory DataFrame that follows the production
schema.  All unit tests use this fixture, which means the suite runs in
<5 seconds and works in CI without external dependencies.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Make the project root (one level above tests/) importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def synthetic_car_df() -> pd.DataFrame:
    """300-row synthetic dataset covering all brand tiers."""
    rng = np.random.default_rng(0)
    n = 300

    brands = rng.choice(
        [
            "volkswagen", "toyota", "audi", "bmw", "mercedes-benz",
            "ford", "kia", "hyundai", "skoda", "opel",
            "porsche", "lexus", "volvo",
            "mini", "alfa romeo",
            "ferrari", "lamborghini", "bentley",
            "dacia", "lada",
        ],
        n,
    )
    production_year = rng.integers(1998, 2025, n)
    power_hp = rng.integers(60, 450, n)
    displacement = rng.integers(900, 5000, n)
    mileage = rng.integers(5_000, 350_000, n)

    age = 2025 - production_year
    base = 8000 + power_hp * 220 - age * 1500 - mileage * 0.05
    noise = rng.normal(0, 5000, n)
    price = np.clip(base + noise, 3000, None)

    return pd.DataFrame({
        "Condition":              "used",
        "Vehicle_brand":          brands,
        "Vehicle_model":          rng.choice(["golf", "corolla", "a4", "320d", "c-class", "panamera"], n),
        "Production_year":        production_year,
        "Mileage_km":             mileage,
        "Power_HP":               power_hp,
        "Displacement_cm3":       displacement,
        "Fuel_type":              rng.choice(["gasoline", "diesel", "hybrid", "electric"], n),
        "Drive":                  rng.choice(["front wheels", "rear wheels", "4x4"], n),
        "Transmission":           rng.choice(["manual", "automatic"], n),
        "Type":                   rng.choice(["sedan", "suv", "combi", "hatchback"], n),
        "Doors_number":           rng.choice([3, 4, 5], n),
        "Colour":                 rng.choice(["black", "white", "silver", "red"], n),
        "Origin_country":         rng.choice(["poland", "germany", "france"], n),
        "First_owner":            rng.integers(0, 2, n),
        "Offer_location":         rng.choice(
            ["Warszawa, Mazowieckie", "Krakow, Malopolskie", "Wroclaw",
             "Gdansk", "Poznan"], n,
        ),
        "Features":               "airbags, abs",
        "price_PLN":              price,
    })
