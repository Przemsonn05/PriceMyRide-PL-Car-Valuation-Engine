"""
src/data_fetcher.py

Production-ready data pipeline for fetching car listings from Otomoto (Poland).

Architecture
------------
- Primary source : Otomoto search results (__NEXT_DATA__ JSON embedded in HTML)
- Each search page returns 32 listings with core fields
- Optional detail mode fetches individual listing pages for full field coverage
- Mock mode generates synthetic data matching the raw scraped schema
- Incremental updates deduplicate by Otomoto offer ID
- Stratified collection fetches brand/fuel-type segments for balanced datasets

Usage
-----
    from src.data_fetcher import fetch_data, fetch_incremental, fetch_balanced_dataset

    # Fetch current listings (search-result fields only)
    df_raw = fetch_data(pages=10)

    # Full detail fetch (slower, all fields including Drive, Colour, etc.)
    df_raw = fetch_data(pages=5, detail_mode=True)

    # Incremental update – appends only new listings to existing dataset
    fetch_incremental(data_path="data/Car_sale_ads_balanced.csv", pages=20)

    # Stratified balanced collection (~200k rows across popular/luxury/EV)
    df_raw = fetch_balanced_dataset(target_rows=200_000)

    # Mock data for testing
    df_raw = fetch_data(mock=True, mock_rows=200)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import random
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OTOMOTO_BASE = "https://www.otomoto.pl"
SEARCH_URL = f"{OTOMOTO_BASE}/osobowe"
PAGE_SIZE = 32  # Otomoto returns 32 listings per page

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "DNT": "1",
}

# Raw field names produced by the scraper (before normalization to schema)
RAW_COLUMNS = [
    "offer_id",
    "offer_url",
    "scraped_at",
    "make",
    "model",
    "version",
    "generation",
    "year",
    "mileage_raw",
    "engine_capacity_raw",
    "engine_power_raw",
    "fuel_type_raw",
    "gearbox_raw",
    "transmission_raw",
    "body_type_raw",
    "door_count_raw",
    "colour_raw",
    "condition_raw",
    "country_origin_raw",
    "original_owner_raw",
    "price_value",
    "price_currency",
    "city",
    "region",
    "features_raw",
    "created_at",
]


# ---------------------------------------------------------------------------
# HTTP session with retry
# ---------------------------------------------------------------------------

def _build_session(retries: int = 3, backoff: float = 1.0) -> requests.Session:
    session = requests.Session()
    session.headers.update(_DEFAULT_HEADERS)
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_next_data(html: str) -> Optional[dict]:
    """Extract and parse the __NEXT_DATA__ JSON from an Otomoto HTML page."""
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("script", id="__NEXT_DATA__")
    if not tag or not tag.string:
        return None
    try:
        return json.loads(tag.string)
    except json.JSONDecodeError:
        return None


def _get_search_edges(next_data: dict) -> list[dict]:
    """Return the list of listing edges from the search page __NEXT_DATA__."""
    try:
        urql_state = next_data["props"]["pageProps"]["urqlState"]
        for _key, val in urql_state.items():
            raw = val.get("data", "")
            if isinstance(raw, str) and "advertSearch" in raw:
                parsed = json.loads(raw)
                return parsed["advertSearch"]["edges"]
    except (KeyError, json.JSONDecodeError):
        pass
    return []


def _params_to_dict(parameters: list[dict]) -> dict[str, str]:
    """Convert a list of {key, displayValue} parameter dicts to a flat map."""
    return {p["key"]: p.get("displayValue", "") for p in parameters if "key" in p}


def _parse_price(price_node: Optional[dict]) -> tuple[str, str]:
    """Return (amount_str, currency) from a price node."""
    if not price_node:
        return ("", "PLN")
    amount = price_node.get("amount") or {}
    value = amount.get("value", price_node.get("value", ""))
    currency = amount.get("currencyCode", price_node.get("currency", "PLN"))
    return (str(value), str(currency))


def _parse_location(location_node: Optional[dict]) -> tuple[str, str]:
    """Return (city, region) from a location node."""
    if not location_node:
        return ("", "")
    city = (location_node.get("city") or {}).get("name", "")
    region = (location_node.get("region") or {}).get("name", "")
    return (city, region)


def _parse_listing_node(node: dict) -> dict:
    """Parse a single listing node from search results into a raw record dict."""
    params = _params_to_dict(node.get("parameters", []))
    price_val, price_curr = _parse_price(node.get("price"))
    city, region = _parse_location(node.get("location"))

    offer_url = node.get("url", "")
    offer_id = str(node.get("id", _url_to_id(offer_url)))

    return {
        "offer_id": offer_id,
        "offer_url": offer_url,
        "scraped_at": datetime.utcnow().isoformat(),
        "make": params.get("make", ""),
        "model": params.get("model", ""),
        "version": params.get("version", ""),
        "generation": params.get("generation", ""),
        "year": params.get("year", ""),
        "mileage_raw": params.get("mileage", ""),
        "engine_capacity_raw": params.get("engine_capacity", ""),
        "engine_power_raw": params.get("engine_power", ""),
        "fuel_type_raw": params.get("fuel_type", ""),
        "gearbox_raw": params.get("gearbox", ""),
        "transmission_raw": params.get("transmission", ""),   # drive type
        "body_type_raw": params.get("body_type", ""),
        "door_count_raw": params.get("door_count", ""),
        "colour_raw": params.get("color", ""),
        "condition_raw": params.get("new_used", ""),
        "country_origin_raw": params.get("country_origin", ""),
        "original_owner_raw": params.get("original_owner", ""),
        "price_value": price_val,
        "price_currency": price_curr,
        "city": city,
        "region": region,
        "features_raw": _collect_features(params),
        "created_at": node.get("createdAt", ""),
    }


def _collect_features(params: dict[str, str]) -> str:
    """
    Collect equipment/feature keys from search-result parameters.
    Equipment items are binary flags; collect those with value 'Tak' (Yes).
    """
    feature_keys = {
        "rear_view_camera", "apple_carplay", "android_auto",
        "cruisecontrol_type", "heated_seat_driver", "heated_seat_passenger",
        "blind_spot_warning", "lane_control_assistant", "navigation_system",
        "air_conditioning_type", "bluetooth_interface", "360_view_camera",
        "service_record", "no_accident", "sunroof", "xenon_lights",
        "led_lights", "electric_windows", "power_steering", "abs",
        "esp", "traction_control", "parking_sensors",
    }
    found = []
    for k, v in params.items():
        if k in feature_keys and v and v.lower() not in ("nie", "no", ""):
            found.append(k)
    return ", ".join(found)


def _url_to_id(url: str) -> str:
    """Generate a stable ID from a URL when numeric ID is unavailable."""
    return hashlib.md5(url.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Detail page fetch (optional, for complete schema coverage)
# ---------------------------------------------------------------------------

def _fetch_detail(url: str, session: requests.Session) -> dict:
    """
    Fetch individual listing page and extract all parametersDict fields.
    Returns a dict of additional fields to overlay on the search-result record.
    """
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
        nd = _extract_next_data(resp.text)
        if not nd:
            return {}
        advert = nd.get("props", {}).get("pageProps", {}).get("advert", {})
        if not advert:
            return {}

        params_dict = advert.get("parametersDict", {})

        def _first_label(key: str) -> str:
            entry = params_dict.get(key, {})
            vals = entry.get("values", [])
            return vals[0].get("label", "") if vals else ""

        # Equipment – flatten all equipment group values
        equipment_labels = []
        for group in advert.get("equipment", []):
            for item in group.get("values", []):
                label = item.get("label", "")
                if label:
                    equipment_labels.append(label)
        features_raw = ", ".join(equipment_labels)

        # Location from individual page
        loc = advert.get("location") or {}
        city = (loc.get("city") or {}).get("name", "")
        region = (loc.get("region") or {}).get("name", "")

        return {
            "generation": _first_label("generation"),
            "transmission_raw": _first_label("transmission"),  # drive type
            "body_type_raw": _first_label("body_type"),
            "door_count_raw": _first_label("door_count"),
            "colour_raw": _first_label("color"),
            "condition_raw": _first_label("new_used"),
            "country_origin_raw": _first_label("country_origin"),
            "original_owner_raw": _first_label("original_owner"),
            "features_raw": features_raw if features_raw else None,
            "city": city if city else None,
            "region": region if region else None,
        }
    except Exception as exc:
        logger.debug("Detail fetch failed for %s: %s", url, exc)
        return {}


# ---------------------------------------------------------------------------
# Core fetch functions
# ---------------------------------------------------------------------------

def _fetch_search_page(
    page: int,
    session: requests.Session,
    delay_range: tuple[float, float] = (1.0, 3.0),
) -> list[dict]:
    """
    Fetch one Otomoto search results page and return raw listing records.
    """
    params = {
        "page": page,
        "search[order]": "created_at_first:desc",
    }
    try:
        time.sleep(random.uniform(*delay_range))
        resp = session.get(SEARCH_URL, params=params, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Search page %d failed: %s", page, exc)
        return []

    nd = _extract_next_data(resp.text)
    if not nd:
        logger.warning("No __NEXT_DATA__ on page %d", page)
        return []

    edges = _get_search_edges(nd)
    records = []
    for edge in edges:
        node = edge.get("node", {})
        if node:
            records.append(_parse_listing_node(node))
    logger.info("Page %d: fetched %d listings", page, len(records))
    return records


def fetch_data(
    pages: int = 5,
    start_page: int = 1,
    delay_range: tuple[float, float] = (1.5, 3.5),
    detail_mode: bool = False,
    detail_delay: tuple[float, float] = (1.0, 2.5),
    mock: bool = False,
    mock_rows: int = 200,
) -> pd.DataFrame:
    """
    Fetch car listings from Otomoto and return a raw DataFrame.

    Parameters
    ----------
    pages : int
        Number of search result pages to fetch (32 listings each).
    start_page : int
        First page number to fetch (1-based).
    delay_range : tuple[float, float]
        Random sleep (seconds) between search-page requests.
    detail_mode : bool
        If True, fetch every individual listing page for full field coverage.
        Much slower (1 extra request per listing), but populates Drive, Colour,
        body type, condition, doors — fields missing from search results.
    detail_delay : tuple[float, float]
        Random sleep between individual listing requests (detail_mode only).
    mock : bool
        Return synthetic data instead of scraping (useful for CI/testing).
    mock_rows : int
        Number of synthetic rows when mock=True.

    Returns
    -------
    pd.DataFrame
        Raw scraped data with columns defined by RAW_COLUMNS.
        Use ``src.data_cleaning.normalize_columns()`` to convert to schema.

    Notes
    -----
    Otomoto only exposes *active* listings. Historical ads (2021–2024) are
    not available via scraping — they were never publicly archived. This
    pipeline collects current and future listings for incremental enrichment.
    """
    if mock:
        logger.info("Mock mode: generating %d synthetic rows", mock_rows)
        return _generate_mock_data(mock_rows)

    session = _build_session()
    all_records: list[dict] = []

    for page_num in range(start_page, start_page + pages):
        records = _fetch_search_page(page_num, session, delay_range)
        if not records:
            logger.info("Empty page %d — stopping early", page_num)
            break

        if detail_mode:
            for rec in records:
                url = rec.get("offer_url", "")
                if url:
                    time.sleep(random.uniform(*detail_delay))
                    extra = _fetch_detail(url, session)
                    # Overlay non-null detail fields on search-result record
                    for k, v in extra.items():
                        if v is not None:
                            rec[k] = v

        all_records.extend(records)

    if not all_records:
        logger.warning("No listings fetched — returning empty DataFrame")
        return pd.DataFrame(columns=RAW_COLUMNS)

    df = pd.DataFrame(all_records)
    # Ensure all expected columns exist
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[RAW_COLUMNS]


def fetch_incremental(
    data_path: str | Path,
    pages: int = 10,
    delay_range: tuple[float, float] = (1.5, 3.5),
    detail_mode: bool = False,
    mock: bool = False,
) -> pd.DataFrame:
    """
    Fetch new listings and append them to an existing dataset (CSV).

    Deduplication is performed using the Otomoto offer ID.  Listings already
    present in ``data_path`` (matched by offer_id) are skipped.

    Parameters
    ----------
    data_path : str or Path
        Path to existing CSV dataset.  The file is expected to have a column
        named ``offer_id`` if it was produced by this pipeline, or the raw
        ``Index`` column from the original dataset (which is treated as a
        proxy — no deduplication is then possible for legacy rows).
    pages : int
        Number of search pages to fetch for new listings.
    delay_range : tuple
        Sleep range between search-page requests.
    detail_mode : bool
        Fetch individual listing pages for full field coverage.
    mock : bool
        Use synthetic data (testing only).

    Returns
    -------
    pd.DataFrame
        The new (deduplicated) rows that were appended to ``data_path``.
    """
    from src.data_cleaning import normalize_columns, clean_data  # local import avoids circular

    data_path = Path(data_path)
    existing_ids: set[str] = set()

    if data_path.exists():
        try:
            existing = pd.read_csv(data_path, usecols=lambda c: c in {"offer_id", "Index"})
            if "offer_id" in existing.columns:
                existing_ids = set(existing["offer_id"].dropna().astype(str))
            logger.info("Existing dataset: %d rows, %d known IDs", len(existing), len(existing_ids))
        except Exception as exc:
            logger.warning("Could not read existing data for dedup: %s", exc)
    else:
        logger.info("No existing dataset at %s — full fetch", data_path)

    df_raw = fetch_data(pages=pages, delay_range=delay_range, detail_mode=detail_mode, mock=mock)

    if df_raw.empty:
        logger.info("No new data fetched")
        return df_raw

    # Deduplicate
    if existing_ids:
        before = len(df_raw)
        df_raw = df_raw[~df_raw["offer_id"].astype(str).isin(existing_ids)]
        logger.info("Deduplication: %d → %d rows (removed %d duplicates)", before, len(df_raw), before - len(df_raw))

    if df_raw.empty:
        logger.info("All fetched listings already in dataset — nothing to append")
        return df_raw

    # Normalize and clean to match schema
    df_schema = clean_data(df_raw)

    # Append to existing file
    data_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not data_path.exists()
    df_schema.to_csv(data_path, mode="a", header=write_header, index=False)
    logger.info("Appended %d new rows to %s", len(df_schema), data_path)

    return df_schema


# ---------------------------------------------------------------------------
# Mock data generator
# ---------------------------------------------------------------------------

def _generate_mock_data(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic raw data that matches the RAW_COLUMNS schema.
    Useful for testing the cleaning/normalization pipeline without network access.
    Pass different *seed* values to produce distinct row sets.
    """
    rng = random.Random(seed)
    current_year = datetime.now().year

    brands_models = [
        ("Volkswagen", "Golf"), ("Volkswagen", "Passat"), ("Toyota", "Corolla"),
        ("Ford", "Focus"), ("Opel", "Astra"), ("Skoda", "Octavia"),
        ("BMW", "3 Series"), ("Audi", "A4"), ("Mercedes-Benz", "C-Class"),
        ("Kia", "Sportage"), ("Hyundai", "Tucson"), ("Peugeot", "308"),
        ("Renault", "Megane"), ("Fiat", "Tipo"), ("Seat", "Leon"),
    ]
    fuel_types = ["Benzyna", "Olej napędowy", "Elektryczny", "Hybryda", "Benzyna + LPG"]
    gearboxes = ["Manualna", "Automatyczna"]
    transmissions = [
        "Przednie", "Tylne", "4x4 (dołączany automatycznie)",
        "4x4 (stały)", "4x4 (dołączany ręcznie)"
    ]
    body_types = ["Sedan", "Kombi", "Hatchback", "SUV", "Minivan", "Coupe"]
    colours = ["Biały", "Czarny", "Szary", "Srebrny", "Niebieski", "Czerwony", "Zielony"]
    conditions = ["Używany", "Nowy"]
    countries = ["Polska", "Niemcy", "Francja", "Włochy", "Japonia", "Stany Zjednoczone"]
    cities = ["Warszawa", "Kraków", "Wrocław", "Poznań", "Gdańsk", "Łódź", "Katowice"]
    features_pool = [
        "ABS", "ESP", "navigation_system", "rear_view_camera",
        "bluetooth_interface", "heated_seat_driver", "apple_carplay",
    ]

    rows = []
    for i in range(n_rows):
        brand, model = rng.choice(brands_models)
        year = rng.randint(2005, current_year)
        mileage = rng.randint(0, 350_000)
        power = rng.randint(70, 400)
        displacement = rng.randint(900, 4500)
        doors = rng.choice([3, 4, 5])
        price = rng.randint(8_000, 250_000)
        offer_id = f"mock_{i:06d}"
        created_at = f"{rng.randint(2022, current_year)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}T12:00:00Z"
        features = ", ".join(rng.sample(features_pool, k=rng.randint(0, len(features_pool))))

        rows.append({
            "offer_id": offer_id,
            "offer_url": f"https://www.otomoto.pl/osobowe/oferta/mock-{offer_id}.html",
            "scraped_at": datetime.utcnow().isoformat(),
            "make": brand,
            "model": model,
            "version": "",
            "generation": "",
            "year": str(year),
            "mileage_raw": f"{mileage:,} km".replace(",", " "),
            "engine_capacity_raw": f"{displacement:,} cm3".replace(",", " "),
            "engine_power_raw": f"{power} KM",
            "fuel_type_raw": rng.choice(fuel_types),
            "gearbox_raw": rng.choice(gearboxes),
            "transmission_raw": rng.choice(transmissions),
            "body_type_raw": rng.choice(body_types),
            "door_count_raw": str(doors),
            "colour_raw": rng.choice(colours),
            "condition_raw": rng.choice(conditions),
            "country_origin_raw": rng.choice(countries),
            "original_owner_raw": rng.choice(["Tak", "Nie"]),
            "price_value": str(price),
            "price_currency": "PLN",
            "city": rng.choice(cities),
            "region": "",
            "features_raw": features,
            "created_at": created_at,
        })

    return pd.DataFrame(rows, columns=RAW_COLUMNS)


# ---------------------------------------------------------------------------
# Stratified / balanced collection
# ---------------------------------------------------------------------------

#: Configuration for stratified dataset collection.
#:
#: Each entry defines one fetch segment:
#:   category      – label used by apply_stratified_sampling() in data_cleaning
#:   filter_key    – "make" (brand), "fuel_type", or "none" (no filter = all brands)
#:   filter_value  – Otomoto-compatible filter value (lowercase); ignored when filter_key="none"
#:   target_share  – fraction of ``target_rows`` allocated to this segment
#:   max_pages     – hard cap on pages fetched regardless of target_share
#:
#: Popular brands : ~57 % — explicit brand filters for the top brands in Poland.
#: General        : ~10 % — unfiltered search catches all remaining brands proportionally
#:                           (Dacia, Citroen, Volvo, Alfa Romeo, Mini, Suzuki, Subaru …).
#: Luxury brands  : ~13 % — oversampled to give models enough rare/high-price examples.
#: Electric       : ~20 % — oversampled to enrich EV representation.
#:
#: max_pages values are sized for a 200 000-row target.
#: pages_per_spec = ceil(target_rows * target_share / PAGE_SIZE) — max_pages is the hard cap.
STRATIFIED_CONFIG: list[dict] = [
    # ----- Popular brands: individually filtered (57 % of target) ----------
    # Core Polish market — high volume, many models per brand
    {"category": "popular",  "filter_key": "make",      "filter_value": "volkswagen",    "target_share": 0.06, "max_pages": 500},
    {"category": "popular",  "filter_key": "make",      "filter_value": "toyota",        "target_share": 0.05, "max_pages": 420},
    {"category": "popular",  "filter_key": "make",      "filter_value": "bmw",           "target_share": 0.04, "max_pages": 380},
    {"category": "popular",  "filter_key": "make",      "filter_value": "audi",          "target_share": 0.04, "max_pages": 380},
    {"category": "popular",  "filter_key": "make",      "filter_value": "mercedes-benz", "target_share": 0.04, "max_pages": 380},
    {"category": "popular",  "filter_key": "make",      "filter_value": "skoda",         "target_share": 0.04, "max_pages": 320},
    {"category": "popular",  "filter_key": "make",      "filter_value": "ford",          "target_share": 0.03, "max_pages": 300},
    {"category": "popular",  "filter_key": "make",      "filter_value": "opel",          "target_share": 0.03, "max_pages": 300},
    {"category": "popular",  "filter_key": "make",      "filter_value": "hyundai",       "target_share": 0.03, "max_pages": 270},
    {"category": "popular",  "filter_key": "make",      "filter_value": "kia",           "target_share": 0.03, "max_pages": 270},
    {"category": "popular",  "filter_key": "make",      "filter_value": "dacia",         "target_share": 0.03, "max_pages": 240},
    {"category": "popular",  "filter_key": "make",      "filter_value": "renault",       "target_share": 0.02, "max_pages": 240},
    {"category": "popular",  "filter_key": "make",      "filter_value": "peugeot",       "target_share": 0.02, "max_pages": 220},
    {"category": "popular",  "filter_key": "make",      "filter_value": "seat",          "target_share": 0.02, "max_pages": 220},
    {"category": "popular",  "filter_key": "make",      "filter_value": "citroen",       "target_share": 0.02, "max_pages": 200},
    {"category": "popular",  "filter_key": "make",      "filter_value": "volvo",         "target_share": 0.02, "max_pages": 200},
    {"category": "popular",  "filter_key": "make",      "filter_value": "fiat",          "target_share": 0.01, "max_pages": 170},
    {"category": "popular",  "filter_key": "make",      "filter_value": "nissan",        "target_share": 0.01, "max_pages": 170},
    {"category": "popular",  "filter_key": "make",      "filter_value": "honda",         "target_share": 0.01, "max_pages": 170},
    {"category": "popular",  "filter_key": "make",      "filter_value": "mazda",         "target_share": 0.01, "max_pages": 170},
    {"category": "popular",  "filter_key": "make",      "filter_value": "alfa-romeo",    "target_share": 0.01, "max_pages": 140},
    {"category": "popular",  "filter_key": "make",      "filter_value": "mini",          "target_share": 0.01, "max_pages": 140},
    {"category": "popular",  "filter_key": "make",      "filter_value": "suzuki",        "target_share": 0.01, "max_pages": 140},
    {"category": "popular",  "filter_key": "make",      "filter_value": "mitsubishi",    "target_share": 0.01, "max_pages": 140},
    {"category": "popular",  "filter_key": "make",      "filter_value": "subaru",        "target_share": 0.01, "max_pages": 120},
    {"category": "popular",  "filter_key": "make",      "filter_value": "land-rover",    "target_share": 0.01, "max_pages": 120},
    # ----- General catch-all: no brand filter (10 % of target) ------------
    # Fetches all remaining brands proportionally (Jeep, Lexus, Cupra, Volvo,
    # Chevrolet, Dodge, Smart, Lancia, Genesis, etc.) — whatever Otomoto lists.
    {"category": "popular",  "filter_key": "none",      "filter_value": "",              "target_share": 0.10, "max_pages": 700},
    # ----- Luxury / rare brands (13 % of target) --------------------------
    {"category": "luxury",   "filter_key": "make",      "filter_value": "porsche",       "target_share": 0.04, "max_pages": 250},
    {"category": "luxury",   "filter_key": "make",      "filter_value": "ferrari",       "target_share": 0.02, "max_pages": 70},
    {"category": "luxury",   "filter_key": "make",      "filter_value": "lamborghini",   "target_share": 0.01, "max_pages": 35},
    {"category": "luxury",   "filter_key": "make",      "filter_value": "bentley",       "target_share": 0.02, "max_pages": 50},
    {"category": "luxury",   "filter_key": "make",      "filter_value": "rolls-royce",   "target_share": 0.01, "max_pages": 35},
    {"category": "luxury",   "filter_key": "make",      "filter_value": "maserati",      "target_share": 0.02, "max_pages": 70},
    {"category": "luxury",   "filter_key": "make",      "filter_value": "aston-martin",  "target_share": 0.01, "max_pages": 35},
    # ----- Electric vehicles: all brands via fuel_type filter (20 % of target)
    {"category": "electric", "filter_key": "fuel_type", "filter_value": "electric",      "target_share": 0.20, "max_pages": 340},
]


def _fetch_with_filters(
    filters: dict,
    pages: int,
    session: requests.Session,
    delay_range: tuple = (1.5, 3.5),
    detail_mode: bool = False,
    detail_delay: tuple = (1.0, 2.5),
) -> list:
    """
    Fetch Otomoto search pages with custom filter parameters.

    Parameters
    ----------
    filters : dict
        Additional query parameters merged into every page request, e.g.
        ``{"search[filter_enum_make][]": "toyota"}``.
    pages : int
        Maximum number of pages to fetch for this filter combination.
    session : requests.Session
        Shared HTTP session with retry/backoff configured.
    delay_range : tuple
        (min, max) seconds of random sleep between page requests.
    detail_mode : bool
        Fetch individual listing pages for full field coverage.
    detail_delay : tuple
        (min, max) seconds between detail page requests.

    Returns
    -------
    list[dict]
        Raw record dicts matching RAW_COLUMNS (without ``_category``).
    """
    all_records: list = []

    for page_num in range(1, pages + 1):
        params: dict = {
            "page": page_num,
            "search[order]": "created_at_first:desc",
        }
        params.update(filters)

        try:
            time.sleep(random.uniform(*delay_range))
            resp = session.get(SEARCH_URL, params=params, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Filtered page %d failed (%s): %s", page_num, filters, exc)
            break

        nd = _extract_next_data(resp.text)
        if not nd:
            logger.warning("No __NEXT_DATA__ on filtered page %d (%s)", page_num, filters)
            break

        edges = _get_search_edges(nd)
        if not edges:
            logger.info("Empty result on page %d for %s — stopping segment", page_num, filters)
            break

        page_records: list = []
        for edge in edges:
            node = edge.get("node", {})
            if node:
                page_records.append(_parse_listing_node(node))

        if detail_mode:
            for rec in page_records:
                url = rec.get("offer_url", "")
                if url:
                    time.sleep(random.uniform(*detail_delay))
                    extra = _fetch_detail(url, session)
                    for k, v in extra.items():
                        if v is not None:
                            rec[k] = v

        logger.info(
            "Filtered page %d/%d: %d listings (%s)",
            page_num, pages, len(page_records), filters,
        )
        all_records.extend(page_records)

    return all_records


def fetch_balanced_dataset(
    target_rows: int = 200_000,
    detail_mode: bool = False,
    delay_range: tuple = (1.5, 3.5),
    detail_delay: tuple = (1.0, 2.5),
    mock: bool = False,
    mock_rows_per_spec: Optional[int] = None,
) -> pd.DataFrame:
    """
    Collect a stratified, balanced dataset from Otomoto.

    Iterates through every entry in STRATIFIED_CONFIG, fetches listings using
    per-segment Otomoto filters (brand or fuel type), and returns the combined
    raw DataFrame.  The number of pages fetched per segment is proportional to
    ``target_rows * target_share``, capped by ``max_pages``.

    After collection, call ``data_cleaning.apply_stratified_sampling()`` to
    enforce the final row count and category distribution.

    Parameters
    ----------
    target_rows : int
        Desired total rows in the output dataset.  Controls page allocation
        per segment; actual row count depends on live Otomoto listings.
        In mock mode this also determines how many synthetic rows are generated
        (``ceil(target_rows / len(STRATIFIED_CONFIG))`` per segment).
    detail_mode : bool
        If True, fetch individual listing pages for full field coverage
        (Drive, Colour, body type, condition, doors).  Much slower.
    delay_range : tuple[float, float]
        (min, max) seconds between search-page requests.
    detail_delay : tuple[float, float]
        (min, max) seconds between detail-page requests (detail_mode only).
    mock : bool
        Return synthetic data without making network requests (for testing).
    mock_rows_per_spec : int or None
        Synthetic rows per STRATIFIED_CONFIG entry in mock mode.
        When ``None`` (default) this is computed automatically as
        ``ceil(target_rows / len(STRATIFIED_CONFIG))`` so that the total
        mock output matches ``target_rows``.

    Returns
    -------
    pd.DataFrame
        Raw combined DataFrame with RAW_COLUMNS plus a ``_category`` column
        (values: "popular", "luxury", "electric") for use by
        ``apply_stratified_sampling()``.

    Notes
    -----
    Rare categories (luxury brands, EVs) may have far fewer Otomoto listings
    than the ``target_share`` implies.  ``apply_stratified_sampling()`` handles
    this gracefully — it uses all available rows from underrepresented
    categories and downsamples popular brands to fill the remainder.

    Example
    -------
    >>> df_raw = fetch_balanced_dataset(target_rows=120_000)
    >>> print(df_raw["_category"].value_counts())
    popular     68320
    electric     4085
    luxury       2140
    Name: _category, dtype: int64
    """
    extra_cols = RAW_COLUMNS + ["_category"]

    if mock:
        import math
        _rows_per_spec = mock_rows_per_spec if mock_rows_per_spec is not None else \
            max(1, math.ceil(target_rows / len(STRATIFIED_CONFIG)))
        logger.info(
            "Mock mode: generating synthetic balanced dataset (%d specs x %d rows = ~%d total)",
            len(STRATIFIED_CONFIG), _rows_per_spec, len(STRATIFIED_CONFIG) * _rows_per_spec,
        )
        frames = []
        for i, spec in enumerate(STRATIFIED_CONFIG):
            # Different seed per spec so rows are genuinely distinct after dedup
            df_spec = _generate_mock_data(_rows_per_spec, seed=42 + i * 1000)
            # Make offer_ids globally unique across specs
            df_spec["offer_id"] = [f"mock_s{i:02d}_{j:06d}" for j in range(len(df_spec))]
            df_spec["_category"] = spec["category"]
            frames.append(df_spec)
        result = pd.concat(frames, ignore_index=True)
        for col in extra_cols:
            if col not in result.columns:
                result[col] = ""
        return result[extra_cols]

    session = _build_session()
    all_records: list = []

    for spec in STRATIFIED_CONFIG:
        category     = spec["category"]
        filter_key   = spec["filter_key"]
        filter_value = spec["filter_value"]
        target_share = spec["target_share"]
        max_pages    = spec["max_pages"]

        # Compute pages proportional to target allocation
        pages_needed  = max(1, int(round(target_rows * target_share / PAGE_SIZE)))
        pages_to_fetch = min(pages_needed, max_pages)

        # Build Otomoto query parameter
        if filter_key == "make":
            filters = {"search[filter_enum_make][]": filter_value}
        elif filter_key == "fuel_type":
            filters = {"search[filter_enum_fuel_type][]": filter_value}
        elif filter_key == "none":
            filters = {}  # no brand/fuel filter — fetches all listings proportionally
        else:
            filters = {filter_key: filter_value}

        logger.info(
            "Segment [%s] %s=%s — fetching %d pages (target_share=%.0f%%)",
            category, filter_key, filter_value, pages_to_fetch, target_share * 100,
        )

        records = _fetch_with_filters(
            filters=filters,
            pages=pages_to_fetch,
            session=session,
            delay_range=delay_range,
            detail_mode=detail_mode,
            detail_delay=detail_delay,
        )

        for rec in records:
            rec["_category"] = category

        all_records.extend(records)
        logger.info(
            "Segment %s/%s done: %d records  (running total: %d)",
            filter_key, filter_value, len(records), len(all_records),
        )

    if not all_records:
        logger.warning("No listings fetched — returning empty DataFrame")
        return pd.DataFrame(columns=extra_cols)

    df = pd.DataFrame(all_records)

    # Ensure all expected columns present
    for col in extra_cols:
        if col not in df.columns:
            df[col] = ""

    # Global deduplication (luxury + EV brands overlap with brand filters)
    before = len(df)
    df = df.drop_duplicates(subset=["offer_id"])
    if before != len(df):
        logger.info(
            "Global dedup: %d → %d rows (removed %d cross-segment duplicates)",
            before, len(df), before - len(df),
        )

    logger.info(
        "fetch_balanced_dataset complete: %d rows  category breakdown:\n%s",
        len(df), df["_category"].value_counts().to_string(),
    )
    return df[extra_cols]
