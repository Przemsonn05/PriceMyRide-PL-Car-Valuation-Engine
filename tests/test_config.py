"""Tests for src.config — the single source of truth for brand tiers."""
from src.config import (
    BRAND_FREQUENCY_FALLBACK,
    IS_PREMIUM_BRANDS,
    LUXURY_BRANDS,
    MASS_MARKET_BRANDS,
    PREMIUM_BRANDS,
    TIER_WEIGHT_MAP,
    ULTRA_LUXURY_BRANDS,
    get_age_category,
    get_brand_popularity,
    get_brand_tier,
    get_performance_category,
    get_usage_category,
)


def test_brand_tiers_are_disjoint():
    """No brand is in two tiers at once - a drift-resistance invariant."""
    pairs = [
        (ULTRA_LUXURY_BRANDS, LUXURY_BRANDS),
        (ULTRA_LUXURY_BRANDS, PREMIUM_BRANDS),
        (ULTRA_LUXURY_BRANDS, MASS_MARKET_BRANDS),
        (LUXURY_BRANDS, PREMIUM_BRANDS),
        (LUXURY_BRANDS, MASS_MARKET_BRANDS),
        (PREMIUM_BRANDS, MASS_MARKET_BRANDS),
    ]
    for a, b in pairs:
        assert a.isdisjoint(b), f"Brands overlap between tiers: {a & b}"


def test_is_premium_equals_union():
    """IS_PREMIUM_BRANDS must match the union of ultra / luxury / premium."""
    assert IS_PREMIUM_BRANDS == (ULTRA_LUXURY_BRANDS | LUXURY_BRANDS | PREMIUM_BRANDS)


def test_get_brand_tier_roundtrip():
    assert get_brand_tier("FERRARI") == "Ultra_Luxury"
    assert get_brand_tier(" bmw ") == "Luxury"
    assert get_brand_tier("mini") == "Premium"
    assert get_brand_tier("toyota") == "Mass_Market"
    assert get_brand_tier("some-unknown-brand") == "Niche"
    assert get_brand_tier(None) == "Niche"
    assert get_brand_tier("") == "Niche"


def test_age_bins_are_monotonic():
    assert get_age_category(0) == "New"
    assert get_age_category(2.99) == "New"
    assert get_age_category(3) == "Recent"
    assert get_age_category(8.99) == "Recent"
    assert get_age_category(9) == "Used"
    assert get_age_category(16.99) == "Used"
    assert get_age_category(17) == "Old"
    assert get_age_category(50) == "Old"
    assert get_age_category(None) == "Used"


def test_usage_and_performance_bins():
    assert get_usage_category(5_000) == "Low"
    assert get_usage_category(15_000) == "Average"
    assert get_usage_category(25_000) == "High"
    assert get_usage_category(40_000) == "Very_High"
    assert get_usage_category(None) == "Unknown"

    assert get_performance_category(40) == "Economy"
    assert get_performance_category(80) == "Standard"
    assert get_performance_category(120) == "Performance"
    assert get_performance_category(200) == "High_Performance"
    assert get_performance_category(None) == "Unknown"


def test_tier_weight_map_covers_all_tiers():
    expected = {"Ultra_Luxury", "Luxury", "Premium", "Mass_Market", "Niche"}
    assert set(TIER_WEIGHT_MAP.keys()) == expected
    for w in TIER_WEIGHT_MAP.values():
        assert w > 0


def test_popularity_thresholds():
    assert get_brand_popularity(3) == "Ultra_Rare"
    assert get_brand_popularity(15) == "Rare"
    assert get_brand_popularity(50) == "Uncommon"
    assert get_brand_popularity(300) == "Common"
    assert get_brand_popularity(5000) == "Popular"


def test_frequency_fallback_populated():
    assert len(BRAND_FREQUENCY_FALLBACK) > 30
    assert "ferrari" in BRAND_FREQUENCY_FALLBACK
    assert BRAND_FREQUENCY_FALLBACK["volkswagen"] > BRAND_FREQUENCY_FALLBACK["ferrari"]
