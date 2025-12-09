# osrs_flipper/filters.py
"""Filtering logic for item selection."""

# Volume thresholds by price tier (max_price, min_volume)
VOLUME_TIERS = [
    (1_000, 2_500_000),        # < 1k gp: 2.5M+ volume
    (10_000, 250_000),         # 1k-10k gp: 250k+ volume
    (100_000, 25_000),         # 10k-100k gp: 25k+ volume
    (1_000_000, 2_500),        # 100k-1M gp: 2.5k+ volume
    (10_000_000, 250),         # 1M-10M gp: 250+ volume
    (float("inf"), 50),        # 10M+ gp: 50+ volume
]


def get_min_volume(price: int) -> int:
    """Get minimum volume threshold for a price tier.

    Args:
        price: Current item price in GP.

    Returns:
        Minimum daily volume required.
    """
    for max_price, min_vol in VOLUME_TIERS:
        if price < max_price:
            return min_vol
    return 50


def passes_volume_filter(price: int, volume: int) -> bool:
    """Check if item passes volume filter for its price tier.

    Args:
        price: Current item price in GP.
        volume: Daily trade volume (item count).

    Returns:
        True if volume meets threshold.
    """
    return volume >= get_min_volume(price)
