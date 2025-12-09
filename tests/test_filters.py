# tests/test_filters.py
import pytest
from osrs_flipper.filters import get_min_volume, passes_volume_filter

# Volume tiers from brainstorming:
# < 1k gp: 2.5M+ volume
# 1k-10k gp: 250k+ volume
# 10k-100k gp: 25k+ volume
# 100k-1M gp: 2.5k+ volume
# 1M-10M gp: 250+ volume
# 10M+ gp: 50+ volume


@pytest.mark.parametrize("price,expected_min_volume", [
    (500, 2_500_000),        # < 1k tier
    (5_000, 250_000),        # 1k-10k tier
    (50_000, 25_000),        # 10k-100k tier
    (500_000, 2_500),        # 100k-1M tier
    (5_000_000, 250),        # 1M-10M tier
    (50_000_000, 50),        # 10M+ tier
])
def test_get_min_volume_returns_correct_tier(price, expected_min_volume):
    assert get_min_volume(price) == expected_min_volume


def test_passes_volume_filter_accepts_above_threshold():
    # 500 gp item needs 2.5M volume, has 3M
    assert passes_volume_filter(price=500, volume=3_000_000) is True


def test_passes_volume_filter_rejects_below_threshold():
    # 500 gp item needs 2.5M volume, has 1M
    assert passes_volume_filter(price=500, volume=1_000_000) is False
