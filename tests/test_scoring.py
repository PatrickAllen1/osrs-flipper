import pytest
from osrs_flipper.scoring import calculate_item_score, calculate_ev


def test_calculate_item_score():
    """Item score combines upside, oversold, liquidity, bounce history."""
    score = calculate_item_score(
        upside_pct=50,
        percentile=10,
        volume_ratio=1.5,  # 1.5x min volume
        bounce_rate=0.8,
    )

    # Score should be positive weighted combination
    assert score > 0
    assert isinstance(score, float)


def test_higher_upside_increases_score():
    """Higher upside should increase score."""
    score_low = calculate_item_score(upside_pct=20, percentile=10, volume_ratio=1.0, bounce_rate=0.5)
    score_high = calculate_item_score(upside_pct=50, percentile=10, volume_ratio=1.0, bounce_rate=0.5)

    assert score_high > score_low


def test_calculate_ev():
    """EV = capital × ROI × confidence."""
    ev = calculate_ev(
        capital=10_000_000,  # 10M
        roi_pct=30,          # 30% ROI
        confidence=0.8,      # 80% confidence
    )

    # EV = 10M × 0.30 × 0.8 = 2.4M
    assert ev == 2_400_000
