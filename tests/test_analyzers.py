# tests/test_analyzers.py
import pytest
from osrs_flipper.analyzers import OversoldAnalyzer, OscillatorAnalyzer


def test_oversold_detector_identifies_near_low():
    """Item within 20% of low with 30%+ upside is oversold."""
    analyzer = OversoldAnalyzer(
        low_threshold_pct=20,  # Within 20% of low
        min_upside_pct=30,     # 30%+ upside required
    )

    result = analyzer.analyze(
        current_price=110,
        six_month_low=100,
        six_month_high=200,
        prices=[100, 110, 120, 130, 140, 150, 140, 130, 120, 110, 100, 110] * 3,  # 36 prices
    )

    assert result["is_oversold"] is True
    assert result["percentile"] < 20
    assert result["upside_pct"] > 30


def test_oversold_detector_rejects_mid_range():
    """Item at 50% of range is not oversold."""
    analyzer = OversoldAnalyzer()

    result = analyzer.analyze(
        current_price=150,
        six_month_low=100,
        six_month_high=200,
        prices=[100, 150, 200, 150] * 10,
    )

    assert result["is_oversold"] is False


def test_oversold_detector_rejects_low_upside():
    """Item with <30% upside is not oversold."""
    analyzer = OversoldAnalyzer()

    result = analyzer.analyze(
        current_price=90,
        six_month_low=85,
        six_month_high=100,  # Only 11% upside
        prices=[85, 90, 95, 100, 95, 90] * 6,
    )

    assert result["is_oversold"] is False


def test_oscillator_detects_bouncing_pattern():
    """Item bouncing between support/resistance is detected."""
    analyzer = OscillatorAnalyzer(
        min_bounces=4,
        max_support_variance_pct=10,
    )

    # Simulating amethyst arrows: 240-340 bounce pattern
    prices = [
        240, 260, 300, 340, 320, 280, 240, 250, 290, 340,
        310, 270, 240, 260, 310, 340, 300, 250, 240, 280,
    ]

    result = analyzer.analyze(prices=prices, current_price=250)

    assert result["is_oscillator"] is True
    assert 230 <= result["support"] <= 250
    assert 330 <= result["resistance"] <= 350
    assert result["bounce_count"] >= 4


def test_oscillator_rejects_trending_price():
    """Trending item is not an oscillator."""
    analyzer = OscillatorAnalyzer()

    # Steady uptrend
    prices = [100 + i * 5 for i in range(30)]

    result = analyzer.analyze(prices=prices, current_price=245)

    assert result["is_oscillator"] is False


def test_oscillator_detects_current_near_support():
    """Flag when current price is near support (buy signal)."""
    analyzer = OscillatorAnalyzer()

    prices = [
        100, 120, 140, 150, 140, 120, 100, 110, 130, 150,
        140, 120, 100, 115, 135, 150, 130, 110, 100, 120,
    ]

    result = analyzer.analyze(prices=prices, current_price=105)

    assert result["near_support"] is True
