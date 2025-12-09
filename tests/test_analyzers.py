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
        prices=[100, 110, 120, 130, 140, 150, 140, 130, 120, 110, 100, 110] * 3,  # 36 prices
        six_month_low=100,
        six_month_high=200,
    )

    assert result["is_oversold"] is True
    assert result["percentile"] < 20
    assert result["upside_pct"] > 30


def test_oversold_detector_rejects_mid_range():
    """Item at 50% of range is not oversold."""
    analyzer = OversoldAnalyzer()

    result = analyzer.analyze(
        current_price=150,
        prices=[100, 150, 200, 150] * 10,
        six_month_low=100,
        six_month_high=200,
    )

    assert result["is_oversold"] is False


def test_oversold_detector_rejects_low_upside():
    """Item with <30% upside is not oversold."""
    analyzer = OversoldAnalyzer()

    result = analyzer.analyze(
        current_price=90,
        prices=[85, 90, 95, 100, 95, 90] * 6,
        six_month_low=85,
        six_month_high=100,  # Only 11% upside
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


class TestOversoldAnalyzerDynamicWindow:
    """Test dynamic lookback window for oversold detection."""

    def test_analyze_with_lookback_days_parameter(self):
        """Analyzer accepts lookback_days parameter."""
        analyzer = OversoldAnalyzer()

        # 90 days of price data
        prices = list(range(100, 190)) + list(range(189, 99, -1))  # 90 + 90 = 180 prices

        # Full window (180 days) - sees entire range
        result_full = analyzer.analyze(
            current_price=110,
            prices=prices,
            lookback_days=180,
        )

        # Short window (30 days) - sees only recent decline
        result_short = analyzer.analyze(
            current_price=110,
            prices=prices,
            lookback_days=30,
        )

        # Both should return valid results
        assert "percentile" in result_full
        assert "percentile" in result_short

    def test_lookback_window_affects_percentile(self):
        """Shorter lookback window changes percentile calculation."""
        analyzer = OversoldAnalyzer()

        # Simulate: old peak at 200, recent range 100-120
        # Day 1-60: prices around 200
        # Day 61-90: prices around 100-120
        old_prices = [200] * 60
        recent_prices = [100, 105, 110, 115, 120, 115, 110, 105, 100, 105] * 3  # 30 prices
        prices = old_prices + recent_prices

        current_price = 110

        # Full 90-day window: 110 is near bottom (200 high, 100 low)
        result_90 = analyzer.analyze(
            current_price=current_price,
            prices=prices,
            lookback_days=90,
        )

        # Recent 30-day window: 110 is mid-range (120 high, 100 low)
        result_30 = analyzer.analyze(
            current_price=current_price,
            prices=prices,
            lookback_days=30,
        )

        # With old peak, 110 looks oversold (low percentile)
        # With recent window only, 110 is mid-range (higher percentile)
        assert result_90["percentile"] < result_30["percentile"]

    def test_lookback_window_uses_tail_of_prices(self):
        """Lookback window uses most recent N days of prices."""
        analyzer = OversoldAnalyzer()

        # 100 days of prices, older data irrelevant
        prices = [500] * 50 + [100, 110, 120, 130, 140, 150, 140, 130, 120, 110] * 5  # 50 + 50

        result = analyzer.analyze(
            current_price=110,
            prices=prices,
            lookback_days=30,  # Only look at last 30 days
        )

        # Should NOT see the 500 peak from early data
        assert result["six_month_high"] <= 150
        assert result["six_month_low"] >= 100

    def test_backward_compatibility_default_lookback(self):
        """Default lookback is 180 days for backward compatibility."""
        analyzer = OversoldAnalyzer()

        prices = [100, 110, 120, 130, 140, 150, 140, 130, 120, 110, 100, 110] * 15  # 180 prices

        # Call without lookback_days parameter
        result = analyzer.analyze(
            current_price=110,
            prices=prices,
        )

        # Should work and use full price history (up to 180)
        assert "percentile" in result
        assert "six_month_high" in result
