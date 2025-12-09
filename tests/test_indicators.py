# tests/test_indicators.py
import pytest
from osrs_flipper.indicators import calculate_rsi, calculate_percentile


def test_rsi_returns_none_for_insufficient_data():
    """RSI needs at least period+1 prices."""
    prices = [100, 105, 102]  # Only 3 prices, need 15 for period=14
    assert calculate_rsi(prices, period=14) is None


def test_rsi_returns_100_for_all_gains():
    """RSI = 100 when all price moves are gains."""
    # 15 consecutive gains
    prices = [100 + i * 10 for i in range(16)]
    rsi = calculate_rsi(prices, period=14)
    assert rsi == 100.0


def test_rsi_returns_0_for_all_losses():
    """RSI = 0 when all price moves are losses."""
    # 15 consecutive losses
    prices = [200 - i * 10 for i in range(16)]
    rsi = calculate_rsi(prices, period=14)
    assert rsi == 0.0


def test_rsi_returns_around_50_for_mixed():
    """RSI near 50 for balanced gains/losses."""
    # Alternating up and down with equal magnitude
    prices = [100, 110, 100, 110, 100, 110, 100, 110, 100, 110, 100, 110, 100, 110, 100, 110]
    rsi = calculate_rsi(prices, period=14)
    assert 45 <= rsi <= 55  # Should be around 50


def test_percentile_at_low_returns_0():
    """Current price at historical low = 0th percentile."""
    low, high, current = 100, 200, 100
    assert calculate_percentile(current, low, high) == 0.0


def test_percentile_at_high_returns_100():
    """Current price at historical high = 100th percentile."""
    low, high, current = 100, 200, 200
    assert calculate_percentile(current, low, high) == 100.0


def test_percentile_at_midpoint_returns_50():
    """Current price at midpoint = 50th percentile."""
    low, high, current = 100, 200, 150
    assert calculate_percentile(current, low, high) == 50.0


def test_percentile_below_low_returns_negative():
    """Current price below historical low = negative percentile."""
    low, high, current = 100, 200, 80
    assert calculate_percentile(current, low, high) == -20.0
