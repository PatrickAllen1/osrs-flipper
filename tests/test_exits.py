import pytest
from osrs_flipper.exits import calculate_exit_strategies


def test_exit_strategies_returns_all_levels():
    """Exit strategies include all percentile levels."""
    exits = calculate_exit_strategies(
        entry_price=100,
        prices=[80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180] * 3,
    )

    assert "conservative" in exits  # 25th percentile
    assert "target" in exits        # 50th percentile
    assert "aggressive" in exits    # 75th percentile
    assert "recent_peak" in exits
    assert "floor_warning" in exits


def test_exit_conservative_below_target():
    """Conservative exit is lower than target."""
    exits = calculate_exit_strategies(
        entry_price=100,
        prices=[80, 100, 120, 140, 160, 180, 200] * 5,
    )

    assert exits["conservative"]["price"] < exits["target"]["price"]
    assert exits["target"]["price"] < exits["aggressive"]["price"]


def test_exit_roi_calculated():
    """Each exit has ROI percentage."""
    exits = calculate_exit_strategies(
        entry_price=90,  # Buying near the low
        prices=[80, 100, 150, 200] * 10,
    )

    assert "roi_pct" in exits["conservative"]
    # Conservative (25th %ile) should be at/below entry, but ROI is calculated correctly
    assert isinstance(exits["conservative"]["roi_pct"], float)
    # Aggressive exit should be profitable
    assert exits["aggressive"]["roi_pct"] > 0
