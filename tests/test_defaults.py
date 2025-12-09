"""Tests for strategy defaults."""
import pytest
from osrs_flipper.defaults import get_default_hold_days, STRATEGY_HOLD_DAYS


class TestStrategyDefaults:
    """Test hold days defaults by strategy."""

    def test_flip_strategy_default(self):
        """Flip strategy defaults to 3 days."""
        assert get_default_hold_days("flip") == 3

    def test_balanced_strategy_default(self):
        """Balanced strategy defaults to 7 days."""
        assert get_default_hold_days("balanced") == 7

    def test_hold_strategy_default(self):
        """Hold strategy defaults to 14 days."""
        assert get_default_hold_days("hold") == 14

    def test_unknown_strategy_returns_balanced(self):
        """Unknown strategy falls back to balanced (7 days)."""
        assert get_default_hold_days("unknown") == 7
        assert get_default_hold_days("") == 7

    def test_strategy_constants_exported(self):
        """Strategy constants are accessible."""
        assert STRATEGY_HOLD_DAYS["flip"] == 3
        assert STRATEGY_HOLD_DAYS["balanced"] == 7
        assert STRATEGY_HOLD_DAYS["hold"] == 14
