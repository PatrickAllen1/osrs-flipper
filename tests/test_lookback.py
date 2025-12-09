"""Tests for lookback window calculation."""
import pytest
import numpy as np
from osrs_flipper.lookback import calculate_lookback_days


class TestCalculateLookbackDays:
    """Test lookback_days = min(hold_days * 4, 180)."""

    @pytest.mark.parametrize("hold_days,expected", [
        (3, 12),    # flip strategy: 3 * 4 = 12
        (7, 28),    # balanced strategy: 7 * 4 = 28
        (14, 56),   # hold strategy: 14 * 4 = 56
        (30, 120),  # extended hold: 30 * 4 = 120
        (50, 180),  # capped at 180: 50 * 4 = 200 -> 180
        (100, 180), # capped at 180
    ])
    def test_lookback_formula(self, hold_days: int, expected: int):
        """Lookback = hold_days * 4, capped at 180."""
        result = calculate_lookback_days(hold_days)
        assert result == expected

    def test_input_validation_positive(self):
        """Hold days must be positive."""
        with pytest.raises(ValueError, match="hold_days must be positive"):
            calculate_lookback_days(0)
        with pytest.raises(ValueError, match="hold_days must be positive"):
            calculate_lookback_days(-5)

    def test_vectorized_input(self):
        """Function works with numpy arrays (vectorized)."""
        hold_days = np.array([3, 7, 14, 50])
        expected = np.array([12, 28, 56, 180])
        result = calculate_lookback_days(hold_days)
        np.testing.assert_array_equal(result, expected)

    def test_return_type_int(self):
        """Returns int for scalar input."""
        result = calculate_lookback_days(7)
        assert isinstance(result, int)
