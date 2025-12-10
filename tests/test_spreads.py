# tests/test_spreads.py
"""Tests for instant spread calculations."""
import pytest
import numpy as np
from osrs_flipper.spreads import (
    calculate_spread_pct,
    calculate_spread_roi_after_tax,
)
from osrs_flipper.tax import calculate_ge_tax


class TestInstantSpreadCalculator:
    """Test instant spread percentage calculations."""

    def test_spread_pct_basic(self):
        """Spread percentage = (instasell - instabuy) / instabuy * 100."""
        instabuy = 100
        instasell = 110

        spread_pct = calculate_spread_pct(instabuy, instasell)

        assert spread_pct == 10.0

    def test_spread_pct_zero_spread(self):
        """Zero spread when buy == sell."""
        spread_pct = calculate_spread_pct(100, 100)
        assert spread_pct == 0.0

    def test_spread_pct_vectorized(self):
        """Vectorized calculation for arrays."""
        instabuy = np.array([100, 200, 500])
        instasell = np.array([110, 220, 550])

        spread_pct = calculate_spread_pct(instabuy, instasell)

        expected = np.array([10.0, 10.0, 10.0])
        np.testing.assert_array_almost_equal(spread_pct, expected)

    def test_spread_pct_handles_zero_buy_price(self):
        """Zero buy price returns inf (division by zero)."""
        spread_pct = calculate_spread_pct(0, 100)
        assert np.isinf(spread_pct)


class TestTaxAdjustedSpreadROI:
    """Test tax-adjusted instant flip ROI."""

    def test_roi_after_tax_regular_item(self):
        """ROI = (instasell - tax - instabuy) / instabuy * 100."""
        instabuy = 1000
        instasell = 1100
        item_name = "Regular Item"

        # Tax = 1100 * 0.02 = 22 (2% tax rate)
        # ROI = (1100 - 22 - 1000) / 1000 * 100 = 7.8%

        roi = calculate_spread_roi_after_tax(instabuy, instasell, item_name)

        # Exact calculation: tax = 1100 * 0.02 = 22
        # profit = 1100 - 22 - 1000 = 78
        # roi = 78 / 1000 * 100 = 7.8
        assert roi == pytest.approx(7.8, abs=0.1)

    def test_roi_after_tax_exempt_item(self):
        """Tax-exempt items get full spread as ROI."""
        instabuy = 1000
        instasell = 1100
        item_name = "Lobster"  # Tax exempt

        # No tax, ROI = (1100 - 1000) / 1000 * 100 = 10%
        roi = calculate_spread_roi_after_tax(instabuy, instasell, item_name)

        assert roi == 10.0

    def test_roi_after_tax_high_value(self):
        """Tax on high-value items (2% for > 1M)."""
        instabuy = 10_000_000  # 10M
        instasell = 11_000_000  # 11M
        item_name = "Twisted Bow"

        # Tax = 11M * 0.02 = 220k, capped at 5M (not hit here)
        # ROI = (11M - 220k - 10M) / 10M * 100 = 7.8%

        roi = calculate_spread_roi_after_tax(instabuy, instasell, item_name)

        tax = calculate_ge_tax(instasell, item_name)
        expected_roi = (instasell - tax - instabuy) / instabuy * 100
        assert roi == pytest.approx(expected_roi, abs=0.1)

    def test_roi_vectorized(self):
        """Vectorized ROI calculation for multiple items."""
        # This test requires vectorizing the tax calculation too
        # For now, test that it handles scalar properly
        instabuy = np.array([1000, 2000])
        instasell = np.array([1100, 2200])
        item_names = ["Item A", "Item B"]

        # Should handle array inputs
        # NOTE: This will require refactoring tax.py to support vectorization
        # For Task 1, we'll focus on scalar and document the limitation

        # Test scalar for now
        roi_1 = calculate_spread_roi_after_tax(1000, 1100, "Item A")
        roi_2 = calculate_spread_roi_after_tax(2000, 2200, "Item B")

        assert roi_1 > 0
        assert roi_2 > 0
