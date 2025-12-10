# tests/test_bsr.py
"""Tests for Buyer/Seller Ratio calculations."""
import pytest
import numpy as np
from osrs_flipper.scanner import calculate_bsr


class TestBSRCalculation:
    """Test buyer/seller ratio calculation."""

    def test_bsr_equal_volume(self):
        """BSR = 1.0 when buy and sell volumes equal."""
        bsr = calculate_bsr(instabuy_vol=1000, instasell_vol=1000)
        assert bsr == 1.0

    def test_bsr_buyers_dominate(self):
        """BSR > 1.0 when buyers outnumber sellers."""
        bsr = calculate_bsr(instabuy_vol=2000, instasell_vol=1000)
        assert bsr == 2.0

    def test_bsr_sellers_dominate(self):
        """BSR < 1.0 when sellers outnumber buyers."""
        bsr = calculate_bsr(instabuy_vol=500, instasell_vol=1000)
        assert bsr == 0.5

    def test_bsr_no_sellers(self):
        """BSR = inf when only buyers (no sellers)."""
        bsr = calculate_bsr(instabuy_vol=1000, instasell_vol=0)
        assert bsr == float("inf")

    def test_bsr_no_buyers(self):
        """BSR = 0.0 when only sellers (no buyers)."""
        bsr = calculate_bsr(instabuy_vol=0, instasell_vol=1000)
        assert bsr == 0.0

    def test_bsr_no_volume(self):
        """BSR = 0.0 when no volume on either side."""
        bsr = calculate_bsr(instabuy_vol=0, instasell_vol=0)
        assert bsr == 0.0

    def test_bsr_vectorized(self):
        """BSR calculation works with numpy arrays."""
        instabuy_vols = np.array([1000, 2000, 500, 1000, 0])
        instasell_vols = np.array([1000, 1000, 1000, 0, 1000])

        bsrs = calculate_bsr(instabuy_vols, instasell_vols)

        expected = np.array([1.0, 2.0, 0.5, np.inf, 0.0])

        # Compare finite values
        finite_mask = np.isfinite(expected)
        np.testing.assert_array_equal(bsrs[finite_mask], expected[finite_mask])

        # Check inf separately
        assert np.isinf(bsrs[3])
