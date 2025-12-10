# tests/test_instant_analyzer.py
"""Tests for instant spread arbitrage analyzer."""
import pytest
from osrs_flipper.instant_analyzer import InstantSpreadAnalyzer


class TestInstantSpreadAnalyzer:
    """Test instant arbitrage opportunity detection."""

    def test_identifies_high_spread_opportunity(self):
        """High spread + strong BSR = instant opportunity."""
        analyzer = InstantSpreadAnalyzer(
            min_spread_pct=5.0,
            min_bsr=1.2,
            max_spread_pct=25.0,
        )

        result = analyzer.analyze(
            instabuy=1000,
            instasell=1100,  # 10% spread
            instabuy_vol=6000,
            instasell_vol=4000,  # BSR = 1.5
            item_name="Test Item",
        )

        assert result["is_instant_opportunity"] is True
        assert result["spread_pct"] == 10.0
        assert result["bsr"] == 1.5
        assert result["instant_roi_after_tax"] > 0

    def test_rejects_low_spread(self):
        """Low spread (<5%) rejected."""
        analyzer = InstantSpreadAnalyzer()

        result = analyzer.analyze(
            instabuy=1000,
            instasell=1030,  # 3% spread (too low)
            instabuy_vol=6000,
            instasell_vol=4000,
            item_name="Test Item",
        )

        assert result["is_instant_opportunity"] is False
        assert result["reject_reason"] == "spread_too_low"

    def test_rejects_weak_bsr(self):
        """Weak BSR (<1.2) rejected even with good spread."""
        analyzer = InstantSpreadAnalyzer()

        result = analyzer.analyze(
            instabuy=1000,
            instasell=1100,  # 10% spread (good)
            instabuy_vol=5000,
            instasell_vol=5000,  # BSR = 1.0 (weak)
            item_name="Test Item",
        )

        assert result["is_instant_opportunity"] is False
        assert result["reject_reason"] == "weak_bsr"

    def test_rejects_suspicious_spread(self):
        """Suspiciously wide spread (>25%) rejected."""
        analyzer = InstantSpreadAnalyzer()

        result = analyzer.analyze(
            instabuy=1000,
            instasell=1300,  # 30% spread (suspicious)
            instabuy_vol=6000,
            instasell_vol=4000,
            item_name="Test Item",
        )

        assert result["is_instant_opportunity"] is False
        assert result["reject_reason"] == "spread_too_wide"

    def test_calculates_tax_adjusted_roi(self):
        """ROI calculation includes GE tax."""
        analyzer = InstantSpreadAnalyzer()

        result = analyzer.analyze(
            instabuy=10_000,
            instasell=11_000,  # 10% spread
            instabuy_vol=100000,
            instasell_vol=50000,
            item_name="Regular Item",
        )

        # Tax on 11k = 11k * 0.02 = 220
        # Profit = 11000 - 220 - 10000 = 780
        # ROI = 780 / 10000 * 100 = 7.8%

        assert result["instant_roi_after_tax"] == pytest.approx(7.8, abs=0.1)

    def test_configurable_thresholds(self):
        """Analyzer accepts custom thresholds."""
        analyzer = InstantSpreadAnalyzer(
            min_spread_pct=10.0,  # Stricter
            min_bsr=1.5,          # Stricter
        )

        # 8% spread would pass default (>5%) but fail here
        result = analyzer.analyze(
            instabuy=1000,
            instasell=1080,
            instabuy_vol=6000,
            instasell_vol=4000,
            item_name="Test",
        )

        assert result["is_instant_opportunity"] is False
