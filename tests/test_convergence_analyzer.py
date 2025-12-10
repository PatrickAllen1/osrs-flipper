# tests/test_convergence_analyzer.py
"""Tests for convergence/mean reversion analyzer."""
import pytest
from osrs_flipper.convergence_analyzer import ConvergenceAnalyzer


class TestConvergenceAnalyzer:
    """Test convergence opportunity detection."""

    def test_identifies_convergence_opportunity(self):
        """Item down across all timeframes = convergence."""
        analyzer = ConvergenceAnalyzer()

        result = analyzer.analyze(
            current_instabuy=100,
            one_day_high=120,
            one_week_high=150,
            one_month_high=200,
            bsr=1.0,  # Neutral (not being dumped)
        )

        # Distance from highs:
        # 1d: (120-100)/120 = 16.7%
        # 1w: (150-100)/150 = 33.3%
        # 1m: (200-100)/200 = 50%

        assert result["is_convergence"] is True
        assert result["distance_from_1d_high"] == pytest.approx(16.7, abs=0.5)
        assert result["distance_from_1w_high"] == pytest.approx(33.3, abs=0.5)
        assert result["distance_from_1m_high"] == pytest.approx(50.0, abs=0.5)

    def test_rejects_item_near_highs(self):
        """Item near recent highs (not oversold)."""
        analyzer = ConvergenceAnalyzer()

        result = analyzer.analyze(
            current_instabuy=115,
            one_day_high=120,
            one_week_high=125,
            one_month_high=130,
            bsr=1.5,
        )

        # Only 4-12% below highs (not enough)
        assert result["is_convergence"] is False
        assert result["reject_reason"] == "not_oversold"

    def test_rejects_item_being_dumped(self):
        """Item with BSR < min threshold (sellers dominate)."""
        analyzer = ConvergenceAnalyzer(min_bsr=0.8)

        result = analyzer.analyze(
            current_instabuy=100,
            one_day_high=150,
            one_week_high=180,
            one_month_high=200,
            bsr=0.5,  # Heavy selling
        )

        assert result["is_convergence"] is False
        assert result["reject_reason"] == "being_dumped"

    def test_calculates_target_price(self):
        """Target = max of recent highs."""
        analyzer = ConvergenceAnalyzer()

        result = analyzer.analyze(
            current_instabuy=100,
            one_day_high=120,
            one_week_high=180,  # Highest
            one_month_high=150,
            bsr=1.0,
        )

        # Target = max(120, 180, 150) = 180
        assert result["target_price"] == 180
        assert result["upside_pct"] == pytest.approx(80.0)  # (180-100)/100

    def test_configurable_thresholds(self):
        """Custom distance thresholds."""
        analyzer = ConvergenceAnalyzer(
            min_distance_1d=20.0,  # Stricter
            min_distance_1w=30.0,
            min_distance_1m=40.0,
        )

        result = analyzer.analyze(
            current_instabuy=100,
            one_day_high=115,  # Only 13% below (would pass default 10%)
            one_week_high=150,
            one_month_high=200,
            bsr=1.0,
        )

        # Fails stricter 1d threshold
        assert result["is_convergence"] is False

    def test_convergence_with_strong_signal(self):
        """All 3 timeframes show oversold + neutral BSR."""
        analyzer = ConvergenceAnalyzer(
            min_distance_1d=10.0,
            min_distance_1w=15.0,
            min_distance_1m=20.0,
        )

        result = analyzer.analyze(
            current_instabuy=100,
            one_day_high=120,   # 16.7% below (> 10%)
            one_week_high=140,  # 28.6% below (> 15%)
            one_month_high=150, # 33.3% below (> 20%)
            bsr=0.9,  # Slight selling but > 0.8 threshold
        )

        assert result["is_convergence"] is True
        assert result["convergence_strength"] == "strong"  # All 3 signals
