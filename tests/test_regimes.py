"""Tests for market regime classifier."""
import pytest
import pandas as pd
import numpy as np
from osrs_flipper.regimes import Regime, RegimeClassifier


class TestRegimeClassifier:
    def test_trending_up_detection(self):
        """Clear uptrend should be classified as TRENDING_UP."""
        # Create steadily increasing prices with higher growth rate (6% per period)
        prices = pd.Series([100 * (1.06 ** i) for i in range(60)])
        classifier = RegimeClassifier()
        regime = classifier.classify(prices)
        assert regime == Regime.TRENDING_UP

    def test_trending_down_detection(self):
        """Clear downtrend should be classified as TRENDING_DOWN."""
        # Create steadily decreasing prices
        prices = pd.Series([200 - i * 5 for i in range(60)])
        classifier = RegimeClassifier()
        regime = classifier.classify(prices)
        assert regime == Regime.TRENDING_DOWN

    def test_mean_reverting_detection(self):
        """Oscillating prices should be MEAN_REVERTING."""
        # Create oscillating prices around a mean
        np.random.seed(42)
        prices = pd.Series([100 + 10 * np.sin(i * 0.5) + np.random.randn() for i in range(100)])
        classifier = RegimeClassifier()
        regime = classifier.classify(prices)
        assert regime == Regime.MEAN_REVERTING

    def test_chaotic_detection(self):
        """High volatility should be classified as CHAOTIC."""
        # Create highly volatile prices around a mean (so no strong trend)
        np.random.seed(42)
        prices = pd.Series([100 + np.random.randn() * 20 for _ in range(60)])
        classifier = RegimeClassifier(volatility_threshold=0.05)  # Standard threshold
        regime = classifier.classify(prices)
        assert regime == Regime.CHAOTIC

    def test_get_simulation_params(self):
        """Each regime should have valid simulation parameters."""
        classifier = RegimeClassifier()
        for regime in Regime:
            params = classifier.get_simulation_params(regime)
            assert "mean_reversion_strength" in params
            assert "momentum_factor" in params
            assert "volatility_multiplier" in params
            assert params["volatility_multiplier"] > 0
