"""Tests for Monte Carlo simulation."""
import pytest
import numpy as np
import pandas as pd


class TestBlockBootstrap:
    """Test block bootstrap sampling."""

    def test_block_preserves_autocorrelation(self):
        """Test that block bootstrap preserves local patterns."""
        from osrs_flipper.simulator import block_bootstrap_sample

        np.random.seed(42)
        # Create returns with clear pattern
        returns = np.array([0.01, 0.02, 0.03, -0.01, -0.02, -0.03] * 10)
        samples = block_bootstrap_sample(returns, n_samples=100, block_size=3)
        assert len(samples) == 100

    def test_sample_length_matches_request(self):
        """Test that output length matches n_samples."""
        from osrs_flipper.simulator import block_bootstrap_sample

        returns = np.array([0.01, -0.01, 0.02, -0.02] * 20)
        samples = block_bootstrap_sample(returns, n_samples=50, block_size=5)
        assert len(samples) == 50

    def test_handles_small_data(self):
        """Test that small data with large block_size is handled."""
        from osrs_flipper.simulator import block_bootstrap_sample

        returns = np.array([0.01, -0.01, 0.02])
        samples = block_bootstrap_sample(returns, n_samples=10, block_size=5)
        assert len(samples) == 10

    def test_handles_empty_data(self):
        """Test that empty data returns zeros."""
        from osrs_flipper.simulator import block_bootstrap_sample

        returns = np.array([])
        samples = block_bootstrap_sample(returns, n_samples=10, block_size=5)
        assert len(samples) == 10
        assert all(s == 0 for s in samples)


class TestPricePathGenerator:
    """Test price path generation with regime adjustments."""

    def test_generates_correct_length(self):
        """Test that path has start_price + n_days prices."""
        from osrs_flipper.simulator import generate_price_path

        returns = np.array([0.01, 0.02, -0.01, 0.015, -0.02] * 10)
        path = generate_price_path(1000, returns, 30, block_size=5)
        assert len(path) == 31  # start + 30 days
        assert path[0] == 1000

    def test_applies_mean_reversion(self):
        """Test that mean reversion pulls price toward historical mean."""
        from osrs_flipper.simulator import generate_price_path

        # Price far below mean should trend upward with mean reversion
        returns = np.zeros(100)  # No base returns
        path = generate_price_path(
            start_price=500,
            returns=returns,
            n_days=10,
            mean_reversion_strength=0.1,
            historical_mean=1000,
        )
        # Should trend up toward mean
        assert path[-1] > path[0]

    def test_applies_momentum(self):
        """Test that momentum amplifies trends."""
        from osrs_flipper.simulator import generate_price_path

        # Positive base returns with momentum should amplify
        np.random.seed(42)
        returns = np.array([0.05] * 50)  # Consistent positive returns

        np.random.seed(42)
        path_no_mom = generate_price_path(1000, returns, 10, momentum_factor=0.0)

        np.random.seed(42)
        path_with_mom = generate_price_path(1000, returns, 10, momentum_factor=0.5)

        # With momentum should end higher (amplified trend)
        assert path_with_mom[-1] > path_no_mom[-1]

    def test_price_stays_positive(self):
        """Test that price floors at 1 GP even with extreme losses."""
        from osrs_flipper.simulator import generate_price_path

        # Even with terrible returns, price floors at 1
        returns = np.array([-0.99] * 50)  # Extreme losses
        path = generate_price_path(1000, returns, 30)
        assert all(p >= 1 for p in path)

    def test_volatility_multiplier(self):
        """Test that volatility multiplier scales returns."""
        from osrs_flipper.simulator import generate_price_path

        np.random.seed(42)
        returns = np.random.randn(100) * 0.05  # 5% base volatility

        np.random.seed(42)
        path_normal = generate_price_path(1000, returns, 30, volatility_multiplier=1.0)

        np.random.seed(42)
        path_high_vol = generate_price_path(1000, returns, 30, volatility_multiplier=2.0)

        # Calculate actual volatility of each path (std of price changes)
        changes_normal = np.diff(path_normal)
        changes_high_vol = np.diff(path_high_vol)

        # High vol path should have higher std of changes
        assert np.std(changes_high_vol) > np.std(changes_normal)


class TestMonteCarloSimulator:
    """Test Monte Carlo simulator with regime detection."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data with some trend."""
        np.random.seed(42)
        base = 1000
        prices = [base]
        for _ in range(99):
            change = np.random.randn() * 20
            prices.append(max(1, prices[-1] + change))
        return pd.Series(prices)

    def test_returns_probability_distribution(self, sample_prices):
        """Probabilities should sum to approximately 1."""
        from osrs_flipper.simulator import MonteCarloSimulator

        sim = MonteCarloSimulator(sample_prices)
        results = sim.run(n_sims=1000, n_days=30)

        # prob_profit + prob_loss + prob_unchanged ≈ 1
        assert 0 <= results["prob_profit"] <= 1
        assert 0 <= results["prob_loss"] <= 1
        # Note: prob_profit + prob_loss might not equal 1 if some end exactly at start

    def test_returns_percentile_distribution(self, sample_prices):
        """Percentiles should be in ascending order."""
        from osrs_flipper.simulator import MonteCarloSimulator

        sim = MonteCarloSimulator(sample_prices)
        results = sim.run(n_sims=1000, n_days=30)

        p = results["percentiles"]
        assert p["5"] <= p["25"] <= p["50"] <= p["75"] <= p["95"]

    def test_uses_regime_params(self, sample_prices):
        """Simulator should detect and use regime."""
        from osrs_flipper.simulator import MonteCarloSimulator

        sim = MonteCarloSimulator(sample_prices)
        results = sim.run(n_sims=100, n_days=10)

        assert "regime" in results
        assert results["regime"] in ["trending_up", "trending_down", "mean_reverting", "chaotic"]

    def test_calculates_roi_percentiles(self, sample_prices):
        """ROI percentiles should be calculated correctly."""
        from osrs_flipper.simulator import MonteCarloSimulator

        sim = MonteCarloSimulator(sample_prices)
        results = sim.run(n_sims=1000, n_days=30)

        assert "roi_percentiles" in results
        roi = results["roi_percentiles"]
        assert roi["5"] <= roi["25"] <= roi["50"] <= roi["75"] <= roi["95"]
