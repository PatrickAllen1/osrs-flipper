"""Tests for feature engineering."""
import pytest
import pandas as pd
import numpy as np


class TestReturnsCalculator:
    """Test suite for returns calculation functions."""

    def test_daily_returns(self):
        """Test basic daily percentage returns calculation."""
        from osrs_flipper.features import calculate_returns

        prices = pd.Series([100, 110, 105, 115, 120])
        returns = calculate_returns(prices)

        # Expected: 10%, -4.55%, 9.52%, 4.35%
        assert len(returns) == 4
        assert abs(returns.iloc[0] - 0.10) < 0.01
        assert abs(returns.iloc[1] - (-0.0455)) < 0.01
        assert abs(returns.iloc[2] - 0.0952) < 0.01
        assert abs(returns.iloc[3] - 0.0435) < 0.01

    def test_log_returns(self):
        """Test logarithmic returns calculation."""
        from osrs_flipper.features import calculate_returns

        prices = pd.Series([100, 110, 105])
        returns = calculate_returns(prices, log_returns=True)

        # log(110/100) = 0.0953, log(105/110) = -0.0465
        assert len(returns) == 2
        assert abs(returns.iloc[0] - 0.0953) < 0.001
        assert abs(returns.iloc[1] - (-0.0465)) < 0.001

    def test_multi_period_returns(self):
        """Test returns calculation with multiple periods."""
        from osrs_flipper.features import calculate_returns

        # Test with 7-period returns (not enough data)
        prices = pd.Series([100, 105, 110, 108, 115, 120, 118])
        returns_7d = calculate_returns(prices, periods=7)

        # 7-period return needs 7 prior values, so only 0 results
        assert len(returns_7d) == 0

        # Test with enough data for 3-period returns
        prices_long = pd.Series([100 + i for i in range(10)])
        returns_3d = calculate_returns(prices_long, periods=3)

        # 10 - 3 = 7 returns
        assert len(returns_3d) == 7

        # Each 3-period return should be (price[i] - price[i-3]) / price[i-3]
        # For i=3: (103 - 100) / 100 = 0.03
        assert abs(returns_3d.iloc[0] - 0.03) < 0.001

    def test_returns_with_zero_price(self):
        """Test that zero prices don't cause divide-by-zero errors."""
        from osrs_flipper.features import calculate_returns

        prices = pd.Series([100, 0, 110])
        returns = calculate_returns(prices)

        # First return: (0 - 100) / 100 = -1.0
        # Second return: (110 - 0) / 0 = inf (should be handled)
        assert len(returns) == 2
        assert returns.iloc[0] == -1.0
        # Check that second value is either inf or NaN (pandas handles this)
        assert np.isinf(returns.iloc[1]) or np.isnan(returns.iloc[1])

    def test_log_returns_with_zero_price(self):
        """Test that log returns handle zero prices gracefully."""
        from osrs_flipper.features import calculate_returns

        prices = pd.Series([100, 0, 110])
        returns = calculate_returns(prices, log_returns=True)

        # log(0/100) = -inf, log(110/0) = inf
        assert len(returns) == 2
        assert np.isinf(returns.iloc[0]) or np.isnan(returns.iloc[0])

    def test_returns_with_negative_price(self):
        """Test that negative prices are handled (though shouldn't occur in OSRS)."""
        from osrs_flipper.features import calculate_returns

        # OSRS prices are always >= 1, but test edge case
        prices = pd.Series([100, -50, 110])
        returns = calculate_returns(prices)

        # Should still calculate, though result may be unusual
        assert len(returns) == 2

    def test_empty_series(self):
        """Test that empty series returns empty series."""
        from osrs_flipper.features import calculate_returns

        prices = pd.Series([])
        returns = calculate_returns(prices)

        assert len(returns) == 0

    def test_single_price(self):
        """Test that single price returns empty series."""
        from osrs_flipper.features import calculate_returns

        prices = pd.Series([100])
        returns = calculate_returns(prices)

        assert len(returns) == 0

    def test_returns_index_preserved(self):
        """Test that the index is properly handled after shift."""
        from osrs_flipper.features import calculate_returns

        # Create series with custom index
        prices = pd.Series([100, 110, 105], index=[10, 20, 30])
        returns = calculate_returns(prices)

        # After dropna, should have indices 20 and 30
        assert len(returns) == 2
        assert 20 in returns.index
        assert 30 in returns.index


class TestVolatilityFeatures:
    """Test suite for volatility calculation functions."""

    def test_rolling_volatility(self):
        """Test rolling standard deviation of returns."""
        from osrs_flipper.features import calculate_volatility

        # Known prices with predictable volatility
        # Constant prices should have volatility = 0
        constant_prices = pd.Series([100] * 20)
        vol = calculate_volatility(constant_prices, window=5)
        assert len(vol) > 0
        # All values should be 0 or very close to 0
        assert (vol == 0.0).all()

        # Volatile prices should have higher volatility
        volatile_prices = pd.Series([100, 120, 90, 110, 95, 115, 85, 105, 100, 120] * 2)
        vol_volatile = calculate_volatility(volatile_prices, window=5)
        assert len(vol_volatile) > 0
        # Should have non-zero volatility
        assert (vol_volatile > 0).any()

    def test_rolling_volatility_window_affects_result(self):
        """Test that window size affects rolling volatility."""
        from osrs_flipper.features import calculate_volatility

        prices = pd.Series([100 + i * (-1) ** i * 10 for i in range(30)])

        vol_short = calculate_volatility(prices, window=5)
        vol_long = calculate_volatility(prices, window=20)

        # Different window sizes should produce different results
        assert len(vol_short) > 0
        assert len(vol_long) > 0
        # At least some values should differ
        assert not np.allclose(vol_short.iloc[-5:], vol_long.iloc[-5:])

    def test_volatility_ratio(self):
        """Test volatility ratio calculation."""
        from osrs_flipper.features import calculate_volatility_ratio

        # High recent volatility vs low historical
        # Start stable, then become volatile
        prices = pd.Series([100] * 25 + [100, 120, 90, 110, 95, 115, 85, 105, 100, 120])
        ratio = calculate_volatility_ratio(prices, short=5, long=20)

        # Ratio should be > 1 (recent more volatile than historical)
        assert ratio > 1.0

        # Low recent volatility vs high historical
        # Need: 25 volatile values so long window (20) captures them, then 10 stable values
        # so short window (5) only sees stable
        volatile_start = [100, 120, 80, 130, 70, 140, 60, 110, 90, 125, 75, 135, 65, 115, 95, 105, 85, 125, 75, 120, 90, 110, 80, 130, 70]
        stable_end = [100, 100.5, 100, 100.5, 100, 100.5, 100, 100.5, 100, 100.5]
        prices_stable = pd.Series(volatile_start + stable_end)
        ratio_stable = calculate_volatility_ratio(prices_stable, short=5, long=20)

        # Ratio should be < 1 (recent less volatile than historical)
        assert ratio_stable < 1.0

    def test_volatility_ratio_handles_zero(self):
        """Test that volatility ratio handles zero long-term volatility."""
        from osrs_flipper.features import calculate_volatility_ratio

        # Constant prices (zero volatility)
        constant_prices = pd.Series([100] * 30)
        ratio = calculate_volatility_ratio(constant_prices, short=5, long=20)

        # Should return 1.0 when long-term volatility is 0
        assert ratio == 1.0


class TestVolumeFeatures:
    """Test suite for volume feature functions."""

    def test_volume_zscore(self):
        """Test volume z-score calculation."""
        from osrs_flipper.features import calculate_volume_zscore

        # Volume at mean should have zscore around 0
        volumes = pd.Series([100] * 30)
        zscore = calculate_volume_zscore(volumes, window=20)

        # Constant volume should have zscore = 0 (but std = 0 case)
        assert len(zscore) > 0
        assert (zscore == 0.0).all()

        # Volume spike should have high zscore
        volumes_spike = pd.Series([100] * 25 + [100, 100, 100, 300, 100])
        zscore_spike = calculate_volume_zscore(volumes_spike, window=20)

        # The spike at position -2 should have high positive zscore
        assert len(zscore_spike) > 0
        # Last few values should include the spike
        assert zscore_spike.iloc[-2] > 2.0

    def test_volume_zscore_handles_zero_std(self):
        """Test that volume zscore handles zero standard deviation."""
        from osrs_flipper.features import calculate_volume_zscore

        # Constant volume (std = 0)
        constant_volume = pd.Series([100] * 30)
        zscore = calculate_volume_zscore(constant_volume, window=20)

        # Should return 0.0 for all values
        assert len(zscore) > 0
        assert (zscore == 0.0).all()

    def test_buyer_momentum(self):
        """Test buyer momentum calculation."""
        from osrs_flipper.features import calculate_buyer_momentum

        # Increasing buyer ratios = positive momentum
        increasing = pd.Series([0.4 + i * 0.01 for i in range(20)])
        momentum_up = calculate_buyer_momentum(increasing, window=7)

        assert len(momentum_up) > 0
        # Should have positive momentum
        assert (momentum_up > 0).all()

        # Decreasing buyer ratios = negative momentum
        decreasing = pd.Series([0.6 - i * 0.01 for i in range(20)])
        momentum_down = calculate_buyer_momentum(decreasing, window=7)

        assert len(momentum_down) > 0
        # Should have negative momentum
        assert (momentum_down < 0).all()


class TestPricePositionFeatures:
    """Test suite for price position features."""

    def test_distance_from_mean(self):
        """Test z-score calculation from rolling mean."""
        from osrs_flipper.features import calculate_distance_from_mean

        # Create series where we know the mean and std
        # Price at mean should have distance ~0
        # Price 2 std above mean should have distance ~2
        prices = pd.Series([100, 100, 100, 100, 100, 120, 100, 100, 100, 100])
        distance = calculate_distance_from_mean(prices, window=5)

        # First 4 values will be NaN (need 5 values for window)
        assert len(distance) == len(prices)

        # At index 5 (price=120), mean of [100,100,100,100,120] = 104, std ≈ 8.94
        # distance ≈ (120 - 104) / 8.94 ≈ 1.79
        assert pd.notna(distance.iloc[5])
        assert distance.iloc[5] > 1.5
        assert distance.iloc[5] < 2.5

    def test_distance_from_mean_handles_zero_std(self):
        """Test that constant prices (std=0) return 0.0."""
        from osrs_flipper.features import calculate_distance_from_mean

        # Constant prices should have std=0
        prices = pd.Series([100] * 10)
        distance = calculate_distance_from_mean(prices, window=5)

        # All non-NaN values should be 0.0 (handled by returning 0 when std=0)
        assert len(distance) == len(prices)
        # After first 4 NaN values, rest should be 0.0
        assert (distance.iloc[4:] == 0.0).all()

    def test_percentile_rank(self):
        """Test percentile rank calculation in rolling window."""
        from osrs_flipper.features import calculate_percentile_rank

        # Test specific case: price 85 in range [80, 120] = 12.5%
        # Create series: [80, 100, 120, 85, ...]
        prices = pd.Series([80, 100, 120, 85, 90])
        percentile = calculate_percentile_rank(prices, window=3)

        assert len(percentile) == len(prices)

        # At index 3 (price=85), window is [100, 120, 85]
        # min=85, max=120, (85-85)/(120-85) = 0/35 = 0%
        assert pd.notna(percentile.iloc[3])
        assert percentile.iloc[3] == 0.0

        # At index 4 (price=90), window is [120, 85, 90]
        # min=85, max=120, (90-85)/(120-85) = 5/35 ≈ 14.3%
        assert pd.notna(percentile.iloc[4])
        assert abs(percentile.iloc[4] - 14.29) < 1.0

    def test_percentile_rank_at_extremes(self):
        """Test percentile at min=0% and max=100%."""
        from osrs_flipper.features import calculate_percentile_rank

        # Create series where last value is at min/max
        prices_at_min = pd.Series([100, 110, 120, 90])  # 90 is min
        percentile = calculate_percentile_rank(prices_at_min, window=4)

        # Last value (90) is minimum, should be 0%
        assert percentile.iloc[-1] == 0.0

        prices_at_max = pd.Series([100, 110, 90, 120])  # 120 is max
        percentile = calculate_percentile_rank(prices_at_max, window=4)

        # Last value (120) is maximum, should be 100%
        assert percentile.iloc[-1] == 100.0

    def test_percentile_rank_handles_zero_range(self):
        """Test that constant prices return 50.0."""
        from osrs_flipper.features import calculate_percentile_rank

        # Constant prices should have range=0
        prices = pd.Series([100] * 10)
        percentile = calculate_percentile_rank(prices, window=5)

        # All non-NaN values should be 50.0 (handled by returning 50.0 when range=0)
        assert len(percentile) == len(prices)
        # After first 4 NaN values, rest should be 50.0
        assert (percentile.iloc[4:] == 50.0).all()

    def test_mean_reversion_half_life(self):
        """Test half-life estimation for mean-reverting series."""
        from osrs_flipper.features import estimate_mean_reversion_half_life

        # Create synthetic mean-reverting series using AR(1) process
        # y_t = 0.7 * y_{t-1} + noise
        # This exhibits mean reversion with beta ≈ 0.7 (which is < 1 but > 0 in AR(1) form)
        # But in deviations-from-mean form, should show negative correlation
        np.random.seed(42)
        n = 100
        y = np.zeros(n)
        y[0] = 100
        for i in range(1, n):
            # Mean reversion: pull toward 100
            y[i] = 100 + 0.6 * (y[i-1] - 100) + np.random.normal(0, 2)

        prices = pd.Series(y)
        half_life = estimate_mean_reversion_half_life(prices)

        # Should have finite half-life (beta < 0 indicates mean reversion)
        assert not np.isinf(half_life)
        assert half_life > 0

    def test_mean_reversion_trending_series(self):
        """Test that trending series return inf (no mean reversion)."""
        from osrs_flipper.features import estimate_mean_reversion_half_life

        # Strong exponential trending series (no mean reversion)
        prices = pd.Series([100 * (1.05 ** i) for i in range(50)])

        half_life = estimate_mean_reversion_half_life(prices)

        # Trending series should have beta >= 0, return inf
        assert np.isinf(half_life)


class TestFeatureExtractor:
    """Test suite for feature extraction pipeline."""

    def test_extract_all_features(self):
        """All expected feature keys should be present."""
        from osrs_flipper.features import FeatureExtractor

        # Create sample DataFrame
        df = pd.DataFrame({
            "timestamp": range(100),
            "mid_price": [100 + i * 0.1 + np.random.randn() * 2 for i in range(100)],
            "total_volume": [10000 + np.random.randint(-1000, 1000) for _ in range(100)],
            "buyer_ratio": [0.5 + np.random.randn() * 0.05 for _ in range(100)],
        })

        extractor = FeatureExtractor()
        features = extractor.extract(df)

        expected_keys = [
            "return_1d", "return_7d", "return_30d", "volatility_14d",
            "volatility_ratio", "volume_zscore", "buyer_momentum",
            "distance_from_mean", "percentile_rank", "mean_reversion_half_life"
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

    def test_feature_values_reasonable(self):
        """Feature values should be in expected ranges."""
        from osrs_flipper.features import FeatureExtractor

        df = pd.DataFrame({
            "timestamp": range(100),
            "mid_price": [100] * 50 + [150] * 50,  # Jump from 100 to 150
            "total_volume": [10000] * 100,
            "buyer_ratio": [0.5] * 100,
        })

        extractor = FeatureExtractor()
        features = extractor.extract(df)

        # Percentile should be 0-100
        assert 0 <= features["percentile_rank"] <= 100

        # Volatility should be non-negative
        assert features["volatility_14d"] >= 0

        # Half-life should be positive (or inf)
        assert features["mean_reversion_half_life"] > 0
