"""Feature engineering for price time series analysis."""
from typing import Any, Dict

import numpy as np
import pandas as pd


def calculate_returns(
    prices: pd.Series,
    periods: int = 1,
    log_returns: bool = False,
) -> pd.Series:
    """Calculate price returns.

    Args:
        prices: Price series.
        periods: Lookback periods (1 = daily, 7 = weekly).
        log_returns: If True, use log returns.

    Returns:
        Series of returns (length = len(prices) - periods).

    Examples:
        >>> prices = pd.Series([100, 110, 105])
        >>> calculate_returns(prices)
        1    0.10
        2   -0.045
        dtype: float64

        >>> calculate_returns(prices, log_returns=True)
        1    0.0953
        2   -0.0465
        dtype: float64

        >>> prices_long = pd.Series([100, 105, 110, 108, 115])
        >>> calculate_returns(prices_long, periods=3)
        3    0.08
        4    0.15
        dtype: float64
    """
    if log_returns:
        # Log returns: ln(P_t / P_{t-periods})
        returns = np.log(prices / prices.shift(periods)).dropna()
    else:
        # Simple percentage returns: (P_t - P_{t-periods}) / P_{t-periods}
        returns = prices.pct_change(periods=periods, fill_method=None).dropna()

    return returns


def calculate_volatility(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate rolling standard deviation of returns.

    Args:
        prices: Price series.
        window: Rolling window size for volatility calculation.

    Returns:
        Series of rolling volatility values.

    Examples:
        >>> prices = pd.Series([100, 110, 105, 115, 120])
        >>> vol = calculate_volatility(prices, window=3)
        >>> len(vol) > 0
        True
    """
    # Calculate returns first
    returns = calculate_returns(prices)

    # Calculate rolling standard deviation of returns
    volatility = returns.rolling(window=window).std()

    # Drop NaN values from the rolling calculation
    return volatility.dropna()


def calculate_volatility_ratio(
    prices: pd.Series,
    short: int = 5,
    long: int = 20,
) -> float:
    """Calculate ratio of short-term to long-term volatility.

    Args:
        prices: Price series.
        short: Short-term window size.
        long: Long-term window size.

    Returns:
        Ratio of short-term volatility to long-term volatility.
        Returns 1.0 if long-term volatility is zero.

    Examples:
        >>> prices = pd.Series([100] * 25 + [100, 120, 90, 110, 95])
        >>> ratio = calculate_volatility_ratio(prices, short=5, long=20)
        >>> ratio > 1.0
        True
    """
    # Calculate short-term and long-term volatility
    vol_short = calculate_volatility(prices, window=short)
    vol_long = calculate_volatility(prices, window=long)

    # Get the most recent values
    if len(vol_short) == 0 or len(vol_long) == 0:
        return 1.0

    short_val = vol_short.iloc[-1]
    long_val = vol_long.iloc[-1]

    # Handle division by zero: return 1.0 if long-term volatility is 0
    if long_val == 0.0:
        return 1.0

    return short_val / long_val


def calculate_volume_zscore(volumes: pd.Series, window: int = 20) -> pd.Series:
    """Calculate rolling z-score of volume.

    Z-score = (volume - rolling_mean) / rolling_std

    Args:
        volumes: Volume series.
        window: Rolling window size for mean and std calculation.

    Returns:
        Series of volume z-scores. Returns 0.0 when rolling std is 0.

    Examples:
        >>> volumes = pd.Series([100] * 25 + [300])
        >>> zscore = calculate_volume_zscore(volumes, window=20)
        >>> zscore.iloc[-1] > 2.0
        True
    """
    # Calculate rolling mean and std
    rolling_mean = volumes.rolling(window=window).mean()
    rolling_std = volumes.rolling(window=window).std()

    # Calculate z-score: (x - mean) / std
    # Handle division by zero: return 0.0 when std is 0
    zscore = (volumes - rolling_mean) / rolling_std

    # Replace inf/nan with 0.0 (happens when std=0)
    zscore = zscore.fillna(0.0)
    zscore = zscore.replace([np.inf, -np.inf], 0.0)

    return zscore


def calculate_buyer_momentum(
    buyer_ratios: pd.Series,
    window: int = 7,
) -> pd.Series:
    """Calculate rolling slope of buyer ratios (momentum).

    Uses simple difference approach: (value - value[window]) / window

    Args:
        buyer_ratios: Series of buyer ratios (0-1).
        window: Rolling window size for momentum calculation.

    Returns:
        Series of buyer momentum values (slope per period).

    Examples:
        >>> increasing = pd.Series([0.4 + i * 0.01 for i in range(20)])
        >>> momentum = calculate_buyer_momentum(increasing, window=7)
        >>> (momentum > 0).all()
        True
    """
    # Calculate momentum as (current - past) / window
    # This gives the average rate of change per period
    momentum = buyer_ratios.diff(window) / window

    # Drop NaN values from the diff operation
    return momentum.dropna()


def calculate_distance_from_mean(
    prices: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Calculate z-score from rolling mean.

    Args:
        prices: Price series.
        window: Rolling window size.

    Returns:
        Series of z-scores: (price - rolling_mean) / rolling_std.
        Returns 0.0 when rolling_std is zero (constant prices).

    Examples:
        >>> prices = pd.Series([100, 100, 100, 100, 120])
        >>> distance = calculate_distance_from_mean(prices, window=5)
        >>> distance.iloc[-1]  # (120 - 104) / 8.94 ≈ 1.79
        1.79
    """
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()

    # Handle zero std (constant prices) by returning 0.0
    distance = (prices - rolling_mean) / rolling_std
    distance = distance.fillna(0.0)  # Fill NaN from window and division by zero

    # Replace inf values (from zero std) with 0.0
    distance = distance.replace([np.inf, -np.inf], 0.0)

    return distance


def calculate_percentile_rank(
    prices: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Calculate percentile rank in rolling window.

    Args:
        prices: Price series.
        window: Rolling window size.

    Returns:
        Series of percentile ranks (0-100).
        Formula: 100 * (price - rolling_min) / (rolling_max - rolling_min).
        Returns 50.0 when range is zero (constant prices).

    Examples:
        >>> prices = pd.Series([80, 100, 120, 85])
        >>> percentile = calculate_percentile_rank(prices, window=3)
        >>> percentile.iloc[-1]  # (85 - 85) / (120 - 85) = 0%
        0.0
    """
    rolling_min = prices.rolling(window=window).min()
    rolling_max = prices.rolling(window=window).max()

    # Calculate percentile: (price - min) / (max - min) * 100
    price_range = rolling_max - rolling_min
    percentile = 100 * (prices - rolling_min) / price_range

    # Handle zero range (constant prices) by returning 50.0
    percentile = percentile.fillna(50.0)
    percentile = percentile.replace([np.inf, -np.inf], 50.0)

    return percentile


def estimate_mean_reversion_half_life(prices: pd.Series) -> float:
    """Estimate half-life of mean reversion using Ornstein-Uhlenbeck regression.

    Uses OLS regression on log prices (as specified in task):
    - delta = log(P_t) - log(P_{t-1})
    - lagged = log(P_{t-1})
    - Regress: delta = alpha + beta * lagged
    - Half-life = -log(2) / beta

    For mean reversion: beta < 0 (price changes negatively related to level).
    For trending: beta >= 0 (no reversion).

    Note: This detects mean reversion in log-space. A price series that oscillates
    around a constant level will have beta < 0.

    Args:
        prices: Price series (must have at least 3 observations).

    Returns:
        Half-life in time periods. Returns float('inf') if no significant mean reversion.

    Examples:
        >>> # Mean-reverting series
        >>> prices = pd.Series([100, 110, 105, 108, 103])
        >>> half_life = estimate_mean_reversion_half_life(prices)
        >>> half_life < float('inf')
        True

        >>> # Trending series
        >>> prices = pd.Series([100, 110, 120, 130, 140])
        >>> half_life = estimate_mean_reversion_half_life(prices)
        >>> half_life
        inf
    """
    if len(prices) < 3:
        return float('inf')

    # Calculate log prices
    log_prices = np.log(prices)

    # Calculate differences: delta = log(P_t) - log(P_{t-1})
    delta = log_prices.diff().dropna()

    # Calculate lagged log prices
    lagged = log_prices.shift(1).dropna()

    # Align the series (drop first element of delta, last element of lagged)
    delta = delta.iloc[1:]
    lagged = lagged.iloc[:-1]

    # Ensure alignment
    if len(delta) != len(lagged) or len(delta) == 0:
        return float('inf')

    # OLS regression: delta = alpha + beta * lagged
    # beta = cov(delta, lagged) / var(lagged)
    covariance = np.cov(delta, lagged)[0, 1]
    variance = np.var(lagged)

    if variance == 0:
        return float('inf')

    beta = covariance / variance

    # If beta >= 0, no mean reversion (trending or random walk)
    # Note: We need beta to be sufficiently negative for true mean reversion
    # A pure trend will have beta slightly negative, so we use a more lenient threshold
    if beta >= -0.0001:  # Very small threshold
        return float('inf')

    # Half-life = -log(2) / beta
    half_life = -np.log(2) / beta

    return half_life


class FeatureExtractor:
    """Extract all features from historical price data."""

    def extract(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract features from historical data DataFrame.

        Args:
            df: DataFrame with columns: timestamp, mid_price, total_volume, buyer_ratio

        Returns:
            Dict with keys: return_1d, return_7d, return_30d, volatility_14d,
            volatility_ratio, volume_zscore, buyer_momentum, distance_from_mean,
            percentile_rank, mean_reversion_half_life
        """
        prices = df["mid_price"]
        volumes = df["total_volume"]
        buyer_ratios = df["buyer_ratio"]

        # Get latest values for each feature
        returns_1d = calculate_returns(prices, periods=1)
        returns_7d = calculate_returns(prices, periods=7)
        returns_30d = calculate_returns(prices, periods=30)

        volatility = calculate_volatility(prices, window=14)
        vol_ratio = calculate_volatility_ratio(prices, short=5, long=20)
        vol_zscore = calculate_volume_zscore(volumes, window=20)
        momentum = calculate_buyer_momentum(buyer_ratios, window=7)
        distance = calculate_distance_from_mean(prices, window=30)
        percentile = calculate_percentile_rank(prices, window=30)
        half_life = estimate_mean_reversion_half_life(prices)

        return {
            "return_1d": returns_1d.iloc[-1] if len(returns_1d) > 0 else 0.0,
            "return_7d": returns_7d.iloc[-1] if len(returns_7d) > 0 else 0.0,
            "return_30d": returns_30d.iloc[-1] if len(returns_30d) > 0 else 0.0,
            "volatility_14d": volatility.iloc[-1] if len(volatility) > 0 else 0.0,
            "volatility_ratio": vol_ratio,
            "volume_zscore": vol_zscore.iloc[-1] if len(vol_zscore) > 0 else 0.0,
            "buyer_momentum": momentum.iloc[-1] if len(momentum) > 0 else 0.0,
            "distance_from_mean": distance.iloc[-1] if len(distance) > 0 else 0.0,
            "percentile_rank": percentile.iloc[-1] if len(percentile) > 0 else 50.0,
            "mean_reversion_half_life": half_life,
        }
