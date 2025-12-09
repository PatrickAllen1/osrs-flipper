# osrs_flipper/analyzers.py
"""Pattern detection analyzers."""
import statistics
from typing import Dict, Any, List, Tuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .indicators import calculate_rsi, calculate_percentile


class OversoldAnalyzer:
    """Detect oversold items near historical lows with recovery potential."""

    def __init__(
        self,
        low_threshold_pct: float = 20,
        min_upside_pct: float = 30,
    ):
        """Initialize analyzer.

        Args:
            low_threshold_pct: Max percentile to consider oversold (default 20).
            min_upside_pct: Minimum upside to historical high (default 30%).
        """
        self.low_threshold_pct = low_threshold_pct
        self.min_upside_pct = min_upside_pct

    def analyze(
        self,
        current_price: int,
        prices: List[int],
        lookback_days: int = 180,
        six_month_low: int | None = None,
        six_month_high: int | None = None,
    ) -> Dict[str, Any]:
        """Analyze item for oversold opportunity.

        Args:
            current_price: Current item price.
            prices: List of historical prices (daily).
            lookback_days: Number of days to look back for range calculation.
            six_month_low: Deprecated - calculated from prices if not provided.
            six_month_high: Deprecated - calculated from prices if not provided.

        Returns:
            Analysis result dict.
        """
        # Use dynamic lookback window (vectorized slice)
        lookback_prices = prices[-lookback_days:] if len(prices) > lookback_days else prices

        # Calculate range from lookback window (no loops)
        window_low = min(lookback_prices) if lookback_prices else current_price
        window_high = max(lookback_prices) if lookback_prices else current_price

        # Support deprecated parameters for backward compatibility
        if six_month_low is not None:
            window_low = six_month_low
        if six_month_high is not None:
            window_high = six_month_high

        percentile = calculate_percentile(current_price, window_low, window_high)
        rsi = calculate_rsi(prices)  # RSI uses full history

        # Calculate upside potential (vectorized division)
        upside_pct = ((window_high - current_price) / current_price) * 100 if current_price > 0 else 0

        is_oversold = (
            percentile <= self.low_threshold_pct
            and upside_pct >= self.min_upside_pct
        )

        return {
            "is_oversold": is_oversold,
            "percentile": round(percentile, 1),
            "rsi": rsi,
            "upside_pct": round(upside_pct, 1),
            "six_month_low": window_low,
            "six_month_high": window_high,
        }


def find_local_extrema(prices: List[int], window: int = 3) -> Tuple[List[int], List[int]]:
    """Find local minima and maxima in price series."""
    prices_arr = np.asarray(prices)
    n = len(prices_arr)

    if n < 2 * window + 1:
        return [], []

    # Create rolling window view - shape: (n - 2*window, 2*window + 1)
    windows = sliding_window_view(prices_arr, 2 * window + 1)

    # Center values being tested
    centers = prices_arr[window:-window]

    # Vectorized min/max comparison
    is_minima = centers == windows.min(axis=1)
    is_maxima = centers == windows.max(axis=1)

    minima = centers[is_minima].tolist()
    maxima = centers[is_maxima].tolist()

    return minima, maxima


class OscillatorAnalyzer:
    """Detect range-bound items bouncing between support/resistance."""

    def __init__(
        self,
        min_bounces: int = 4,
        max_support_variance_pct: float = 10,
        support_proximity_pct: float = 5,
    ):
        self.min_bounces = min_bounces
        self.max_support_variance_pct = max_support_variance_pct
        self.support_proximity_pct = support_proximity_pct

    def analyze(self, prices: List[int], current_price: int) -> Dict[str, Any]:
        if len(prices) < 10:
            return {"is_oscillator": False, "reason": "insufficient_data"}

        minima, maxima = find_local_extrema(prices)

        if len(minima) < 2 or len(maxima) < 2:
            return {"is_oscillator": False, "reason": "no_bounces"}

        support = statistics.mean(minima) if minima else prices[-1]
        resistance = statistics.mean(maxima) if maxima else prices[-1]

        if len(minima) >= 2 and support > 0:
            support_std = statistics.stdev(minima)
            support_variance_pct = (support_std / support) * 100
        else:
            support_variance_pct = float("inf")

        bounce_count = len(minima) + len(maxima)

        is_oscillator = (
            bounce_count >= self.min_bounces
            and support_variance_pct <= self.max_support_variance_pct
            and resistance > support
        )

        if support > 0:
            distance_from_support = ((current_price - support) / support) * 100
            near_support = distance_from_support <= self.support_proximity_pct
        else:
            near_support = False

        range_pct = ((resistance - support) / support) * 100 if support > 0 else 0

        return {
            "is_oscillator": is_oscillator,
            "support": round(support),
            "resistance": round(resistance),
            "range_pct": round(range_pct, 1),
            "bounce_count": bounce_count,
            "support_variance_pct": round(support_variance_pct, 1),
            "near_support": near_support,
        }
