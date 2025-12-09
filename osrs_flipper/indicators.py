# osrs_flipper/indicators.py
"""Technical indicators for price analysis."""
from typing import List, Optional
import numpy as np


def calculate_rsi(prices: List[int], period: int = 14) -> Optional[float]:
    """Calculate Relative Strength Index.

    Args:
        prices: List of prices (oldest first).
        period: RSI period (default 14).

    Returns:
        RSI value 0-100, or None if insufficient data.
    """
    if len(prices) < period + 1:
        return None

    # Vectorized calculation of gains and losses
    prices_arr = np.asarray(prices)
    diffs = np.diff(prices_arr)
    gains = np.maximum(diffs, 0)
    losses = np.abs(np.minimum(diffs, 0))

    if len(gains) < period:
        return None

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    if avg_gain == 0:
        return 0.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 1)


def calculate_percentile(current: int, low: int, high: int) -> float:
    """Calculate where current price sits in historical range.

    Args:
        current: Current price.
        low: Historical low price.
        high: Historical high price.

    Returns:
        Percentile 0-100 (can be negative if below low).
    """
    if high == low:
        return 50.0  # No range, assume middle

    return round(((current - low) / (high - low)) * 100, 1)
