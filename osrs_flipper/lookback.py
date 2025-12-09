"""Lookback window calculation for dynamic price range analysis."""
from typing import Union
import numpy as np

MAX_LOOKBACK_DAYS = 180
LOOKBACK_MULTIPLIER = 4


def calculate_lookback_days(hold_days: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Calculate lookback window based on hold time.

    Formula: lookback_days = min(hold_days * 4, 180)

    Args:
        hold_days: Expected hold period in days (scalar or array).

    Returns:
        Lookback window in days (same type as input).

    Raises:
        ValueError: If hold_days is not positive.
    """
    # Vectorized validation
    if np.any(np.asarray(hold_days) <= 0):
        raise ValueError("hold_days must be positive")

    # Vectorized calculation (no loops)
    result = np.minimum(np.asarray(hold_days) * LOOKBACK_MULTIPLIER, MAX_LOOKBACK_DAYS)

    # Return scalar if input was scalar
    if np.isscalar(hold_days):
        return int(result)
    return result
