"""Buyer/Seller Ratio (BSR) calculation."""
from typing import Union
import numpy as np


def calculate_bsr(
    instabuy_vol: Union[int, float, np.ndarray],
    instasell_vol: Union[int, float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Calculate Buyer/Seller Ratio (BSR).

    Args:
        instabuy_vol: Volume of buyers (paying ask price)
        instasell_vol: Volume of sellers (hitting bid price)

    Returns:
        BSR = instabuy_vol / instasell_vol
        - BSR > 1.0: Buyers dominate (bullish)
        - BSR = 1.0: Balanced
        - BSR < 1.0: Sellers dominate (bearish)
        - BSR = inf: Only buyers, no sellers
        - BSR = 0.0: Only sellers or no volume

    Examples:
        >>> calculate_bsr(2000, 1000)
        2.0
        >>> calculate_bsr(np.array([1000, 2000]), np.array([1000, 1000]))
        array([1., 2.])
    """
    # Vectorized calculation
    instabuy = np.asarray(instabuy_vol, dtype=float)
    instasell = np.asarray(instasell_vol, dtype=float)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        bsr = instabuy / instasell

    # Handle edge cases
    # When instasell = 0 and instabuy > 0, result is inf (correct)
    # When both = 0, result is nan, convert to 0.0
    if np.isscalar(instabuy_vol):
        return 0.0 if np.isnan(bsr) else float(bsr)
    else:
        bsr = np.where(np.isnan(bsr), 0.0, bsr)
        return bsr
