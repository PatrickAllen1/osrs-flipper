# osrs_flipper/spreads.py
"""Instant spread calculations for arbitrage opportunities."""
import numpy as np
from typing import Union
from .tax import calculate_ge_tax


def calculate_spread_pct(
    instabuy: Union[int, float, np.ndarray],
    instasell: Union[int, float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Calculate instant spread percentage.

    Args:
        instabuy: Instant buy price (buy from seller at this price)
        instasell: Instant sell price (sell to buyer at this price)

    Returns:
        Spread percentage: (instasell - instabuy) / instabuy * 100

    Examples:
        >>> calculate_spread_pct(100, 110)
        10.0
        >>> calculate_spread_pct(np.array([100, 200]), np.array([110, 220]))
        array([10., 10.])
    """
    # Vectorized calculation
    instabuy_arr = np.asarray(instabuy, dtype=float)
    instasell_arr = np.asarray(instasell, dtype=float)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        spread_pct = ((instasell_arr - instabuy_arr) / instabuy_arr) * 100

    # Return scalar if input was scalar
    if np.isscalar(instabuy):
        return float(spread_pct)
    return spread_pct


def calculate_spread_roi_after_tax(
    instabuy: Union[int, float],
    instasell: Union[int, float],
    item_name: str,
) -> float:
    """Calculate tax-adjusted ROI for instant flip.

    Args:
        instabuy: Price to buy at
        instasell: Price to sell at
        item_name: Item name (for tax calculation)

    Returns:
        Tax-adjusted ROI percentage

    Formula:
        profit = instasell - calculate_ge_tax(instasell, item_name) - instabuy
        roi = (profit / instabuy) * 100

    Examples:
        >>> calculate_spread_roi_after_tax(1000, 1100, "Regular Item")
        7.8  # After 2% tax on 1100gp
    """
    tax = calculate_ge_tax(instasell, item_name)
    profit = instasell - tax - instabuy
    roi = (profit / instabuy) * 100 if instabuy > 0 else 0.0
    return round(roi, 2)
