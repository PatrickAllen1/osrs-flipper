# osrs_flipper/timeframes.py
"""Multi-timeframe price analysis."""
import numpy as np
from typing import Dict, Any, Optional
from .api import OSRSClient


def fetch_timeframe_highs(
    client: OSRSClient,
    item_id: int,
    current_instabuy: Optional[int] = None,
) -> Dict[str, Any]:
    """Fetch high prices across 1d, 1w, 1m timeframes.

    Args:
        client: OSRS API client
        item_id: Item ID to fetch
        current_instabuy: Current instant buy price (for distance calculation)

    Returns:
        Dictionary with:
        - 1d_high: Highest price in last 24 hours
        - 1w_high: Highest price in last 7 days
        - 1m_high: Highest price in last 30 days
        - distance_from_1d_high: % below 1d high
        - distance_from_1w_high: % below 1w high
        - distance_from_1m_high: % below 1m high

    Data Flow:
        API (1h timestep, ~720 points for 30d)
        → Extract avgHighPrice per point
        → Vectorized max over windows (24h, 168h, 720h)
        → Calculate distances
    """
    # Fetch 1h resolution data (efficient for 1m lookback)
    timeseries = client.fetch_timeseries(item_id, timestep="1h")

    if not timeseries:
        # No data available
        return {
            "1d_high": 0,
            "1w_high": 0,
            "1m_high": 0,
            "distance_from_1d_high": 0.0,
            "distance_from_1w_high": 0.0,
            "distance_from_1m_high": 0.0,
        }

    # Extract highs (vectorized)
    highs = np.array([
        point.get("avgHighPrice", 0) for point in timeseries
    ], dtype=float)

    # Calculate window highs (vectorized max over slices)
    # 1d = last 24 hours, 1w = last 168 hours, 1m = all data (up to 720)
    n = len(highs)

    one_day_high = np.max(highs[-24:]) if n >= 1 else 0
    one_week_high = np.max(highs[-168:]) if n >= 1 else 0
    one_month_high = np.max(highs) if n >= 1 else 0

    # Handle NaN values (can occur if timeseries has no valid data)
    one_day_high = 0 if np.isnan(one_day_high) else one_day_high
    one_week_high = 0 if np.isnan(one_week_high) else one_week_high
    one_month_high = 0 if np.isnan(one_month_high) else one_month_high

    result = {
        "1d_high": int(one_day_high),
        "1w_high": int(one_week_high),
        "1m_high": int(one_month_high),
    }

    # Calculate distances if current price provided
    if current_instabuy is not None:
        for period, high in [("1d", one_day_high), ("1w", one_week_high), ("1m", one_month_high)]:
            if high > 0:
                distance = ((high - current_instabuy) / high) * 100
                result[f"distance_from_{period}_high"] = round(float(distance), 1)
            else:
                result[f"distance_from_{period}_high"] = 0.0

    return result
