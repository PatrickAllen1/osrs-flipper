"""Single item deep analysis module.

This module provides comprehensive analysis of individual items by combining
instant spread analysis, convergence analysis, and oversold detection.

Data Flow:
    IN: item_id (int)
    FETCH: mapping, latest, volumes, timeseries from OSRSClient
    ANALYZE:
        - instant: InstantSpreadAnalyzer (spread + BSR)
        - convergence: ConvergenceAnalyzer (timeframe highs)
        - oversold: OversoldAnalyzer (if historical data available)
    OUT: {
        item_id, name, instabuy, instasell, bsr,
        instant: {...},
        convergence: {...},
        oversold: {...} [optional]
    }
"""
from typing import Dict, Any
from ..api import OSRSClient
from ..instant_analyzer import InstantSpreadAnalyzer
from ..convergence_analyzer import ConvergenceAnalyzer
from ..analyzers import OversoldAnalyzer
from ..timeframes import fetch_timeframe_highs
from ..bsr import calculate_bsr


def analyze_single_item(item_id: int) -> Dict[str, Any]:
    """Perform comprehensive analysis on a single item.

    Runs all available analyzers (instant, convergence, oversold) and returns
    complete analysis data. This is used for deep-dive analysis on specific items.

    Args:
        item_id: Item ID to analyze

    Returns:
        Analysis result with all available data

    Raises:
        ValueError: If item not found or missing required price data

    Data Flow:
        IN: item_id
        FETCH:
            - mapping → item name, limit
            - latest → instabuy, instasell
            - volumes → buyer/seller volumes
            - timeseries (1h) → timeframe highs for convergence
            - timeseries (24h) → price history for oversold
        ANALYZE:
            - InstantSpreadAnalyzer(instabuy, instasell, volumes)
            - ConvergenceAnalyzer(instabuy, 1d/1w/1m highs, bsr)
            - OversoldAnalyzer(prices, current_price) [if sufficient history]
        OUT: {
            item_id: int,
            name: str,
            instabuy: int,
            instasell: int,
            instabuy_vol: int,
            instasell_vol: int,
            bsr: float,
            instant: {
                spread_pct: float,
                bsr: float,
                is_instant_opportunity: bool,
                instant_roi_after_tax: float [if opportunity],
                reject_reason: str [if rejected]
            },
            convergence: {
                is_convergence: bool,
                distance_from_1d_high: float,
                distance_from_1w_high: float,
                distance_from_1m_high: float,
                target_price: int [if convergence],
                upside_pct: float [if convergence],
                convergence_strength: str [if convergence],
                reject_reason: str [if rejected]
            },
            oversold: {  # Optional - only if sufficient history
                is_oversold: bool,
                percentile: float,
                rsi: float,
                upside_pct: float,
                six_month_low: int,
                six_month_high: int
            }
        }
    """
    client = OSRSClient()

    # Fetch item mapping
    mapping = client.fetch_mapping()
    if item_id not in mapping:
        raise ValueError(f"Item {item_id} not found in mapping")

    item = mapping[item_id]
    name = item.get("name", "Unknown")

    # Fetch latest prices
    latest = client.fetch_latest()
    price_data = latest.get(str(item_id), {})
    if not price_data:
        raise ValueError(f"No price data for item {item_id}")

    instabuy = price_data.get("low")
    instasell = price_data.get("high")

    if not instabuy or not instasell:
        raise ValueError(f"Missing buy/sell prices for item {item_id}")

    # Fetch volumes
    volumes = client.fetch_volumes()
    vol_data = volumes.get(str(item_id), {})
    instabuy_vol = vol_data.get("highPriceVolume", 0) or 0
    instasell_vol = vol_data.get("lowPriceVolume", 0) or 0

    # Calculate BSR
    bsr = calculate_bsr(instabuy_vol, instasell_vol)

    # Build result dict
    result = {
        "item_id": item_id,
        "name": name,
        "instabuy": instabuy,
        "instasell": instasell,
        "instabuy_vol": instabuy_vol,
        "instasell_vol": instasell_vol,
        "bsr": round(bsr, 2) if bsr != float("inf") else 99.9,
    }

    # === INSTANT ANALYSIS ===
    instant_analyzer = InstantSpreadAnalyzer()
    result["instant"] = instant_analyzer.analyze(
        instabuy=instabuy,
        instasell=instasell,
        instabuy_vol=instabuy_vol,
        instasell_vol=instasell_vol,
        item_name=name
    )

    # === CONVERGENCE ANALYSIS ===
    try:
        timeframe_highs = fetch_timeframe_highs(client, item_id, instabuy)

        convergence_analyzer = ConvergenceAnalyzer()
        result["convergence"] = convergence_analyzer.analyze(
            current_instabuy=instabuy,
            one_day_high=timeframe_highs["1d_high"],
            one_week_high=timeframe_highs["1w_high"],
            one_month_high=timeframe_highs["1m_high"],
            bsr=bsr
        )
    except Exception:
        # No historical data available
        result["convergence"] = {
            "is_convergence": False,
            "reject_reason": "no_historical_data"
        }

    # === OVERSOLD ANALYSIS ===
    # Requires 24h timeseries data (at least 30 days of history)
    try:
        history = client.fetch_timeseries(item_id, timestep="24h")

        if len(history) >= 30:
            # Extract midpoint prices from timeseries
            prices = []
            for point in history:
                h = point.get("avgHighPrice")
                l = point.get("avgLowPrice")
                if h and l:
                    prices.append((h + l) // 2)

            if len(prices) >= 30:
                current_price = (instasell + instabuy) // 2

                oversold_analyzer = OversoldAnalyzer()
                result["oversold"] = oversold_analyzer.analyze(
                    current_price=current_price,
                    prices=prices,
                    lookback_days=180
                )
    except Exception:
        # Not enough history for oversold analysis - this is OK
        # Don't include oversold key if insufficient data
        pass

    return result
