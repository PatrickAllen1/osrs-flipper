"""Exit strategy calculations."""
from typing import Dict, Any, List
import statistics


def calculate_exit_strategies(
    entry_price: int,
    prices: List[int],
    floor_drop_pct: float = 7,
) -> Dict[str, Any]:
    """Calculate exit strategy levels based on historical prices.

    Args:
        entry_price: Entry/buy price.
        prices: Historical price list.
        floor_drop_pct: Percentage drop to trigger floor warning.

    Returns:
        Dict with exit levels and ROI calculations.
    """
    sorted_prices = sorted(prices)
    n = len(sorted_prices)

    def percentile_price(pct: float) -> int:
        idx = int((pct / 100) * (n - 1))
        return sorted_prices[idx]

    def calc_roi(exit_price: int) -> float:
        if entry_price <= 0:
            return 0.0
        return round(((exit_price - entry_price) / entry_price) * 100, 1)

    conservative_price = percentile_price(25)
    target_price = percentile_price(50)
    aggressive_price = percentile_price(75)
    recent_peak = max(prices[-30:]) if len(prices) >= 30 else max(prices)
    floor_price = int(entry_price * (1 - floor_drop_pct / 100))

    return {
        "conservative": {
            "price": conservative_price,
            "roi_pct": calc_roi(conservative_price),
            "percentile": 25,
        },
        "target": {
            "price": target_price,
            "roi_pct": calc_roi(target_price),
            "percentile": 50,
        },
        "aggressive": {
            "price": aggressive_price,
            "roi_pct": calc_roi(aggressive_price),
            "percentile": 75,
        },
        "recent_peak": {
            "price": recent_peak,
            "roi_pct": calc_roi(recent_peak),
        },
        "floor_warning": {
            "price": floor_price,
            "drop_pct": floor_drop_pct,
        },
    }
