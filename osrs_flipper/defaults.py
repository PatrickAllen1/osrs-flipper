"""Default values for scanner configuration."""
from typing import Dict

STRATEGY_HOLD_DAYS: Dict[str, int] = {
    "flip": 3,
    "balanced": 7,
    "hold": 14,
}

DEFAULT_MIN_ROI: float = 20.0


def get_default_hold_days(strategy: str) -> int:
    """Get default hold days for a strategy.

    Args:
        strategy: Strategy name (flip, balanced, hold).

    Returns:
        Default hold days for the strategy.
        Falls back to balanced (7) for unknown strategies.
    """
    return STRATEGY_HOLD_DAYS.get(strategy, STRATEGY_HOLD_DAYS["balanced"])
