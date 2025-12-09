"""EV and scoring calculations."""

# Score weights
WEIGHT_UPSIDE = 0.4
WEIGHT_OVERSOLD = 0.3
WEIGHT_LIQUIDITY = 0.2
WEIGHT_BOUNCE = 0.1


def calculate_item_score(
    upside_pct: float,
    percentile: float,
    volume_ratio: float,
    bounce_rate: float,
) -> float:
    """Calculate composite score for an item.

    Args:
        upside_pct: Upside potential percentage.
        percentile: Where price sits in historical range (lower = more oversold).
        volume_ratio: Volume / min_required_volume (capped at 2).
        bounce_rate: Historical bounce success rate (0-1).

    Returns:
        Composite score.
    """
    # Normalize components to 0-100 scale
    upside_score = min(upside_pct, 100)  # Cap at 100%
    oversold_score = 100 - percentile     # Lower percentile = higher score
    liquidity_score = min(volume_ratio, 2) * 50  # 2x volume = 100
    bounce_score = bounce_rate * 100

    score = (
        upside_score * WEIGHT_UPSIDE
        + oversold_score * WEIGHT_OVERSOLD
        + liquidity_score * WEIGHT_LIQUIDITY
        + bounce_score * WEIGHT_BOUNCE
    )

    return round(score, 2)


def calculate_ev(capital: int, roi_pct: float, confidence: float) -> int:
    """Calculate expected value.

    Args:
        capital: Capital allocated.
        roi_pct: Expected ROI percentage.
        confidence: Confidence factor (0-1).

    Returns:
        Expected value in GP.
    """
    return int(capital * (roi_pct / 100) * confidence)
