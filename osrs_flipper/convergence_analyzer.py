# osrs_flipper/convergence_analyzer.py
"""Convergence/mean reversion opportunity analyzer."""
from typing import Dict, Any


class ConvergenceAnalyzer:
    """Detect mean reversion opportunities across timeframes.

    Strategy: Items that crashed across 1d/1w/1m but likely to revert.

    Criteria:
    - Current price significantly below 1d/1w/1m highs (convergence signal)
    - BSR >= min_bsr (not being dumped by sellers)
    - Target = recent highs (not ancient 6-month peaks)
    """

    def __init__(
        self,
        min_distance_1d: float = 10.0,
        min_distance_1w: float = 15.0,
        min_distance_1m: float = 20.0,
        min_bsr: float = 0.8,
    ):
        """Initialize analyzer.

        Args:
            min_distance_1d: Min % below 1d high (default 10%)
            min_distance_1w: Min % below 1w high (default 15%)
            min_distance_1m: Min % below 1m high (default 20%)
            min_bsr: Min BSR to avoid dump scenarios (default 0.8)
        """
        self.min_distance_1d = min_distance_1d
        self.min_distance_1w = min_distance_1w
        self.min_distance_1m = min_distance_1m
        self.min_bsr = min_bsr

    def analyze(
        self,
        current_instabuy: int,
        one_day_high: int,
        one_week_high: int,
        one_month_high: int,
        bsr: float,
    ) -> Dict[str, Any]:
        """Analyze item for convergence opportunity.

        Args:
            current_instabuy: Current instant buy price
            one_day_high: Highest price in last 24h
            one_week_high: Highest price in last 7d
            one_month_high: Highest price in last 30d
            bsr: Buyer/seller ratio

        Returns:
            Analysis result with:
            - is_convergence: bool
            - distance_from_1d_high: float
            - distance_from_1w_high: float
            - distance_from_1m_high: float
            - target_price: int (max of highs)
            - upside_pct: float
            - convergence_strength: str ("strong"/"moderate"/"weak")
            - reject_reason: str (if rejected)

        Data Flow:
            Current price + highs → calculate distances → check thresholds → target/upside
        """
        # Calculate distances from highs
        def calc_distance(high):
            return ((high - current_instabuy) / high) * 100 if high > 0 else 0.0

        dist_1d = calc_distance(one_day_high)
        dist_1w = calc_distance(one_week_high)
        dist_1m = calc_distance(one_month_high)

        result = {
            "distance_from_1d_high": round(dist_1d, 1),
            "distance_from_1w_high": round(dist_1w, 1),
            "distance_from_1m_high": round(dist_1m, 1),
            "bsr": round(bsr, 2) if bsr != float("inf") else 99.9,
        }

        # Check BSR threshold (avoid dumps)
        if bsr < self.min_bsr:
            result["is_convergence"] = False
            result["reject_reason"] = "being_dumped"
            return result

        # Check convergence signal (oversold across timeframes)
        signals = 0
        if dist_1d >= self.min_distance_1d:
            signals += 1
        if dist_1w >= self.min_distance_1w:
            signals += 1
        if dist_1m >= self.min_distance_1m:
            signals += 1

        if signals < 3:
            result["is_convergence"] = False
            result["reject_reason"] = "not_oversold"
            return result

        # Calculate target and upside
        target_price = max(one_day_high, one_week_high, one_month_high)
        upside_pct = ((target_price - current_instabuy) / current_instabuy) * 100

        result["is_convergence"] = True
        result["target_price"] = target_price
        result["upside_pct"] = round(upside_pct, 1)

        # Convergence strength
        if signals == 3:
            result["convergence_strength"] = "strong"
        elif signals == 2:
            result["convergence_strength"] = "moderate"
        else:
            result["convergence_strength"] = "weak"

        return result
