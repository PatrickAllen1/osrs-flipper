# osrs_flipper/instant_analyzer.py
"""Instant spread arbitrage opportunity analyzer."""
from typing import Dict, Any
from .spreads import calculate_spread_pct, calculate_spread_roi_after_tax
from .bsr import calculate_bsr


class InstantSpreadAnalyzer:
    """Detect instant arbitrage opportunities.

    Strategy: Same-day flip on high spread with strong buyer demand.

    Criteria:
    - Instant spread >= min_spread_pct (default 5%)
    - BSR >= min_bsr (default 1.2, buyers dominate)
    - Spread <= max_spread_pct (default 25%, avoid suspiciously wide)
    """

    def __init__(
        self,
        min_spread_pct: float = 5.0,
        min_bsr: float = 1.2,
        max_spread_pct: float = 25.0,
    ):
        """Initialize analyzer.

        Args:
            min_spread_pct: Minimum instant spread to consider (default 5%)
            min_bsr: Minimum buyer/seller ratio (default 1.2)
            max_spread_pct: Maximum spread to avoid suspicious outliers (default 25%)
        """
        self.min_spread_pct = min_spread_pct
        self.min_bsr = min_bsr
        self.max_spread_pct = max_spread_pct

    def analyze(
        self,
        instabuy: int,
        instasell: int,
        instabuy_vol: int,
        instasell_vol: int,
        item_name: str,
    ) -> Dict[str, Any]:
        """Analyze item for instant arbitrage opportunity.

        Args:
            instabuy: Instant buy price
            instasell: Instant sell price
            instabuy_vol: Buyer volume
            instasell_vol: Seller volume
            item_name: Item name (for tax calculation)

        Returns:
            Analysis result with:
            - is_instant_opportunity: bool
            - spread_pct: float
            - bsr: float
            - instant_roi_after_tax: float
            - reject_reason: str (if rejected)

        Data Flow:
            Prices + volumes → spread_pct, BSR → threshold checks → tax-adjusted ROI
        """
        spread_pct = calculate_spread_pct(instabuy, instasell)
        bsr = calculate_bsr(instabuy_vol, instasell_vol)

        result = {
            "spread_pct": round(spread_pct, 2),
            "bsr": round(bsr, 2) if bsr != float("inf") else 99.9,
        }

        # Check thresholds
        if spread_pct < self.min_spread_pct:
            result["is_instant_opportunity"] = False
            result["reject_reason"] = "spread_too_low"
            return result

        if spread_pct > self.max_spread_pct:
            result["is_instant_opportunity"] = False
            result["reject_reason"] = "spread_too_wide"
            return result

        if bsr < self.min_bsr:
            result["is_instant_opportunity"] = False
            result["reject_reason"] = "weak_bsr"
            return result

        # Calculate tax-adjusted ROI
        roi = calculate_spread_roi_after_tax(instabuy, instasell, item_name)
        result["instant_roi_after_tax"] = roi
        result["is_instant_opportunity"] = True

        return result
