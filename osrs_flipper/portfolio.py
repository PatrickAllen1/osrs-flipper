# osrs_flipper/portfolio.py
"""Portfolio preset management."""
from typing import Dict, Any, List, Optional
import numpy as np

PRESETS = {
    "grinder": {
        "name": "Grinder",
        "description": "8 quick flips, ~2 day avg hold",
        "flip_ratio": 0.85,
        "hold_ratio": 0.15,
        "min_volume_mult": 2.0,
        "max_percentile": 25,
        "min_upside": 15,
    },
    "balanced": {
        "name": "Balanced",
        "description": "4 flip / 4 hold, ~5 day avg hold",
        "flip_ratio": 0.5,
        "hold_ratio": 0.5,
        "min_volume_mult": 1.0,
        "max_percentile": 20,
        "min_upside": 25,
    },
    "diamondhands": {
        "name": "Diamond Hands",
        "description": "8 mid-term holds, ~12 day avg hold",
        "flip_ratio": 0.15,
        "hold_ratio": 0.85,
        "min_volume_mult": 0.5,
        "max_percentile": 15,
        "min_upside": 35,
    },
}


class PortfolioManager:
    """Manages portfolio presets and recommendations."""

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a preset by name."""
        return PRESETS.get(name)

    def list_presets(self) -> Dict[str, Dict[str, Any]]:
        """List all available presets."""
        return PRESETS.copy()

    def recommend(self, opportunities: List[Dict[str, Any]]) -> str:
        """Recommend a preset based on market conditions.

        Args:
            opportunities: List of scanned opportunities.

        Returns:
            Recommended preset name.
        """
        if not opportunities:
            return "balanced"

        # Count deeply oversold items (high upside potential)
        deeply_oversold = sum(
            1 for opp in opportunities
            if opp.get("oversold", {}).get("upside_pct", 0) >= 40
        )

        # Count oscillator opportunities
        oscillators = sum(
            1 for opp in opportunities
            if opp.get("oscillator", {}).get("is_oscillator")
        )

        total = len(opportunities)

        # Recommend based on market composition
        if oscillators > total * 0.5:
            return "grinder"  # Many oscillators = quick flip opportunity
        elif deeply_oversold > total * 0.3:
            return "diamondhands"  # Many deep value plays
        else:
            return "balanced"


class ParetoFrontier:
    """Portfolio-level Pareto frontier for ROI vs hold time optimization."""

    def score_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add ROI per day scoring to items."""
        result = []
        for item in items:
            scored = item.copy()
            hold_days = item.get("expected_hold_days", 1)
            roi_pct = item.get("roi_pct", 0)
            scored["roi_per_day"] = round(roi_pct / hold_days, 2) if hold_days > 0 else 0
            result.append(scored)
        return result

    def get_efficient_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get Pareto-efficient items (non-dominated)."""
        if not items:
            return []

        n = len(items)
        roi = np.array([item["roi_pct"] for item in items])
        hold = np.array([item["expected_hold_days"] for item in items])

        # Broadcasting comparison: (n, n) matrices
        # roi_ge[i,j] = True if roi[j] >= roi[i]
        roi_ge = roi[:, np.newaxis] <= roi[np.newaxis, :]
        hold_le = hold[:, np.newaxis] >= hold[np.newaxis, :]

        # Strict inequality in at least one dimension
        roi_gt = roi[:, np.newaxis] < roi[np.newaxis, :]
        hold_lt = hold[:, np.newaxis] > hold[np.newaxis, :]

        # j dominates i if: roi[j] >= roi[i] AND hold[j] <= hold[i] AND (strict in one)
        dominates = roi_ge & hold_le & (roi_gt | hold_lt)

        # Exclude self-comparison
        np.fill_diagonal(dominates, False)

        # Item is dominated if ANY other dominates it
        is_dominated = dominates.any(axis=1)

        return [items[i] for i in range(n) if not is_dominated[i]]

    def portfolio_score(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio-level Pareto position."""
        if not portfolio:
            return {"avg_roi_pct": 0, "avg_hold_days": 0, "efficiency_score": 0}

        avg_roi = sum(i.get("roi_pct", 0) for i in portfolio) / len(portfolio)
        avg_hold = sum(i.get("expected_hold_days", 1) for i in portfolio) / len(portfolio)
        efficiency = avg_roi / avg_hold if avg_hold > 0 else 0

        return {
            "avg_roi_pct": round(avg_roi, 1),
            "avg_hold_days": round(avg_hold, 1),
            "efficiency_score": round(efficiency, 2),
        }
