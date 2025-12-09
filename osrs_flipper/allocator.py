"""Slot allocation engine."""
from typing import List, Dict, Any

# Strategy configurations
STRATEGIES = {
    "flip": {"flip_ratio": 0.7, "hold_ratio": 0.3},
    "hold": {"flip_ratio": 0.3, "hold_ratio": 0.7},
    "balanced": {"flip_ratio": 0.4, "hold_ratio": 0.6},
}


class SlotAllocator:
    """Allocates capital across GE slots."""

    def __init__(self, strategy: str = "balanced"):
        self.strategy = strategy
        self.config = STRATEGIES.get(strategy, STRATEGIES["balanced"])

    def allocate(
        self,
        opportunities: List[Dict[str, Any]],
        cash: int,
        slots: int,
        rotations: int,
    ) -> List[Dict[str, Any]]:
        """Allocate capital to opportunities.

        Args:
            opportunities: Scored opportunity list.
            cash: Total cash to allocate.
            slots: Available GE slots.
            rotations: Buy limit rotations.

        Returns:
            List of slot allocations.
        """
        if not opportunities:
            return []

        # Sort by score descending
        sorted_opps = sorted(
            opportunities,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )

        allocations = []
        remaining_cash = cash

        for opp in sorted_opps[:slots]:
            if remaining_cash <= 0:
                break

            price = opp["current_price"]
            buy_limit = opp.get("buy_limit") or 1

            # Max quantity based on rotations
            max_qty = buy_limit * rotations

            # Max capital for this slot (fair share of remaining)
            slots_remaining = slots - len(allocations)
            fair_share = remaining_cash // max(slots_remaining, 1)

            # Capital limited by buy limit
            max_capital_by_limit = max_qty * price

            # Actual capital for this slot
            slot_capital = min(fair_share, max_capital_by_limit, remaining_cash)

            # Calculate quantity
            quantity = slot_capital // price if price > 0 else 0
            quantity = min(quantity, max_qty)

            actual_capital = quantity * price

            if quantity > 0:
                # Get exit strategy info
                exits = opp.get("exits", {})
                target_exit = exits.get("target", {})
                oversold = opp.get("oversold", {})

                allocations.append({
                    "slot": len(allocations) + 1,
                    "name": opp["name"],
                    "item_id": opp.get("item_id"),
                    "buy_price": price,
                    "quantity": quantity,
                    "capital": actual_capital,
                    "buy_limit": buy_limit,
                    "rotations": rotations,
                    # Selection reasons
                    "percentile": oversold.get("percentile", 0),
                    "upside_pct": oversold.get("upside_pct", 0),
                    "buyer_momentum": opp.get("buyer_momentum", 0),
                    # Exit strategy
                    "target_price": target_exit.get("price", 0),
                    "target_roi_pct": target_exit.get("roi_pct", 0),
                    "expected_hold_days": opp.get("expected_hold_days", 0),
                    "score": opp.get("score", 0),
                })

                remaining_cash -= actual_capital

        return allocations
