"""Item scanning service."""
from typing import List, Dict, Any, Optional, Callable
from .api import OSRSClient
from .analyzers import OversoldAnalyzer, OscillatorAnalyzer
from .exits import calculate_exit_strategies
from .filters import passes_volume_filter
from .tax import is_tax_exempt, calculate_ge_tax


class ItemScanner:
    """Scans items for flip opportunities."""

    def __init__(self, client: OSRSClient):
        self.client = client
        self.oversold_analyzer = OversoldAnalyzer()
        self.oscillator_analyzer = OscillatorAnalyzer()

    def scan(
        self,
        mode: str = "all",
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Scan for flip opportunities.

        Args:
            mode: "oversold", "oscillator", or "all"
            limit: Max items to scan (None for all)
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of opportunity dicts.
        """
        mapping = self.client.fetch_mapping()
        latest = self.client.fetch_latest()
        volumes = self.client.fetch_volumes()

        opportunities = []
        item_ids = list(mapping.keys())[:limit] if limit else list(mapping.keys())
        total = len(item_ids)

        for i, item_id in enumerate(item_ids):
            if progress_callback:
                progress_callback(i + 1, total)

            result = self._analyze_item(item_id, mapping, latest, volumes, mode)
            if result:
                opportunities.append(result)

        return opportunities

    def _analyze_item(self, item_id: int, mapping: Dict, latest: Dict, volumes: Dict, mode: str) -> Optional[Dict[str, Any]]:
        if item_id not in mapping:
            return None

        item = mapping[item_id]
        name = item.get("name", "Unknown")
        if "(noted)" in name.lower():
            return None

        price_data = latest.get(str(item_id), {})
        if not price_data:
            return None

        high = price_data.get("high")
        low = price_data.get("low")
        if not high or not low:
            return None

        current_price = (high + low) // 2
        if current_price <= 0:
            return None

        vol_data = volumes.get(str(item_id), {})
        instabuy_vol = vol_data.get("highPriceVolume", 0) or 0  # buyers paying ask
        instasell_vol = vol_data.get("lowPriceVolume", 0) or 0  # sellers hitting bid
        daily_volume = instabuy_vol + instasell_vol

        # Buyer momentum: >1.0 means buyers outnumber sellers
        if instasell_vol > 0:
            buyer_momentum = instabuy_vol / instasell_vol
        else:
            buyer_momentum = float("inf") if instabuy_vol > 0 else 0.0

        if not passes_volume_filter(current_price, daily_volume):
            return None

        try:
            history = self.client.fetch_timeseries(item_id)
        except Exception:
            return None

        if len(history) < 30:
            return None

        prices = []
        for point in history:
            h = point.get("avgHighPrice")
            l = point.get("avgLowPrice")
            if h and l:
                prices.append((h + l) // 2)

        if len(prices) < 30:
            return None

        six_month_low = min(prices)
        six_month_high = max(prices)

        result = {
            "item_id": item_id,
            "name": name,
            "current_price": current_price,
            "daily_volume": daily_volume,
            "instabuy_vol": instabuy_vol,
            "instasell_vol": instasell_vol,
            "buyer_momentum": round(buyer_momentum, 2) if buyer_momentum != float("inf") else 99.9,
            "buy_limit": item.get("limit"),
            "is_tax_exempt": is_tax_exempt(name),
        }

        if mode in ("oversold", "all"):
            oversold = self.oversold_analyzer.analyze(current_price, six_month_low, six_month_high, prices)
            result["oversold"] = oversold

        if mode in ("oscillator", "all"):
            oscillator = self.oscillator_analyzer.analyze(prices, current_price)
            result["oscillator"] = oscillator

        is_opportunity = False
        if mode == "oversold" and result.get("oversold", {}).get("is_oversold"):
            is_opportunity = True
        elif mode == "oscillator" and result.get("oscillator", {}).get("is_oscillator"):
            is_opportunity = True
        elif mode == "all":
            is_opportunity = (result.get("oversold", {}).get("is_oversold") or result.get("oscillator", {}).get("is_oscillator"))

        if is_opportunity:
            # Add exit strategies
            result["exits"] = calculate_exit_strategies(current_price, prices)

            # Calculate tax-adjusted upside if exits are available
            if result.get("exits"):
                target_price = result["exits"].get("target", {}).get("price", 0)
                if target_price > 0:
                    # Calculate tax on the target sell price
                    tax = calculate_ge_tax(target_price, name)
                    # Tax-adjusted profit = target_price - tax - current_price
                    tax_adjusted_profit = target_price - tax - current_price
                    # Tax-adjusted upside percentage
                    tax_adjusted_upside = (tax_adjusted_profit / current_price) * 100 if current_price > 0 else 0
                    result["tax_adjusted_upside_pct"] = round(tax_adjusted_upside, 2)

            # Estimate hold time based on volume turnover
            # Higher volume = faster fills = shorter hold
            buy_limit = item.get("limit") or 1
            if daily_volume > 0:
                # Days to fill buy limit based on daily volume
                days_to_fill = max(1, buy_limit / (daily_volume / 2))  # Assume we get half the volume
                # Add time for price to recover (based on how oversold)
                percentile = result.get("oversold", {}).get("percentile", 50)
                recovery_factor = max(1, (50 - percentile) / 10)  # More oversold = longer recovery
                result["expected_hold_days"] = round(days_to_fill + recovery_factor, 1)
            else:
                result["expected_hold_days"] = 14  # Default for low volume

        return result if is_opportunity else None
