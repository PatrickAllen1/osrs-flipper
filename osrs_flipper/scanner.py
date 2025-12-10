"""Item scanning service."""
from typing import List, Dict, Any, Optional, Callable
from .api import OSRSClient
from .analyzers import OversoldAnalyzer, OscillatorAnalyzer
from .instant_analyzer import InstantSpreadAnalyzer
from .convergence_analyzer import ConvergenceAnalyzer
from .timeframes import fetch_timeframe_highs
from .bsr import calculate_bsr
from .exits import calculate_exit_strategies
from .filters import passes_volume_filter
from .tax import is_tax_exempt, calculate_ge_tax


class ItemScanner:
    """Scans items for flip opportunities."""

    def __init__(self, client: OSRSClient):
        self.client = client
        # Legacy analyzers
        self.oversold_analyzer = OversoldAnalyzer()
        self.oscillator_analyzer = OscillatorAnalyzer()
        # New analyzers
        self.instant_analyzer = InstantSpreadAnalyzer()
        self.convergence_analyzer = ConvergenceAnalyzer()

    def scan(
        self,
        mode: str = "both",  # Changed default from "all"
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        lookback_days: int = 180,
        min_roi: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Scan for flip opportunities.

        Args:
            mode: "instant", "convergence", "both", "oversold", "oscillator", or "all"
            limit: Max items to scan (None for all)
            progress_callback: Optional callback(current, total) for progress updates
            lookback_days: Days to look back (for legacy modes)
            min_roi: Minimum tax-adjusted ROI % to include (None for no filter)

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

            result = self._analyze_item(item_id, mapping, latest, volumes, mode, lookback_days)
            if result:
                # Apply min_roi filter
                if min_roi is not None:
                    # Check instant ROI or convergence upside or legacy tax_adjusted_upside
                    instant_roi = result.get("instant", {}).get("instant_roi_after_tax", 0)
                    convergence_upside = result.get("convergence", {}).get("upside_pct", 0)
                    legacy_roi = result.get("tax_adjusted_upside_pct", 0)

                    max_roi = max(instant_roi, convergence_upside, legacy_roi)
                    if max_roi < min_roi:
                        continue

                opportunities.append(result)

        return opportunities

    def _analyze_item(
        self,
        item_id: int,
        mapping: Dict,
        latest: Dict,
        volumes: Dict,
        mode: str,
        lookback_days: int = 180,
    ) -> Optional[Dict[str, Any]]:
        if item_id not in mapping:
            return None

        item = mapping[item_id]
        name = item.get("name", "Unknown")
        if "(noted)" in name.lower():
            return None

        price_data = latest.get(str(item_id), {})
        if not price_data:
            return None

        # Use actual instant buy/sell prices (not midpoint)
        instasell = price_data.get("high")
        instabuy = price_data.get("low")
        if not instasell or not instabuy:
            return None

        if instabuy <= 0 or instasell <= 0:
            return None

        vol_data = volumes.get(str(item_id), {})
        instabuy_vol = vol_data.get("highPriceVolume", 0) or 0
        instasell_vol = vol_data.get("lowPriceVolume", 0) or 0
        daily_volume = instabuy_vol + instasell_vol

        bsr = calculate_bsr(instabuy_vol, instasell_vol)

        if not passes_volume_filter(instabuy, daily_volume):
            return None

        result = {
            "item_id": item_id,
            "name": name,
            "instabuy": instabuy,
            "instasell": instasell,
            "daily_volume": daily_volume,
            "instabuy_vol": instabuy_vol,
            "instasell_vol": instasell_vol,
            "bsr": round(bsr, 2) if bsr != float("inf") else 99.9,
            "buy_limit": item.get("limit"),
            "is_tax_exempt": is_tax_exempt(name),
        }

        is_opportunity = False

        # New modes: instant, convergence, both
        if mode in ("instant", "both"):
            instant_result = self.instant_analyzer.analyze(
                instabuy=instabuy,
                instasell=instasell,
                instabuy_vol=instabuy_vol,
                instasell_vol=instasell_vol,
                item_name=name,
            )
            result["instant"] = instant_result
            if instant_result["is_instant_opportunity"]:
                is_opportunity = True

        if mode in ("convergence", "both"):
            # Fetch multi-timeframe highs
            timeframe_highs = fetch_timeframe_highs(self.client, item_id, instabuy)

            convergence_result = self.convergence_analyzer.analyze(
                current_instabuy=instabuy,
                one_day_high=timeframe_highs["1d_high"],
                one_week_high=timeframe_highs["1w_high"],
                one_month_high=timeframe_highs["1m_high"],
                bsr=bsr,
            )
            result["convergence"] = convergence_result
            if convergence_result["is_convergence"]:
                is_opportunity = True

        # Legacy modes (oversold, oscillator, all)
        if mode in ("oversold", "oscillator", "all"):
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

            current_price = (instasell + instabuy) // 2

            if mode in ("oversold", "all"):
                oversold = self.oversold_analyzer.analyze(
                    current_price=current_price,
                    prices=prices,
                    lookback_days=lookback_days,
                )
                result["oversold"] = oversold

            if mode in ("oscillator", "all"):
                oscillator = self.oscillator_analyzer.analyze(prices, current_price)
                result["oscillator"] = oscillator

            # Check legacy opportunity criteria
            if mode == "oversold" and result.get("oversold", {}).get("is_oversold"):
                is_opportunity = True
            elif mode == "oscillator" and result.get("oscillator", {}).get("is_oscillator"):
                is_opportunity = True
            elif mode == "all":
                is_opportunity = (result.get("oversold", {}).get("is_oversold") or result.get("oscillator", {}).get("is_oscillator"))

            # Add legacy exit strategies if opportunity
            if is_opportunity and mode in ("oversold", "oscillator", "all"):
                result["exits"] = calculate_exit_strategies(current_price, prices)
                if result.get("exits"):
                    target_price = result["exits"].get("target", {}).get("price", 0)
                    if target_price > 0:
                        tax = calculate_ge_tax(target_price, name)
                        tax_adjusted_profit = target_price - tax - current_price
                        tax_adjusted_upside = (tax_adjusted_profit / current_price) * 100 if current_price > 0 else 0
                        result["tax_adjusted_upside_pct"] = round(tax_adjusted_upside, 2)

        return result if is_opportunity else None
