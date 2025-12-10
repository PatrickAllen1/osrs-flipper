"""Scanner service with caching for API server."""
import time
import logging
from typing import List, Dict, Any, Optional
from ..api import OSRSClient
from ..scanner import ItemScanner

logger = logging.getLogger(__name__)


class ScannerService:
    """Singleton service wrapping ItemScanner with TTL-based caching.

    Maintains cached scan results to avoid redundant API calls and computations.
    Cache invalidates after cache_ttl seconds.

    Data Flow:
        scan(mode, min_roi, limit, force_scan)
        → check cache validity (if not force_scan)
        → if valid: return cached_opportunities
        → if invalid: scanner.scan() → update cache → return opportunities
    """

    def __init__(self, cache_ttl: int = 300):
        """Initialize scanner service.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 300 = 5 min)
        """
        self.client = OSRSClient()
        self.scanner = ItemScanner(self.client)

        # Cache state
        self.cached_opportunities: List[Dict[str, Any]] = []
        self.last_scan_time: float = 0.0
        self.cache_ttl: int = cache_ttl

        logger.info(f"ScannerService initialized with {cache_ttl}s cache TTL")

    def scan(
        self,
        mode: str = "both",
        min_roi: float = 20.0,
        limit: int = 20,
        force_scan: bool = False
    ) -> List[Dict[str, Any]]:
        """Get flip opportunities (cached or fresh).

        Args:
            mode: Scan mode (instant/convergence/both/oversold/oscillator/all)
            min_roi: Minimum ROI % filter
            limit: Max items to scan
            force_scan: Force fresh scan bypassing cache

        Returns:
            List of opportunity dicts

        Data Flow:
            IN: mode, min_roi, limit, force_scan
            CHECK: is_cache_valid() using (current_time - last_scan_time) < cache_ttl
            IF force_scan OR not valid: scanner.scan(mode, limit, min_roi) → cached_opportunities → return
            ELSE: return cached_opportunities
        """
        if force_scan:
            logger.info("Force scan requested")
            return self._scan_fresh(mode, min_roi, limit)

        if self._is_cache_valid():
            logger.info(f"Cache hit (age: {self.get_cache_age():.1f}s)")
            return self._filter_cached(mode, min_roi, limit)

        logger.info("Cache miss - scanning")
        return self._scan_fresh(mode, min_roi, limit)

    def get_cached_opportunities(self) -> List[Dict[str, Any]]:
        """Get cached opportunities without triggering scan.

        Returns:
            Cached opportunities list (may be empty if never scanned)
        """
        return self.cached_opportunities

    def get_cache_age(self) -> float:
        """Get cache age in seconds.

        Returns:
            Seconds since last scan (0.0 if never scanned)
        """
        if self.last_scan_time == 0.0:
            return 0.0
        return time.time() - self.last_scan_time

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid.

        Returns:
            True if cache exists and is within TTL
        """
        if not self.cached_opportunities:
            return False

        age = self.get_cache_age()
        return age < self.cache_ttl

    def _scan_fresh(
        self,
        mode: str,
        min_roi: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute fresh scan and update cache.

        Args:
            mode: Scan mode
            min_roi: Min ROI filter
            limit: Max items

        Returns:
            Fresh opportunities
        """
        opportunities = self.scanner.scan(
            mode=mode,
            limit=limit,
            min_roi=min_roi
        )

        # Update cache
        self.cached_opportunities = opportunities
        self.last_scan_time = time.time()

        logger.info(f"Scanned {limit} items, found {len(opportunities)} opportunities")

        return opportunities

    def _filter_cached(
        self,
        mode: str,
        min_roi: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Filter cached opportunities by mode and min_roi.

        Args:
            mode: Desired mode filter (instant/convergence/both/oversold/oscillator/all)
            min_roi: Min ROI threshold
            limit: Max results

        Returns:
            Filtered cached opportunities

        Data Flow:
            IN: cached_opportunities (List[Dict]), mode, min_roi, limit
            FILTER MODE:
                - instant: has "instant" key
                - convergence: has "convergence" key
                - both: has "instant" OR "convergence"
                - oversold/oscillator/all: has "oversold" OR "oscillator"
            FILTER ROI: max(instant_roi, conv_upside, legacy_upside) >= min_roi
            LIMIT: [:limit]
            OUT: filtered List[Dict]
        """
        filtered = []

        for opp in self.cached_opportunities:
            # Filter by mode
            if not self._matches_mode(opp, mode):
                continue

            # Filter by min ROI
            if not self._meets_min_roi(opp, min_roi):
                continue

            filtered.append(opp)

            # Apply limit
            if len(filtered) >= limit:
                break

        return filtered

    def _matches_mode(self, opp: Dict[str, Any], mode: str) -> bool:
        """Check if opportunity matches requested mode.

        Args:
            opp: Opportunity dict
            mode: Requested mode

        Returns:
            True if matches mode
        """
        if mode == "instant":
            return "instant" in opp
        elif mode == "convergence":
            return "convergence" in opp
        elif mode == "both":
            return "instant" in opp or "convergence" in opp
        elif mode == "oversold":
            return "oversold" in opp
        elif mode == "oscillator":
            return "oscillator" in opp
        elif mode == "all":
            return "oversold" in opp or "oscillator" in opp

        return False

    def _meets_min_roi(self, opp: Dict[str, Any], min_roi: float) -> bool:
        """Check if opportunity meets min ROI threshold.

        Args:
            opp: Opportunity dict
            min_roi: Minimum ROI %

        Returns:
            True if meets threshold
        """
        # Extract best ROI from available data
        instant_roi = opp.get("instant", {}).get("instant_roi_after_tax", 0.0)
        conv_upside = opp.get("convergence", {}).get("upside_pct", 0.0)
        legacy_upside = opp.get("tax_adjusted_upside_pct", 0.0)

        best_roi = max(instant_roi, conv_upside, legacy_upside)

        return best_roi >= min_roi
