"""OSRS Wiki API client for price and volume data."""
import time
import requests
from typing import Dict, Any

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"
HEADERS = {"User-Agent": "osrs-flipper - github.com/user/osrs-flipper"}


class OSRSClient:
    """Client for OSRS Wiki Real-Time Prices API."""

    def __init__(self, base_url: str = BASE_URL, rate_limit: float = 0.1):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.rate_limit = rate_limit
        self._last_request = 0.0

    def _rate_limited_get(self, url: str, **kwargs):
        """Make rate-limited GET request."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        self._last_request = time.time()
        return self.session.get(url, **kwargs)

    def fetch_mapping(self) -> Dict[int, Dict[str, Any]]:
        """Fetch item ID to name/limit mapping."""
        resp = self._rate_limited_get(f"{self.base_url}/mapping")
        resp.raise_for_status()
        return {item["id"]: item for item in resp.json()}

    def fetch_latest(self) -> Dict[str, Dict[str, Any]]:
        """Fetch latest prices for all items."""
        resp = self._rate_limited_get(f"{self.base_url}/latest")
        resp.raise_for_status()
        return resp.json().get("data", {})

    def fetch_volumes(self) -> Dict[str, Dict[str, Any]]:
        """Fetch 24h volume data."""
        resp = self._rate_limited_get(f"{self.base_url}/24h")
        resp.raise_for_status()
        return resp.json().get("data", {})

    def fetch_timeseries(
        self,
        item_id: int,
        timestep: str = "24h",
        timestamp: int = None,
    ) -> list:
        """Fetch historical price data for an item.

        Args:
            item_id: OSRS Wiki item ID
            timestep: Time resolution - "5m", "1h", "6h", or "24h"
            timestamp: Optional Unix timestamp - only return data after this time

        Returns:
            List of price data points
        """
        params = {"id": item_id, "timestep": timestep}
        if timestamp is not None:
            params["timestamp"] = timestamp

        resp = self._rate_limited_get(
            f"{self.base_url}/timeseries",
            params=params,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
