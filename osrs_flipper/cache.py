"""Historical data caching for OSRS Grand Exchange timeseries."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


class HistoricalCache:
    """Cache for historical OSRS GE timeseries data.

    Stores timeseries data as JSON files with naming format: {item_id}_{timestep}.json
    Supports incremental updates with automatic deduplication by timestamp.
    """

    def __init__(self, cache_dir: str = "./cache"):
        """Initialize cache with specified directory.

        Args:
            cache_dir: Directory path for cache storage. Created if it doesn't exist.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, item_id: int, timestep: str) -> Path:
        """Get the cache file path for an item/timestep combination.

        Args:
            item_id: OSRS item ID
            timestep: Time resolution (e.g., "5m", "1h", "6h", "24h")

        Returns:
            Path to cache file
        """
        filename = f"{item_id}_{timestep}.json"
        return self.cache_dir / filename

    def store(self, item_id: int, timestep: str, data: List[Dict]) -> None:
        """Store timeseries data to cache.

        Args:
            item_id: OSRS item ID
            timestep: Time resolution (e.g., "5m", "1h", "6h", "24h")
            data: List of timeseries data points (dicts with timestamp field)
        """
        cache_path = self._get_cache_path(item_id, timestep)
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    def get(self, item_id: int, timestep: str) -> Optional[List[Dict]]:
        """Retrieve timeseries data from cache.

        Args:
            item_id: OSRS item ID
            timestep: Time resolution (e.g., "5m", "1h", "6h", "24h")

        Returns:
            List of timeseries data points, or None if not cached
        """
        cache_path = self._get_cache_path(item_id, timestep)
        if not cache_path.exists():
            return None

        with open(cache_path, 'r') as f:
            return json.load(f)

    def get_latest_timestamp(self, item_id: int, timestep: str) -> Optional[int]:
        """Get the latest (maximum) timestamp from cached data.

        Args:
            item_id: OSRS item ID
            timestep: Time resolution (e.g., "5m", "1h", "6h", "24h")

        Returns:
            Maximum timestamp value, or None if no data cached
        """
        data = self.get(item_id, timestep)
        if data is None or len(data) == 0:
            return None

        return max(d["timestamp"] for d in data)

    def append(self, item_id: int, timestep: str, new_data: List[Dict]) -> None:
        """Append new data to cached timeseries, deduplicating by timestamp.

        Args:
            item_id: OSRS item ID
            timestep: Time resolution (e.g., "5m", "1h", "6h", "24h")
            new_data: List of new timeseries data points to append
        """
        existing_data = self.get(item_id, timestep)
        if existing_data is None:
            existing_data = []

        # Deduplicate by timestamp using a dict
        merged = {d["timestamp"]: d for d in existing_data}
        for d in new_data:
            merged[d["timestamp"]] = d

        # Sort by timestamp and store
        sorted_data = sorted(merged.values(), key=lambda x: x["timestamp"])
        self.store(item_id, timestep, sorted_data)
