"""History fetcher service with caching and DataFrame support."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from osrs_flipper.api import OSRSClient
from osrs_flipper.cache import HistoricalCache


class HistoryFetcher:
    """Service for fetching historical OSRS GE data with intelligent caching.

    Automatically handles incremental updates by fetching only new data points
    after the latest cached timestamp. Provides both raw data and DataFrame
    interfaces with derived fields.
    """

    def __init__(
        self, client: Optional[OSRSClient] = None, cache: Optional[HistoricalCache] = None
    ):
        """Initialize history fetcher with optional client and cache.

        Args:
            client: OSRS API client. If None, creates default OSRSClient.
            cache: Historical cache. If None, creates default HistoricalCache.
        """
        self.client = client if client is not None else OSRSClient()
        self.cache = cache if cache is not None else HistoricalCache()

    def get_history(
        self, item_id: int, timestep: str = "24h", force_refresh: bool = False
    ) -> List[Dict]:
        """Get historical data for an item with intelligent caching.

        Args:
            item_id: OSRS Wiki item ID
            timestep: Time resolution - "5m", "1h", "6h", or "24h"
            force_refresh: If True, ignore cache and fetch all data fresh

        Returns:
            List of timeseries data points with fields:
                timestamp, avgHighPrice, avgLowPrice, highPriceVolume, lowPriceVolume
        """
        # Force refresh: fetch all data and replace cache
        if force_refresh:
            data = self.client.fetch_timeseries(
                item_id=item_id, timestep=timestep, timestamp=None
            )
            self.cache.store(item_id, timestep, data)
            return data

        # Check cache
        cached_data = self.cache.get(item_id, timestep)

        # No cache: fetch all and store
        if cached_data is None:
            data = self.client.fetch_timeseries(
                item_id=item_id, timestep=timestep, timestamp=None
            )
            self.cache.store(item_id, timestep, data)
            return data

        # Cache exists: fetch incremental from latest timestamp
        latest_timestamp = self.cache.get_latest_timestamp(item_id, timestep)
        new_data = self.client.fetch_timeseries(
            item_id=item_id, timestep=timestep, timestamp=latest_timestamp
        )

        # Append new data to cache (deduplication handled by cache)
        if new_data:
            self.cache.append(item_id, timestep, new_data)

        return cached_data

    def get_dataframe(self, item_id: int, timestep: str = "24h") -> pd.DataFrame:
        """Get historical data as DataFrame with derived columns.

        Args:
            item_id: OSRS Wiki item ID
            timestep: Time resolution - "5m", "1h", "6h", or "24h"

        Returns:
            DataFrame with columns:
                - timestamp: Unix timestamp
                - high: Average high price
                - low: Average low price
                - mid_price: (high + low) // 2
                - high_volume: Volume at high price
                - low_volume: Volume at low price
                - total_volume: high_volume + low_volume
                - buyer_ratio: high_volume / total_volume (0.0 if total_volume == 0)
        """
        # Get raw data
        data = self.get_history(item_id, timestep)

        # Handle empty data
        if not data:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "high",
                    "low",
                    "mid_price",
                    "high_volume",
                    "low_volume",
                    "total_volume",
                    "buyer_ratio",
                ]
            )

        # Convert to DataFrame with renamed columns
        df = pd.DataFrame(data)
        df = df.rename(
            columns={
                "avgHighPrice": "high",
                "avgLowPrice": "low",
                "highPriceVolume": "high_volume",
                "lowPriceVolume": "low_volume",
            }
        )

        # Calculate derived fields
        df["mid_price"] = (df["high"] + df["low"]) // 2
        df["total_volume"] = df["high_volume"] + df["low_volume"]

        # Calculate buyer ratio with division by zero handling
        df["buyer_ratio"] = np.where(
            df["total_volume"] > 0,
            df["high_volume"] / df["total_volume"],
            0.0
        )

        # Select and order columns
        df = df[
            [
                "timestamp",
                "high",
                "low",
                "mid_price",
                "high_volume",
                "low_volume",
                "total_volume",
                "buyer_ratio",
            ]
        ]

        return df
