# tests/test_timeframes.py
"""Tests for multi-timeframe price analysis."""
import pytest
import responses
import numpy as np
from osrs_flipper.timeframes import fetch_timeframe_highs
from osrs_flipper.api import OSRSClient

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"


class TestTimeframeHighs:
    """Test multi-timeframe high price extraction."""

    @responses.activate
    def test_fetch_1d_1w_1m_highs(self):
        """Fetch highs for 1 day, 1 week, 1 month windows."""
        # Mock 30 days of 1h data (720 hours)
        timeseries_data = []
        for i in range(720):
            # Simulate price declining from 200 to 100
            price = 200 - (i * 100 // 720)
            timeseries_data.append({
                "timestamp": i * 3600,  # hourly
                "avgHighPrice": price,
                "avgLowPrice": price - 10,
            })

        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": timeseries_data},
        )

        client = OSRSClient()
        highs = fetch_timeframe_highs(client, item_id=123)

        # 1d high: last 24 hours (indices -24 to -1)
        # 1w high: last 168 hours (indices -168 to -1)
        # 1m high: last 720 hours (all data)

        assert "1d_high" in highs
        assert "1w_high" in highs
        assert "1m_high" in highs

        # 1m high should be from earliest data (highest price)
        assert highs["1m_high"] == pytest.approx(200, abs=5)

        # 1d high should be from recent data (lower price)
        assert highs["1d_high"] < highs["1w_high"]
        assert highs["1w_high"] < highs["1m_high"]

    @responses.activate
    def test_fetch_handles_missing_data(self):
        """Handle missing or sparse data gracefully."""
        # Only 10 hours of data
        timeseries_data = [
            {"timestamp": i * 3600, "avgHighPrice": 150, "avgLowPrice": 140}
            for i in range(10)
        ]

        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": timeseries_data},
        )

        client = OSRSClient()
        highs = fetch_timeframe_highs(client, item_id=123)

        # With only 10 hours, all timeframes collapse to same range
        assert highs["1d_high"] == 150
        assert highs["1w_high"] == 150
        assert highs["1m_high"] == 150

    @responses.activate
    def test_fetch_uses_1h_timestep(self):
        """Fetcher uses 1h resolution for efficiency."""
        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": []},
            match=[responses.matchers.query_param_matcher({"id": "123", "timestep": "1h"})]
        )

        client = OSRSClient()
        fetch_timeframe_highs(client, item_id=123)

        # Assertion: request matcher verified timestep=1h

    @responses.activate
    def test_calculates_distance_from_highs(self):
        """Calculate percentage distance from each timeframe high."""
        # Current instabuy: 100
        # 1d high: 120, 1w high: 150, 1m high: 200
        timeseries_data = []

        # First 552 hours (720 - 168): price at 200 (oldest data to 1w ago)
        for i in range(552):
            timeseries_data.append({"timestamp": i * 3600, "avgHighPrice": 200, "avgLowPrice": 190})

        # Next 144 hours (552-696): price at 150 (1w ago to 1d ago)
        for i in range(552, 696):
            timeseries_data.append({"timestamp": i * 3600, "avgHighPrice": 150, "avgLowPrice": 140})

        # Last 24 hours (696-720): price at 120
        for i in range(696, 720):
            timeseries_data.append({"timestamp": i * 3600, "avgHighPrice": 120, "avgLowPrice": 110})

        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": timeseries_data},
        )

        client = OSRSClient()
        current_instabuy = 100

        highs = fetch_timeframe_highs(client, item_id=123, current_instabuy=current_instabuy)

        # Distance = (high - current) / high * 100
        assert highs["distance_from_1d_high"] == pytest.approx((120 - 100) / 120 * 100, abs=1)
        assert highs["distance_from_1w_high"] == pytest.approx((150 - 100) / 150 * 100, abs=1)
        assert highs["distance_from_1m_high"] == pytest.approx((200 - 100) / 200 * 100, abs=1)
