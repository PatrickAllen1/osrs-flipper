"""Tests for single item analysis."""
import pytest
from unittest.mock import Mock, patch
import responses
from osrs_flipper.server.item_analyzer import analyze_single_item

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"


class TestSingleItemAnalysis:
    """Test single item deep analysis."""

    @responses.activate
    def test_analyze_item_with_all_data(self):
        """Analyze item with instant, convergence, and historical data."""
        # Mock API responses
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 123, "name": "Test Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"123": {"high": 110, "low": 100}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"123": {"highPriceVolume": 5000000, "lowPriceVolume": 3000000}}
        })

        # 1h timeseries for convergence
        timeseries_1h = []
        for i in range(168):  # 1 week
            timeseries_1h.append({"timestamp": i, "avgHighPrice": 150, "avgLowPrice": 140})
        for i in range(168, 240):  # Recent drop
            timeseries_1h.append({"timestamp": i, "avgHighPrice": 115, "avgLowPrice": 105})

        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": timeseries_1h},
            match=[responses.matchers.query_param_matcher({"id": 123, "timestep": "1h"})]
        )

        # 24h timeseries for oversold
        timeseries_24h = [
            {"timestamp": i, "avgHighPrice": 150, "avgLowPrice": 140}
            for i in range(90)
        ] + [
            {"timestamp": i, "avgHighPrice": 115, "avgLowPrice": 105}
            for i in range(90, 180)
        ]

        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": timeseries_24h},
            match=[responses.matchers.query_param_matcher({"id": 123, "timestep": "24h"})]
        )

        # Analyze
        result = analyze_single_item(item_id=123)

        assert result["item_id"] == 123
        assert result["name"] == "Test Item"
        assert result["instabuy"] == 100
        assert result["instasell"] == 110

        # Should have instant analysis
        assert "instant" in result
        assert result["instant"]["spread_pct"] == 10.0

        # Should have convergence analysis
        assert "convergence" in result
        assert result["convergence"]["is_convergence"] is True

        # Should have oversold analysis
        assert "oversold" in result
        assert "percentile" in result["oversold"]

    @responses.activate
    def test_analyze_item_without_history(self):
        """Analyze item with only instant data (no history)."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 456, "name": "New Item", "limit": 50}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"456": {"high": 200, "low": 180}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"456": {"highPriceVolume": 1000000, "lowPriceVolume": 800000}}
        })

        # Empty history
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": []})

        result = analyze_single_item(item_id=456)

        assert result["item_id"] == 456
        assert result["name"] == "New Item"

        # Should have instant analysis
        assert "instant" in result

        # Should have convergence with no highs (no data)
        assert "convergence" in result

        # Should NOT have oversold (insufficient history)
        assert "oversold" not in result
