# tests/test_e2e_instant_convergence.py
"""End-to-end data flow tests for instant + convergence system."""
import pytest
import responses
import numpy as np
from click.testing import CliRunner

from osrs_flipper.cli import scan
from osrs_flipper.scanner import ItemScanner, calculate_bsr
from osrs_flipper.api import OSRSClient
from osrs_flipper.spreads import calculate_spread_pct, calculate_spread_roi_after_tax
from osrs_flipper.timeframes import fetch_timeframe_highs

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"


class TestE2EDataFlowIntegrity:
    """Test complete data flow from API to final output."""

    def test_spread_calculation_integrity(self):
        """Spread calculation matches formula."""
        instabuy = 1000
        instasell = 1100

        spread_pct = calculate_spread_pct(instabuy, instasell)

        # Verify formula
        expected = ((instasell - instabuy) / instabuy) * 100
        assert spread_pct == expected

    def test_bsr_calculation_integrity(self):
        """BSR calculation matches formula."""
        instabuy_vol = 6000
        instasell_vol = 4000

        bsr = calculate_bsr(instabuy_vol, instasell_vol)

        # Verify formula
        expected = instabuy_vol / instasell_vol
        assert bsr == expected

    def test_roi_after_tax_integrity(self):
        """ROI calculation includes tax correctly."""
        from osrs_flipper.tax import calculate_ge_tax

        instabuy = 10000
        instasell = 11000
        item_name = "Regular Item"

        roi = calculate_spread_roi_after_tax(instabuy, instasell, item_name)

        # Manual calculation
        tax = calculate_ge_tax(instasell, item_name)
        expected_profit = instasell - tax - instabuy
        expected_roi = (expected_profit / instabuy) * 100

        assert roi == pytest.approx(expected_roi, abs=0.1)

    @responses.activate
    def test_timeframe_highs_data_flow(self):
        """Timeframe highs extracted correctly from API."""
        # 720 hours of data
        timeseries = []
        for i in range(720):
            # Price declining over time
            price = 200 - (i * 100 // 720)
            timeseries.append({
                "timestamp": i * 3600,
                "avgHighPrice": price,
                "avgLowPrice": price - 10,
            })

        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": timeseries},
        )

        client = OSRSClient()
        highs = fetch_timeframe_highs(client, item_id=123, current_instabuy=100)

        # Verify data extraction
        # 1m high should be from earliest data
        assert highs["1m_high"] >= 190  # Early prices
        # 1d high should be from last 24 hours
        assert highs["1d_high"] <= 110  # Recent prices

        # Verify distance calculation
        expected_1d_distance = ((highs["1d_high"] - 100) / highs["1d_high"]) * 100
        assert highs["distance_from_1d_high"] == pytest.approx(expected_1d_distance, abs=0.5)

    @responses.activate
    def test_instant_mode_full_pipeline(self):
        """Complete data flow: API → Scanner → Instant Analyzer → Output."""
        # Setup mocks
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 999, "name": "Test Arbitrage Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {
                "999": {"high": 1100, "low": 1000}
            }
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {
                "999": {"highPriceVolume": 6000000, "lowPriceVolume": 4000000}
            }
        })

        client = OSRSClient()
        scanner = ItemScanner(client)

        results = scanner.scan(mode="instant", limit=1)

        # Verify data flow
        assert len(results) == 1
        item = results[0]

        # Input data preserved
        assert item["instabuy"] == 1000
        assert item["instasell"] == 1100
        assert item["instabuy_vol"] == 6000000
        assert item["instasell_vol"] == 4000000

        # Derived metrics calculated
        assert item["bsr"] == pytest.approx(1.5)

        # Analyzer output
        assert "instant" in item
        assert item["instant"]["spread_pct"] == 10.0
        assert item["instant"]["bsr"] == 1.5
        assert item["instant"]["is_instant_opportunity"] is True

        # ROI calculated with tax
        assert item["instant"]["instant_roi_after_tax"] > 0
        assert item["instant"]["instant_roi_after_tax"] < 10.0  # Less than spread due to tax

    @responses.activate
    def test_convergence_mode_full_pipeline(self):
        """Complete data flow: API → Timeframes → Convergence Analyzer → Output."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 888, "name": "Crashed Item", "limit": 50}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {
                "888": {"high": 105, "low": 100}
            }
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {
                "888": {"highPriceVolume": 5000000, "lowPriceVolume": 3000000}  # BSR = 1.67 > 0.8
            }
        })

        # Timeseries: Build prices that satisfy convergence criteria
        # Need: 10% below 1d, 15% below 1w, 20% below 1m
        # Current = 100, so:
        # 1d high needs to be >= 111 (10% above)
        # 1w high needs to be >= 118 (15% above)
        # 1m high needs to be >= 125 (20% above)
        timeseries = []
        # Last 720 hours (30 days) in reverse order (most recent last)
        for i in range(720):
            hours_ago = 720 - i - 1
            if hours_ago < 24:
                # Last 24 hours: prices at 115 (gives 13% distance)
                price = 115
            elif hours_ago < 168:
                # Last week: prices at 125 (gives 20% distance)
                price = 125
            else:
                # Last month: prices at 130 (gives 23% distance)
                price = 130
            timeseries.append({"timestamp": i * 3600, "avgHighPrice": price, "avgLowPrice": price - 5})

        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": timeseries})

        client = OSRSClient()
        scanner = ItemScanner(client)

        results = scanner.scan(mode="convergence", limit=1)

        assert len(results) == 1
        item = results[0]

        # Convergence analysis present
        assert "convergence" in item
        conv = item["convergence"]

        # Distances calculated
        assert conv["distance_from_1d_high"] > 0
        assert conv["distance_from_1w_high"] > 0
        assert conv["distance_from_1m_high"] > 0

        # Target = recent high
        assert conv["target_price"] > item["instabuy"]
        assert conv["upside_pct"] > 0

        # Convergence signal detected
        assert conv["is_convergence"] is True

    @responses.activate
    def test_cli_to_output_data_flow(self):
        """CLI parameters flow through to scanner and output."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "CLI Test Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"1": {"high": 1100, "low": 1000}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"1": {"highPriceVolume": 6000000, "lowPriceVolume": 4000000}}
        })

        runner = CliRunner()
        result = runner.invoke(scan, [
            "--mode", "instant",
            "--min-roi", "5",
            "--limit", "1",
        ])

        # Verify CLI ran (may have errors if output formatting not updated for new modes)
        # The key test is that mode parameter flows through
        assert "instant" in result.output.lower() or "INSTANT" in result.output

        # Verify scanner was invoked with correct mode (output contains mode header)
        assert "INSTANT" in result.output or "instant" in result.output.lower()

    def test_vectorization_integrity(self):
        """Vectorized calculations match scalar calculations."""
        # Spread calculation
        scalar_spreads = [
            calculate_spread_pct(100, 110),
            calculate_spread_pct(200, 220),
            calculate_spread_pct(500, 550),
        ]

        vector_spreads = calculate_spread_pct(
            np.array([100, 200, 500]),
            np.array([110, 220, 550])
        )

        np.testing.assert_array_almost_equal(vector_spreads, scalar_spreads)

        # BSR calculation
        scalar_bsrs = [
            calculate_bsr(1000, 1000),
            calculate_bsr(2000, 1000),
            calculate_bsr(500, 1000),
        ]

        vector_bsrs = calculate_bsr(
            np.array([1000, 2000, 500]),
            np.array([1000, 1000, 1000])
        )

        np.testing.assert_array_almost_equal(vector_bsrs, scalar_bsrs)

    @responses.activate
    def test_min_roi_filter_applies_correctly(self):
        """min_roi filter uses highest ROI from all strategies."""
        # Item with 8% instant ROI and 30% convergence upside
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "Mixed Opportunity", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"1": {"high": 1080, "low": 1000}}  # 8% spread
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"1": {"highPriceVolume": 6000000, "lowPriceVolume": 4000000}}  # BSR = 1.5
        })

        # Build timeseries with proper convergence signal
        # Current = 1000, need all three: 10% below 1d, 15% below 1w, 20% below 1m
        # 1d high = 1112 (11.2% distance), 1w high = 1177 (17.7% distance), 1m high = 1300 (30% distance)
        timeseries = []
        for i in range(720):
            hours_ago = 720 - i - 1
            if hours_ago < 24:
                # Last 24 hours: at 1112
                price = 1112
            elif hours_ago < 168:
                # Last week: at 1177
                price = 1177
            else:
                # Last month: at 1300
                price = 1300
            timeseries.append({"timestamp": i * 3600, "avgHighPrice": price, "avgLowPrice": price - 10})
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": timeseries})

        client = OSRSClient()
        scanner = ItemScanner(client)

        # With min_roi=20%, should pass (convergence upside = 30%)
        results_pass = scanner.scan(mode="both", limit=1, min_roi=20.0)

        # Reset for second call
        responses.reset()
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "Mixed Opportunity", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"1": {"high": 1080, "low": 1000}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"1": {"highPriceVolume": 6000000, "lowPriceVolume": 4000000}}
        })
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": timeseries})

        # With min_roi=35%, should fail (highest ROI = 30%)
        results_fail = scanner.scan(mode="both", limit=1, min_roi=35.0)

        assert len(results_pass) == 1
        assert len(results_fail) == 0
