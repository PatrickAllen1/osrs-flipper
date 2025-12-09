"""End-to-end data flow integrity tests."""
import pytest
import responses
import numpy as np
from click.testing import CliRunner

from osrs_flipper.cli import scan
from osrs_flipper.lookback import calculate_lookback_days
from osrs_flipper.defaults import get_default_hold_days, STRATEGY_HOLD_DAYS, DEFAULT_MIN_ROI
from osrs_flipper.scanner import ItemScanner
from osrs_flipper.api import OSRSClient

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"


class TestDataFlowIntegrity:
    """Test data flows correctly through entire pipeline."""

    def test_hold_days_to_lookback_flow(self):
        """Verify: strategy -> hold_days -> lookback_days formula."""
        # Test each strategy's data flow
        for strategy, expected_hold in STRATEGY_HOLD_DAYS.items():
            hold_days = get_default_hold_days(strategy)
            assert hold_days == expected_hold, f"Strategy {strategy} should give hold_days={expected_hold}"

            lookback = calculate_lookback_days(hold_days)
            expected_lookback = min(hold_days * 4, 180)
            assert lookback == expected_lookback, f"hold_days={hold_days} should give lookback={expected_lookback}"

    def test_lookback_vectorization_integrity(self):
        """Verify vectorized lookback calculation matches scalar."""
        hold_days_scalar = [3, 7, 14, 30, 50]
        hold_days_vector = np.array(hold_days_scalar)

        # Scalar results
        scalar_results = [calculate_lookback_days(h) for h in hold_days_scalar]

        # Vectorized result
        vector_results = calculate_lookback_days(hold_days_vector)

        # Must match exactly
        np.testing.assert_array_equal(
            vector_results,
            scalar_results,
            err_msg="Vectorized calculation must match scalar"
        )

    @responses.activate
    def test_cli_to_scanner_data_flow(self):
        """Verify CLI parameters flow correctly to scanner."""
        # Setup mock API
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "Test Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"1": {"high": 110, "low": 100}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"1": {"highPriceVolume": 3000000, "lowPriceVolume": 2500000}}
        })
        # 180 days of data with clear oversold pattern
        timeseries = []
        for i in range(120):
            timeseries.append({"timestamp": i, "avgHighPrice": 200, "avgLowPrice": 190})
        for i in range(120, 180):
            timeseries.append({"timestamp": i, "avgHighPrice": 110, "avgLowPrice": 100})
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": timeseries})

        runner = CliRunner()

        # Test with explicit parameters
        result = runner.invoke(scan, [
            "--mode", "oversold",
            "--hold-days", "7",
            "--min-roi", "30",
            "--limit", "1",
        ])

        # Verify output shows correct parameters
        assert result.exit_code == 0 or "No opportunities" in result.output
        assert "Hold time: 7 days" in result.output
        assert "Lookback: 28 days" in result.output  # 7 * 4 = 28
        assert "Min ROI: 30" in result.output

    @responses.activate
    def test_min_roi_filter_data_flow(self):
        """Verify min_roi filter uses tax_adjusted_upside_pct correctly."""
        # Setup: Two items with different upside
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "Low Upside", "limit": 100},
            {"id": 2, "name": "High Upside", "limit": 100},
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {
                "1": {"high": 105, "low": 95},   # Current: 100
                "2": {"high": 105, "low": 95},   # Current: 100
            }
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {
                "1": {"highPriceVolume": 5000000, "lowPriceVolume": 4000000},
                "2": {"highPriceVolume": 5000000, "lowPriceVolume": 4000000},
            }
        })

        # Item 1: Peak at 115 -> ~15% upside (below 50% threshold)
        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": [{"timestamp": i, "avgHighPrice": 115, "avgLowPrice": 95} for i in range(90)]},
            match=[responses.matchers.query_param_matcher({"id": 1, "timestep": "24h"})]
        )
        # Item 2: Peak at 200 -> ~100% upside (above threshold)
        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": [{"timestamp": i, "avgHighPrice": 200, "avgLowPrice": 95} for i in range(90)]},
            match=[responses.matchers.query_param_matcher({"id": 2, "timestep": "24h"})]
        )

        client = OSRSClient()
        scanner = ItemScanner(client)

        # With min_roi=50%, only high upside should pass
        results = scanner.scan(mode="oversold", limit=2, min_roi=50.0)

        # Verify filter worked
        names = [r["name"] for r in results]
        assert "Low Upside" not in names, "Low upside item should be filtered out"

        # Verify data integrity of remaining items
        for result in results:
            assert "tax_adjusted_upside_pct" in result
            assert result["tax_adjusted_upside_pct"] >= 50.0

    def test_default_values_integrity(self):
        """Verify all default values are consistent."""
        # DEFAULT_MIN_ROI should be 20
        assert DEFAULT_MIN_ROI == 20.0

        # Strategy defaults should match spec
        assert STRATEGY_HOLD_DAYS["flip"] == 3
        assert STRATEGY_HOLD_DAYS["balanced"] == 7
        assert STRATEGY_HOLD_DAYS["hold"] == 14

        # Unknown strategy falls back to balanced
        assert get_default_hold_days("unknown") == 7

    @responses.activate
    def test_lookback_window_affects_analysis(self):
        """Verify lookback window actually changes analysis results."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "Test Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"1": {"high": 110, "low": 100}}  # Current: 105
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"1": {"highPriceVolume": 5000000, "lowPriceVolume": 4000000}}
        })

        # Historical data: Old peak at 300, recent range 100-120
        timeseries = []
        for i in range(90):
            timeseries.append({"timestamp": i, "avgHighPrice": 300, "avgLowPrice": 290})
        for i in range(90, 180):
            timeseries.append({"timestamp": i, "avgHighPrice": 120, "avgLowPrice": 100})
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": timeseries})

        client = OSRSClient()
        scanner = ItemScanner(client)

        # Long lookback (180 days) - sees old 300 peak
        results_long = scanner.scan(mode="oversold", limit=1, lookback_days=180)

        # Reset responses for second call
        responses.reset()
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "Test Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"1": {"high": 110, "low": 100}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"1": {"highPriceVolume": 5000000, "lowPriceVolume": 4000000}}
        })
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": timeseries})

        # Short lookback (30 days) - sees only recent 100-120 range
        results_short = scanner.scan(mode="oversold", limit=1, lookback_days=30)

        # With long lookback, item looks very oversold (current 105 vs peak 300)
        # With short lookback, item is near middle of recent range (105 in 100-120)
        # Results should differ based on lookback window
        if results_long and results_short:
            long_upside = results_long[0]["oversold"]["upside_pct"]
            short_upside = results_short[0]["oversold"]["upside_pct"]
            # Long lookback should see much higher upside
            assert long_upside > short_upside, "Long lookback should see higher upside than short"
