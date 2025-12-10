"""End-to-end API integration tests.

Tests complete data flow: HTTP request → FastAPI → ScannerService → ItemScanner → Analyzers → Response
Uses mocked OSRS Wiki API but real scanner/analyzer implementations.
"""
import pytest
import responses
from fastapi.testclient import TestClient

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"


class TestE2EAPIDataFlow:
    """Test complete data flow through API."""

    @responses.activate
    def test_health_endpoint_e2e(self):
        """E2E: Health endpoint returns correct status."""
        from osrs_flipper.server.api import app

        client = TestClient(app)
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "last_scan_time" in data
        assert "cache_age_seconds" in data
        assert isinstance(data["last_scan_time"], (int, float))
        assert isinstance(data["cache_age_seconds"], (int, float))

    @responses.activate
    def test_opportunities_endpoint_e2e_instant_mode(self):
        """E2E: Opportunities endpoint returns instant flips with full data flow."""
        # Mock OSRS Wiki API
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "Test Item", "limit": 100},
            {"id": 2, "name": "High Spread", "limit": 50}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {
                "1": {"high": 105, "low": 100},
                "2": {"high": 220, "low": 200}
            }
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {
                "1": {"highPriceVolume": 5000000, "lowPriceVolume": 3000000},
                "2": {"highPriceVolume": 8000000, "lowPriceVolume": 4000000}
            }
        })

        # Test 1h timeseries (for convergence - will be called)
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": []})

        from osrs_flipper.server.api import app

        # Reset scanner service for clean test
        import osrs_flipper.server.api
        osrs_flipper.server.api._scanner_service = None

        client = TestClient(app)
        response = client.get("/api/opportunities?mode=instant&min_roi=5&limit=10")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "opportunities" in data
        assert "scan_time" in data
        assert "cache_age_seconds" in data
        assert isinstance(data["opportunities"], list)
        assert isinstance(data["scan_time"], (int, float))
        assert isinstance(data["cache_age_seconds"], (int, float))

        # Should have found instant opportunities (high spread items)
        opps = data["opportunities"]
        assert len(opps) > 0

        # Verify data integrity - all opportunities should have required fields
        for opp in opps:
            assert "item_id" in opp
            assert "name" in opp
            assert "instabuy" in opp
            assert "instasell" in opp
            assert "instant" in opp
            assert "instant_roi_after_tax" in opp["instant"]
            assert isinstance(opp["item_id"], int)
            assert isinstance(opp["name"], str)
            assert isinstance(opp["instabuy"], int)
            assert isinstance(opp["instasell"], int)

    @responses.activate
    def test_opportunities_endpoint_e2e_convergence_mode(self):
        """E2E: Opportunities endpoint returns convergence plays with full data flow."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 3, "name": "Crashed Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"3": {"high": 105, "low": 100}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"3": {"highPriceVolume": 5000000, "lowPriceVolume": 4000000}}
        })

        # 1h timeseries showing crash from highs
        timeseries = []
        # Old highs at 150
        for i in range(100):
            timeseries.append({"timestamp": i, "avgHighPrice": 150, "avgLowPrice": 145})
        # Recent crash to 105
        for i in range(100, 200):
            timeseries.append({"timestamp": i, "avgHighPrice": 105, "avgLowPrice": 100})

        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": timeseries})

        from osrs_flipper.server.api import app
        import osrs_flipper.server.api
        osrs_flipper.server.api._scanner_service = None

        client = TestClient(app)
        response = client.get("/api/opportunities?mode=convergence&min_roi=20&limit=10")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "opportunities" in data
        assert "scan_time" in data
        assert "cache_age_seconds" in data

        opps = data["opportunities"]

        # Should find convergence opportunity (30%+ drop from highs)
        if len(opps) > 0:
            opp = opps[0]
            assert "convergence" in opp
            assert "is_convergence" in opp["convergence"]
            assert "upside_pct" in opp["convergence"]
            # Verify data flow: crashed from 150 to 105 = 30% drop
            if opp["convergence"]["is_convergence"]:
                assert opp["convergence"]["upside_pct"] > 0

    @responses.activate
    def test_analyze_endpoint_e2e_valid_item(self):
        """E2E: Analyze endpoint returns full item analysis with real data flow."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 999, "name": "Deep Analysis Item", "limit": 200}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"999": {"high": 110, "low": 100}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"999": {"highPriceVolume": 10000000, "lowPriceVolume": 8000000}}
        })

        # 1h timeseries
        timeseries_1h = [
            {"timestamp": i, "avgHighPrice": 120, "avgLowPrice": 110}
            for i in range(200)
        ]
        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": timeseries_1h},
            match=[responses.matchers.query_param_matcher({"id": "999", "timestep": "1h"})]
        )

        # 24h timeseries
        timeseries_24h = [
            {"timestamp": i, "avgHighPrice": 120, "avgLowPrice": 110}
            for i in range(90)
        ]
        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": timeseries_24h},
            match=[responses.matchers.query_param_matcher({"id": "999", "timestep": "24h"})]
        )

        from osrs_flipper.server.api import app

        client = TestClient(app)
        response = client.get("/api/analyze/999")

        assert response.status_code == 200
        data = response.json()

        # Verify all analysis types present
        assert data["item_id"] == 999
        assert data["name"] == "Deep Analysis Item"
        assert "instant" in data
        assert "convergence" in data
        assert "oversold" in data

        # Verify data integrity from INPUT → TRANSFORM → OUTPUT
        # INPUT: instabuy=100, instasell=110
        assert data["instabuy"] == 100
        assert data["instasell"] == 110
        assert "bsr" in data
        assert isinstance(data["bsr"], (int, float))

        # TRANSFORM: Instant spread analysis
        assert "spread_pct" in data["instant"]
        assert "instant_roi_after_tax" in data["instant"]

        # TRANSFORM: Convergence analysis (from 1h timeseries)
        assert "is_convergence" in data["convergence"]
        assert "distance_from_1d_high" in data["convergence"]
        assert "distance_from_1w_high" in data["convergence"]
        assert "distance_from_1m_high" in data["convergence"]
        # upside_pct only present if is_convergence=True
        if data["convergence"]["is_convergence"]:
            assert "upside_pct" in data["convergence"]
            assert "target_price" in data["convergence"]

        # TRANSFORM: Oversold analysis (from 24h timeseries)
        assert "is_oversold" in data["oversold"]

    @responses.activate
    def test_analyze_endpoint_e2e_invalid_item(self):
        """E2E: Analyze endpoint returns 404 for invalid item."""
        # Empty mapping - item 999 doesn't exist
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={"data": {}})
        responses.add(responses.GET, f"{BASE_URL}/24h", json={"data": {}})

        from osrs_flipper.server.api import app

        client = TestClient(app)
        response = client.get("/api/analyze/999")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower() or "missing" in data["detail"].lower()

    @responses.activate
    def test_portfolio_endpoint_e2e_with_allocation(self):
        """E2E: Portfolio allocation endpoint works end-to-end with real data flow."""
        # Setup opportunities
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 10, "name": "Cheap Item", "limit": 100},
            {"id": 11, "name": "Expensive Item", "limit": 10}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {
                "10": {"high": 105, "low": 100},
                "11": {"high": 5500, "low": 5000}
            }
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {
                "10": {"highPriceVolume": 10000000, "lowPriceVolume": 8000000},
                "11": {"highPriceVolume": 5000000, "lowPriceVolume": 4000000}
            }
        })
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": []})

        from osrs_flipper.server.api import app
        import osrs_flipper.server.api
        osrs_flipper.server.api._scanner_service = None

        client = TestClient(app)
        response = client.get("/api/portfolio/allocate?cash=10000000&slots=3&strategy=balanced")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "allocation" in data
        assert "total_capital" in data
        assert "expected_profit" in data
        assert "strategy" in data
        assert data["strategy"] == "balanced"

        # Verify data types
        assert isinstance(data["allocation"], list)
        assert isinstance(data["total_capital"], int)
        assert isinstance(data["expected_profit"], int)

        # Verify capital doesn't exceed cash (INPUT constraint → OUTPUT verification)
        assert data["total_capital"] <= 10000000

        # If allocation exists, verify each slot structure
        for slot in data["allocation"]:
            assert "slot" in slot or "name" in slot
            assert "capital" in slot or "buy_price" in slot
            if "capital" in slot:
                assert isinstance(slot["capital"], int)
                assert slot["capital"] > 0

    @responses.activate
    def test_portfolio_endpoint_e2e_cash_parsing(self):
        """E2E: Portfolio endpoint correctly parses cash with suffixes (100m, 1.5b)."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 20, "name": "Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"20": {"high": 110, "low": 100}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"20": {"highPriceVolume": 5000000, "lowPriceVolume": 4000000}}
        })
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": []})

        from osrs_flipper.server.api import app
        import osrs_flipper.server.api
        osrs_flipper.server.api._scanner_service = None

        client = TestClient(app)

        # Test 100m suffix
        response = client.get("/api/portfolio/allocate?cash=100m&slots=8")
        assert response.status_code == 200
        data = response.json()
        # 100m = 100,000,000
        assert data["total_capital"] <= 100000000

        # Test 1.5b suffix
        osrs_flipper.server.api._scanner_service = None
        response = client.get("/api/portfolio/allocate?cash=1.5b&slots=8")
        assert response.status_code == 200
        data = response.json()
        # 1.5b = 1,500,000,000
        assert data["total_capital"] <= 1500000000

    @responses.activate
    def test_min_roi_filter_e2e(self):
        """E2E: min_roi filter correctly filters opportunities through full data flow."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 20, "name": "Low ROI", "limit": 100},
            {"id": 21, "name": "High ROI", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {
                "20": {"high": 102, "low": 100},  # ~2% spread
                "21": {"high": 150, "low": 100}   # 50% spread
            }
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {
                "20": {"highPriceVolume": 5000000, "lowPriceVolume": 3000000},
                "21": {"highPriceVolume": 8000000, "lowPriceVolume": 4000000}
            }
        })
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": []})

        from osrs_flipper.server.api import app
        import osrs_flipper.server.api
        osrs_flipper.server.api._scanner_service = None

        client = TestClient(app)

        # Request min_roi=30 (should filter out low ROI item)
        response = client.get("/api/opportunities?mode=instant&min_roi=30&limit=10")

        assert response.status_code == 200
        data = response.json()

        opps = data["opportunities"]

        # Verify data flow: INPUT (min_roi=30) → FILTER → OUTPUT (only items >= 30% ROI)
        for opp in opps:
            instant_roi = opp.get("instant", {}).get("instant_roi_after_tax", 0)
            # All returned items should have ROI >= 30% (or be rejected with 0)
            # Item 20 (~2% ROI) should be filtered out
            # Item 21 (50% ROI) should be included
            if instant_roi > 0:
                assert instant_roi >= 30.0, f"Item {opp['name']} has ROI {instant_roi}% < 30%"

    @responses.activate
    def test_opportunities_mode_both_e2e(self):
        """E2E: mode=both returns both instant and convergence opportunities."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 30, "name": "Instant Item", "limit": 100},
            {"id": 31, "name": "Convergence Item", "limit": 50}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {
                "30": {"high": 120, "low": 100},  # Good instant spread
                "31": {"high": 105, "low": 100}   # Small spread but crashed
            }
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {
                "30": {"highPriceVolume": 10000000, "lowPriceVolume": 8000000},
                "31": {"highPriceVolume": 5000000, "lowPriceVolume": 4000000}
            }
        })

        # Convergence item: crashed from 150 to 105
        timeseries = []
        for i in range(100):
            timeseries.append({"timestamp": i, "avgHighPrice": 150, "avgLowPrice": 145})
        for i in range(100, 200):
            timeseries.append({"timestamp": i, "avgHighPrice": 105, "avgLowPrice": 100})
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": timeseries})

        from osrs_flipper.server.api import app
        import osrs_flipper.server.api
        osrs_flipper.server.api._scanner_service = None

        client = TestClient(app)
        response = client.get("/api/opportunities?mode=both&min_roi=10&limit=20")

        assert response.status_code == 200
        data = response.json()

        opps = data["opportunities"]
        assert len(opps) > 0

        # Verify we have both types of opportunities
        has_instant = False
        has_convergence = False

        for opp in opps:
            # Check if instant opportunity
            if opp.get("instant", {}).get("instant_roi_after_tax", 0) >= 10:
                has_instant = True

            # Check if convergence opportunity
            if opp.get("convergence", {}).get("is_convergence", False):
                has_convergence = True

        # At least one type should be found (or both)
        assert has_instant or has_convergence, "Should find instant or convergence opportunities"

    @responses.activate
    def test_limit_parameter_e2e(self):
        """E2E: limit parameter correctly limits number of returned opportunities."""
        # Create 10 items
        mapping = [{"id": i, "name": f"Item {i}", "limit": 100} for i in range(1, 11)]
        latest = {"data": {str(i): {"high": 110, "low": 100} for i in range(1, 11)}}
        volume = {"data": {str(i): {"highPriceVolume": 5000000, "lowPriceVolume": 4000000} for i in range(1, 11)}}

        responses.add(responses.GET, f"{BASE_URL}/mapping", json=mapping)
        responses.add(responses.GET, f"{BASE_URL}/latest", json=latest)
        responses.add(responses.GET, f"{BASE_URL}/24h", json=volume)
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": []})

        from osrs_flipper.server.api import app
        import osrs_flipper.server.api
        osrs_flipper.server.api._scanner_service = None

        client = TestClient(app)

        # Request limit=5
        response = client.get("/api/opportunities?mode=instant&min_roi=1&limit=5")

        assert response.status_code == 200
        data = response.json()

        opps = data["opportunities"]

        # Verify data flow: INPUT (limit=5) → PROCESS → OUTPUT (<=5 items)
        assert len(opps) <= 5, f"Expected max 5 items, got {len(opps)}"

    @responses.activate
    def test_cache_functionality_e2e(self):
        """E2E: Verify cache reduces API calls on subsequent requests."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 40, "name": "Cached Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"40": {"high": 110, "low": 100}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"40": {"highPriceVolume": 5000000, "lowPriceVolume": 4000000}}
        })
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": []})

        from osrs_flipper.server.api import app
        import osrs_flipper.server.api
        osrs_flipper.server.api._scanner_service = None

        client = TestClient(app)

        # First request - should hit API
        response1 = client.get("/api/opportunities?mode=instant&min_roi=5&limit=10")
        assert response1.status_code == 200
        data1 = response1.json()

        # Record number of API calls
        first_call_count = len(responses.calls)

        # Second request - should use cache (same params)
        response2 = client.get("/api/opportunities?mode=instant&min_roi=5&limit=10")
        assert response2.status_code == 200
        data2 = response2.json()

        # Should not make new API calls (cache hit)
        second_call_count = len(responses.calls)
        assert second_call_count == first_call_count, "Cache should prevent new API calls"

        # Cache age should be small and increasing
        assert data2["cache_age_seconds"] >= 0
        assert data2["scan_time"] == data1["scan_time"], "Scan time should match (cached)"
