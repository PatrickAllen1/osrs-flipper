import pytest
import responses
from osrs_flipper.scanner import ItemScanner
from osrs_flipper.api import OSRSClient

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"


@responses.activate
def test_scanner_finds_oversold_items():
    """Scanner identifies oversold opportunities."""
    responses.add(responses.GET, f"{BASE_URL}/mapping", json=[{"id": 1, "name": "Test Item", "limit": 100}])
    responses.add(responses.GET, f"{BASE_URL}/latest", json={"data": {"1": {"high": 110, "low": 100}}})
    responses.add(responses.GET, f"{BASE_URL}/24h", json={"data": {"1": {"highPriceVolume": 3000000, "lowPriceVolume": 2500000}}})
    responses.add(responses.GET, f"{BASE_URL}/timeseries", json={
        "data": [{"timestamp": i, "avgHighPrice": 100 + (i % 100), "avgLowPrice": 95 + (i % 100)} for i in range(90)]
    })

    client = OSRSClient()
    scanner = ItemScanner(client)
    results = scanner.scan(mode="oversold", limit=10)

    assert isinstance(results, list)


@responses.activate
def test_scanner_respects_volume_filter():
    """Scanner filters out low volume items."""
    responses.add(responses.GET, f"{BASE_URL}/mapping", json=[{"id": 1, "name": "Low Volume Item", "limit": 100}])
    responses.add(responses.GET, f"{BASE_URL}/latest", json={"data": {"1": {"high": 110, "low": 100}}})
    responses.add(responses.GET, f"{BASE_URL}/24h", json={"data": {"1": {"highPriceVolume": 1000, "lowPriceVolume": 1000}}})

    client = OSRSClient()
    scanner = ItemScanner(client)
    results = scanner.scan(mode="oversold", limit=10)

    assert len(results) == 0


@responses.activate
def test_scanner_progress_callback():
    """Scanner calls progress callback during scan."""
    responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
        {"id": 1, "name": "Item 1", "limit": 100},
        {"id": 2, "name": "Item 2", "limit": 100},
        {"id": 3, "name": "Item 3", "limit": 100},
    ])
    responses.add(responses.GET, f"{BASE_URL}/latest", json={"data": {
        "1": {"high": 110, "low": 100},
        "2": {"high": 220, "low": 200},
        "3": {"high": 330, "low": 300},
    }})
    responses.add(responses.GET, f"{BASE_URL}/24h", json={"data": {
        "1": {"highPriceVolume": 1000, "lowPriceVolume": 1000},
        "2": {"highPriceVolume": 1000, "lowPriceVolume": 1000},
        "3": {"highPriceVolume": 1000, "lowPriceVolume": 1000},
    }})

    client = OSRSClient()
    scanner = ItemScanner(client)

    # Track progress callback calls
    progress_calls = []

    def track_progress(current, total):
        progress_calls.append((current, total))

    scanner.scan(mode="oversold", limit=3, progress_callback=track_progress)

    # Verify callback was called for each item
    assert len(progress_calls) == 3
    assert progress_calls[0] == (1, 3)
    assert progress_calls[1] == (2, 3)
    assert progress_calls[2] == (3, 3)


@responses.activate
def test_scanner_includes_tax_fields():
    """All opportunities should have is_tax_exempt and tax_adjusted_upside_pct."""
    # Setup realistic mock data that will trigger oversold detection
    responses.add(
        responses.GET,
        f"{BASE_URL}/mapping",
        json=[
            {"id": 4151, "name": "Abyssal whip", "limit": 70},  # Not tax-exempt
            {"id": 1, "name": "Bronze arrow", "limit": 9000},  # Tax-exempt
        ]
    )

    responses.add(
        responses.GET,
        f"{BASE_URL}/latest",
        json={"data": {
            "4151": {"high": 1100000, "low": 1000000},  # Current: 1050000
            "1": {"high": 7, "low": 5},  # Current: 6
        }}
    )

    responses.add(
        responses.GET,
        f"{BASE_URL}/24h",
        json={"data": {
            "4151": {"highPriceVolume": 300000, "lowPriceVolume": 200000},  # 500k volume (enough for 1M item)
            "1": {"highPriceVolume": 3000000, "lowPriceVolume": 2000000},  # 5M volume (enough for low price item)
        }}
    )

    # Timeseries: Item was at 2M, crashed to 1M (oversold)
    # Low=1M, High=2M, Current=1.05M -> percentile=5%, upside=90%
    whip_timeseries = []
    for i in range(60):
        # Historical high at 2M
        whip_timeseries.append({"timestamp": i, "avgHighPrice": 2000000, "avgLowPrice": 1900000})
    for i in range(60, 90):
        # Recent crash to 1M
        whip_timeseries.append({"timestamp": i, "avgHighPrice": 1100000, "avgLowPrice": 1000000})

    # Arrow: was at 12, crashed to 6
    arrow_timeseries = []
    for i in range(60):
        arrow_timeseries.append({"timestamp": i, "avgHighPrice": 12, "avgLowPrice": 11})
    for i in range(60, 90):
        arrow_timeseries.append({"timestamp": i, "avgHighPrice": 7, "avgLowPrice": 5})

    # Add timeseries responses with matchers
    responses.add(
        responses.GET,
        f"{BASE_URL}/timeseries",
        json={"data": whip_timeseries},
        match=[responses.matchers.query_param_matcher({"id": 4151, "timestep": "24h"})]
    )

    responses.add(
        responses.GET,
        f"{BASE_URL}/timeseries",
        json={"data": arrow_timeseries},
        match=[responses.matchers.query_param_matcher({"id": 1, "timestep": "24h"})]
    )

    client = OSRSClient()
    scanner = ItemScanner(client)
    opportunities = scanner.scan(mode="oversold", limit=2)

    # Should find opportunities
    assert len(opportunities) > 0, "Scanner should find oversold opportunities"

    for opp in opportunities:
        # All opportunities must have tax exemption field
        assert "is_tax_exempt" in opp, f"Opportunity for {opp['name']} missing is_tax_exempt field"
        assert isinstance(opp["is_tax_exempt"], bool)

        # If opportunity has exits with a target price, should have tax-adjusted upside
        if "exits" in opp and opp.get("exits") and opp["exits"].get("target"):
            assert "tax_adjusted_upside_pct" in opp, f"Opportunity for {opp['name']} missing tax_adjusted_upside_pct"

            raw_upside = opp.get("oversold", {}).get("upside_pct", 0)
            adjusted_upside = opp["tax_adjusted_upside_pct"]

            # Tax should reduce upside for non-exempt items
            if not opp["is_tax_exempt"]:
                assert adjusted_upside <= raw_upside, (
                    f"Tax should reduce upside for {opp['name']}: "
                    f"raw={raw_upside}, adjusted={adjusted_upside}"
                )
            else:
                # Tax-exempt items should have equal raw and adjusted upside
                assert abs(adjusted_upside - raw_upside) < 0.1, (
                    f"Tax-exempt {opp['name']} should have equal upside: "
                    f"raw={raw_upside}, adjusted={adjusted_upside}"
                )


@responses.activate
def test_scanner_instant_mode():
    """Scanner finds instant arbitrage opportunities."""
    # Mock API responses
    responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
        {"id": 1, "name": "High Spread Item", "limit": 100}
    ])
    responses.add(responses.GET, f"{BASE_URL}/latest", json={
        "data": {
            "1": {"high": 1100, "low": 1000}  # 10% spread
        }
    })
    responses.add(responses.GET, f"{BASE_URL}/24h", json={
        "data": {
            "1": {"highPriceVolume": 6000000, "lowPriceVolume": 4000000}  # BSR = 1.5
        }
    })

    client = OSRSClient()
    scanner = ItemScanner(client)

    results = scanner.scan(mode="instant", limit=1)

    assert len(results) >= 1
    assert results[0]["instant"]["is_instant_opportunity"] is True
    assert results[0]["instant"]["spread_pct"] == 10.0


@responses.activate
def test_scanner_convergence_mode():
    """Scanner finds convergence opportunities."""
    responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
        {"id": 1, "name": "Crashed Item", "limit": 100}
    ])
    responses.add(responses.GET, f"{BASE_URL}/latest", json={
        "data": {
            "1": {"high": 105, "low": 100}  # Current ~100
        }
    })
    responses.add(responses.GET, f"{BASE_URL}/24h", json={
        "data": {
            "1": {"highPriceVolume": 5000000, "lowPriceVolume": 5000000}  # BSR = 1.0
        }
    })
    # Timeseries: was at 200, now at 100
    timeseries = []
    for i in range(720):
        # First 600 hours: price at 200
        # Last 120 hours: price declining to 100
        if i < 600:
            price = 200
        else:
            price = 200 - ((i - 600) * 100 // 120)
        timeseries.append({"timestamp": i * 3600, "avgHighPrice": price, "avgLowPrice": price - 10})

    responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": timeseries})

    client = OSRSClient()
    scanner = ItemScanner(client)

    results = scanner.scan(mode="convergence", limit=1)

    assert len(results) >= 1
    assert results[0]["convergence"]["is_convergence"] is True


@responses.activate
def test_scanner_both_mode():
    """Scanner finds items matching both strategies."""
    # Item with high spread AND crashed from recent highs
    responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
        {"id": 1, "name": "Perfect Item", "limit": 100}
    ])
    responses.add(responses.GET, f"{BASE_URL}/latest", json={
        "data": {
            "1": {"high": 1100, "low": 1000}  # 10% spread, current ~1050
        }
    })
    responses.add(responses.GET, f"{BASE_URL}/24h", json={
        "data": {
            "1": {"highPriceVolume": 6000000, "lowPriceVolume": 4000000}
        }
    })

    # Was at 1500, now at 1050
    timeseries = [{"timestamp": i * 3600, "avgHighPrice": 1500 - (i * 450 // 720), "avgLowPrice": 1450 - (i * 450 // 720)} for i in range(720)]
    responses.add(responses.GET, f"{BASE_URL}/timeseries", json={"data": timeseries})

    client = OSRSClient()
    scanner = ItemScanner(client)

    results = scanner.scan(mode="both", limit=1)

    if results:
        assert "instant" in results[0]
        assert "convergence" in results[0]
        # At least one should be True
        assert results[0]["instant"]["is_instant_opportunity"] or results[0]["convergence"]["is_convergence"]
