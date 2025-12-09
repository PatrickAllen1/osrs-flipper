"""Tests for API client."""
import pytest
import responses
from osrs_flipper.api import OSRSClient

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"


@responses.activate
def test_fetch_mapping_returns_item_dict():
    """Mapping endpoint returns dict keyed by item ID."""
    responses.add(
        responses.GET,
        f"{BASE_URL}/mapping",
        json=[
            {"id": 4151, "name": "Abyssal whip", "limit": 70},
            {"id": 2, "name": "Cannonball", "limit": 10000},
        ],
        status=200,
    )

    client = OSRSClient()
    mapping = client.fetch_mapping()

    assert 4151 in mapping
    assert mapping[4151]["name"] == "Abyssal whip"
    assert mapping[4151]["limit"] == 70


@responses.activate
def test_fetch_latest_returns_price_data():
    """Latest endpoint returns current high/low prices."""
    responses.add(
        responses.GET,
        f"{BASE_URL}/latest",
        json={
            "data": {
                "4151": {"high": 1900000, "low": 1850000, "highTime": 1234567890, "lowTime": 1234567880},
                "2": {"high": 150, "low": 145, "highTime": 1234567890, "lowTime": 1234567880},
            }
        },
        status=200,
    )

    client = OSRSClient()
    latest = client.fetch_latest()

    assert "4151" in latest
    assert latest["4151"]["high"] == 1900000
    assert latest["4151"]["low"] == 1850000


@responses.activate
def test_fetch_volumes_returns_24h_data():
    """24h endpoint returns volume data."""
    responses.add(
        responses.GET,
        f"{BASE_URL}/24h",
        json={
            "data": {
                "4151": {"highPriceVolume": 5000, "lowPriceVolume": 4800},
                "2": {"highPriceVolume": 2500000, "lowPriceVolume": 2400000},
            }
        },
        status=200,
    )

    client = OSRSClient()
    volumes = client.fetch_volumes()

    assert "4151" in volumes
    assert volumes["4151"]["highPriceVolume"] == 5000
    assert volumes["2"]["lowPriceVolume"] == 2400000


@responses.activate
def test_fetch_timeseries_returns_historical_prices():
    """Timeseries endpoint returns daily price history."""
    responses.add(
        responses.GET,
        f"{BASE_URL}/timeseries",
        json={
            "data": [
                {"timestamp": 1700000000, "avgHighPrice": 1900000, "avgLowPrice": 1850000},
                {"timestamp": 1700086400, "avgHighPrice": 1920000, "avgLowPrice": 1870000},
            ]
        },
        status=200,
    )

    client = OSRSClient()
    history = client.fetch_timeseries(item_id=4151, timestep="24h")

    assert len(history) == 2
    assert history[0]["avgHighPrice"] == 1900000
    assert history[1]["avgLowPrice"] == 1870000


@responses.activate
def test_fetch_timeseries_5min_resolution():
    """Should fetch 5-minute resolution data."""
    responses.add(
        responses.GET,
        f"{BASE_URL}/timeseries",
        json={
            "data": [
                {"timestamp": 1700000000, "avgHighPrice": 1900000, "avgLowPrice": 1850000},
                {"timestamp": 1700000300, "avgHighPrice": 1901000, "avgLowPrice": 1851000},
            ]
        },
        status=200,
    )

    client = OSRSClient()
    data = client.fetch_timeseries(4151, timestep="5m")
    assert isinstance(data, list)
    assert len(data) == 2


@responses.activate
def test_fetch_timeseries_1h_resolution():
    """Should fetch 1-hour resolution data."""
    responses.add(
        responses.GET,
        f"{BASE_URL}/timeseries",
        json={
            "data": [
                {"timestamp": 1700000000, "avgHighPrice": 1900000, "avgLowPrice": 1850000},
                {"timestamp": 1700003600, "avgHighPrice": 1905000, "avgLowPrice": 1855000},
            ]
        },
        status=200,
    )

    client = OSRSClient()
    data = client.fetch_timeseries(4151, timestep="1h")
    assert isinstance(data, list)
    assert len(data) == 2


@responses.activate
def test_fetch_timeseries_with_timestamp_filter():
    """Should fetch data after a specific timestamp."""
    seven_days_ago = 1700000000
    responses.add(
        responses.GET,
        f"{BASE_URL}/timeseries",
        json={
            "data": [
                {"timestamp": 1700000000, "avgHighPrice": 1900000, "avgLowPrice": 1850000},
                {"timestamp": 1700086400, "avgHighPrice": 1920000, "avgLowPrice": 1870000},
            ]
        },
        status=200,
    )

    client = OSRSClient()
    data = client.fetch_timeseries(4151, timestep="1h", timestamp=seven_days_ago)
    assert isinstance(data, list)
    for point in data:
        assert point.get("timestamp", 0) >= seven_days_ago
