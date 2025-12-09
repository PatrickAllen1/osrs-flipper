# OSRS GE Flip Scanner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI tool that finds profitable Grand Exchange flip opportunities using percentile-based oversold detection and range-bound oscillator patterns, with portfolio optimization across 8 GE slots.

**Architecture:** Modular Python CLI with separate concerns: API client for data fetching, analyzers for pattern detection (oversold + oscillator), optimizer for slot allocation with EV scoring, and portfolio manager for presets/recommendations. SQLite for persistence.

**Tech Stack:** Python 3.11+, Click (CLI), requests (API), SQLite (persistence), pandas (data analysis)

---

## Phase 1: Project Structure & API Client

### Task 1: Initialize Project Structure ✅ DONE

**Files:**
- Create: `osrs_flipper/__init__.py`
- Create: `osrs_flipper/cli.py`
- Create: `osrs_flipper/api.py`
- Create: `tests/__init__.py`
- Create: `tests/test_api.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "osrs-flipper"
version = "0.1.0"
description = "OSRS Grand Exchange flip scanner and portfolio optimizer"
requires-python = ">=3.11"
dependencies = [
    "click>=8.0",
    "requests>=2.28",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "responses>=0.23",
]

[project.scripts]
osrs-flip = "osrs_flipper.cli:main"
```

**Step 2: Create requirements.txt**

```
click>=8.0
requests>=2.28
pandas>=2.0
pytest>=7.0
pytest-cov>=4.0
responses>=0.23
```

**Step 3: Create package structure**

```bash
mkdir -p osrs_flipper tests
touch osrs_flipper/__init__.py tests/__init__.py
```

**Step 4: Install dependencies**

```bash
pip install -e ".[dev]"
```

**Step 5: Commit**

```bash
git add pyproject.toml requirements.txt osrs_flipper/ tests/
git commit -m "chore: initialize project structure"
```

---

### Task 2: API Client - Mapping Endpoint ✅ DONE

**Files:**
- Modify: `osrs_flipper/api.py`
- Modify: `tests/test_api.py`

**Step 1: Write the failing test**

```python
# tests/test_api.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_fetch_mapping_returns_item_dict -v`
Expected: FAIL with "cannot import name 'OSRSClient'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/api.py
"""OSRS Wiki API client for price and volume data."""
import requests
from typing import Dict, Any

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"
HEADERS = {"User-Agent": "osrs-flipper - github.com/user/osrs-flipper"}


class OSRSClient:
    """Client for OSRS Wiki Real-Time Prices API."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def fetch_mapping(self) -> Dict[int, Dict[str, Any]]:
        """Fetch item ID to name/limit mapping.

        Returns:
            Dict keyed by item ID with name, limit, etc.
        """
        resp = self.session.get(f"{self.base_url}/mapping")
        resp.raise_for_status()
        return {item["id"]: item for item in resp.json()}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py::test_fetch_mapping_returns_item_dict -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/api.py tests/test_api.py
git commit -m "feat(api): add mapping endpoint"
```

---

### Task 3: API Client - Latest Prices Endpoint ✅ DONE

**Files:**
- Modify: `osrs_flipper/api.py`
- Modify: `tests/test_api.py`

**Step 1: Write the failing test**

```python
# tests/test_api.py (append)
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_fetch_latest_returns_price_data -v`
Expected: FAIL with "OSRSClient has no attribute 'fetch_latest'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/api.py (add method to OSRSClient)
    def fetch_latest(self) -> Dict[str, Dict[str, Any]]:
        """Fetch latest prices for all items.

        Returns:
            Dict keyed by item ID string with high/low prices.
        """
        resp = self.session.get(f"{self.base_url}/latest")
        resp.raise_for_status()
        return resp.json().get("data", {})
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py::test_fetch_latest_returns_price_data -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/api.py tests/test_api.py
git commit -m "feat(api): add latest prices endpoint"
```

---

### Task 4: API Client - 24h Volume Endpoint ✅ DONE

**Files:**
- Modify: `osrs_flipper/api.py`
- Modify: `tests/test_api.py`

**Step 1: Write the failing test**

```python
# tests/test_api.py (append)
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_fetch_volumes_returns_24h_data -v`
Expected: FAIL with "OSRSClient has no attribute 'fetch_volumes'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/api.py (add method to OSRSClient)
    def fetch_volumes(self) -> Dict[str, Dict[str, Any]]:
        """Fetch 24h volume data.

        Returns:
            Dict keyed by item ID with highPriceVolume/lowPriceVolume.
        """
        resp = self.session.get(f"{self.base_url}/24h")
        resp.raise_for_status()
        return resp.json().get("data", {})
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py::test_fetch_volumes_returns_24h_data -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/api.py tests/test_api.py
git commit -m "feat(api): add 24h volume endpoint"
```

---

### Task 5: API Client - Timeseries Endpoint ✅ DONE

**Files:**
- Modify: `osrs_flipper/api.py`
- Modify: `tests/test_api.py`

**Step 1: Write the failing test**

```python
# tests/test_api.py (append)
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_fetch_timeseries_returns_historical_prices -v`
Expected: FAIL with "OSRSClient has no attribute 'fetch_timeseries'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/api.py (add method to OSRSClient)
    def fetch_timeseries(self, item_id: int, timestep: str = "24h") -> list:
        """Fetch historical price data for an item.

        Args:
            item_id: The item ID to fetch history for.
            timestep: Time resolution ("5m", "1h", "6h", "24h").

        Returns:
            List of price points with timestamp, avgHighPrice, avgLowPrice.
        """
        resp = self.session.get(
            f"{self.base_url}/timeseries",
            params={"id": item_id, "timestep": timestep},
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py::test_fetch_timeseries_returns_historical_prices -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/api.py tests/test_api.py
git commit -m "feat(api): add timeseries endpoint"
```

---

## Phase 2: Core Detection Logic

### Task 6: Volume Tier Filter ✅ DONE

**Files:**
- Create: `osrs_flipper/filters.py`
- Create: `tests/test_filters.py`

**Step 1: Write the failing test**

```python
# tests/test_filters.py
import pytest
from osrs_flipper.filters import get_min_volume, passes_volume_filter

# Volume tiers from brainstorming:
# < 1k gp: 2.5M+ volume
# 1k-10k gp: 250k+ volume
# 10k-100k gp: 25k+ volume
# 100k-1M gp: 2.5k+ volume
# 1M-10M gp: 250+ volume
# 10M+ gp: 50+ volume


@pytest.mark.parametrize("price,expected_min_volume", [
    (500, 2_500_000),        # < 1k tier
    (5_000, 250_000),        # 1k-10k tier
    (50_000, 25_000),        # 10k-100k tier
    (500_000, 2_500),        # 100k-1M tier
    (5_000_000, 250),        # 1M-10M tier
    (50_000_000, 50),        # 10M+ tier
])
def test_get_min_volume_returns_correct_tier(price, expected_min_volume):
    assert get_min_volume(price) == expected_min_volume


def test_passes_volume_filter_accepts_above_threshold():
    # 500 gp item needs 2.5M volume, has 3M
    assert passes_volume_filter(price=500, volume=3_000_000) is True


def test_passes_volume_filter_rejects_below_threshold():
    # 500 gp item needs 2.5M volume, has 1M
    assert passes_volume_filter(price=500, volume=1_000_000) is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_filters.py -v`
Expected: FAIL with "cannot import name 'get_min_volume'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/filters.py
"""Filtering logic for item selection."""

# Volume thresholds by price tier (max_price, min_volume)
VOLUME_TIERS = [
    (1_000, 2_500_000),        # < 1k gp: 2.5M+ volume
    (10_000, 250_000),         # 1k-10k gp: 250k+ volume
    (100_000, 25_000),         # 10k-100k gp: 25k+ volume
    (1_000_000, 2_500),        # 100k-1M gp: 2.5k+ volume
    (10_000_000, 250),         # 1M-10M gp: 250+ volume
    (float("inf"), 50),        # 10M+ gp: 50+ volume
]


def get_min_volume(price: int) -> int:
    """Get minimum volume threshold for a price tier.

    Args:
        price: Current item price in GP.

    Returns:
        Minimum daily volume required.
    """
    for max_price, min_vol in VOLUME_TIERS:
        if price < max_price:
            return min_vol
    return 50


def passes_volume_filter(price: int, volume: int) -> bool:
    """Check if item passes volume filter for its price tier.

    Args:
        price: Current item price in GP.
        volume: Daily trade volume (item count).

    Returns:
        True if volume meets threshold.
    """
    return volume >= get_min_volume(price)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_filters.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/filters.py tests/test_filters.py
git commit -m "feat(filters): add volume tier filtering"
```

---

### Task 7: RSI Calculator ✅ DONE

**Files:**
- Create: `osrs_flipper/indicators.py`
- Create: `tests/test_indicators.py`

**Step 1: Write the failing test**

```python
# tests/test_indicators.py
import pytest
from osrs_flipper.indicators import calculate_rsi


def test_rsi_returns_none_for_insufficient_data():
    """RSI needs at least period+1 prices."""
    prices = [100, 105, 102]  # Only 3 prices, need 15 for period=14
    assert calculate_rsi(prices, period=14) is None


def test_rsi_returns_100_for_all_gains():
    """RSI = 100 when all price moves are gains."""
    # 15 consecutive gains
    prices = [100 + i * 10 for i in range(16)]
    rsi = calculate_rsi(prices, period=14)
    assert rsi == 100.0


def test_rsi_returns_0_for_all_losses():
    """RSI = 0 when all price moves are losses."""
    # 15 consecutive losses
    prices = [200 - i * 10 for i in range(16)]
    rsi = calculate_rsi(prices, period=14)
    assert rsi == 0.0


def test_rsi_returns_around_50_for_mixed():
    """RSI near 50 for balanced gains/losses."""
    # Alternating up and down with equal magnitude
    prices = [100, 110, 100, 110, 100, 110, 100, 110, 100, 110, 100, 110, 100, 110, 100, 110]
    rsi = calculate_rsi(prices, period=14)
    assert 45 <= rsi <= 55  # Should be around 50
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_indicators.py -v`
Expected: FAIL with "cannot import name 'calculate_rsi'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/indicators.py
"""Technical indicators for price analysis."""
from typing import List, Optional


def calculate_rsi(prices: List[int], period: int = 14) -> Optional[float]:
    """Calculate Relative Strength Index.

    Args:
        prices: List of prices (oldest first).
        period: RSI period (default 14).

    Returns:
        RSI value 0-100, or None if insufficient data.
    """
    if len(prices) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    if len(gains) < period:
        return None

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100.0

    if avg_gain == 0:
        return 0.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 1)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_indicators.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/indicators.py tests/test_indicators.py
git commit -m "feat(indicators): add RSI calculator"
```

---

### Task 8: Percentile Calculator ✅ DONE

**Files:**
- Modify: `osrs_flipper/indicators.py`
- Modify: `tests/test_indicators.py`

**Step 1: Write the failing test**

```python
# tests/test_indicators.py (append)
from osrs_flipper.indicators import calculate_percentile


def test_percentile_at_low_returns_0():
    """Current price at historical low = 0th percentile."""
    low, high, current = 100, 200, 100
    assert calculate_percentile(current, low, high) == 0.0


def test_percentile_at_high_returns_100():
    """Current price at historical high = 100th percentile."""
    low, high, current = 100, 200, 200
    assert calculate_percentile(current, low, high) == 100.0


def test_percentile_at_midpoint_returns_50():
    """Current price at midpoint = 50th percentile."""
    low, high, current = 100, 200, 150
    assert calculate_percentile(current, low, high) == 50.0


def test_percentile_below_low_returns_negative():
    """Current price below historical low = negative percentile."""
    low, high, current = 100, 200, 80
    assert calculate_percentile(current, low, high) == -20.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_indicators.py::test_percentile_at_low_returns_0 -v`
Expected: FAIL with "cannot import name 'calculate_percentile'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/indicators.py (add function)
def calculate_percentile(current: int, low: int, high: int) -> float:
    """Calculate where current price sits in historical range.

    Args:
        current: Current price.
        low: Historical low price.
        high: Historical high price.

    Returns:
        Percentile 0-100 (can be negative if below low).
    """
    if high == low:
        return 50.0  # No range, assume middle

    return round(((current - low) / (high - low)) * 100, 1)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_indicators.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/indicators.py tests/test_indicators.py
git commit -m "feat(indicators): add percentile calculator"
```

---

### Task 9: Oversold Detector ✅ DONE

**Files:**
- Create: `osrs_flipper/analyzers.py`
- Create: `tests/test_analyzers.py`

**Step 1: Write the failing test**

```python
# tests/test_analyzers.py
import pytest
from osrs_flipper.analyzers import OversoldAnalyzer


def test_oversold_detector_identifies_near_low():
    """Item within 20% of low with 30%+ upside is oversold."""
    analyzer = OversoldAnalyzer(
        low_threshold_pct=20,  # Within 20% of low
        min_upside_pct=30,     # 30%+ upside required
    )

    result = analyzer.analyze(
        current_price=110,
        six_month_low=100,
        six_month_high=200,
        prices=[100, 110, 120, 130, 140, 150, 140, 130, 120, 110, 100, 110] * 3,  # 36 prices
    )

    assert result["is_oversold"] is True
    assert result["percentile"] < 20
    assert result["upside_pct"] > 30


def test_oversold_detector_rejects_mid_range():
    """Item at 50% of range is not oversold."""
    analyzer = OversoldAnalyzer()

    result = analyzer.analyze(
        current_price=150,
        six_month_low=100,
        six_month_high=200,
        prices=[100, 150, 200, 150] * 10,
    )

    assert result["is_oversold"] is False


def test_oversold_detector_rejects_low_upside():
    """Item with <30% upside is not oversold."""
    analyzer = OversoldAnalyzer()

    result = analyzer.analyze(
        current_price=90,
        six_month_low=85,
        six_month_high=100,  # Only 11% upside
        prices=[85, 90, 95, 100, 95, 90] * 6,
    )

    assert result["is_oversold"] is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzers.py -v`
Expected: FAIL with "cannot import name 'OversoldAnalyzer'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/analyzers.py
"""Pattern detection analyzers."""
from typing import Dict, Any, List
from .indicators import calculate_rsi, calculate_percentile


class OversoldAnalyzer:
    """Detect oversold items near historical lows with recovery potential."""

    def __init__(
        self,
        low_threshold_pct: float = 20,
        min_upside_pct: float = 30,
    ):
        """Initialize analyzer.

        Args:
            low_threshold_pct: Max percentile to consider oversold (default 20).
            min_upside_pct: Minimum upside to historical high (default 30%).
        """
        self.low_threshold_pct = low_threshold_pct
        self.min_upside_pct = min_upside_pct

    def analyze(
        self,
        current_price: int,
        six_month_low: int,
        six_month_high: int,
        prices: List[int],
    ) -> Dict[str, Any]:
        """Analyze item for oversold opportunity.

        Args:
            current_price: Current item price.
            six_month_low: Lowest price in period.
            six_month_high: Highest price in period.
            prices: List of historical prices.

        Returns:
            Analysis result dict.
        """
        percentile = calculate_percentile(current_price, six_month_low, six_month_high)
        rsi = calculate_rsi(prices)

        # Calculate upside potential
        if current_price > 0:
            upside_pct = ((six_month_high - current_price) / current_price) * 100
        else:
            upside_pct = 0

        is_oversold = (
            percentile <= self.low_threshold_pct
            and upside_pct >= self.min_upside_pct
        )

        return {
            "is_oversold": is_oversold,
            "percentile": round(percentile, 1),
            "rsi": rsi,
            "upside_pct": round(upside_pct, 1),
            "six_month_low": six_month_low,
            "six_month_high": six_month_high,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/analyzers.py tests/test_analyzers.py
git commit -m "feat(analyzers): add oversold detector"
```

---

### Task 10: Oscillator Detector (Range-Bound)

**Files:**
- Modify: `osrs_flipper/analyzers.py`
- Modify: `tests/test_analyzers.py`

**Step 1: Write the failing test**

```python
# tests/test_analyzers.py (append)
from osrs_flipper.analyzers import OscillatorAnalyzer


def test_oscillator_detects_bouncing_pattern():
    """Item bouncing between support/resistance is detected."""
    analyzer = OscillatorAnalyzer(
        min_bounces=4,
        max_support_variance_pct=10,
    )

    # Simulating amethyst arrows: 240-340 bounce pattern
    prices = [
        240, 260, 300, 340, 320, 280, 240, 250, 290, 340,
        310, 270, 240, 260, 310, 340, 300, 250, 240, 280,
    ]

    result = analyzer.analyze(prices=prices, current_price=250)

    assert result["is_oscillator"] is True
    assert 230 <= result["support"] <= 250
    assert 330 <= result["resistance"] <= 350
    assert result["bounce_count"] >= 4


def test_oscillator_rejects_trending_price():
    """Trending item is not an oscillator."""
    analyzer = OscillatorAnalyzer()

    # Steady uptrend
    prices = [100 + i * 5 for i in range(30)]

    result = analyzer.analyze(prices=prices, current_price=245)

    assert result["is_oscillator"] is False


def test_oscillator_detects_current_near_support():
    """Flag when current price is near support (buy signal)."""
    analyzer = OscillatorAnalyzer()

    prices = [
        100, 120, 140, 150, 140, 120, 100, 110, 130, 150,
        140, 120, 100, 115, 135, 150, 130, 110, 100, 120,
    ]

    result = analyzer.analyze(prices=prices, current_price=105)

    assert result["near_support"] is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyzers.py::test_oscillator_detects_bouncing_pattern -v`
Expected: FAIL with "cannot import name 'OscillatorAnalyzer'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/analyzers.py (add class)
import statistics
from typing import List, Tuple


def find_local_extrema(prices: List[int], window: int = 3) -> Tuple[List[int], List[int]]:
    """Find local minima and maxima in price series.

    Args:
        prices: List of prices.
        window: Window size for detecting peaks/troughs.

    Returns:
        Tuple of (minima, maxima) lists.
    """
    minima = []
    maxima = []

    for i in range(window, len(prices) - window):
        window_slice = prices[i - window : i + window + 1]
        if prices[i] == min(window_slice):
            minima.append(prices[i])
        elif prices[i] == max(window_slice):
            maxima.append(prices[i])

    return minima, maxima


class OscillatorAnalyzer:
    """Detect range-bound items bouncing between support/resistance."""

    def __init__(
        self,
        min_bounces: int = 4,
        max_support_variance_pct: float = 10,
        support_proximity_pct: float = 5,
    ):
        """Initialize analyzer.

        Args:
            min_bounces: Minimum bounces off support/resistance.
            max_support_variance_pct: Max variance in support cluster.
            support_proximity_pct: Threshold for "near support".
        """
        self.min_bounces = min_bounces
        self.max_support_variance_pct = max_support_variance_pct
        self.support_proximity_pct = support_proximity_pct

    def analyze(
        self,
        prices: List[int],
        current_price: int,
    ) -> Dict[str, Any]:
        """Analyze item for oscillator pattern.

        Args:
            prices: Historical price series.
            current_price: Current item price.

        Returns:
            Analysis result dict.
        """
        if len(prices) < 10:
            return {"is_oscillator": False, "reason": "insufficient_data"}

        minima, maxima = find_local_extrema(prices)

        if len(minima) < 2 or len(maxima) < 2:
            return {"is_oscillator": False, "reason": "no_bounces"}

        # Calculate support and resistance levels
        support = statistics.mean(minima) if minima else prices[-1]
        resistance = statistics.mean(maxima) if maxima else prices[-1]

        # Calculate variance in support level
        if len(minima) >= 2 and support > 0:
            support_std = statistics.stdev(minima)
            support_variance_pct = (support_std / support) * 100
        else:
            support_variance_pct = float("inf")

        bounce_count = len(minima) + len(maxima)

        is_oscillator = (
            bounce_count >= self.min_bounces
            and support_variance_pct <= self.max_support_variance_pct
            and resistance > support
        )

        # Check if current price is near support
        if support > 0:
            distance_from_support = ((current_price - support) / support) * 100
            near_support = distance_from_support <= self.support_proximity_pct
        else:
            near_support = False

        range_pct = ((resistance - support) / support) * 100 if support > 0 else 0

        return {
            "is_oscillator": is_oscillator,
            "support": round(support),
            "resistance": round(resistance),
            "range_pct": round(range_pct, 1),
            "bounce_count": bounce_count,
            "support_variance_pct": round(support_variance_pct, 1),
            "near_support": near_support,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyzers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/analyzers.py tests/test_analyzers.py
git commit -m "feat(analyzers): add oscillator detector"
```

---

## Phase 3: CLI Framework

### Task 11: Cash Parser ✅ DONE

**Files:**
- Create: `osrs_flipper/utils.py`
- Create: `tests/test_utils.py`

**Step 1: Write the failing test**

```python
# tests/test_utils.py
import pytest
from osrs_flipper.utils import parse_cash


@pytest.mark.parametrize("input_str,expected", [
    ("120m", 120_000_000),
    ("120M", 120_000_000),
    ("1.5b", 1_500_000_000),
    ("1.5B", 1_500_000_000),
    ("50k", 50_000),
    ("50K", 50_000),
    ("1000000", 1_000_000),
    ("1,000,000", 1_000_000),
])
def test_parse_cash_handles_suffixes(input_str, expected):
    assert parse_cash(input_str) == expected


def test_parse_cash_raises_on_invalid():
    with pytest.raises(ValueError):
        parse_cash("invalid")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_utils.py -v`
Expected: FAIL with "cannot import name 'parse_cash'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/utils.py
"""Utility functions."""
import re


def parse_cash(amount_str: str) -> int:
    """Parse cash amount string with suffixes.

    Supports: "120m", "1.5b", "50k", "1000000", "1,000,000"

    Args:
        amount_str: Cash amount as string.

    Returns:
        Amount in GP as integer.

    Raises:
        ValueError: If format is invalid.
    """
    amount_str = amount_str.strip().lower().replace(",", "")

    suffixes = {
        "k": 1_000,
        "m": 1_000_000,
        "b": 1_000_000_000,
    }

    # Check for suffix
    if amount_str[-1] in suffixes:
        try:
            value = float(amount_str[:-1])
            return int(value * suffixes[amount_str[-1]])
        except ValueError:
            raise ValueError(f"Invalid cash format: {amount_str}")

    # Plain number
    try:
        return int(float(amount_str))
    except ValueError:
        raise ValueError(f"Invalid cash format: {amount_str}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_utils.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/utils.py tests/test_utils.py
git commit -m "feat(utils): add cash string parser"
```

---

### Task 12: Basic CLI Structure ✅ DONE

**Files:**
- Modify: `osrs_flipper/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
from click.testing import CliRunner
from osrs_flipper.cli import main, scan


def test_cli_main_shows_help():
    """Main command shows help."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "OSRS Grand Exchange" in result.output


def test_scan_command_exists():
    """Scan subcommand is available."""
    runner = CliRunner()
    result = runner.invoke(scan, ["--help"])

    assert result.exit_code == 0
    assert "--mode" in result.output
    assert "--cash" in result.output
    assert "--slots" in result.output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with "cannot import name 'main'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/cli.py
"""Command-line interface for OSRS Flip Scanner."""
import click
from .utils import parse_cash


@click.group()
@click.version_option(version="0.1.0")
def main():
    """OSRS Grand Exchange flip scanner and portfolio optimizer."""
    pass


@main.command()
@click.option(
    "--mode",
    type=click.Choice(["oversold", "oscillator", "all"]),
    default="all",
    help="Scan mode: oversold, oscillator, or all",
)
@click.option(
    "--cash",
    type=str,
    default=None,
    help="Cash stack to allocate (e.g., '120m', '1.5b')",
)
@click.option(
    "--slots",
    type=click.IntRange(1, 8),
    default=8,
    help="Available GE slots (1-8)",
)
@click.option(
    "--rotations",
    type=int,
    default=3,
    help="Buy limit rotations (default: 3)",
)
@click.option(
    "--strategy",
    type=click.Choice(["flip", "hold", "balanced"]),
    default="balanced",
    help="Allocation strategy",
)
@click.option(
    "--export",
    is_flag=True,
    help="Export results to CSV",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./output",
    help="Output directory for exports",
)
def scan(mode, cash, slots, rotations, strategy, export, output_dir):
    """Scan for flip opportunities."""
    click.echo(f"Scanning in {mode} mode...")

    if cash:
        try:
            cash_gp = parse_cash(cash)
            click.echo(f"Cash: {cash_gp:,} GP")
            click.echo(f"Slots: {slots}, Rotations: {rotations}, Strategy: {strategy}")
        except ValueError as e:
            raise click.BadParameter(str(e))

    # TODO: Implement actual scanning
    click.echo("Scan complete.")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/cli.py tests/test_cli.py
git commit -m "feat(cli): add basic CLI structure with scan command"
```

---

### Task 13: Portfolio CLI Command ✅ DONE

**Files:**
- Modify: `osrs_flipper/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py (append)
from osrs_flipper.cli import portfolio


def test_portfolio_command_exists():
    """Portfolio subcommand is available."""
    runner = CliRunner()
    result = runner.invoke(portfolio, ["--help"])

    assert result.exit_code == 0
    assert "--list" in result.output
    assert "--use" in result.output
    assert "--recommend" in result.output
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_portfolio_command_exists -v`
Expected: FAIL with "cannot import name 'portfolio'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/cli.py (add command)
@main.command()
@click.option(
    "--list",
    "list_presets",
    is_flag=True,
    help="List saved portfolio presets",
)
@click.option(
    "--use",
    type=str,
    default=None,
    help="Use a preset (grinder, balanced, diamondhands)",
)
@click.option(
    "--recommend",
    is_flag=True,
    help="Auto-recommend portfolio based on market",
)
@click.option(
    "--cash",
    type=str,
    default=None,
    help="Cash stack to allocate",
)
@click.option(
    "--slots",
    type=click.IntRange(1, 8),
    default=8,
    help="Available GE slots (1-8)",
)
@click.option(
    "--save",
    type=str,
    default=None,
    help="Save current allocation as preset",
)
def portfolio(list_presets, use, recommend, cash, slots, save):
    """Manage portfolio presets and recommendations."""
    if list_presets:
        click.echo("Available presets:")
        click.echo("  grinder      - 8 quick flips, ~2 day avg hold")
        click.echo("  balanced     - 4 flip / 4 hold, ~5 day avg hold")
        click.echo("  diamondhands - 8 mid-term holds, ~12 day avg hold")
        return

    if recommend:
        if not cash:
            raise click.UsageError("--cash required with --recommend")
        click.echo("Analyzing market conditions...")
        # TODO: Implement recommendation
        click.echo("Recommendation: balanced")
        return

    if use:
        if not cash:
            raise click.UsageError("--cash required with --use")
        click.echo(f"Loading preset: {use}")
        # TODO: Implement preset loading
        return

    click.echo("Use --list, --use, or --recommend")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/cli.py tests/test_cli.py
git commit -m "feat(cli): add portfolio command"
```

---

## Phase 4: Scanner Implementation

### Task 14: Item Scanner Service ✅ DONE

**Files:**
- Create: `osrs_flipper/scanner.py`
- Create: `tests/test_scanner.py`

**Step 1: Write the failing test**

```python
# tests/test_scanner.py
import pytest
import responses
from osrs_flipper.scanner import ItemScanner
from osrs_flipper.api import OSRSClient

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"


@responses.activate
def test_scanner_finds_oversold_items():
    """Scanner identifies oversold opportunities."""
    # Mock API responses
    responses.add(
        responses.GET,
        f"{BASE_URL}/mapping",
        json=[{"id": 1, "name": "Test Item", "limit": 100}],
    )
    responses.add(
        responses.GET,
        f"{BASE_URL}/latest",
        json={"data": {"1": {"high": 110, "low": 100}}},
    )
    responses.add(
        responses.GET,
        f"{BASE_URL}/24h",
        json={"data": {"1": {"highPriceVolume": 3000000, "lowPriceVolume": 2500000}}},
    )
    responses.add(
        responses.GET,
        f"{BASE_URL}/timeseries",
        json={
            "data": [
                {"timestamp": i, "avgHighPrice": 100 + (i % 100), "avgLowPrice": 95 + (i % 100)}
                for i in range(90)
            ]
        },
    )

    client = OSRSClient()
    scanner = ItemScanner(client)

    results = scanner.scan(mode="oversold", limit=10)

    assert isinstance(results, list)


@responses.activate
def test_scanner_respects_volume_filter():
    """Scanner filters out low volume items."""
    responses.add(
        responses.GET,
        f"{BASE_URL}/mapping",
        json=[{"id": 1, "name": "Low Volume Item", "limit": 100}],
    )
    responses.add(
        responses.GET,
        f"{BASE_URL}/latest",
        json={"data": {"1": {"high": 110, "low": 100}}},  # ~105 GP
    )
    responses.add(
        responses.GET,
        f"{BASE_URL}/24h",
        json={"data": {"1": {"highPriceVolume": 1000, "lowPriceVolume": 1000}}},  # 2k volume, needs 2.5M
    )

    client = OSRSClient()
    scanner = ItemScanner(client)

    results = scanner.scan(mode="oversold", limit=10)

    # Should find nothing due to volume filter
    assert len(results) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scanner.py -v`
Expected: FAIL with "cannot import name 'ItemScanner'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/scanner.py
"""Item scanning service."""
from typing import List, Dict, Any, Optional
from .api import OSRSClient
from .analyzers import OversoldAnalyzer, OscillatorAnalyzer
from .filters import passes_volume_filter


class ItemScanner:
    """Scans items for flip opportunities."""

    def __init__(self, client: OSRSClient):
        self.client = client
        self.oversold_analyzer = OversoldAnalyzer()
        self.oscillator_analyzer = OscillatorAnalyzer()

    def scan(
        self,
        mode: str = "all",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Scan for flip opportunities.

        Args:
            mode: "oversold", "oscillator", or "all"
            limit: Max items to scan (None for all)

        Returns:
            List of opportunity dicts.
        """
        # Fetch data
        mapping = self.client.fetch_mapping()
        latest = self.client.fetch_latest()
        volumes = self.client.fetch_volumes()

        opportunities = []
        item_ids = list(mapping.keys())[:limit] if limit else list(mapping.keys())

        for item_id in item_ids:
            result = self._analyze_item(
                item_id=item_id,
                mapping=mapping,
                latest=latest,
                volumes=volumes,
                mode=mode,
            )
            if result:
                opportunities.append(result)

        return opportunities

    def _analyze_item(
        self,
        item_id: int,
        mapping: Dict,
        latest: Dict,
        volumes: Dict,
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        """Analyze single item."""
        if item_id not in mapping:
            return None

        item = mapping[item_id]
        name = item.get("name", "Unknown")

        # Skip noted items
        if "(noted)" in name.lower():
            return None

        # Get current price
        price_data = latest.get(str(item_id), {})
        if not price_data:
            return None

        high = price_data.get("high")
        low = price_data.get("low")
        if not high or not low:
            return None

        current_price = (high + low) // 2
        if current_price <= 0:
            return None

        # Check volume
        vol_data = volumes.get(str(item_id), {})
        high_vol = vol_data.get("highPriceVolume", 0) or 0
        low_vol = vol_data.get("lowPriceVolume", 0) or 0
        daily_volume = high_vol + low_vol

        if not passes_volume_filter(current_price, daily_volume):
            return None

        # Fetch historical data
        try:
            history = self.client.fetch_timeseries(item_id)
        except Exception:
            return None

        if len(history) < 30:
            return None

        # Extract prices
        prices = []
        for point in history:
            h = point.get("avgHighPrice")
            l = point.get("avgLowPrice")
            if h and l:
                prices.append((h + l) // 2)

        if len(prices) < 30:
            return None

        six_month_low = min(prices)
        six_month_high = max(prices)

        result = {
            "item_id": item_id,
            "name": name,
            "current_price": current_price,
            "daily_volume": daily_volume,
            "buy_limit": item.get("limit"),
        }

        # Run analyzers based on mode
        if mode in ("oversold", "all"):
            oversold = self.oversold_analyzer.analyze(
                current_price=current_price,
                six_month_low=six_month_low,
                six_month_high=six_month_high,
                prices=prices,
            )
            result["oversold"] = oversold

        if mode in ("oscillator", "all"):
            oscillator = self.oscillator_analyzer.analyze(
                prices=prices,
                current_price=current_price,
            )
            result["oscillator"] = oscillator

        # Check if any pattern matched
        is_opportunity = False
        if mode == "oversold" and result.get("oversold", {}).get("is_oversold"):
            is_opportunity = True
        elif mode == "oscillator" and result.get("oscillator", {}).get("is_oscillator"):
            is_opportunity = True
        elif mode == "all":
            is_opportunity = (
                result.get("oversold", {}).get("is_oversold")
                or result.get("oscillator", {}).get("is_oscillator")
            )

        return result if is_opportunity else None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scanner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/scanner.py tests/test_scanner.py
git commit -m "feat(scanner): add item scanner service"
```

---

## Phase 5: Allocation Engine

### Task 15: EV Scoring ✅ DONE

**Files:**
- Create: `osrs_flipper/scoring.py`
- Create: `tests/test_scoring.py`

**Step 1: Write the failing test**

```python
# tests/test_scoring.py
import pytest
from osrs_flipper.scoring import calculate_item_score, calculate_ev


def test_calculate_item_score():
    """Item score combines upside, oversold, liquidity, bounce history."""
    score = calculate_item_score(
        upside_pct=50,
        percentile=10,
        volume_ratio=1.5,  # 1.5x min volume
        bounce_rate=0.8,
    )

    # Score should be positive weighted combination
    assert score > 0
    assert isinstance(score, float)


def test_higher_upside_increases_score():
    """Higher upside should increase score."""
    score_low = calculate_item_score(upside_pct=20, percentile=10, volume_ratio=1.0, bounce_rate=0.5)
    score_high = calculate_item_score(upside_pct=50, percentile=10, volume_ratio=1.0, bounce_rate=0.5)

    assert score_high > score_low


def test_calculate_ev():
    """EV = capital × ROI × confidence."""
    ev = calculate_ev(
        capital=10_000_000,  # 10M
        roi_pct=30,          # 30% ROI
        confidence=0.8,      # 80% confidence
    )

    # EV = 10M × 0.30 × 0.8 = 2.4M
    assert ev == 2_400_000
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL with "cannot import name 'calculate_item_score'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/scoring.py
"""EV and scoring calculations."""

# Score weights
WEIGHT_UPSIDE = 0.4
WEIGHT_OVERSOLD = 0.3
WEIGHT_LIQUIDITY = 0.2
WEIGHT_BOUNCE = 0.1


def calculate_item_score(
    upside_pct: float,
    percentile: float,
    volume_ratio: float,
    bounce_rate: float,
) -> float:
    """Calculate composite score for an item.

    Args:
        upside_pct: Upside potential percentage.
        percentile: Where price sits in historical range (lower = more oversold).
        volume_ratio: Volume / min_required_volume (capped at 2).
        bounce_rate: Historical bounce success rate (0-1).

    Returns:
        Composite score.
    """
    # Normalize components to 0-100 scale
    upside_score = min(upside_pct, 100)  # Cap at 100%
    oversold_score = 100 - percentile     # Lower percentile = higher score
    liquidity_score = min(volume_ratio, 2) * 50  # 2x volume = 100
    bounce_score = bounce_rate * 100

    score = (
        upside_score * WEIGHT_UPSIDE
        + oversold_score * WEIGHT_OVERSOLD
        + liquidity_score * WEIGHT_LIQUIDITY
        + bounce_score * WEIGHT_BOUNCE
    )

    return round(score, 2)


def calculate_ev(capital: int, roi_pct: float, confidence: float) -> int:
    """Calculate expected value.

    Args:
        capital: Capital allocated.
        roi_pct: Expected ROI percentage.
        confidence: Confidence factor (0-1).

    Returns:
        Expected value in GP.
    """
    return int(capital * (roi_pct / 100) * confidence)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/scoring.py tests/test_scoring.py
git commit -m "feat(scoring): add EV and item scoring"
```

---

### Task 16: Exit Strategy Calculator ✅ DONE

**Files:**
- Create: `osrs_flipper/exits.py`
- Create: `tests/test_exits.py`

**Step 1: Write the failing test**

```python
# tests/test_exits.py
import pytest
from osrs_flipper.exits import calculate_exit_strategies


def test_exit_strategies_returns_all_levels():
    """Exit strategies include all percentile levels."""
    exits = calculate_exit_strategies(
        entry_price=100,
        prices=[80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180] * 3,
    )

    assert "conservative" in exits  # 25th percentile
    assert "target" in exits        # 50th percentile
    assert "aggressive" in exits    # 75th percentile
    assert "recent_peak" in exits
    assert "floor_warning" in exits


def test_exit_conservative_below_target():
    """Conservative exit is lower than target."""
    exits = calculate_exit_strategies(
        entry_price=100,
        prices=[80, 100, 120, 140, 160, 180, 200] * 5,
    )

    assert exits["conservative"]["price"] < exits["target"]["price"]
    assert exits["target"]["price"] < exits["aggressive"]["price"]


def test_exit_roi_calculated():
    """Each exit has ROI percentage."""
    exits = calculate_exit_strategies(
        entry_price=100,
        prices=[80, 100, 150, 200] * 10,
    )

    assert "roi_pct" in exits["conservative"]
    assert exits["conservative"]["roi_pct"] > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_exits.py -v`
Expected: FAIL with "cannot import name 'calculate_exit_strategies'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/exits.py
"""Exit strategy calculations."""
from typing import Dict, Any, List
import statistics


def calculate_exit_strategies(
    entry_price: int,
    prices: List[int],
    floor_drop_pct: float = 7,
) -> Dict[str, Any]:
    """Calculate exit strategy levels based on historical prices.

    Args:
        entry_price: Entry/buy price.
        prices: Historical price list.
        floor_drop_pct: Percentage drop to trigger floor warning.

    Returns:
        Dict with exit levels and ROI calculations.
    """
    sorted_prices = sorted(prices)
    n = len(sorted_prices)

    def percentile_price(pct: float) -> int:
        idx = int((pct / 100) * (n - 1))
        return sorted_prices[idx]

    def calc_roi(exit_price: int) -> float:
        if entry_price <= 0:
            return 0.0
        return round(((exit_price - entry_price) / entry_price) * 100, 1)

    conservative_price = percentile_price(25)
    target_price = percentile_price(50)
    aggressive_price = percentile_price(75)
    recent_peak = max(prices[-30:]) if len(prices) >= 30 else max(prices)
    floor_price = int(entry_price * (1 - floor_drop_pct / 100))

    return {
        "conservative": {
            "price": conservative_price,
            "roi_pct": calc_roi(conservative_price),
            "percentile": 25,
        },
        "target": {
            "price": target_price,
            "roi_pct": calc_roi(target_price),
            "percentile": 50,
        },
        "aggressive": {
            "price": aggressive_price,
            "roi_pct": calc_roi(aggressive_price),
            "percentile": 75,
        },
        "recent_peak": {
            "price": recent_peak,
            "roi_pct": calc_roi(recent_peak),
        },
        "floor_warning": {
            "price": floor_price,
            "drop_pct": floor_drop_pct,
        },
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_exits.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/exits.py tests/test_exits.py
git commit -m "feat(exits): add exit strategy calculator"
```

---

### Task 17: Slot Allocator ✅ DONE

**Files:**
- Create: `osrs_flipper/allocator.py`
- Create: `tests/test_allocator.py`

**Step 1: Write the failing test**

```python
# tests/test_allocator.py
import pytest
from osrs_flipper.allocator import SlotAllocator


def test_allocator_respects_slot_limit():
    """Allocator uses at most the specified slots."""
    allocator = SlotAllocator(strategy="balanced")

    opportunities = [
        {"name": f"Item {i}", "current_price": 1000, "buy_limit": 100, "score": 50 - i}
        for i in range(10)
    ]

    allocation = allocator.allocate(
        opportunities=opportunities,
        cash=10_000_000,
        slots=5,
        rotations=3,
    )

    assert len(allocation) <= 5


def test_allocator_respects_cash_limit():
    """Total allocated capital does not exceed cash."""
    allocator = SlotAllocator(strategy="balanced")

    opportunities = [
        {"name": "Expensive", "current_price": 1_000_000, "buy_limit": 100, "score": 90},
        {"name": "Cheap", "current_price": 1_000, "buy_limit": 10000, "score": 80},
    ]

    allocation = allocator.allocate(
        opportunities=opportunities,
        cash=5_000_000,
        slots=8,
        rotations=3,
    )

    total_capital = sum(slot["capital"] for slot in allocation)
    assert total_capital <= 5_000_000


def test_allocator_calculates_quantity():
    """Allocator calculates buy quantity based on rotations."""
    allocator = SlotAllocator(strategy="balanced")

    opportunities = [
        {"name": "Test", "current_price": 10_000, "buy_limit": 100, "score": 80},
    ]

    allocation = allocator.allocate(
        opportunities=opportunities,
        cash=10_000_000,
        slots=8,
        rotations=3,
    )

    # With 3 rotations and buy_limit=100, max qty = 300
    assert allocation[0]["quantity"] <= 300
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_allocator.py -v`
Expected: FAIL with "cannot import name 'SlotAllocator'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/allocator.py
"""Slot allocation engine."""
from typing import List, Dict, Any

# Strategy configurations
STRATEGIES = {
    "flip": {"flip_ratio": 0.7, "hold_ratio": 0.3},
    "hold": {"flip_ratio": 0.3, "hold_ratio": 0.7},
    "balanced": {"flip_ratio": 0.4, "hold_ratio": 0.6},
}


class SlotAllocator:
    """Allocates capital across GE slots."""

    def __init__(self, strategy: str = "balanced"):
        self.strategy = strategy
        self.config = STRATEGIES.get(strategy, STRATEGIES["balanced"])

    def allocate(
        self,
        opportunities: List[Dict[str, Any]],
        cash: int,
        slots: int,
        rotations: int,
    ) -> List[Dict[str, Any]]:
        """Allocate capital to opportunities.

        Args:
            opportunities: Scored opportunity list.
            cash: Total cash to allocate.
            slots: Available GE slots.
            rotations: Buy limit rotations.

        Returns:
            List of slot allocations.
        """
        if not opportunities:
            return []

        # Sort by score descending
        sorted_opps = sorted(
            opportunities,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )

        allocations = []
        remaining_cash = cash

        for opp in sorted_opps[:slots]:
            if remaining_cash <= 0:
                break

            price = opp["current_price"]
            buy_limit = opp.get("buy_limit") or 1

            # Max quantity based on rotations
            max_qty = buy_limit * rotations

            # Max capital for this slot (fair share of remaining)
            slots_remaining = slots - len(allocations)
            fair_share = remaining_cash // max(slots_remaining, 1)

            # Capital limited by buy limit
            max_capital_by_limit = max_qty * price

            # Actual capital for this slot
            slot_capital = min(fair_share, max_capital_by_limit, remaining_cash)

            # Calculate quantity
            quantity = slot_capital // price if price > 0 else 0
            quantity = min(quantity, max_qty)

            actual_capital = quantity * price

            if quantity > 0:
                allocations.append({
                    "slot": len(allocations) + 1,
                    "name": opp["name"],
                    "item_id": opp.get("item_id"),
                    "buy_price": price,
                    "quantity": quantity,
                    "capital": actual_capital,
                    "buy_limit": buy_limit,
                    "rotations": rotations,
                })

                remaining_cash -= actual_capital

        return allocations
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_allocator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/allocator.py tests/test_allocator.py
git commit -m "feat(allocator): add slot allocation engine"
```

---

## Phase 6: Portfolio System

### Task 18: Portfolio Presets ✅ DONE

**Files:**
- Create: `osrs_flipper/portfolio.py`
- Create: `tests/test_portfolio.py`

**Step 1: Write the failing test**

```python
# tests/test_portfolio.py
import pytest
from osrs_flipper.portfolio import PortfolioManager, PRESETS


def test_presets_exist():
    """Standard presets are defined."""
    assert "grinder" in PRESETS
    assert "balanced" in PRESETS
    assert "diamondhands" in PRESETS


def test_portfolio_manager_loads_preset():
    """Portfolio manager can load a preset."""
    manager = PortfolioManager()

    preset = manager.get_preset("grinder")

    assert preset is not None
    assert "flip_ratio" in preset
    assert "hold_ratio" in preset


def test_portfolio_manager_recommend():
    """Portfolio manager recommends based on opportunities."""
    manager = PortfolioManager()

    # Many oversold items = recommend balanced/hold
    opportunities = [
        {"oversold": {"is_oversold": True, "upside_pct": 40}}
        for _ in range(10)
    ]

    recommendation = manager.recommend(opportunities)

    assert recommendation in ["grinder", "balanced", "diamondhands"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_portfolio.py -v`
Expected: FAIL with "cannot import name 'PortfolioManager'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/portfolio.py
"""Portfolio preset management."""
from typing import Dict, Any, List, Optional

PRESETS = {
    "grinder": {
        "name": "Grinder",
        "description": "8 quick flips, ~2 day avg hold",
        "flip_ratio": 0.85,
        "hold_ratio": 0.15,
        "min_volume_mult": 2.0,
        "max_percentile": 25,
        "min_upside": 15,
    },
    "balanced": {
        "name": "Balanced",
        "description": "4 flip / 4 hold, ~5 day avg hold",
        "flip_ratio": 0.5,
        "hold_ratio": 0.5,
        "min_volume_mult": 1.0,
        "max_percentile": 20,
        "min_upside": 25,
    },
    "diamondhands": {
        "name": "Diamond Hands",
        "description": "8 mid-term holds, ~12 day avg hold",
        "flip_ratio": 0.15,
        "hold_ratio": 0.85,
        "min_volume_mult": 0.5,
        "max_percentile": 15,
        "min_upside": 35,
    },
}


class PortfolioManager:
    """Manages portfolio presets and recommendations."""

    def get_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a preset by name."""
        return PRESETS.get(name)

    def list_presets(self) -> Dict[str, Dict[str, Any]]:
        """List all available presets."""
        return PRESETS.copy()

    def recommend(self, opportunities: List[Dict[str, Any]]) -> str:
        """Recommend a preset based on market conditions.

        Args:
            opportunities: List of scanned opportunities.

        Returns:
            Recommended preset name.
        """
        if not opportunities:
            return "balanced"

        # Count deeply oversold items (high upside potential)
        deeply_oversold = sum(
            1 for opp in opportunities
            if opp.get("oversold", {}).get("upside_pct", 0) >= 40
        )

        # Count oscillator opportunities
        oscillators = sum(
            1 for opp in opportunities
            if opp.get("oscillator", {}).get("is_oscillator")
        )

        total = len(opportunities)

        # Recommend based on market composition
        if oscillators > total * 0.5:
            return "grinder"  # Many oscillators = quick flip opportunity
        elif deeply_oversold > total * 0.3:
            return "diamondhands"  # Many deep value plays
        else:
            return "balanced"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_portfolio.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/portfolio.py tests/test_portfolio.py
git commit -m "feat(portfolio): add preset management and recommendations"
```

---

### Task 18b: Pareto Frontier Optimization

**Files:**
- Modify: `osrs_flipper/portfolio.py`
- Modify: `tests/test_portfolio.py`

**Step 1: Write the failing test**

```python
# tests/test_portfolio.py (append)
from osrs_flipper.portfolio import ParetoFrontier


def test_pareto_frontier_calculates_roi_per_day():
    """Pareto frontier calculates time-adjusted ROI."""
    frontier = ParetoFrontier()

    items = [
        {"name": "Quick Flip", "roi_pct": 15, "expected_hold_days": 2},
        {"name": "Mid Hold", "roi_pct": 30, "expected_hold_days": 7},
        {"name": "Long Hold", "roi_pct": 50, "expected_hold_days": 14},
    ]

    scored = frontier.score_items(items)

    # Quick flip: 15% / 2 = 7.5% per day
    assert scored[0]["roi_per_day"] == 7.5
    # Long hold: 50% / 14 = 3.57% per day
    assert abs(scored[2]["roi_per_day"] - 3.57) < 0.1


def test_pareto_frontier_identifies_efficient_items():
    """Pareto frontier identifies non-dominated items."""
    frontier = ParetoFrontier()

    items = [
        {"name": "Efficient", "roi_pct": 30, "expected_hold_days": 5},    # On frontier
        {"name": "Dominated", "roi_pct": 20, "expected_hold_days": 7},    # Dominated
        {"name": "Also Efficient", "roi_pct": 50, "expected_hold_days": 12},  # On frontier
    ]

    efficient = frontier.get_efficient_items(items)

    efficient_names = [i["name"] for i in efficient]
    assert "Efficient" in efficient_names
    assert "Also Efficient" in efficient_names
    assert "Dominated" not in efficient_names


def test_pareto_frontier_portfolio_score():
    """Portfolio-level Pareto position calculated correctly."""
    frontier = ParetoFrontier()

    portfolio = [
        {"roi_pct": 20, "expected_hold_days": 3},
        {"roi_pct": 40, "expected_hold_days": 10},
    ]

    score = frontier.portfolio_score(portfolio)

    # Avg ROI: 30%, Avg hold: 6.5 days
    assert score["avg_roi_pct"] == 30
    assert score["avg_hold_days"] == 6.5
    assert "efficiency_score" in score
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_portfolio.py::test_pareto_frontier_calculates_roi_per_day -v`
Expected: FAIL with "cannot import name 'ParetoFrontier'"

**Step 3: Write minimal implementation**

```python
# osrs_flipper/portfolio.py (add class)
class ParetoFrontier:
    """Portfolio-level Pareto frontier for ROI vs hold time optimization."""

    def score_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add ROI per day scoring to items.

        Args:
            items: List of items with roi_pct and expected_hold_days.

        Returns:
            Items with roi_per_day added.
        """
        result = []
        for item in items:
            scored = item.copy()
            hold_days = item.get("expected_hold_days", 1)
            roi_pct = item.get("roi_pct", 0)
            scored["roi_per_day"] = round(roi_pct / hold_days, 2) if hold_days > 0 else 0
            result.append(scored)
        return result

    def get_efficient_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get Pareto-efficient items (non-dominated).

        An item is efficient if no other item has both:
        - Higher ROI AND shorter hold time

        Args:
            items: List of items with roi_pct and expected_hold_days.

        Returns:
            List of non-dominated items.
        """
        efficient = []
        for item in items:
            dominated = False
            for other in items:
                if other is item:
                    continue
                # Other dominates if it has >= ROI and <= hold time (with at least one strict)
                if (other["roi_pct"] >= item["roi_pct"] and
                    other["expected_hold_days"] <= item["expected_hold_days"] and
                    (other["roi_pct"] > item["roi_pct"] or
                     other["expected_hold_days"] < item["expected_hold_days"])):
                    dominated = True
                    break
            if not dominated:
                efficient.append(item)
        return efficient

    def portfolio_score(self, portfolio: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio-level Pareto position.

        Args:
            portfolio: List of items in portfolio.

        Returns:
            Portfolio metrics: avg_roi_pct, avg_hold_days, efficiency_score.
        """
        if not portfolio:
            return {"avg_roi_pct": 0, "avg_hold_days": 0, "efficiency_score": 0}

        avg_roi = sum(i.get("roi_pct", 0) for i in portfolio) / len(portfolio)
        avg_hold = sum(i.get("expected_hold_days", 1) for i in portfolio) / len(portfolio)

        # Efficiency score: ROI per day, higher is better
        efficiency = avg_roi / avg_hold if avg_hold > 0 else 0

        return {
            "avg_roi_pct": round(avg_roi, 1),
            "avg_hold_days": round(avg_hold, 1),
            "efficiency_score": round(efficiency, 2),
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_portfolio.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add osrs_flipper/portfolio.py tests/test_portfolio.py
git commit -m "feat(portfolio): add Pareto frontier optimization"
```

---

## Phase 7: Integration & Export

### Task 19: Wire Scanner to CLI

**Files:**
- Modify: `osrs_flipper/cli.py`

**Step 1: Update scan command implementation**

```python
# osrs_flipper/cli.py - replace scan command
import os
from datetime import datetime
from .api import OSRSClient
from .scanner import ItemScanner
from .allocator import SlotAllocator
from .scoring import calculate_item_score
from .exits import calculate_exit_strategies


@main.command()
@click.option("--mode", type=click.Choice(["oversold", "oscillator", "all"]), default="all")
@click.option("--cash", type=str, default=None)
@click.option("--slots", type=click.IntRange(1, 8), default=8)
@click.option("--rotations", type=int, default=3)
@click.option("--strategy", type=click.Choice(["flip", "hold", "balanced"]), default="balanced")
@click.option("--export", is_flag=True)
@click.option("--output-dir", type=click.Path(), default="./output")
def scan(mode, cash, slots, rotations, strategy, export, output_dir):
    """Scan for flip opportunities."""
    click.echo(f"OSRS Flip Scanner - {mode.upper()} mode")
    click.echo("=" * 60)

    # Initialize
    client = OSRSClient()
    scanner = ItemScanner(client)

    click.echo("Fetching data...")
    opportunities = scanner.scan(mode=mode)

    click.echo(f"Found {len(opportunities)} opportunities")

    if not opportunities:
        click.echo("No opportunities found matching criteria.")
        return

    # Add scores
    for opp in opportunities:
        oversold = opp.get("oversold", {})
        opp["score"] = calculate_item_score(
            upside_pct=oversold.get("upside_pct", 0),
            percentile=oversold.get("percentile", 50),
            volume_ratio=1.0,  # TODO: calculate actual ratio
            bounce_rate=0.5,   # TODO: calculate from history
        )

    # Sort by score
    opportunities.sort(key=lambda x: x["score"], reverse=True)

    if cash:
        # Allocation mode
        cash_gp = parse_cash(cash)
        click.echo(f"\nAllocating {cash_gp:,} GP across {slots} slots...")

        allocator = SlotAllocator(strategy=strategy)
        allocation = allocator.allocate(
            opportunities=opportunities,
            cash=cash_gp,
            slots=slots,
            rotations=rotations,
        )

        _print_allocation(allocation, strategy)

        if export:
            _export_allocation(allocation, output_dir)
    else:
        # List mode
        _print_opportunities(opportunities[:20])

        if export:
            _export_opportunities(opportunities, output_dir)


def _print_opportunities(opportunities):
    """Print opportunity list."""
    click.echo(f"\n{'Item':<30} {'Price':>10} {'%ile':>6} {'RSI':>5} {'Upside':>8} {'Score':>6}")
    click.echo("-" * 70)

    for opp in opportunities:
        oversold = opp.get("oversold", {})
        rsi = oversold.get("rsi")
        rsi_str = f"{rsi:.0f}" if rsi else "N/A"

        click.echo(
            f"{opp['name']:<30} "
            f"{opp['current_price']:>10,} "
            f"{oversold.get('percentile', 0):>5.1f}% "
            f"{rsi_str:>5} "
            f"{oversold.get('upside_pct', 0):>7.1f}% "
            f"{opp.get('score', 0):>6.1f}"
        )


def _print_allocation(allocation, strategy):
    """Print allocation table."""
    click.echo(f"\nAllocation ({strategy}):")
    click.echo(f"{'Slot':<5} {'Item':<25} {'Price':>10} {'Qty':>8} {'Capital':>12}")
    click.echo("-" * 65)

    total = 0
    for slot in allocation:
        click.echo(
            f"{slot['slot']:<5} "
            f"{slot['name']:<25} "
            f"{slot['buy_price']:>10,} "
            f"{slot['quantity']:>8,} "
            f"{slot['capital']:>12,}"
        )
        total += slot["capital"]

    click.echo("-" * 65)
    click.echo(f"{'Total':<5} {'':<25} {'':<10} {'':<8} {total:>12,}")


def _export_allocation(allocation, output_dir):
    """Export allocation to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filepath = os.path.join(output_dir, f"allocation_{timestamp}.csv")

    with open(filepath, "w") as f:
        f.write("slot,item,item_id,buy_price,quantity,capital,buy_limit,rotations\n")
        for slot in allocation:
            f.write(
                f"{slot['slot']},{slot['name']},{slot.get('item_id', '')},"
                f"{slot['buy_price']},{slot['quantity']},{slot['capital']},"
                f"{slot['buy_limit']},{slot['rotations']}\n"
            )

    click.echo(f"\nExported to: {filepath}")


def _export_opportunities(opportunities, output_dir):
    """Export opportunities to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filepath = os.path.join(output_dir, f"scan_{timestamp}.csv")

    with open(filepath, "w") as f:
        f.write("item,item_id,price,percentile,rsi,upside_pct,volume,score\n")
        for opp in opportunities:
            oversold = opp.get("oversold", {})
            f.write(
                f"{opp['name']},{opp['item_id']},{opp['current_price']},"
                f"{oversold.get('percentile', '')},"
                f"{oversold.get('rsi', '')},"
                f"{oversold.get('upside_pct', '')},"
                f"{opp['daily_volume']},"
                f"{opp.get('score', '')}\n"
            )

    click.echo(f"\nExported to: {filepath}")
```

**Step 2: Run full integration test**

```bash
pip install -e ".[dev]"
osrs-flip scan --mode oversold
```

**Step 3: Commit**

```bash
git add osrs_flipper/cli.py
git commit -m "feat(cli): wire scanner and allocator to CLI"
```

---

### Task 20: Wire Portfolio to CLI

**Files:**
- Modify: `osrs_flipper/cli.py`

**Step 1: Update portfolio command implementation**

```python
# osrs_flipper/cli.py - update portfolio command
from .portfolio import PortfolioManager, PRESETS


@main.command()
@click.option("--list", "list_presets", is_flag=True)
@click.option("--use", type=str, default=None)
@click.option("--recommend", is_flag=True)
@click.option("--cash", type=str, default=None)
@click.option("--slots", type=click.IntRange(1, 8), default=8)
@click.option("--save", type=str, default=None)
def portfolio(list_presets, use, recommend, cash, slots, save):
    """Manage portfolio presets and recommendations."""
    manager = PortfolioManager()

    if list_presets:
        click.echo("Available presets:")
        click.echo("-" * 50)
        for name, preset in PRESETS.items():
            click.echo(f"  {name:<15} - {preset['description']}")
        return

    if recommend:
        if not cash:
            raise click.UsageError("--cash required with --recommend")

        click.echo("Analyzing market conditions...")

        # Scan for opportunities
        client = OSRSClient()
        scanner = ItemScanner(client)
        opportunities = scanner.scan(mode="all")

        recommendation = manager.recommend(opportunities)
        preset = manager.get_preset(recommendation)

        click.echo(f"\nRecommendation: {recommendation}")
        click.echo(f"  {preset['description']}")
        click.echo(f"\nRun: osrs-flip scan --cash {cash} --strategy {recommendation}")
        return

    if use:
        if not cash:
            raise click.UsageError("--cash required with --use")

        preset = manager.get_preset(use)
        if not preset:
            raise click.BadParameter(f"Unknown preset: {use}")

        click.echo(f"Loading preset: {use}")
        click.echo(f"  {preset['description']}")

        # Map preset to strategy
        strategy = "balanced"
        if preset["flip_ratio"] > 0.6:
            strategy = "flip"
        elif preset["hold_ratio"] > 0.6:
            strategy = "hold"

        # Run scan with preset parameters
        ctx = click.get_current_context()
        ctx.invoke(
            scan,
            mode="all",
            cash=cash,
            slots=slots,
            strategy=strategy,
        )
        return

    click.echo("Use --list, --use <preset>, or --recommend")
```

**Step 2: Test portfolio command**

```bash
osrs-flip portfolio --list
osrs-flip portfolio --recommend --cash 120m
```

**Step 3: Commit**

```bash
git add osrs_flipper/cli.py
git commit -m "feat(cli): wire portfolio manager to CLI"
```

---

## Phase 8: Polish

### Task 21: Add Rate Limiting ✅ DONE

**Files:**
- Modify: `osrs_flipper/api.py`

**Step 1: Add rate limiting to API client**

```python
# osrs_flipper/api.py - add to OSRSClient
import time

class OSRSClient:
    def __init__(self, base_url: str = BASE_URL, rate_limit: float = 0.1):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.rate_limit = rate_limit
        self._last_request = 0

    def _rate_limited_get(self, url: str, **kwargs):
        """Make rate-limited GET request."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        self._last_request = time.time()
        return self.session.get(url, **kwargs)

    # Update all methods to use _rate_limited_get instead of self.session.get
```

**Step 2: Commit**

```bash
git add osrs_flipper/api.py
git commit -m "feat(api): add rate limiting"
```

---

### Task 22: Add Progress Indicator

**Files:**
- Modify: `osrs_flipper/scanner.py`

**Step 1: Add progress callback to scanner**

```python
# osrs_flipper/scanner.py - update scan method
from typing import Callable, Optional

def scan(
    self,
    mode: str = "all",
    limit: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    # ... existing code ...

    for i, item_id in enumerate(item_ids):
        if progress_callback:
            progress_callback(i + 1, len(item_ids))

        # ... rest of loop ...
```

**Step 2: Use click progress bar in CLI**

```python
# osrs_flipper/cli.py - in scan command
with click.progressbar(length=100, label="Scanning") as bar:
    def update_progress(current, total):
        bar.update(int((current / total) * 100) - bar.pos)

    opportunities = scanner.scan(mode=mode, progress_callback=update_progress)
```

**Step 3: Commit**

```bash
git add osrs_flipper/scanner.py osrs_flipper/cli.py
git commit -m "feat: add progress indicator to scanner"
```

---

### Task 23: Final Integration Test

**Step 1: Run full end-to-end test**

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Test CLI commands
osrs-flip --help
osrs-flip scan --mode oversold
osrs-flip scan --mode oscillator
osrs-flip scan --cash 120m --slots 5 --strategy balanced
osrs-flip scan --cash 50m --export --output-dir ./test-output
osrs-flip portfolio --list
osrs-flip portfolio --recommend --cash 120m
```

**Step 2: Verify output**

- Scan shows ranked opportunities
- Allocation mode shows slot-by-slot breakdown
- Export creates CSV files
- Portfolio presets work

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete OSRS flip scanner v0.1.0"
```

---

## Summary

**Total Tasks:** 23
**Estimated Implementation Time:** 4-6 hours with TDD

**Key Files:**
- `osrs_flipper/api.py` - OSRS Wiki API client
- `osrs_flipper/filters.py` - Volume tier filtering
- `osrs_flipper/indicators.py` - RSI, percentile calculations
- `osrs_flipper/analyzers.py` - Oversold + oscillator detection
- `osrs_flipper/scanner.py` - Item scanning service
- `osrs_flipper/scoring.py` - EV/ROI calculations
- `osrs_flipper/exits.py` - Exit strategy levels
- `osrs_flipper/allocator.py` - Slot allocation engine
- `osrs_flipper/portfolio.py` - Preset management
- `osrs_flipper/cli.py` - Click CLI interface
- `osrs_flipper/utils.py` - Cash parsing utilities

**CLI Commands:**
```bash
osrs-flip scan [--mode] [--cash] [--slots] [--rotations] [--strategy] [--export] [--output-dir]
osrs-flip portfolio [--list] [--use] [--recommend] [--cash] [--slots] [--save]
```
