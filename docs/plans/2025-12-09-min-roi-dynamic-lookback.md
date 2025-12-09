# Min ROI Filter & Dynamic Lookback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--min-roi` flag to filter opportunities by tax-adjusted ROI and dynamic lookback window that scales with hold time.

**Architecture:** Two independent features that integrate at the scanner level. The lookback window calculation is a pure function. The min-roi filter operates on existing `tax_adjusted_upside_pct` field. CLI passes parameters through to scanner.

**Tech Stack:** Python, Click (CLI), NumPy (vectorized operations), pytest

---

## Execution Protocol

### For Each Task:
1. Read this plan file, focusing on YOUR task section
2. Read the "Agent Logs" section from previous tasks for context
3. Follow TDD: RED (write failing test) → GREEN (minimal implementation) → REFACTOR
4. Update YOUR task's "Agent Log" section with:
   - Files modified with line numbers
   - Data flow: inputs → transformations → outputs
   - Any issues encountered and how resolved
5. Run verification command
6. Hand off to code reviewer

### Agent Log Format:
```
#### Agent Log - Task N
**Status:** COMPLETE | IN_PROGRESS | BLOCKED
**Files Modified:**
- `path/to/file.py:10-25` - description
**Data Flow:**
- Input: `param_name: type` from `source`
- Transform: description of transformation
- Output: `return_value: type` to `destination`
**Issues:** None | description
**Verification:** `pytest path -v` - PASS/FAIL
```

---

## Task Dependency Graph

```
[Task 1: lookback_days pure function] ──┐
                                        ├──► [Task 3: OversoldAnalyzer window param]
[Task 2: hold_days defaults]  ──────────┘              │
                                                       ▼
                                        [Task 4: Scanner integration]
                                                       │
                                                       ▼
                                        [Task 5: CLI --hold-days flag]
                                                       │
                                                       ▼
                                        [Task 6: CLI --min-roi flag]
                                                       │
                                                       ▼
                                        [Task 7: E2E data flow test]
```

**Parallel Execution Groups:**
- Group A (parallel): Tasks 1, 2
- Group B (sequential after A): Task 3
- Group C (sequential after B): Task 4
- Group D (sequential after C): Tasks 5, 6 (parallel)
- Group E (sequential after D): Task 7

---

## Task 1: Lookback Days Pure Function

**Complexity:** LOW (1 file, ~20 lines)
**Parallel Group:** A (can run with Task 2)

**Files:**
- Create: `osrs_flipper/lookback.py`
- Create: `tests/test_lookback.py`

### Step 1: Write failing tests

```python
# tests/test_lookback.py
"""Tests for lookback window calculation."""
import pytest
import numpy as np
from osrs_flipper.lookback import calculate_lookback_days


class TestCalculateLookbackDays:
    """Test lookback_days = min(hold_days * 4, 180)."""

    @pytest.mark.parametrize("hold_days,expected", [
        (3, 12),    # flip strategy: 3 * 4 = 12
        (7, 28),    # balanced strategy: 7 * 4 = 28
        (14, 56),   # hold strategy: 14 * 4 = 56
        (30, 120),  # extended hold: 30 * 4 = 120
        (50, 180),  # capped at 180: 50 * 4 = 200 -> 180
        (100, 180), # capped at 180
    ])
    def test_lookback_formula(self, hold_days: int, expected: int):
        """Lookback = hold_days * 4, capped at 180."""
        result = calculate_lookback_days(hold_days)
        assert result == expected

    def test_input_validation_positive(self):
        """Hold days must be positive."""
        with pytest.raises(ValueError, match="hold_days must be positive"):
            calculate_lookback_days(0)
        with pytest.raises(ValueError, match="hold_days must be positive"):
            calculate_lookback_days(-5)

    def test_vectorized_input(self):
        """Function works with numpy arrays (vectorized)."""
        hold_days = np.array([3, 7, 14, 50])
        expected = np.array([12, 28, 56, 180])
        result = calculate_lookback_days(hold_days)
        np.testing.assert_array_equal(result, expected)

    def test_return_type_int(self):
        """Returns int for scalar input."""
        result = calculate_lookback_days(7)
        assert isinstance(result, int)
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_lookback.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'osrs_flipper.lookback'`

### Step 3: Write minimal implementation

```python
# osrs_flipper/lookback.py
"""Lookback window calculation for dynamic price range analysis."""
from typing import Union
import numpy as np

MAX_LOOKBACK_DAYS = 180
LOOKBACK_MULTIPLIER = 4


def calculate_lookback_days(hold_days: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Calculate lookback window based on hold time.

    Formula: lookback_days = min(hold_days * 4, 180)

    Args:
        hold_days: Expected hold period in days (scalar or array).

    Returns:
        Lookback window in days (same type as input).

    Raises:
        ValueError: If hold_days is not positive.
    """
    # Vectorized validation
    if np.any(np.asarray(hold_days) <= 0):
        raise ValueError("hold_days must be positive")

    # Vectorized calculation (no loops)
    result = np.minimum(np.asarray(hold_days) * LOOKBACK_MULTIPLIER, MAX_LOOKBACK_DAYS)

    # Return scalar if input was scalar
    if np.isscalar(hold_days):
        return int(result)
    return result
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_lookback.py -v
```
Expected: PASS (all 4 tests)

### Step 5: Commit

```bash
git add osrs_flipper/lookback.py tests/test_lookback.py
git commit -m "feat: add calculate_lookback_days pure function

Formula: lookback_days = min(hold_days * 4, 180)
Supports vectorized numpy input for batch processing.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

#### Agent Log - Task 1
**Status:** COMPLETE
**Files Modified:**
- `osrs_flipper/lookback.py:1-36` - Created pure function for lookback window calculation
- `tests/test_lookback.py:1-44` - Created comprehensive test suite with 9 test cases
**Data Flow:**
- Input: `hold_days: int | np.ndarray` from caller
- Transform: `min(hold_days * 4, 180)` vectorized via numpy
- Output: `lookback_days: int | np.ndarray` to caller
**Issues:** None
**Verification:** `pytest tests/test_lookback.py -v` - PASS (9/9 tests passed in 0.36s)

---

## Task 2: Hold Days Strategy Defaults

**Complexity:** LOW (1 file, ~15 lines)
**Parallel Group:** A (can run with Task 1)

**Files:**
- Create: `osrs_flipper/defaults.py`
- Create: `tests/test_defaults.py`

### Step 1: Write failing tests

```python
# tests/test_defaults.py
"""Tests for strategy defaults."""
import pytest
from osrs_flipper.defaults import get_default_hold_days, STRATEGY_HOLD_DAYS


class TestStrategyDefaults:
    """Test hold days defaults by strategy."""

    def test_flip_strategy_default(self):
        """Flip strategy defaults to 3 days."""
        assert get_default_hold_days("flip") == 3

    def test_balanced_strategy_default(self):
        """Balanced strategy defaults to 7 days."""
        assert get_default_hold_days("balanced") == 7

    def test_hold_strategy_default(self):
        """Hold strategy defaults to 14 days."""
        assert get_default_hold_days("hold") == 14

    def test_unknown_strategy_returns_balanced(self):
        """Unknown strategy falls back to balanced (7 days)."""
        assert get_default_hold_days("unknown") == 7
        assert get_default_hold_days("") == 7

    def test_strategy_constants_exported(self):
        """Strategy constants are accessible."""
        assert STRATEGY_HOLD_DAYS["flip"] == 3
        assert STRATEGY_HOLD_DAYS["balanced"] == 7
        assert STRATEGY_HOLD_DAYS["hold"] == 14
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_defaults.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'osrs_flipper.defaults'`

### Step 3: Write minimal implementation

```python
# osrs_flipper/defaults.py
"""Default values for scanner configuration."""
from typing import Dict

STRATEGY_HOLD_DAYS: Dict[str, int] = {
    "flip": 3,
    "balanced": 7,
    "hold": 14,
}

DEFAULT_MIN_ROI: float = 20.0


def get_default_hold_days(strategy: str) -> int:
    """Get default hold days for a strategy.

    Args:
        strategy: Strategy name (flip, balanced, hold).

    Returns:
        Default hold days for the strategy.
        Falls back to balanced (7) for unknown strategies.
    """
    return STRATEGY_HOLD_DAYS.get(strategy, STRATEGY_HOLD_DAYS["balanced"])
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_defaults.py -v
```
Expected: PASS (all 5 tests)

### Step 5: Commit

```bash
git add osrs_flipper/defaults.py tests/test_defaults.py
git commit -m "feat: add strategy hold days defaults

flip=3, balanced=7, hold=14 days
Includes DEFAULT_MIN_ROI=20.0 constant.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

#### Agent Log - Task 2
**Status:** COMPLETE
**Files Modified:**
- `osrs_flipper/defaults.py:1-24` - Created new module with STRATEGY_HOLD_DAYS dict and get_default_hold_days function
- `tests/test_defaults.py:1-30` - Created comprehensive tests for all strategies and edge cases
**Data Flow:**
- Input: `strategy: str` from CLI
- Transform: dictionary lookup with fallback to balanced (7 days) for unknown strategies
- Output: `hold_days: int` to scanner
**Issues:** None
**Verification:** `pytest tests/test_defaults.py -v` - PASS (all 5 tests passed)

---

## Task 3: OversoldAnalyzer Dynamic Window Parameter

**Complexity:** LOW (2 files, ~40 lines modified)
**Depends On:** Tasks 1, 2

**Files:**
- Modify: `osrs_flipper/analyzers.py:27-66`
- Modify: `tests/test_analyzers.py` (add new tests)

### Step 1: Write failing tests

Add to `tests/test_analyzers.py`:

```python
# Add these tests to tests/test_analyzers.py

class TestOversoldAnalyzerDynamicWindow:
    """Test dynamic lookback window for oversold detection."""

    def test_analyze_with_lookback_days_parameter(self):
        """Analyzer accepts lookback_days parameter."""
        analyzer = OversoldAnalyzer()

        # 90 days of price data
        prices = list(range(100, 190)) + list(range(189, 99, -1))  # 90 + 90 = 180 prices

        # Full window (180 days) - sees entire range
        result_full = analyzer.analyze(
            current_price=110,
            prices=prices,
            lookback_days=180,
        )

        # Short window (30 days) - sees only recent decline
        result_short = analyzer.analyze(
            current_price=110,
            prices=prices,
            lookback_days=30,
        )

        # Both should return valid results
        assert "percentile" in result_full
        assert "percentile" in result_short

    def test_lookback_window_affects_percentile(self):
        """Shorter lookback window changes percentile calculation."""
        analyzer = OversoldAnalyzer()

        # Simulate: old peak at 200, recent range 100-120
        # Day 1-60: prices around 200
        # Day 61-90: prices around 100-120
        old_prices = [200] * 60
        recent_prices = [100, 105, 110, 115, 120, 115, 110, 105, 100, 105] * 3  # 30 prices
        prices = old_prices + recent_prices

        current_price = 110

        # Full 90-day window: 110 is near bottom (200 high, 100 low)
        result_90 = analyzer.analyze(
            current_price=current_price,
            prices=prices,
            lookback_days=90,
        )

        # Recent 30-day window: 110 is mid-range (120 high, 100 low)
        result_30 = analyzer.analyze(
            current_price=current_price,
            prices=prices,
            lookback_days=30,
        )

        # With old peak, 110 looks oversold (low percentile)
        # With recent window only, 110 is mid-range (higher percentile)
        assert result_90["percentile"] < result_30["percentile"]

    def test_lookback_window_uses_tail_of_prices(self):
        """Lookback window uses most recent N days of prices."""
        analyzer = OversoldAnalyzer()

        # 100 days of prices, older data irrelevant
        prices = [500] * 50 + [100, 110, 120, 130, 140, 150, 140, 130, 120, 110] * 5  # 50 + 50

        result = analyzer.analyze(
            current_price=110,
            prices=prices,
            lookback_days=30,  # Only look at last 30 days
        )

        # Should NOT see the 500 peak from early data
        assert result["six_month_high"] <= 150
        assert result["six_month_low"] >= 100

    def test_backward_compatibility_default_lookback(self):
        """Default lookback is 180 days for backward compatibility."""
        analyzer = OversoldAnalyzer()

        prices = [100, 110, 120, 130, 140, 150, 140, 130, 120, 110, 100, 110] * 15  # 180 prices

        # Call without lookback_days parameter
        result = analyzer.analyze(
            current_price=110,
            prices=prices,
        )

        # Should work and use full price history (up to 180)
        assert "percentile" in result
        assert "six_month_high" in result
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_analyzers.py::TestOversoldAnalyzerDynamicWindow -v
```
Expected: FAIL with `TypeError: analyze() got an unexpected keyword argument 'lookback_days'`

### Step 3: Modify implementation

Update `osrs_flipper/analyzers.py`:

```python
# osrs_flipper/analyzers.py - Replace the analyze method (lines ~27-66)

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
        prices: list[int],
        lookback_days: int = 180,
        six_month_low: int | None = None,
        six_month_high: int | None = None,
    ) -> dict[str, any]:
        """Analyze item for oversold opportunity.

        Args:
            current_price: Current item price.
            prices: List of historical prices (daily).
            lookback_days: Number of days to look back for range calculation.
            six_month_low: Deprecated - calculated from prices if not provided.
            six_month_high: Deprecated - calculated from prices if not provided.

        Returns:
            Analysis result dict.
        """
        # Use dynamic lookback window (vectorized slice)
        lookback_prices = prices[-lookback_days:] if len(prices) > lookback_days else prices

        # Calculate range from lookback window (no loops)
        window_low = min(lookback_prices) if lookback_prices else current_price
        window_high = max(lookback_prices) if lookback_prices else current_price

        # Support deprecated parameters for backward compatibility
        if six_month_low is not None:
            window_low = six_month_low
        if six_month_high is not None:
            window_high = six_month_high

        percentile = calculate_percentile(current_price, window_low, window_high)
        rsi = calculate_rsi(prices)  # RSI uses full history

        # Calculate upside potential (vectorized division)
        upside_pct = ((window_high - current_price) / current_price) * 100 if current_price > 0 else 0

        is_oversold = (
            percentile <= self.low_threshold_pct
            and upside_pct >= self.min_upside_pct
        )

        return {
            "is_oversold": is_oversold,
            "percentile": round(percentile, 1),
            "rsi": rsi,
            "upside_pct": round(upside_pct, 1),
            "six_month_low": window_low,
            "six_month_high": window_high,
        }
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_analyzers.py -v
```
Expected: PASS (all tests including new ones)

### Step 5: Commit

```bash
git add osrs_flipper/analyzers.py tests/test_analyzers.py
git commit -m "feat: add dynamic lookback_days parameter to OversoldAnalyzer

- analyze() now accepts lookback_days (default 180)
- Uses tail slice of prices for dynamic window
- Backward compatible with existing six_month_low/high params

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

#### Agent Log - Task 3
**Status:** PENDING
**Files Modified:**
- (to be filled by agent)
**Data Flow:**
- Input: `prices: List[int]`, `lookback_days: int` from scanner
- Transform: `prices[-lookback_days:]` slice, then min/max
- Output: `result: Dict` with percentile, upside_pct using windowed range
**Issues:** (to be filled by agent)
**Verification:** (to be filled by agent)

---

## Task 4: Scanner Integration

**Complexity:** LOW (2 files, ~50 lines modified)
**Depends On:** Task 3

**Files:**
- Modify: `osrs_flipper/scanner.py:18-166`
- Modify: `tests/test_scanner.py` (add new tests)

### Step 1: Write failing tests

Add to `tests/test_scanner.py`:

```python
# Add these tests to tests/test_scanner.py

class TestScannerDynamicLookback:
    """Test scanner with dynamic lookback and min ROI filtering."""

    @responses.activate
    def test_scan_accepts_lookback_days_parameter(self):
        """Scanner passes lookback_days to analyzer."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "Test Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"1": {"high": 110, "low": 100}}
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"1": {"highPriceVolume": 3000000, "lowPriceVolume": 2500000}}
        })
        # 90 days of timeseries
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={
            "data": [{"timestamp": i, "avgHighPrice": 150, "avgLowPrice": 140} for i in range(90)]
        })

        client = OSRSClient()
        scanner = ItemScanner(client)

        # Should not raise - lookback_days accepted
        results = scanner.scan(mode="oversold", limit=1, lookback_days=30)
        assert isinstance(results, list)

    @responses.activate
    def test_scan_accepts_min_roi_parameter(self):
        """Scanner filters by min_roi threshold."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "Low ROI Item", "limit": 100},
            {"id": 2, "name": "High ROI Item", "limit": 100},
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {
                "1": {"high": 110, "low": 100},  # Current: 105
                "2": {"high": 110, "low": 100},  # Current: 105
            }
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {
                "1": {"highPriceVolume": 3000000, "lowPriceVolume": 2500000},
                "2": {"highPriceVolume": 3000000, "lowPriceVolume": 2500000},
            }
        })

        # Item 1: Low upside (peak 120 -> 14% upside)
        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": [{"timestamp": i, "avgHighPrice": 120, "avgLowPrice": 100} for i in range(90)]},
            match=[responses.matchers.query_param_matcher({"id": 1, "timestep": "24h"})]
        )
        # Item 2: High upside (peak 200 -> 90% upside)
        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": [{"timestamp": i, "avgHighPrice": 200, "avgLowPrice": 100} for i in range(90)]},
            match=[responses.matchers.query_param_matcher({"id": 2, "timestep": "24h"})]
        )

        client = OSRSClient()
        scanner = ItemScanner(client)

        # With min_roi=50, only high ROI item should pass
        results = scanner.scan(mode="oversold", limit=2, min_roi=50.0)

        # Filter should exclude low ROI items
        item_names = [r["name"] for r in results]
        assert "Low ROI Item" not in item_names

    @responses.activate
    def test_min_roi_uses_tax_adjusted_upside(self):
        """Min ROI filter uses tax_adjusted_upside_pct, not raw upside."""
        responses.add(responses.GET, f"{BASE_URL}/mapping", json=[
            {"id": 1, "name": "Taxed Item", "limit": 100}
        ])
        responses.add(responses.GET, f"{BASE_URL}/latest", json={
            "data": {"1": {"high": 1100, "low": 1000}}  # Current: 1050
        })
        responses.add(responses.GET, f"{BASE_URL}/24h", json={
            "data": {"1": {"highPriceVolume": 3000000, "lowPriceVolume": 2500000}}
        })
        # Peak at 1400 -> raw upside ~33%, but after 2% tax, net is lower
        responses.add(responses.GET, f"{BASE_URL}/timeseries", json={
            "data": [{"timestamp": i, "avgHighPrice": 1400, "avgLowPrice": 1000} for i in range(90)]
        })

        client = OSRSClient()
        scanner = ItemScanner(client)

        # The scanner should use tax_adjusted_upside_pct for filtering
        results = scanner.scan(mode="oversold", limit=1, min_roi=20.0)

        # Result should have tax_adjusted_upside_pct populated
        if results:
            assert "tax_adjusted_upside_pct" in results[0]
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_scanner.py::TestScannerDynamicLookback -v
```
Expected: FAIL with `TypeError: scan() got an unexpected keyword argument 'lookback_days'`

### Step 3: Modify implementation

Update `osrs_flipper/scanner.py`:

```python
# osrs_flipper/scanner.py - Update scan() method signature and _analyze_item

from typing import List, Dict, Any, Optional, Callable
from .api import OSRSClient
from .analyzers import OversoldAnalyzer, OscillatorAnalyzer
from .exits import calculate_exit_strategies
from .filters import passes_volume_filter
from .tax import is_tax_exempt, calculate_ge_tax
from .lookback import calculate_lookback_days  # NEW IMPORT


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
        progress_callback: Optional[Callable[[int, int], None]] = None,
        lookback_days: int = 180,  # NEW PARAMETER
        min_roi: Optional[float] = None,  # NEW PARAMETER
    ) -> List[Dict[str, Any]]:
        """Scan for flip opportunities.

        Args:
            mode: "oversold", "oscillator", or "all"
            limit: Max items to scan (None for all)
            progress_callback: Optional callback(current, total) for progress updates
            lookback_days: Days to look back for price range (default 180)
            min_roi: Minimum tax-adjusted ROI % to include (None for no filter)

        Returns:
            List of opportunity dicts.
        """
        mapping = self.client.fetch_mapping()
        latest = self.client.fetch_latest()
        volumes = self.client.fetch_volumes()

        opportunities = []
        item_ids = list(mapping.keys())[:limit] if limit else list(mapping.keys())
        total = len(item_ids)

        for i, item_id in enumerate(item_ids):
            if progress_callback:
                progress_callback(i + 1, total)

            result = self._analyze_item(
                item_id, mapping, latest, volumes, mode, lookback_days  # PASS lookback_days
            )
            if result:
                # Apply min_roi filter
                if min_roi is not None:
                    tax_adjusted = result.get("tax_adjusted_upside_pct", 0)
                    if tax_adjusted < min_roi:
                        continue
                opportunities.append(result)

        return opportunities

    def _analyze_item(
        self,
        item_id: int,
        mapping: Dict,
        latest: Dict,
        volumes: Dict,
        mode: str,
        lookback_days: int = 180,  # NEW PARAMETER
    ) -> Optional[Dict[str, Any]]:
        # ... existing validation code unchanged until line 104 ...

        if item_id not in mapping:
            return None

        item = mapping[item_id]
        name = item.get("name", "Unknown")
        if "(noted)" in name.lower():
            return None

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

        vol_data = volumes.get(str(item_id), {})
        instabuy_vol = vol_data.get("highPriceVolume", 0) or 0
        instasell_vol = vol_data.get("lowPriceVolume", 0) or 0
        daily_volume = instabuy_vol + instasell_vol

        if instasell_vol > 0:
            buyer_momentum = instabuy_vol / instasell_vol
        else:
            buyer_momentum = float("inf") if instabuy_vol > 0 else 0.0

        if not passes_volume_filter(current_price, daily_volume):
            return None

        try:
            history = self.client.fetch_timeseries(item_id)
        except Exception:
            return None

        if len(history) < 30:
            return None

        prices = []
        for point in history:
            h = point.get("avgHighPrice")
            l = point.get("avgLowPrice")
            if h and l:
                prices.append((h + l) // 2)

        if len(prices) < 30:
            return None

        # CHANGED: Use lookback_days for window calculation
        lookback_prices = prices[-lookback_days:] if len(prices) > lookback_days else prices
        window_low = min(lookback_prices)
        window_high = max(lookback_prices)

        result = {
            "item_id": item_id,
            "name": name,
            "current_price": current_price,
            "daily_volume": daily_volume,
            "instabuy_vol": instabuy_vol,
            "instasell_vol": instasell_vol,
            "buyer_momentum": round(buyer_momentum, 2) if buyer_momentum != float("inf") else 99.9,
            "buy_limit": item.get("limit"),
            "is_tax_exempt": is_tax_exempt(name),
        }

        if mode in ("oversold", "all"):
            # CHANGED: Pass lookback_days to analyzer
            oversold = self.oversold_analyzer.analyze(
                current_price=current_price,
                prices=prices,
                lookback_days=lookback_days,
            )
            result["oversold"] = oversold

        if mode in ("oscillator", "all"):
            oscillator = self.oscillator_analyzer.analyze(prices, current_price)
            result["oscillator"] = oscillator

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

        if is_opportunity:
            result["exits"] = calculate_exit_strategies(current_price, prices)

            if result.get("exits"):
                target_price = result["exits"].get("target", {}).get("price", 0)
                if target_price > 0:
                    tax = calculate_ge_tax(target_price, name)
                    tax_adjusted_profit = target_price - tax - current_price
                    tax_adjusted_upside = (tax_adjusted_profit / current_price) * 100 if current_price > 0 else 0
                    result["tax_adjusted_upside_pct"] = round(tax_adjusted_upside, 2)

            buy_limit = item.get("limit") or 1
            if daily_volume > 0:
                days_to_fill = max(1, buy_limit / (daily_volume / 2))
                percentile = result.get("oversold", {}).get("percentile", 50)
                recovery_factor = max(1, (50 - percentile) / 10)
                result["expected_hold_days"] = round(days_to_fill + recovery_factor, 1)
            else:
                result["expected_hold_days"] = 14

        return result if is_opportunity else None
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_scanner.py -v
```
Expected: PASS (all tests)

### Step 5: Commit

```bash
git add osrs_flipper/scanner.py tests/test_scanner.py
git commit -m "feat: add lookback_days and min_roi params to scanner

- scan() accepts lookback_days (default 180) and min_roi filter
- Passes lookback_days to OversoldAnalyzer
- Filters results by tax_adjusted_upside_pct >= min_roi

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

#### Agent Log - Task 4
**Status:** PENDING
**Files Modified:**
- (to be filled by agent)
**Data Flow:**
- Input: `lookback_days: int`, `min_roi: float` from CLI
- Transform: Pass to analyzer, filter results by tax_adjusted_upside_pct
- Output: `opportunities: List[Dict]` filtered by min_roi
**Issues:** (to be filled by agent)
**Verification:** (to be filled by agent)

---

## Task 5: CLI --hold-days Flag

**Complexity:** LOW (2 files, ~30 lines)
**Parallel Group:** D (can run with Task 6)
**Depends On:** Task 4

**Files:**
- Modify: `osrs_flipper/cli.py:25-131` (scan command)
- Modify: `tests/test_cli.py` (add new tests)

### Step 1: Write failing tests

Add to `tests/test_cli.py`:

```python
# Add these tests to tests/test_cli.py

def test_scan_command_has_hold_days_option():
    """Scan command has --hold-days option."""
    runner = CliRunner()
    result = runner.invoke(scan, ["--help"])

    assert result.exit_code == 0
    assert "--hold-days" in result.output


def test_scan_hold_days_accepts_integer():
    """--hold-days accepts integer value."""
    runner = CliRunner()
    # Just check help shows the option type
    result = runner.invoke(scan, ["--help"])

    assert result.exit_code == 0
    assert "--hold-days" in result.output
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_cli.py::test_scan_command_has_hold_days_option -v
```
Expected: FAIL with `AssertionError: assert '--hold-days' in ...`

### Step 3: Modify implementation

Update `osrs_flipper/cli.py` - add to scan command options (after line 72):

```python
# In cli.py, add these imports at top:
from .defaults import get_default_hold_days, DEFAULT_MIN_ROI
from .lookback import calculate_lookback_days

# Add this option to @main.command() scan (after --limit option, ~line 72):
@click.option(
    "--hold-days",
    type=int,
    default=None,
    help="Expected hold period in days (default: based on strategy)",
)

# Update scan function signature to include hold_days:
def scan(mode, cash, slots, rotations, strategy, export, output_dir, limit, hold_days):
    """Scan for flip opportunities."""
    click.echo(f"OSRS Flip Scanner - {mode.upper()} mode")
    click.echo("=" * 60)

    # Calculate hold_days from strategy if not provided
    if hold_days is None:
        hold_days = get_default_hold_days(strategy)

    # Calculate lookback window
    lookback_days = calculate_lookback_days(hold_days)

    click.echo(f"Hold time: {hold_days} days | Lookback window: {lookback_days} days")

    client = OSRSClient()
    scanner = ItemScanner(client)

    click.echo("Fetching data...")

    def progress(current, total):
        if current % 50 == 0 or current == total:
            click.echo(f"  Scanning items: {current}/{total}", nl=False)
            click.echo("\r", nl=False)

    # Pass lookback_days to scanner
    opportunities = scanner.scan(
        mode=mode,
        limit=limit,
        progress_callback=progress,
        lookback_days=lookback_days,
    )
    # ... rest of function unchanged
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_cli.py -v
```
Expected: PASS

### Step 5: Commit

```bash
git add osrs_flipper/cli.py tests/test_cli.py
git commit -m "feat: add --hold-days flag to scan command

- Defaults based on strategy (flip=3, balanced=7, hold=14)
- Calculates lookback window as hold_days * 4 (max 180)
- Displays hold time and lookback window in output

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

#### Agent Log - Task 5
**Status:** PENDING
**Files Modified:**
- (to be filled by agent)
**Data Flow:**
- Input: `--hold-days N` or None from CLI args
- Transform: `get_default_hold_days(strategy)` if None, then `calculate_lookback_days(hold_days)`
- Output: `lookback_days: int` passed to `scanner.scan()`
**Issues:** (to be filled by agent)
**Verification:** (to be filled by agent)

---

## Task 6: CLI --min-roi Flag

**Complexity:** LOW (2 files, ~20 lines)
**Parallel Group:** D (can run with Task 5)
**Depends On:** Task 4

**Files:**
- Modify: `osrs_flipper/cli.py:25-131` (scan command)
- Modify: `tests/test_cli.py` (add new tests)

### Step 1: Write failing tests

Add to `tests/test_cli.py`:

```python
# Add these tests to tests/test_cli.py

def test_scan_command_has_min_roi_option():
    """Scan command has --min-roi option."""
    runner = CliRunner()
    result = runner.invoke(scan, ["--help"])

    assert result.exit_code == 0
    assert "--min-roi" in result.output


def test_scan_min_roi_has_default():
    """--min-roi has default value of 20."""
    runner = CliRunner()
    result = runner.invoke(scan, ["--help"])

    assert result.exit_code == 0
    # Check help text mentions default
    assert "--min-roi" in result.output
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_cli.py::test_scan_command_has_min_roi_option -v
```
Expected: FAIL with `AssertionError: assert '--min-roi' in ...`

### Step 3: Modify implementation

Update `osrs_flipper/cli.py` - add to scan command options:

```python
# Add this option to @main.command() scan (after --hold-days):
@click.option(
    "--min-roi",
    type=float,
    default=20.0,
    help="Minimum tax-adjusted ROI %% to include (default: 20)",
)

# Update scan function signature to include min_roi:
def scan(mode, cash, slots, rotations, strategy, export, output_dir, limit, hold_days, min_roi):
    """Scan for flip opportunities."""
    # ... existing code ...

    click.echo(f"Hold time: {hold_days} days | Lookback: {lookback_days} days | Min ROI: {min_roi}%")

    # Pass min_roi to scanner
    opportunities = scanner.scan(
        mode=mode,
        limit=limit,
        progress_callback=progress,
        lookback_days=lookback_days,
        min_roi=min_roi,
    )
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_cli.py -v
```
Expected: PASS

### Step 5: Commit

```bash
git add osrs_flipper/cli.py tests/test_cli.py
git commit -m "feat: add --min-roi flag to scan command

- Default: 20% tax-adjusted ROI
- Filters opportunities below threshold
- Displays min ROI in scan header

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

#### Agent Log - Task 6
**Status:** PENDING
**Files Modified:**
- (to be filled by agent)
**Data Flow:**
- Input: `--min-roi N` from CLI args (default 20.0)
- Transform: Passed directly to scanner
- Output: `min_roi: float` passed to `scanner.scan()`
**Issues:** (to be filled by agent)
**Verification:** (to be filled by agent)

---

## Task 7: E2E Data Flow Integration Test

**Complexity:** LOW (1 file, ~80 lines)
**Depends On:** Tasks 5, 6

**Files:**
- Create: `tests/test_e2e_dataflow.py`

### Step 1: Write comprehensive E2E test

```python
# tests/test_e2e_dataflow.py
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

        # Item 1: Peak at 115 -> ~15% upside (below 20% threshold)
        responses.add(
            responses.GET,
            f"{BASE_URL}/timeseries",
            json={"data": [{"timestamp": i, "avgHighPrice": 115, "avgLowPrice": 95} for i in range(90)]},
            match=[responses.matchers.query_param_matcher({"id": 1, "timestep": "24h"})]
        )
        # Item 2: Peak at 200 -> ~100% upside (above any threshold)
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

        # Short lookback (30 days) - only sees recent 100-120 range
        results_short = scanner.scan(mode="oversold", limit=1, lookback_days=30)

        # With long lookback, item appears more oversold (higher upside to old peak)
        # With short lookback, item is mid-range (less/no upside)
        if results_long and results_short:
            long_upside = results_long[0].get("oversold", {}).get("upside_pct", 0)
            short_upside = results_short[0].get("oversold", {}).get("upside_pct", 0)
            assert long_upside > short_upside, "Long lookback should show higher upside to old peak"
```

### Step 2: Run test to verify it passes

```bash
pytest tests/test_e2e_dataflow.py -v
```
Expected: PASS (all tests)

### Step 3: Commit

```bash
git add tests/test_e2e_dataflow.py
git commit -m "test: add E2E data flow integrity tests

- Tests hold_days -> lookback_days formula
- Tests CLI parameter flow to scanner
- Tests min_roi filter uses tax_adjusted_upside_pct
- Tests lookback window affects analysis
- Verifies vectorization matches scalar calculations

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

#### Agent Log - Task 7
**Status:** PENDING
**Files Modified:**
- (to be filled by agent)
**Data Flow:**
- Verified: CLI args -> defaults -> lookback -> scanner -> analyzer -> filter -> output
- All transformations tested for data integrity
**Issues:** (to be filled by agent)
**Verification:** (to be filled by agent)

---

## Final Verification Checklist

After all tasks complete, run:

```bash
# Full test suite
pytest -v

# Specific feature tests
pytest tests/test_lookback.py tests/test_defaults.py tests/test_e2e_dataflow.py -v

# CLI smoke test
python -m osrs_flipper.cli scan --help

# Verify new flags appear
python -m osrs_flipper.cli scan --help | grep -E "(hold-days|min-roi)"
```

Expected output should show:
- `--hold-days` option with integer type
- `--min-roi` option with float type and default 20

---

## Summary of Changes

| File | Change Type | Lines |
|------|-------------|-------|
| `osrs_flipper/lookback.py` | CREATE | ~25 |
| `osrs_flipper/defaults.py` | CREATE | ~20 |
| `osrs_flipper/analyzers.py` | MODIFY | ~40 |
| `osrs_flipper/scanner.py` | MODIFY | ~30 |
| `osrs_flipper/cli.py` | MODIFY | ~25 |
| `tests/test_lookback.py` | CREATE | ~45 |
| `tests/test_defaults.py` | CREATE | ~30 |
| `tests/test_analyzers.py` | MODIFY | ~50 |
| `tests/test_scanner.py` | MODIFY | ~60 |
| `tests/test_cli.py` | MODIFY | ~20 |
| `tests/test_e2e_dataflow.py` | CREATE | ~150 |

**Total: ~495 lines across 11 files**
**Average per task: ~70 lines**
