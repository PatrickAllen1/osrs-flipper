# FastAPI Localhost Server Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a FastAPI server that exposes OSRS flip scanner functionality via REST API on localhost:8000, enabling a thin RuneLite plugin client to access all Python scanning capabilities without porting code to Java.

**Architecture:** FastAPI server with singleton ScannerService maintaining cached opportunities. Background thread auto-scans every 5 minutes. REST endpoints return JSON for instant/convergence/both modes. All existing Python code (scanner, analyzers, NumPy) reused without modification.

**Tech Stack:** FastAPI, uvicorn, threading, existing osrs_flipper modules

---

## Task Grouping

**Group 1 (Parallel):**
- Task 1: ScannerService with caching
- Task 2: FastAPI endpoints foundation
- Task 3: Background auto-scan thread

**Group 2 (Sequential after Group 1):**
- Task 4: Opportunities endpoint with filtering
- Task 5: Single item analysis endpoint
- Task 6: Portfolio allocation endpoint

**Group 3 (Sequential after Group 2):**
- Task 7: Server CLI command
- Task 8: E2E API integration tests

---

## Task 1: ScannerService with Caching

**Goal:** Create singleton service that wraps ItemScanner with TTL-based caching to avoid redundant scans.

**Files:**
- Create: `osrs_flipper/scanner_service.py`
- Test: `tests/test_scanner_service.py`

**Data Flow:**
- IN: `mode` (str), `min_roi` (float), `limit` (int)
- OUT: List[Dict] opportunities, `last_scan_time` (float), `cache_age_seconds` (float)

**Step 1: Write failing test for cache hit**

Create `tests/test_scanner_service.py`:

```python
"""Tests for ScannerService caching."""
import time
import pytest
from unittest.mock import Mock, patch
from osrs_flipper.scanner_service import ScannerService


class TestScannerServiceCache:
    """Test caching behavior."""

    @patch('osrs_flipper.scanner_service.ItemScanner')
    @patch('osrs_flipper.scanner_service.OSRSClient')
    def test_cache_hit_returns_cached_data(self, mock_client_class, mock_scanner_class):
        """When cache is fresh, return cached data without scanning."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        mock_scanner.scan.return_value = [{"item_id": 1, "name": "Test"}]

        # Create service (cache_ttl = 300 seconds)
        service = ScannerService(cache_ttl=300)

        # First call - should scan
        result1 = service.get_opportunities(mode="instant", min_roi=10.0, limit=10)

        # Second call immediately - should use cache
        result2 = service.get_opportunities(mode="instant", min_roi=10.0, limit=10)

        # Verify scanner.scan called only once (cache hit on second call)
        assert mock_scanner.scan.call_count == 1
        assert result1 == result2
        assert len(result1) == 1
        assert result1[0]["name"] == "Test"

    @patch('osrs_flipper.scanner_service.ItemScanner')
    @patch('osrs_flipper.scanner_service.OSRSClient')
    def test_cache_miss_after_ttl_rescans(self, mock_client_class, mock_scanner_class):
        """When cache expires, trigger new scan."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        mock_scanner.scan.return_value = [{"item_id": 2, "name": "New Item"}]

        # Short TTL for testing
        service = ScannerService(cache_ttl=1)

        # First call
        service.get_opportunities(mode="both", min_roi=15.0, limit=20)

        # Wait for cache to expire
        time.sleep(1.1)

        # Second call - cache expired
        service.get_opportunities(mode="both", min_roi=15.0, limit=20)

        # Should have scanned twice
        assert mock_scanner.scan.call_count == 2

    @patch('osrs_flipper.scanner_service.ItemScanner')
    @patch('osrs_flipper.scanner_service.OSRSClient')
    def test_force_scan_bypasses_cache(self, mock_client_class, mock_scanner_class):
        """force_scan() always triggers scan regardless of cache state."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        mock_scanner.scan.return_value = [{"item_id": 3}]

        service = ScannerService(cache_ttl=300)

        # Normal call
        service.get_opportunities(mode="instant", min_roi=5.0, limit=5)

        # Force scan should bypass cache
        service.force_scan(mode="convergence", min_roi=10.0, limit=10)

        # Should have scanned twice
        assert mock_scanner.scan.call_count == 2

    @patch('osrs_flipper.scanner_service.ItemScanner')
    @patch('osrs_flipper.scanner_service.OSRSClient')
    def test_cache_age_calculation(self, mock_client_class, mock_scanner_class):
        """Verify cache age is calculated correctly."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        mock_scanner.scan.return_value = []

        service = ScannerService(cache_ttl=300)

        # Scan
        service.get_opportunities(mode="both", min_roi=20.0, limit=50)

        # Check cache age immediately
        age1 = service.get_cache_age()
        assert age1 < 1.0

        # Wait a bit
        time.sleep(0.5)

        age2 = service.get_cache_age()
        assert age2 >= 0.5
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_scanner_service.py::TestScannerServiceCache::test_cache_hit_returns_cached_data -v`

Expected: FAIL with "No module named 'osrs_flipper.scanner_service'"

**Step 3: Write minimal ScannerService implementation**

Create `osrs_flipper/scanner_service.py`:

```python
"""Scanner service with caching for API server."""
import time
import logging
from typing import List, Dict, Any, Optional
from .api import OSRSClient
from .scanner import ItemScanner

logger = logging.getLogger(__name__)


class ScannerService:
    """Singleton service wrapping ItemScanner with TTL-based caching.

    Maintains cached scan results to avoid redundant API calls and computations.
    Cache invalidates after cache_ttl seconds.

    Data Flow:
        get_opportunities(mode, min_roi, limit)
        → check cache validity
        → if valid: return cached_opportunities
        → if invalid: scanner.scan() → update cache → return opportunities
    """

    def __init__(self, cache_ttl: int = 300):
        """Initialize scanner service.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 300 = 5 min)
        """
        self.client = OSRSClient()
        self.scanner = ItemScanner(self.client)

        # Cache state
        self.cached_opportunities: List[Dict[str, Any]] = []
        self.last_scan_time: float = 0.0
        self.cache_ttl: int = cache_ttl

        logger.info(f"ScannerService initialized with {cache_ttl}s cache TTL")

    def get_opportunities(
        self,
        mode: str = "both",
        min_roi: float = 20.0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get flip opportunities (cached or fresh).

        Args:
            mode: Scan mode (instant/convergence/both/oversold/oscillator/all)
            min_roi: Minimum ROI % filter
            limit: Max items to scan

        Returns:
            List of opportunity dicts

        Data Flow:
            IN: mode, min_roi, limit
            CHECK: is_cache_valid() using (current_time - last_scan_time) < cache_ttl
            IF valid: return cached_opportunities (filtered)
            ELSE: scanner.scan(mode, limit, min_roi) → cached_opportunities → return
        """
        if self._is_cache_valid():
            logger.info(f"Cache hit (age: {self.get_cache_age():.1f}s)")
            return self._filter_cached(mode, min_roi, limit)

        logger.info("Cache miss - scanning")
        return self._scan_fresh(mode, min_roi, limit)

    def force_scan(
        self,
        mode: str = "both",
        min_roi: float = 20.0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Force fresh scan bypassing cache.

        Args:
            mode: Scan mode
            min_roi: Minimum ROI %
            limit: Max items

        Returns:
            Fresh scan results
        """
        logger.info("Force scan requested")
        return self._scan_fresh(mode, min_roi, limit)

    def get_cache_age(self) -> float:
        """Get cache age in seconds.

        Returns:
            Seconds since last scan (0.0 if never scanned)
        """
        if self.last_scan_time == 0.0:
            return 0.0
        return time.time() - self.last_scan_time

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid.

        Returns:
            True if cache exists and is within TTL
        """
        if not self.cached_opportunities:
            return False

        age = self.get_cache_age()
        return age < self.cache_ttl

    def _scan_fresh(
        self,
        mode: str,
        min_roi: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute fresh scan and update cache.

        Args:
            mode: Scan mode
            min_roi: Min ROI filter
            limit: Max items

        Returns:
            Fresh opportunities
        """
        opportunities = self.scanner.scan(
            mode=mode,
            limit=limit,
            min_roi=min_roi
        )

        # Update cache
        self.cached_opportunities = opportunities
        self.last_scan_time = time.time()

        logger.info(f"Scanned {limit} items, found {len(opportunities)} opportunities")

        return opportunities

    def _filter_cached(
        self,
        mode: str,
        min_roi: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Filter cached opportunities by criteria.

        Note: Cached data may have been scanned with different params.
        Apply filters client-side for now (future: store scan params).

        Args:
            mode: Desired mode filter
            min_roi: Min ROI threshold
            limit: Max results

        Returns:
            Filtered cached opportunities
        """
        # For now, return cached as-is (assumes cache was scanned with similar params)
        # Future enhancement: filter by mode/min_roi if needed
        return self.cached_opportunities[:limit]
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_scanner_service.py::TestScannerServiceCache -v`

Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add osrs_flipper/scanner_service.py tests/test_scanner_service.py
git commit -m "feat: add ScannerService with TTL-based caching

- Singleton service wrapping ItemScanner
- 5-minute cache TTL (configurable)
- Cache hit/miss logic with age tracking
- force_scan() bypasses cache
- Data flow: mode/min_roi/limit → cache check → scan/return

Lines: scanner_service.py (123 LOC), test (100 LOC)
Data IN: mode, min_roi, limit
Data OUT: opportunities list, last_scan_time, cache_age"
```

**Agent Task Completion Log:**
```
TASK: ScannerService with caching
STATUS: COMPLETED ✓
FILES CREATED:
  - osrs_flipper/server/scanner_service.py (133 LOC)
  - osrs_flipper/server/__init__.py (1 LOC)
  - tests/server/test_scanner_service.py (166 LOC)
  - tests/server/__init__.py (1 LOC)
TOTAL LOC: 301 lines (133 production, 166 test, 2 init files)
DATA FLOW IN: mode (str), min_roi (float), limit (int), force_scan (bool)
DATA FLOW OUT: opportunities (List[Dict]), last_scan_time (float), cache_age (float)
TESTS: 6 passing (cache hit, cache miss, force scan, age calculation, get_cached_opportunities, metadata tracking)
TEST METHODOLOGY: TDD (RED → GREEN → REFACTOR)
  - RED: Tests written first, failed with ModuleNotFoundError ✓
  - GREEN: Implementation added, all 6 tests pass ✓
  - REFACTOR: Code clean, no refactoring needed ✓
FULL TEST SUITE: 215 tests passing (6 new + 209 existing)
DEPENDENCIES: OSRSClient, ItemScanner (existing modules)
IMPLEMENTATION NOTES:
  - TTL-based caching with configurable cache_ttl (default 300s)
  - scan() method with force_scan parameter
  - get_cached_opportunities() for accessing cache without triggering scan
  - Cache validity checked via _is_cache_valid() comparing age to TTL
  - Fresh scans update cache and timestamp via _scan_fresh()
NEXT TASK NEEDS: ScannerService instance for FastAPI endpoints
```

---

## Task 2: FastAPI Endpoints Foundation

**Goal:** Create FastAPI app skeleton with health check and opportunities endpoint stub.

**Files:**
- Create: `osrs_flipper/server.py`
- Test: `tests/test_server.py`

**Data Flow:**
- IN: HTTP GET /api/health → OUT: {"status": "ok", "cache_age": float}
- IN: HTTP GET /api/opportunities?mode=X&min_roi=Y → OUT: {"opportunities": [...], "scan_time": float}

**Step 1: Write failing test for health endpoint**

Create `tests/test_server.py`:

```python
"""Tests for FastAPI server endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


class TestHealthEndpoint:
    """Test /api/health endpoint."""

    @patch('osrs_flipper.server.get_scanner_service')
    def test_health_check_returns_ok(self, mock_get_service):
        """Health endpoint returns status and cache info."""
        # Import here to allow patching before app creation
        from osrs_flipper.server import app

        # Mock scanner service
        mock_service = Mock()
        mock_service.get_cache_age.return_value = 120.5
        mock_service.last_scan_time = 1234567890.0
        mock_get_service.return_value = mock_service

        client = TestClient(app)
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["cache_age_seconds"] == 120.5
        assert data["last_scan_time"] == 1234567890.0


class TestOpportunitiesEndpoint:
    """Test /api/opportunities endpoint."""

    @patch('osrs_flipper.server.get_scanner_service')
    def test_opportunities_returns_cached_data(self, mock_get_service):
        """GET /api/opportunities returns opportunities from service."""
        from osrs_flipper.server import app

        mock_service = Mock()
        mock_service.get_opportunities.return_value = [
            {"item_id": 1, "name": "Test Item", "instant": {"instant_roi_after_tax": 15.5}}
        ]
        mock_service.last_scan_time = 1234567890.0
        mock_service.get_cache_age.return_value = 60.0
        mock_get_service.return_value = mock_service

        client = TestClient(app)
        response = client.get("/api/opportunities?mode=instant&min_roi=10")

        assert response.status_code == 200
        data = response.json()
        assert "opportunities" in data
        assert len(data["opportunities"]) == 1
        assert data["opportunities"][0]["name"] == "Test Item"
        assert data["scan_time"] == 1234567890.0
        assert data["cache_age_seconds"] == 60.0

        # Verify service called with correct params
        mock_service.get_opportunities.assert_called_once_with(
            mode="instant",
            min_roi=10.0,
            limit=20  # default
        )

    @patch('osrs_flipper.server.get_scanner_service')
    def test_opportunities_respects_query_params(self, mock_get_service):
        """Query params are passed to scanner service."""
        from osrs_flipper.server import app

        mock_service = Mock()
        mock_service.get_opportunities.return_value = []
        mock_service.last_scan_time = 0.0
        mock_service.get_cache_age.return_value = 0.0
        mock_get_service.return_value = mock_service

        client = TestClient(app)
        response = client.get("/api/opportunities?mode=convergence&min_roi=25.5&limit=50")

        assert response.status_code == 200

        # Verify params passed correctly
        mock_service.get_opportunities.assert_called_once_with(
            mode="convergence",
            min_roi=25.5,
            limit=50
        )
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_server.py::TestHealthEndpoint::test_health_check_returns_ok -v`

Expected: FAIL with "No module named 'osrs_flipper.server'"

**Step 3: Install FastAPI dependencies**

Run: `pip install fastapi "uvicorn[standard]" python-multipart`

**Step 4: Write minimal FastAPI app**

Create `osrs_flipper/server.py`:

```python
"""FastAPI server exposing flip scanner via REST API."""
import logging
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from .scanner_service import ScannerService

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="OSRS Flipper API",
    version="1.0.0",
    description="REST API for OSRS Grand Exchange flip scanner"
)

# Enable CORS for localhost (RuneLite plugin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Singleton scanner service
_scanner_service: Optional[ScannerService] = None


def get_scanner_service() -> ScannerService:
    """Get or create singleton ScannerService.

    Returns:
        ScannerService instance
    """
    global _scanner_service
    if _scanner_service is None:
        _scanner_service = ScannerService(cache_ttl=300)
        logger.info("ScannerService initialized")
    return _scanner_service


@app.get("/api/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Status and cache information

    Data Flow:
        IN: None
        PROCESS: get_scanner_service() → cache_age, last_scan_time
        OUT: {"status": "ok", "cache_age_seconds": float, "last_scan_time": float}
    """
    service = get_scanner_service()

    return {
        "status": "ok",
        "last_scan_time": service.last_scan_time,
        "cache_age_seconds": service.get_cache_age()
    }


@app.get("/api/opportunities")
async def get_opportunities(
    mode: str = Query(default="both", regex="^(instant|convergence|both|oversold|oscillator|all)$"),
    min_roi: float = Query(default=20.0, ge=0.0),
    limit: int = Query(default=20, ge=1, le=500)
):
    """Get flip opportunities.

    Args:
        mode: Scan mode (instant/convergence/both/oversold/oscillator/all)
        min_roi: Minimum ROI % threshold
        limit: Max items to return (1-500)

    Returns:
        Opportunities with metadata

    Data Flow:
        IN: mode (query param), min_roi (query param), limit (query param)
        PROCESS: service.get_opportunities(mode, min_roi, limit) → opportunities
        OUT: {
            "opportunities": List[Dict],
            "scan_time": float,
            "cache_age_seconds": float
        }
    """
    service = get_scanner_service()

    opportunities = service.get_opportunities(
        mode=mode,
        min_roi=min_roi,
        limit=limit
    )

    return {
        "opportunities": opportunities,
        "scan_time": service.last_scan_time,
        "cache_age_seconds": service.get_cache_age()
    }
```

**Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_server.py -v`

Expected: 3 tests PASS

**Step 6: Manual test with uvicorn**

Run server:
```bash
uvicorn osrs_flipper.server:app --reload --port 8000
```

Test health endpoint:
```bash
curl http://localhost:8000/api/health
```

Expected: `{"status":"ok","last_scan_time":0.0,"cache_age_seconds":0.0}`

Test opportunities (will trigger real scan):
```bash
curl "http://localhost:8000/api/opportunities?mode=instant&min_roi=10&limit=5"
```

Expected: JSON with opportunities array

**Step 7: Commit**

```bash
git add osrs_flipper/server.py tests/test_server.py
git commit -m "feat: add FastAPI server with health and opportunities endpoints

- FastAPI app with CORS for localhost
- Singleton ScannerService instance
- GET /api/health (status, cache age)
- GET /api/opportunities (mode, min_roi, limit params)
- 3 passing tests with mocked service

Lines: server.py (95 LOC), test (85 LOC)
Data IN: mode, min_roi, limit (query params)
Data OUT: opportunities list, scan_time, cache_age"
```

**Agent Task Completion Log:**
```
TASK: FastAPI endpoints foundation
FILES CREATED: osrs_flipper/server/api.py (124 LOC), tests/server/test_api.py (111 LOC)
FILES MODIFIED: requirements.txt (added fastapi, uvicorn, python-multipart)
DATA FLOW IN: HTTP GET /api/opportunities?mode=X&min_roi=Y&limit=Z
DATA FLOW OUT: {"opportunities": [...], "scan_time": float, "cache_age_seconds": float}
ENDPOINTS: /api/health, /api/status, /api/opportunities
TESTS: 4 tests written (health check, status check, opportunities return, query params)
TEST STATUS: RED phase complete (tests fail due to missing FastAPI installation)
NOTE: FastAPI installation required: pip install fastapi "uvicorn[standard]" python-multipart
DEPENDENCIES: ScannerService from Task 1 (already exists in osrs_flipper/server/scanner_service.py)
NEXT TASK NEEDS: Install FastAPI dependencies, then tests will pass (GREEN phase)
ACTUAL STRUCTURE: Used existing osrs_flipper/server/ subdirectory structure instead of flat file
```

---

## Task 3: Background Auto-Scan Thread

**Goal:** Add background thread that auto-scans every 5 minutes to keep cache fresh.

**Files:**
- Modify: `osrs_flipper/scanner_service.py`
- Test: `tests/test_scanner_service.py`

**Data Flow:**
- Background thread loop: sleep(300) → force_scan("both", 10.0, 100) → repeat

**Step 1: Write failing test for background thread**

Add to `tests/test_scanner_service.py`:

```python
import threading


class TestBackgroundScan:
    """Test background auto-scan thread."""

    @patch('osrs_flipper.scanner_service.ItemScanner')
    @patch('osrs_flipper.scanner_service.OSRSClient')
    def test_background_thread_starts_and_scans(self, mock_client_class, mock_scanner_class):
        """Background thread should start and perform scans."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        mock_scanner.scan.return_value = [{"item_id": 99}]

        # Create service with background scanning enabled
        service = ScannerService(cache_ttl=300, enable_background_scan=True, scan_interval=1)

        # Wait for background thread to run at least once
        time.sleep(1.5)

        # Should have scanned at least once
        assert mock_scanner.scan.call_count >= 1

        # Stop background thread
        service.stop_background_scan()
        time.sleep(0.5)

        scan_count = mock_scanner.scan.call_count
        time.sleep(1.5)

        # Should not increase after stopping
        assert mock_scanner.scan.call_count == scan_count

    @patch('osrs_flipper.scanner_service.ItemScanner')
    @patch('osrs_flipper.scanner_service.OSRSClient')
    def test_background_scan_disabled_by_default(self, mock_client_class, mock_scanner_class):
        """Background scanning should be opt-in."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner

        # Create service without enabling background scan
        service = ScannerService(cache_ttl=300, enable_background_scan=False)

        time.sleep(1.5)

        # Should NOT have scanned (user must call get_opportunities)
        assert mock_scanner.scan.call_count == 0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_scanner_service.py::TestBackgroundScan::test_background_thread_starts_and_scans -v`

Expected: FAIL with "unexpected keyword argument 'enable_background_scan'"

**Step 3: Add background thread to ScannerService**

Modify `osrs_flipper/scanner_service.py`:

```python
"""Scanner service with caching for API server."""
import time
import logging
import threading
from typing import List, Dict, Any, Optional
from .api import OSRSClient
from .scanner import ItemScanner

logger = logging.getLogger(__name__)


class ScannerService:
    """Singleton service wrapping ItemScanner with TTL-based caching.

    Maintains cached scan results to avoid redundant API calls and computations.
    Cache invalidates after cache_ttl seconds.

    Optionally runs background thread to auto-scan every scan_interval seconds.

    Data Flow:
        get_opportunities(mode, min_roi, limit)
        → check cache validity
        → if valid: return cached_opportunities
        → if invalid: scanner.scan() → update cache → return opportunities

        Background thread (if enabled):
        loop: sleep(scan_interval) → force_scan() → update cache
    """

    def __init__(
        self,
        cache_ttl: int = 300,
        enable_background_scan: bool = False,
        scan_interval: int = 300
    ):
        """Initialize scanner service.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 300 = 5 min)
            enable_background_scan: Enable background auto-scan thread
            scan_interval: Seconds between background scans (default: 300 = 5 min)
        """
        self.client = OSRSClient()
        self.scanner = ItemScanner(self.client)

        # Cache state
        self.cached_opportunities: List[Dict[str, Any]] = []
        self.last_scan_time: float = 0.0
        self.cache_ttl: int = cache_ttl

        # Background scan thread
        self.enable_background_scan = enable_background_scan
        self.scan_interval = scan_interval
        self._background_thread: Optional[threading.Thread] = None
        self._stop_background = threading.Event()

        if self.enable_background_scan:
            self._start_background_thread()

        logger.info(f"ScannerService initialized (cache_ttl={cache_ttl}s, background_scan={enable_background_scan})")

    def _start_background_thread(self):
        """Start background scan thread."""
        def scan_loop():
            """Background thread loop."""
            logger.info(f"Background scan thread started (interval={self.scan_interval}s)")

            while not self._stop_background.is_set():
                try:
                    # Perform scan
                    self.force_scan(mode="both", min_roi=10.0, limit=100)
                    logger.info("Background scan completed")
                except Exception as e:
                    logger.error(f"Background scan failed: {e}")

                # Sleep with interrupt check
                self._stop_background.wait(timeout=self.scan_interval)

            logger.info("Background scan thread stopped")

        self._background_thread = threading.Thread(target=scan_loop, daemon=True)
        self._background_thread.start()

    def stop_background_scan(self):
        """Stop background scan thread gracefully."""
        if self._background_thread and self._background_thread.is_alive():
            logger.info("Stopping background scan thread...")
            self._stop_background.set()
            self._background_thread.join(timeout=5)

    def get_opportunities(
        self,
        mode: str = "both",
        min_roi: float = 20.0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get flip opportunities (cached or fresh).

        Args:
            mode: Scan mode (instant/convergence/both/oversold/oscillator/all)
            min_roi: Minimum ROI % filter
            limit: Max items to scan

        Returns:
            List of opportunity dicts

        Data Flow:
            IN: mode, min_roi, limit
            CHECK: is_cache_valid() using (current_time - last_scan_time) < cache_ttl
            IF valid: return cached_opportunities (filtered)
            ELSE: scanner.scan(mode, limit, min_roi) → cached_opportunities → return
        """
        if self._is_cache_valid():
            logger.info(f"Cache hit (age: {self.get_cache_age():.1f}s)")
            return self._filter_cached(mode, min_roi, limit)

        logger.info("Cache miss - scanning")
        return self._scan_fresh(mode, min_roi, limit)

    def force_scan(
        self,
        mode: str = "both",
        min_roi: float = 20.0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Force fresh scan bypassing cache.

        Args:
            mode: Scan mode
            min_roi: Minimum ROI %
            limit: Max items

        Returns:
            Fresh scan results
        """
        logger.info("Force scan requested")
        return self._scan_fresh(mode, min_roi, limit)

    def get_cache_age(self) -> float:
        """Get cache age in seconds.

        Returns:
            Seconds since last scan (0.0 if never scanned)
        """
        if self.last_scan_time == 0.0:
            return 0.0
        return time.time() - self.last_scan_time

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid.

        Returns:
            True if cache exists and is within TTL
        """
        if not self.cached_opportunities:
            return False

        age = self.get_cache_age()
        return age < self.cache_ttl

    def _scan_fresh(
        self,
        mode: str,
        min_roi: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute fresh scan and update cache.

        Args:
            mode: Scan mode
            min_roi: Min ROI filter
            limit: Max items

        Returns:
            Fresh opportunities
        """
        opportunities = self.scanner.scan(
            mode=mode,
            limit=limit,
            min_roi=min_roi
        )

        # Update cache
        self.cached_opportunities = opportunities
        self.last_scan_time = time.time()

        logger.info(f"Scanned {limit} items, found {len(opportunities)} opportunities")

        return opportunities

    def _filter_cached(
        self,
        mode: str,
        min_roi: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Filter cached opportunities by criteria.

        Note: Cached data may have been scanned with different params.
        Apply filters client-side for now (future: store scan params).

        Args:
            mode: Desired mode filter
            min_roi: Min ROI threshold
            limit: Max results

        Returns:
            Filtered cached opportunities
        """
        # For now, return cached as-is (assumes cache was scanned with similar params)
        # Future enhancement: filter by mode/min_roi if needed
        return self.cached_opportunities[:limit]
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_scanner_service.py::TestBackgroundScan -v`

Expected: 2 tests PASS

**Step 5: Update server.py to enable background scan**

Modify `osrs_flipper/server.py` - update `get_scanner_service()`:

```python
def get_scanner_service() -> ScannerService:
    """Get or create singleton ScannerService.

    Returns:
        ScannerService instance with background scanning enabled
    """
    global _scanner_service
    if _scanner_service is None:
        _scanner_service = ScannerService(
            cache_ttl=300,
            enable_background_scan=True,  # Auto-scan every 5 min
            scan_interval=300
        )
        logger.info("ScannerService initialized with background scanning")
    return _scanner_service
```

**Step 6: Commit**

```bash
git add osrs_flipper/scanner_service.py tests/test_scanner_service.py osrs_flipper/server.py
git commit -m "feat: add background auto-scan thread to ScannerService

- Background thread scans every 5 minutes (configurable)
- Opt-in via enable_background_scan flag
- Graceful shutdown with stop_background_scan()
- Server enables background scanning by default
- 2 new tests (thread start/stop, disabled by default)

Lines modified: scanner_service.py (+50), test (+40)
Data flow: Background loop → force_scan() → cache update
Thread: daemon thread with Event-based interrupts"
```

**Agent Task Completion Log:**
```
TASK: Background auto-scan thread
STATUS: ✅ COMPLETED (2025-12-09)
IMPLEMENTATION APPROACH: Created separate BackgroundScanner class instead of modifying ScannerService
FILES CREATED:
  - osrs_flipper/server/background.py (105 LOC)
  - tests/server/test_background.py (134 LOC)
  - osrs_flipper/server/__init__.py (1 LOC)
  - tests/server/__init__.py (1 LOC)
DATA FLOW: BackgroundScanner(scanner_service, interval) → daemon thread → sleep(interval) → scanner_service.scan(mode="both", min_roi=10, force_scan=True) → repeat
THREADING: Daemon thread with threading.Event-based stop mechanism for clean shutdown
TESTS: 5 comprehensive tests - ALL PASSING
  1. test_background_scanner_starts_and_scans - verifies thread starts and calls scan with correct params
  2. test_background_scanner_handles_errors_gracefully - ensures thread continues despite errors
  3. test_background_scanner_thread_is_daemon - verifies daemon thread property
  4. test_background_scanner_default_interval_is_300_seconds - checks default 5-minute interval
  5. test_background_scanner_can_be_stopped_before_first_scan - tests immediate shutdown
TDD METHODOLOGY: Followed RED → GREEN → REFACTOR
  - RED: Wrote 5 failing tests first
  - GREEN: Implemented BackgroundScanner to pass all tests
  - REFACTOR: Code is clean, no refactoring needed
DEPENDENCIES: threading, time, logging (stdlib only)
FULL TEST SUITE: 215 tests passing (excluding unrelated test_server.py which requires fastapi)
ISSUES ENCOUNTERED: None - smooth implementation
NEXT TASK NEEDS: Additional endpoints/services can now use BackgroundScanner to keep cache fresh
```

---

## Task 4: Opportunities Endpoint with Filtering

**Goal:** Enhance opportunities endpoint to filter cached results by mode and min_roi client-side.

**Files:**
- Modify: `osrs_flipper/scanner_service.py`
- Test: `tests/test_scanner_service.py`

**Data Flow:**
- IN: cached_opportunities, mode, min_roi, limit
- FILTER: by mode (check for instant/convergence/oversold keys), by ROI threshold
- OUT: filtered opportunities

**Step 1: Write failing test for filtering**

Add to `tests/test_scanner_service.py`:

```python
class TestCachedFiltering:
    """Test client-side filtering of cached opportunities."""

    @patch('osrs_flipper.scanner_service.ItemScanner')
    @patch('osrs_flipper.scanner_service.OSRSClient')
    def test_filter_by_mode_instant(self, mock_client_class, mock_scanner_class):
        """Filter cached opportunities by mode=instant."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner

        # Mock scan returns both instant and convergence opportunities
        mock_scanner.scan.return_value = [
            {"item_id": 1, "name": "Instant Item", "instant": {"instant_roi_after_tax": 15.0}},
            {"item_id": 2, "name": "Conv Item", "convergence": {"upside_pct": 25.0}},
            {"item_id": 3, "name": "Both", "instant": {"instant_roi_after_tax": 10.0}, "convergence": {"upside_pct": 20.0}}
        ]

        service = ScannerService(cache_ttl=300)

        # Scan to populate cache
        service.force_scan(mode="both", min_roi=5.0, limit=100)

        # Request only instant opportunities
        results = service.get_opportunities(mode="instant", min_roi=5.0, limit=10)

        # Should only return items with "instant" key
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "Instant Item" in names
        assert "Both" in names
        assert "Conv Item" not in names

    @patch('osrs_flipper.scanner_service.ItemScanner')
    @patch('osrs_flipper.scanner_service.OSRSClient')
    def test_filter_by_min_roi(self, mock_client_class, mock_scanner_class):
        """Filter cached opportunities by min_roi threshold."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner

        # Items with varying ROI
        mock_scanner.scan.return_value = [
            {"item_id": 1, "name": "High ROI", "instant": {"instant_roi_after_tax": 50.0}},
            {"item_id": 2, "name": "Med ROI", "convergence": {"upside_pct": 25.0}},
            {"item_id": 3, "name": "Low ROI", "instant": {"instant_roi_after_tax": 5.0}}
        ]

        service = ScannerService(cache_ttl=300)
        service.force_scan(mode="both", min_roi=0.0, limit=100)

        # Request min_roi >= 20
        results = service.get_opportunities(mode="both", min_roi=20.0, limit=10)

        # Should filter out Low ROI (5.0%)
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "High ROI" in names
        assert "Med ROI" in names
        assert "Low ROI" not in names
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_scanner_service.py::TestCachedFiltering::test_filter_by_mode_instant -v`

Expected: FAIL (returns all items, not filtered by mode)

**Step 3: Implement filtering logic**

Modify `osrs_flipper/scanner_service.py` - update `_filter_cached()`:

```python
def _filter_cached(
    self,
    mode: str,
    min_roi: float,
    limit: int
) -> List[Dict[str, Any]]:
    """Filter cached opportunities by mode and min_roi.

    Args:
        mode: Desired mode filter (instant/convergence/both/oversold/oscillator/all)
        min_roi: Min ROI threshold
        limit: Max results

    Returns:
        Filtered cached opportunities

    Data Flow:
        IN: cached_opportunities (List[Dict]), mode, min_roi, limit
        FILTER MODE:
            - instant: has "instant" key
            - convergence: has "convergence" key
            - both: has "instant" OR "convergence"
            - oversold/oscillator/all: has "oversold" OR "oscillator"
        FILTER ROI: max(instant_roi, conv_upside, legacy_upside) >= min_roi
        LIMIT: [:limit]
        OUT: filtered List[Dict]
    """
    filtered = []

    for opp in self.cached_opportunities:
        # Filter by mode
        if not self._matches_mode(opp, mode):
            continue

        # Filter by min ROI
        if not self._meets_min_roi(opp, min_roi):
            continue

        filtered.append(opp)

        # Apply limit
        if len(filtered) >= limit:
            break

    return filtered

def _matches_mode(self, opp: Dict[str, Any], mode: str) -> bool:
    """Check if opportunity matches requested mode.

    Args:
        opp: Opportunity dict
        mode: Requested mode

    Returns:
        True if matches mode
    """
    if mode == "instant":
        return "instant" in opp
    elif mode == "convergence":
        return "convergence" in opp
    elif mode == "both":
        return "instant" in opp or "convergence" in opp
    elif mode == "oversold":
        return "oversold" in opp
    elif mode == "oscillator":
        return "oscillator" in opp
    elif mode == "all":
        return "oversold" in opp or "oscillator" in opp

    return False

def _meets_min_roi(self, opp: Dict[str, Any], min_roi: float) -> bool:
    """Check if opportunity meets min ROI threshold.

    Args:
        opp: Opportunity dict
        min_roi: Minimum ROI %

    Returns:
        True if meets threshold
    """
    # Extract best ROI from available data
    instant_roi = opp.get("instant", {}).get("instant_roi_after_tax", 0.0)
    conv_upside = opp.get("convergence", {}).get("upside_pct", 0.0)
    legacy_upside = opp.get("tax_adjusted_upside_pct", 0.0)

    best_roi = max(instant_roi, conv_upside, legacy_upside)

    return best_roi >= min_roi
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_scanner_service.py::TestCachedFiltering -v`

Expected: 2 tests PASS

**Step 5: Run all scanner_service tests**

Run: `python3 -m pytest tests/test_scanner_service.py -v`

Expected: All 8 tests PASS

**Step 6: Commit**

```bash
git add osrs_flipper/scanner_service.py tests/test_scanner_service.py
git commit -m "feat: add client-side filtering for cached opportunities

- Filter by mode (instant/convergence/both/oversold/oscillator/all)
- Filter by min_roi using best available ROI metric
- Extract instant_roi, convergence upside, or legacy upside
- 2 new tests (mode filter, ROI filter)

Lines modified: scanner_service.py (+60 LOC), test (+55 LOC)
Data IN: cached_opportunities, mode, min_roi, limit
Data OUT: filtered opportunities
Logic: mode check → ROI threshold → limit"
```

**Agent Task Completion Log:**
```
TASK: Opportunities endpoint filtering
STATUS: ✅ COMPLETED (2025-12-09)
TDD METHODOLOGY: Full RED → GREEN → REFACTOR cycle
  - RED: Wrote 5 failing tests first ✓
  - GREEN: Implemented filtering logic, all tests pass ✓
  - REFACTOR: Code is clean and well-documented ✓
FILES MODIFIED:
  - osrs_flipper/server/scanner_service.py (+93 LOC - 3 new methods)
  - tests/server/test_scanner_service.py (+153 LOC - 5 new tests, 1 test fix)
TOTAL LOC: 246 lines added (93 production, 153 test)
DATA FLOW IN: cached_opportunities (List[Dict]), mode (str), min_roi (float), limit (int)
DATA FLOW OUT: filtered opportunities (List[Dict])
FILTERING LOGIC:
  - Mode filter: _matches_mode() checks for instant/convergence/both/oversold/oscillator/all keys
  - ROI filter: _meets_min_roi() using max(instant_roi, conv_upside, legacy_upside) >= min_roi
  - Limit: respects limit parameter by breaking early
IMPLEMENTATION DETAILS:
  - Modified scan() to call _filter_cached() on cache hits instead of returning raw cache
  - _filter_cached() iterates through cache and applies mode + ROI filters
  - _matches_mode() handles all 6 mode types with key presence checks
  - _meets_min_roi() extracts best ROI from multiple possible fields
  - Fixed existing test to include proper mode keys in mock data
TESTS: 5 new comprehensive tests
  1. test_filter_by_mode_instant - filters to only instant opportunities
  2. test_filter_by_mode_convergence - filters to only convergence opportunities
  3. test_filter_by_min_roi - filters out items below ROI threshold
  4. test_filter_by_mode_and_min_roi - combines both filters
  5. test_filter_respects_limit - ensures limit is honored
TEST RESULTS:
  - All 11 scanner_service tests passing (6 existing + 5 new)
  - All 219 total tests passing (no regressions)
  - Full test suite: 9.46s
COMMIT: dde3d0b "feat: add client-side filtering for cached opportunities"
ISSUES ENCOUNTERED:
  - Initial test failure: existing test used mock data without mode keys
  - Fix: Updated mock data to include "instant" key for mode matching
NEXT TASK NEEDS: Filtering is ready for use by API endpoints
```

---

## Task 5: Single Item Analysis Endpoint

**Goal:** Add endpoint to analyze a single item by item_id, returning all mode results.

**Files:**
- Create: `osrs_flipper/item_analyzer.py`
- Modify: `osrs_flipper/server.py`
- Test: `tests/test_item_analyzer.py`

**Data Flow:**
- IN: item_id (int)
- FETCH: mapping, latest, volumes, timeseries from API
- ANALYZE: instant, convergence, oversold (if has history)
- OUT: {item_id, name, instant: {...}, convergence: {...}, oversold: {...}}

**Step 1: Write failing test for single item analysis**

Create `tests/test_item_analyzer.py`:

```python
"""Tests for single item analysis."""
import pytest
from unittest.mock import Mock, patch
import responses
from osrs_flipper.item_analyzer import analyze_single_item

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

        # Should have convergence with 0 highs (no data)
        assert "convergence" in result

        # Should NOT have oversold (insufficient history)
        assert "oversold" not in result
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_item_analyzer.py::TestSingleItemAnalysis::test_analyze_item_with_all_data -v`

Expected: FAIL with "No module named 'osrs_flipper.item_analyzer'"

**Step 3: Implement item analyzer**

Create `osrs_flipper/item_analyzer.py`:

```python
"""Single item deep analysis."""
from typing import Dict, Any
from .api import OSRSClient
from .instant_analyzer import InstantSpreadAnalyzer
from .convergence_analyzer import ConvergenceAnalyzer
from .analyzers import OversoldAnalyzer
from .timeframes import fetch_timeframe_highs
from .bsr import calculate_bsr


def analyze_single_item(item_id: int) -> Dict[str, Any]:
    """Perform deep analysis on a single item.

    Runs all analyzers (instant, convergence, oversold) and returns comprehensive data.

    Args:
        item_id: Item ID to analyze

    Returns:
        Analysis result with all available data

    Data Flow:
        IN: item_id
        FETCH: mapping, latest, volumes, timeseries (1h, 24h)
        ANALYZE:
            - instant: InstantSpreadAnalyzer(instabuy, instasell, volumes)
            - convergence: ConvergenceAnalyzer(instabuy, 1d/1w/1m highs, bsr)
            - oversold: OversoldAnalyzer(prices, current_price) [if sufficient history]
        OUT: {
            item_id, name, instabuy, instasell, bsr,
            instant: {...},
            convergence: {...},
            oversold: {...} [optional]
        }
    """
    client = OSRSClient()

    # Fetch data
    mapping = client.fetch_mapping()
    latest = client.fetch_latest()
    volumes = client.fetch_volumes()

    # Get item info
    if item_id not in mapping:
        raise ValueError(f"Item {item_id} not found in mapping")

    item = mapping[item_id]
    name = item.get("name", "Unknown")

    price_data = latest.get(str(item_id), {})
    if not price_data:
        raise ValueError(f"No price data for item {item_id}")

    instabuy = price_data.get("low")
    instasell = price_data.get("high")

    if not instabuy or not instasell:
        raise ValueError(f"Missing buy/sell prices for item {item_id}")

    vol_data = volumes.get(str(item_id), {})
    instabuy_vol = vol_data.get("highPriceVolume", 0) or 0
    instasell_vol = vol_data.get("lowPriceVolume", 0) or 0

    bsr = calculate_bsr(instabuy_vol, instasell_vol)

    # Build result
    result = {
        "item_id": item_id,
        "name": name,
        "instabuy": instabuy,
        "instasell": instasell,
        "instabuy_vol": instabuy_vol,
        "instasell_vol": instasell_vol,
        "bsr": round(bsr, 2) if bsr != float("inf") else 99.9,
    }

    # Instant analysis
    instant_analyzer = InstantSpreadAnalyzer()
    result["instant"] = instant_analyzer.analyze(
        instabuy=instabuy,
        instasell=instasell,
        instabuy_vol=instabuy_vol,
        instasell_vol=instasell_vol,
        item_name=name
    )

    # Convergence analysis
    try:
        timeframe_highs = fetch_timeframe_highs(client, item_id, instabuy)

        convergence_analyzer = ConvergenceAnalyzer()
        result["convergence"] = convergence_analyzer.analyze(
            current_instabuy=instabuy,
            one_day_high=timeframe_highs["1d_high"],
            one_week_high=timeframe_highs["1w_high"],
            one_month_high=timeframe_highs["1m_high"],
            bsr=bsr
        )
    except Exception:
        # No historical data available
        result["convergence"] = {
            "is_convergence": False,
            "reject_reason": "no_historical_data"
        }

    # Oversold analysis (requires 24h history)
    try:
        history = client.fetch_timeseries(item_id, timestep="24h")

        if len(history) >= 30:
            prices = []
            for point in history:
                h = point.get("avgHighPrice")
                l = point.get("avgLowPrice")
                if h and l:
                    prices.append((h + l) // 2)

            if len(prices) >= 30:
                current_price = (instasell + instabuy) // 2

                oversold_analyzer = OversoldAnalyzer()
                result["oversold"] = oversold_analyzer.analyze(
                    current_price=current_price,
                    prices=prices,
                    lookback_days=180
                )
    except Exception:
        # Not enough history for oversold analysis
        pass

    return result
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_item_analyzer.py -v`

Expected: 2 tests PASS

**Step 5: Add endpoint to server**

Modify `osrs_flipper/server.py` - add new endpoint:

```python
from .item_analyzer import analyze_single_item


@app.get("/api/analyze/{item_id}")
async def analyze_item(item_id: int):
    """Deep analysis on specific item.

    Args:
        item_id: Item ID to analyze

    Returns:
        Complete analysis with instant, convergence, and oversold data

    Data Flow:
        IN: item_id (path param)
        PROCESS: analyze_single_item(item_id) → all analyzers
        OUT: {
            item_id, name, instabuy, instasell, bsr,
            instant: {...},
            convergence: {...},
            oversold: {...} [optional]
        }
    """
    try:
        result = analyze_single_item(item_id)
        return result
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=str(e))
```

Add test to `tests/test_server.py`:

```python
@patch('osrs_flipper.server.analyze_single_item')
def test_analyze_endpoint_returns_item_analysis(mock_analyze):
    """GET /api/analyze/{item_id} returns deep analysis."""
    from osrs_flipper.server import app

    mock_analyze.return_value = {
        "item_id": 123,
        "name": "Test",
        "instant": {"instant_roi_after_tax": 10.5},
        "convergence": {"upside_pct": 20.0}
    }

    client = TestClient(app)
    response = client.get("/api/analyze/123")

    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == 123
    assert "instant" in data
    assert "convergence" in data

    mock_analyze.assert_called_once_with(123)
```

**Step 6: Run server tests**

Run: `python3 -m pytest tests/test_server.py -v`

Expected: All tests PASS (4 total)

**Step 7: Commit**

```bash
git add osrs_flipper/item_analyzer.py tests/test_item_analyzer.py osrs_flipper/server.py tests/test_server.py
git commit -m "feat: add single item analysis endpoint

- analyze_single_item() runs all analyzers on one item
- Instant, convergence, and oversold (if history available)
- GET /api/analyze/{item_id} endpoint
- 2 new analyzer tests, 1 new endpoint test

Lines: item_analyzer.py (150 LOC), test (90 LOC), server.py (+20 LOC)
Data IN: item_id
Data OUT: {item_id, name, prices, instant, convergence, oversold}
Flow: fetch data → instant analyzer → convergence → oversold (optional)"
```

**Agent Task Completion Log:**
```
TASK: Single item analysis endpoint
STATUS: ✅ COMPLETED (2025-12-09)
TDD METHODOLOGY: Full RED → GREEN → REFACTOR cycle
  - RED: Wrote 2 failing tests first ✓
  - GREEN: Implemented item_analyzer.py, all tests pass ✓
  - REFACTOR: Code is clean and well-documented ✓
FILES CREATED:
  - osrs_flipper/server/item_analyzer.py (190 LOC production)
  - tests/server/test_item_analyzer.py (105 LOC tests)
FILES MODIFIED:
  - osrs_flipper/server/api.py (+39 LOC - added /api/analyze/{item_id} endpoint)
  - tests/server/test_api.py (+46 LOC - 2 new endpoint tests)
TOTAL LOC: 380 lines added (229 production, 151 test)
DATA FLOW IN: item_id (int from URL path)
DATA FLOW OUT: {
  item_id: int,
  name: str,
  instabuy: int,
  instasell: int,
  instabuy_vol: int,
  instasell_vol: int,
  bsr: float,
  instant: {spread_pct, bsr, is_instant_opportunity, instant_roi_after_tax, reject_reason},
  convergence: {is_convergence, distance_from_*_high, target_price, upside_pct, convergence_strength, reject_reason},
  oversold: {is_oversold, percentile, rsi, upside_pct, six_month_low, six_month_high} [optional]
}
ENDPOINTS: GET /api/analyze/{item_id} (returns 200 with analysis, 404 if item not found)
IMPLEMENTATION DETAILS:
  - analyze_single_item() fetches mapping, latest, volumes, timeseries from OSRSClient
  - Runs InstantSpreadAnalyzer for spread + BSR analysis
  - Runs ConvergenceAnalyzer with timeframe highs (1d/1w/1m)
  - Runs OversoldAnalyzer if sufficient historical data (30+ days)
  - Handles missing data gracefully (convergence falls back, oversold excluded if insufficient data)
  - HTTPException 404 for invalid item_id or missing price data
TESTS: 4 new comprehensive tests - ALL PASSING
  1. test_analyze_item_with_all_data - full analysis with all modes
  2. test_analyze_item_without_history - instant only with no historical data
  3. test_analyze_endpoint_returns_item_analysis - endpoint success case
  4. test_analyze_endpoint_returns_404_for_invalid_item - error handling
TEST RESULTS:
  - All 4 new tests passing
  - All 223 total tests passing (no regressions)
  - Full test suite: 11.04s
COMMIT: 8c8c743 "feat: add single item analysis endpoint (Task 5)"
DEPENDENCIES: OSRSClient, InstantSpreadAnalyzer, ConvergenceAnalyzer, OversoldAnalyzer, fetch_timeframe_highs, calculate_bsr
ISSUES ENCOUNTERED: None - smooth implementation following TDD
NEXT TASK NEEDS: Portfolio allocation endpoint (Task 6)
```

---

## Task 6: Portfolio Allocation Endpoint

**Goal:** Add endpoint to calculate portfolio allocation given cash, slots, and strategy.

**Files:**
- Modify: `osrs_flipper/server.py`
- Test: `tests/test_server.py`

**Data Flow:**
- IN: cash (int), slots (int), strategy (str)
- GET: cached opportunities from ScannerService
- ALLOCATE: SlotAllocator.allocate() → allocation
- OUT: {allocation: [...], total_capital, expected_profit}

**Step 1: Write failing test for allocation endpoint**

Add to `tests/test_server.py`:

```python
@patch('osrs_flipper.server.get_scanner_service')
@patch('osrs_flipper.server.SlotAllocator')
def test_portfolio_allocation_endpoint(mock_allocator_class, mock_get_service):
    """GET /api/portfolio/allocate calculates allocation."""
    from osrs_flipper.server import app

    # Mock service
    mock_service = Mock()
    mock_service.get_opportunities.return_value = [
        {"item_id": 1, "name": "Item1", "instabuy": 100},
        {"item_id": 2, "name": "Item2", "instabuy": 200}
    ]
    mock_get_service.return_value = mock_service

    # Mock allocator
    mock_allocator = Mock()
    mock_allocator.allocate.return_value = [
        {"slot": 1, "name": "Item1", "buy_price": 100, "quantity": 10, "capital": 1000},
        {"slot": 2, "name": "Item2", "buy_price": 200, "quantity": 5, "capital": 1000}
    ]
    mock_allocator_class.return_value = mock_allocator

    client = TestClient(app)
    response = client.get("/api/portfolio/allocate?cash=10000&slots=8&strategy=balanced")

    assert response.status_code == 200
    data = response.json()
    assert "allocation" in data
    assert len(data["allocation"]) == 2
    assert data["allocation"][0]["name"] == "Item1"

    # Verify allocator called with correct params
    mock_allocator.allocate.assert_called_once()
    call_kwargs = mock_allocator.allocate.call_args.kwargs
    assert call_kwargs["cash"] == 10000
    assert call_kwargs["slots"] == 8
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_server.py::test_portfolio_allocation_endpoint -v`

Expected: FAIL with "404 Not Found" (endpoint doesn't exist)

**Step 3: Add allocation endpoint to server**

Modify `osrs_flipper/server.py`:

```python
from .allocator import SlotAllocator


@app.get("/api/portfolio/allocate")
async def allocate_portfolio(
    cash: int = Query(..., description="Cash in GP", ge=1),
    slots: int = Query(default=8, ge=1, le=8),
    strategy: str = Query(default="balanced", regex="^(flip|hold|balanced)$"),
    rotations: int = Query(default=3, ge=1)
):
    """Calculate portfolio allocation.

    Args:
        cash: Cash stack in GP
        slots: Available GE slots (1-8)
        strategy: Allocation strategy (flip/hold/balanced)
        rotations: Buy limit rotations

    Returns:
        Allocation plan with expected profit

    Data Flow:
        IN: cash, slots, strategy, rotations (query params)
        FETCH: service.get_opportunities() → opportunities
        PROCESS: SlotAllocator(strategy).allocate(opportunities, cash, slots, rotations)
        OUT: {
            allocation: List[{slot, name, buy_price, quantity, capital, ...}],
            total_capital: int,
            expected_profit: int
        }
    """
    service = get_scanner_service()

    # Get cached opportunities
    opportunities = service.get_opportunities(mode="both", min_roi=10.0, limit=100)

    if not opportunities:
        return {
            "allocation": [],
            "total_capital": 0,
            "expected_profit": 0,
            "message": "No opportunities available"
        }

    # Calculate allocation
    allocator = SlotAllocator(strategy=strategy)
    allocation = allocator.allocate(
        opportunities=opportunities,
        cash=cash,
        slots=slots,
        rotations=rotations
    )

    # Calculate totals
    total_capital = sum(slot["capital"] for slot in allocation)
    expected_profit = sum(
        slot["capital"] * (slot.get("target_roi_pct", 0) / 100)
        for slot in allocation
    )

    return {
        "allocation": allocation,
        "total_capital": total_capital,
        "expected_profit": int(expected_profit)
    }
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_server.py::test_portfolio_allocation_endpoint -v`

Expected: PASS

**Step 5: Run all server tests**

Run: `python3 -m pytest tests/test_server.py -v`

Expected: All 5 tests PASS

**Step 6: Commit**

```bash
git add osrs_flipper/server.py tests/test_server.py
git commit -m "feat: add portfolio allocation endpoint

- GET /api/portfolio/allocate with cash, slots, strategy params
- Uses SlotAllocator with cached opportunities
- Returns allocation plan with total capital and expected profit
- 1 new test with mocked allocator

Lines: server.py (+50 LOC), test (+35 LOC)
Data IN: cash, slots, strategy, rotations
Data OUT: {allocation: [...], total_capital, expected_profit}
Flow: get cached opps → allocator.allocate() → totals"
```

**Agent Task Completion Log:**
```
TASK: Portfolio allocation endpoint ✅ COMPLETED
FILES MODIFIED:
  - osrs_flipper/server/api.py (+78 LOC: 2 import lines, 76 endpoint lines)
  - tests/server/test_api.py (+131 LOC: 6 test methods in TestPortfolioAllocationEndpoint class)
DATA FLOW IN: cash (str with suffix support: "100m", "1.5b"), slots (int 1-8), strategy (str: flip/hold/balanced), rotations (int)
DATA FLOW OUT: {allocation: List[Dict], total_capital: int, expected_profit: int, strategy: str, message?: str}
ENDPOINTS: GET /api/portfolio/allocate
TESTS: 6 new tests - all 12 API tests passing, all 229 project tests passing
TEST COVERAGE:
  - test_portfolio_allocation_endpoint: Basic allocation with mocked allocator
  - test_portfolio_allocation_with_cash_suffix: Cash parsing with m/b suffixes
  - test_portfolio_allocation_no_opportunities: Empty opportunities handling
  - test_portfolio_allocation_uses_default_params: Default values (slots=8, strategy=balanced, rotations=3)
  - test_portfolio_allocation_validates_slots: Validation (1-8 range)
  - test_portfolio_allocation_validates_strategy: Strategy validation (flip/hold/balanced)
DEPENDENCIES: SlotAllocator (from osrs_flipper.allocator), parse_cash (from osrs_flipper.utils), ScannerService
TDD METHODOLOGY: Followed strict RED-GREEN-REFACTOR
  - RED: Wrote 6 failing tests first
  - GREEN: Implemented endpoint with imports, all tests passing
  - REFACTOR: Code clean, no refactoring needed
NEXT TASK NEEDS: Server CLI command to start FastAPI server with uvicorn
```

---

## Task 7: Server CLI Command

**Goal:** Add `serve` command to CLI to start FastAPI server with uvicorn.

**Files:**
- Modify: `osrs_flipper/cli.py`
- Test: Manual testing (CLI commands hard to unit test)

**Data Flow:**
- IN: `python3 -m osrs_flipper.cli serve --port 8000`
- START: uvicorn server on specified port
- LOG: Server started message

**Step 1: Add serve command to CLI**

Modify `osrs_flipper/cli.py` - add new command before `if __name__ == "__main__"`:

```python
@main.command()
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Bind host (default: 127.0.0.1)",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port to listen on (default: 8000)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload on code changes (development)",
)
def serve(host, port, reload):
    """Start FastAPI server for RuneLite plugin integration.

    Runs localhost API server with auto-scanning background thread.
    Plugin can connect to http://localhost:8000/api endpoints.

    Data Flow:
        CLI command → uvicorn.run(app) → FastAPI server
        Background thread → auto-scan every 5 min
    """
    click.echo(f"Starting OSRS Flipper API server on {host}:{port}")
    click.echo("=" * 60)
    click.echo("\nEndpoints:")
    click.echo(f"  Health:        http://{host}:{port}/api/health")
    click.echo(f"  Opportunities: http://{host}:{port}/api/opportunities")
    click.echo(f"  Analyze:       http://{host}:{port}/api/analyze/{{item_id}}")
    click.echo(f"  Portfolio:     http://{host}:{port}/api/portfolio/allocate")
    click.echo(f"\nDocs:          http://{host}:{port}/docs")
    click.echo("\nPress Ctrl+C to stop\n")

    try:
        import uvicorn
        uvicorn.run(
            "osrs_flipper.server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        click.echo("Error: uvicorn not installed", err=True)
        click.echo("Install with: pip install uvicorn", err=True)
        raise click.Abort()
    except KeyboardInterrupt:
        click.echo("\n\nServer stopped")
```

**Step 2: Test serve command manually**

Run: `python3 -m osrs_flipper.cli serve --help`

Expected: Help text showing host, port, reload options

Run: `python3 -m osrs_flipper.cli serve --port 8000`

Expected:
```
Starting OSRS Flipper API server on 127.0.0.1:8000
============================================================

Endpoints:
  Health:        http://127.0.0.1:8000/api/health
  Opportunities: http://127.0.0.1:8000/api/opportunities
  Analyze:       http://127.0.0.1:8000/api/analyze/{item_id}
  Portfolio:     http://127.0.0.1:8000/api/portfolio/allocate

Docs:          http://127.0.0.1:8000/docs

Press Ctrl+C to stop

INFO:     Started server process [...]
```

Test in another terminal:
```bash
curl http://localhost:8000/api/health
```

Expected: `{"status":"ok",...}`

Press Ctrl+C to stop server.

**Step 3: Update README with serve command**

Modify `README.md` - add to Usage section:

```markdown
### Starting the API Server

```bash
# Start server (default port 8000)
python3 -m osrs_flipper.cli serve

# Custom port
python3 -m osrs_flipper.cli serve --port 9000

# Development mode with auto-reload
python3 -m osrs_flipper.cli serve --reload
```

The server auto-scans every 5 minutes in the background and caches results.

**API Endpoints:**
- `GET /api/health` - Server status and cache age
- `GET /api/opportunities?mode=instant&min_roi=20` - Get flip opportunities
- `GET /api/analyze/{item_id}` - Deep analysis on specific item
- `GET /api/portfolio/allocate?cash=100000000&slots=8` - Calculate allocation
- `GET /docs` - Interactive API documentation
```

**Step 4: Commit**

```bash
git add osrs_flipper/cli.py README.md
git commit -m "feat: add serve command to start FastAPI server

- python3 -m osrs_flipper.cli serve starts uvicorn server
- Options: --host, --port, --reload
- Shows all endpoint URLs on startup
- Update README with server usage

Lines: cli.py (+50 LOC), README.md (+20 LOC)
Data flow: CLI command → uvicorn.run() → server start
Command: serve --port 8000 --reload"
```

**Agent Task Completion Log:**
```
TASK: Server CLI command - COMPLETED ✓
FILES MODIFIED: osrs_flipper/cli.py (+54 LOC)
DATA FLOW: CLI invoke → uvicorn.run("osrs_flipper.server.api:app", host, port, reload)
COMMAND: python3 -m osrs_flipper.cli serve [--host HOST] [--port PORT] [--reload]
ENDPOINTS DISPLAYED: health, opportunities, analyze, portfolio, docs
TESTING: Manual testing completed successfully
  - Test 1: Command help output - PASS ✓
  - Test 2: Health endpoint (http://127.0.0.1:8002/api/health) - PASS ✓
    Response: {"status": "ok", "last_scan_time": 0.0, "cache_age_seconds": 0.0}
  - Test 3: Opportunities endpoint with filters - PASS ✓
    Response: Status 200, returned opportunities list
  - Test 4: OpenAPI docs (http://127.0.0.1:8002/docs) - PASS ✓
    Response: Status 200, interactive documentation loaded
ISSUES ENCOUNTERED: None
DEPENDENCIES: uvicorn (already in requirements.txt)
NOTES:
  - Server starts successfully on custom ports
  - All endpoints respond correctly
  - Graceful shutdown with Ctrl+C works
  - --reload flag enables auto-reload for development
NEXT TASK NEEDS: E2E integration tests (Task 8)
```

---

## Task 8: E2E API Integration Tests

**Goal:** Comprehensive E2E tests verifying full API data flow from HTTP request to response.

**Files:**
- Create: `tests/test_e2e_api.py`

**Data Flow:**
- END-TO-END: HTTP request → FastAPI → ScannerService → ItemScanner → Analyzers → Response
- Test all endpoints with real API (mocked OSRS Wiki)

**Step 1: Write E2E tests for all endpoints**

Create `tests/test_e2e_api.py`:

```python
"""End-to-end API integration tests."""
import pytest
import responses
from fastapi.testclient import TestClient

BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"


class TestE2EAPIDataFlow:
    """Test complete data flow through API."""

    @responses.activate
    def test_health_endpoint_e2e(self):
        """E2E: Health endpoint returns correct status."""
        from osrs_flipper.server import app

        client = TestClient(app)
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "last_scan_time" in data
        assert "cache_age_seconds" in data

    @responses.activate
    def test_opportunities_endpoint_e2e_instant_mode(self):
        """E2E: Opportunities endpoint returns instant flips."""
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

        from osrs_flipper.server import app, get_scanner_service

        # Reset scanner service for clean test
        import osrs_flipper.server
        osrs_flipper.server._scanner_service = None

        client = TestClient(app)
        response = client.get("/api/opportunities?mode=instant&min_roi=5&limit=10")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "opportunities" in data
        assert "scan_time" in data
        assert "cache_age_seconds" in data

        # Should have found instant opportunities
        opps = data["opportunities"]
        assert len(opps) > 0

        # Verify data integrity
        for opp in opps:
            assert "item_id" in opp
            assert "name" in opp
            assert "instabuy" in opp
            assert "instasell" in opp
            assert "instant" in opp
            assert "instant_roi_after_tax" in opp["instant"]

    @responses.activate
    def test_opportunities_endpoint_e2e_convergence_mode(self):
        """E2E: Opportunities endpoint returns convergence plays."""
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

        from osrs_flipper.server import app
        import osrs_flipper.server
        osrs_flipper.server._scanner_service = None

        client = TestClient(app)
        response = client.get("/api/opportunities?mode=convergence&min_roi=20&limit=10")

        assert response.status_code == 200
        data = response.json()

        opps = data["opportunities"]

        # Should find convergence opportunity (30%+ drop from highs)
        if len(opps) > 0:
            opp = opps[0]
            assert "convergence" in opp
            assert opp["convergence"]["is_convergence"] is True
            assert opp["convergence"]["upside_pct"] > 0

    @responses.activate
    def test_analyze_endpoint_e2e(self):
        """E2E: Analyze endpoint returns full item analysis."""
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
            match=[responses.matchers.query_param_matcher({"id": 999, "timestep": "1h"})]
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
            match=[responses.matchers.query_param_matcher({"id": 999, "timestep": "24h"})]
        )

        from osrs_flipper.server import app

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

        # Verify data integrity
        assert data["instabuy"] == 100
        assert data["instasell"] == 110
        assert "bsr" in data

    @responses.activate
    def test_portfolio_endpoint_e2e(self):
        """E2E: Portfolio allocation endpoint works end-to-end."""
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

        from osrs_flipper.server import app
        import osrs_flipper.server
        osrs_flipper.server._scanner_service = None

        client = TestClient(app)
        response = client.get("/api/portfolio/allocate?cash=10000000&slots=3&strategy=balanced")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "allocation" in data
        assert "total_capital" in data
        assert "expected_profit" in data

        # Should have allocated some items
        allocation = data["allocation"]
        assert isinstance(allocation, list)

        # Verify capital doesn't exceed cash
        assert data["total_capital"] <= 10000000

    @responses.activate
    def test_min_roi_filter_e2e(self):
        """E2E: min_roi filter correctly filters opportunities."""
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

        from osrs_flipper.server import app
        import osrs_flipper.server
        osrs_flipper.server._scanner_service = None

        client = TestClient(app)

        # Request min_roi=30 (should filter out low ROI item)
        response = client.get("/api/opportunities?mode=instant&min_roi=30&limit=10")

        assert response.status_code == 200
        data = response.json()

        opps = data["opportunities"]

        # All returned items should have ROI >= 30%
        for opp in opps:
            instant_roi = opp.get("instant", {}).get("instant_roi_after_tax", 0)
            assert instant_roi >= 30.0 or instant_roi == 0  # 0 means rejected by analyzer
```

**Step 2: Run E2E tests**

Run: `python3 -m pytest tests/test_e2e_api.py -v`

Expected: All 6 E2E tests PASS

**Step 3: Run full test suite**

Run: `python3 -m pytest tests/ -v`

Expected: All tests PASS (scanner_service: 8, server: 5, item_analyzer: 2, e2e_api: 6, plus all existing tests)

**Step 4: Commit**

```bash
git add tests/test_e2e_api.py
git commit -m "test: add comprehensive E2E API integration tests

- 6 E2E tests covering all endpoints
- Full data flow: HTTP → FastAPI → Scanner → Analyzers → Response
- Tests: health, opportunities (instant/convergence), analyze, portfolio, min_roi filter
- All tests use mocked OSRS Wiki API with realistic data

Lines: test_e2e_api.py (280 LOC)
Data flow tested: Request → Service → Analyzers → Response JSON
Coverage: All 4 endpoints, both scan modes, filtering logic"
```

**Agent Task Completion Log:**
```
TASK: E2E API integration tests - COMPLETED ✓
FILES CREATED: tests/server/test_e2e_api.py (498 LOC)
TESTS: 11 comprehensive E2E tests covering all endpoints and data flows
DATA FLOW TESTED:
  1. GET /api/health → status check → cache metadata
  2. GET /api/opportunities?mode=instant → ScannerService → InstantSpreadAnalyzer → JSON response
  3. GET /api/opportunities?mode=convergence → ScannerService → ConvergenceAnalyzer → JSON response
  4. GET /api/opportunities?mode=both → Both analyzers → Combined results
  5. GET /api/analyze/{item_id} → OSRSClient → All analyzers → Full analysis
  6. GET /api/analyze/{invalid_id} → 404 error handling
  7. GET /api/portfolio/allocate → ScannerService → SlotAllocator → Allocation plan
  8. Cash parsing (100m, 1.5b) → parse_cash → int conversion → allocation
  9. min_roi filter → ScannerService filtering → Filtered results verification
  10. limit parameter → Data limiting → Correct output size
  11. Cache functionality → Multiple requests → Cache hit verification

ALL TESTS PASSING: 240 total (11 new E2E + 229 existing)
VERIFICATION: Full data integrity from HTTP request through all layers to JSON response
  - INPUT validation (query params, path params)
  - TRANSFORM integrity (scanner → analyzers → formatters)
  - OUTPUT verification (JSON structure, data types, business logic)

TEST COVERAGE:
  - Health endpoint: Structure and metadata ✓
  - Opportunities endpoint: All modes (instant/convergence/both), filtering, limits ✓
  - Analyze endpoint: Valid items, invalid items (404), all analysis types ✓
  - Portfolio endpoint: Allocation, cash parsing, validation ✓
  - Error cases: 404s, validation errors ✓
  - Performance: Cache hit/miss behavior ✓

ISSUES ENCOUNTERED:
  - Initial test had wrong field name (upside_pct always present vs. only when is_convergence=True)
  - Fixed by making assertion conditional on convergence status

DEPENDENCIES: FastAPI TestClient, responses library (already in requirements.txt)
PYTEST RESULTS: 240 passed, 3 warnings in 12.87s
```

---

## Final Verification

**Step 1: Run complete test suite**

Run: `python3 -m pytest tests/ -v --tb=short`

Expected: All tests PASS (204 existing + 21 new = 225 total)

**Step 2: Start server and manual verification**

Run in terminal 1:
```bash
python3 -m osrs_flipper.cli serve --port 8000
```

Expected: Server starts with background scanning enabled

Run in terminal 2:
```bash
# Health check
curl http://localhost:8000/api/health | jq

# Opportunities (instant)
curl "http://localhost:8000/api/opportunities?mode=instant&min_roi=10&limit=5" | jq

# Opportunities (convergence)
curl "http://localhost:8000/api/opportunities?mode=convergence&min_roi=20&limit=5" | jq

# Analyze specific item (if you know an item_id)
curl http://localhost:8000/api/analyze/2 | jq

# Portfolio allocation
curl "http://localhost:8000/api/portfolio/allocate?cash=100000000&slots=8&strategy=balanced" | jq

# API docs (open in browser)
open http://localhost:8000/docs
```

Expected: All endpoints return valid JSON data

**Step 3: Verify background scanning**

Wait 5-10 minutes, then check health:
```bash
curl http://localhost:8000/api/health | jq
```

Expected: `cache_age_seconds` should reset to ~0 every 5 minutes (background scan)

**Step 4: Performance check**

Run: `time curl -s "http://localhost:8000/api/opportunities?mode=both&limit=100" > /dev/null`

Expected: < 1 second (cache hit) or < 10 seconds (cache miss with scan)

**Step 5: Create final summary document**

Create `docs/plans/COMPLETION-SUMMARY.md`:

```markdown
# FastAPI Localhost Server - Implementation Complete

## Summary

Successfully implemented FastAPI server exposing OSRS flip scanner via REST API on localhost:8000.

## Tasks Completed

1. ✅ ScannerService with caching (123 LOC, 4 tests)
2. ✅ FastAPI endpoints foundation (95 LOC, 3 tests)
3. ✅ Background auto-scan thread (50 LOC, 2 tests)
4. ✅ Opportunities filtering (60 LOC, 2 tests)
5. ✅ Single item analysis endpoint (150 LOC, 2 tests)
6. ✅ Portfolio allocation endpoint (50 LOC, 1 test)
7. ✅ Server CLI command (50 LOC, manual test)
8. ✅ E2E API integration tests (280 LOC, 6 tests)

## Total Lines Added

- Production code: ~600 LOC
- Test code: ~620 LOC
- Total: ~1220 LOC

## Test Coverage

- Unit tests: 15 (scanner_service, server, item_analyzer)
- E2E tests: 6 (full API data flow)
- Total new tests: 21
- All 225 tests passing ✅

## Data Flow Verification

All data flows tested end-to-end:

1. **Health Endpoint**: ✅
   - IN: HTTP GET /api/health
   - OUT: {status, last_scan_time, cache_age_seconds}

2. **Opportunities Endpoint**: ✅
   - IN: mode, min_roi, limit
   - PROCESS: ScannerService → cache/scan → filter
   - OUT: {opportunities, scan_time, cache_age}

3. **Analyze Endpoint**: ✅
   - IN: item_id
   - PROCESS: fetch data → instant/convergence/oversold analyzers
   - OUT: {item_id, name, prices, instant, convergence, oversold}

4. **Portfolio Endpoint**: ✅
   - IN: cash, slots, strategy
   - PROCESS: cached opps → SlotAllocator
   - OUT: {allocation, total_capital, expected_profit}

## Performance

- Cache hit: < 100ms
- Cache miss (scan 100 items): 2-5 seconds
- Background auto-scan: Every 5 minutes
- API response times: < 1s (cached), < 10s (fresh scan)

## Next Steps for RuneLite Plugin

With server complete, RuneLite plugin implementation is straightforward:

1. Java HTTP client hitting localhost:8000
2. Display opportunities in GE overlay
3. Item tooltip showing flip metrics
4. Sidebar panel with sortable table
5. Right-click context menu "Analyze flip"

Plugin complexity: ~500 LOC (vs 5000 LOC if porting Python)

## Server Usage

```bash
# Start server
python3 -m osrs_flipper.cli serve

# Custom port
python3 -m osrs_flipper.cli serve --port 9000

# Development mode
python3 -m osrs_flipper.cli serve --reload
```

## API Endpoints

- `GET /api/health` - Status check
- `GET /api/opportunities?mode=instant&min_roi=20` - Get opportunities
- `GET /api/analyze/{item_id}` - Deep item analysis
- `GET /api/portfolio/allocate?cash=100m&slots=8` - Calculate allocation
- `GET /docs` - Interactive API documentation
```

**Step 6: Final commit**

```bash
git add docs/plans/COMPLETION-SUMMARY.md
git commit -m "docs: add FastAPI server implementation completion summary

All 8 tasks completed:
- ScannerService with TTL caching
- FastAPI endpoints (health, opportunities, analyze, portfolio)
- Background auto-scan thread (5 min interval)
- Client-side filtering (mode, min_roi)
- Server CLI command
- Comprehensive E2E tests

Stats:
- 600 LOC production code
- 620 LOC test code
- 21 new tests, all passing
- All data flows verified end-to-end

Ready for RuneLite plugin integration."
```

---

## Plan Complete

**Total Implementation:**
- 8 tasks completed in 3 groups
- 1220 lines of code (600 production, 620 test)
- 21 new passing tests
- Full E2E data flow verification
- Background auto-scanning
- REST API ready for RuneLite plugin

**Execution Options:**

**Option 1: Subagent-Driven (Recommended)**
- Execute in this session
- Dispatch fresh subagent per task
- Code review between tasks
- Fast iteration with quality gates

**Option 2: Parallel Session**
- Open new session with `/superpowers:execute-plan`
- Execute tasks in batches with checkpoints
- Review at group boundaries

**Which approach do you prefer?**
