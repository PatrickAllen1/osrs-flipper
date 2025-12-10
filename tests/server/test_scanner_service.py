"""Tests for ScannerService caching."""
import time
import pytest
from unittest.mock import Mock, patch
from osrs_flipper.server.scanner_service import ScannerService


class TestScannerServiceCache:
    """Test caching behavior."""

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
    def test_cache_hit_returns_cached_data(self, mock_client_class, mock_scanner_class):
        """When cache is fresh, return cached data without scanning."""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        # Mock data with instant key to match mode="instant"
        mock_scanner.scan.return_value = [{"item_id": 1, "name": "Test", "instant": {"instant_roi_after_tax": 15.0}}]

        # Create service (cache_ttl = 300 seconds)
        service = ScannerService(cache_ttl=300)

        # First call - should scan
        result1 = service.scan(mode="instant", min_roi=10.0, limit=10)

        # Second call immediately - should use cache
        result2 = service.scan(mode="instant", min_roi=10.0, limit=10)

        # Verify scanner.scan called only once (cache hit on second call)
        assert mock_scanner.scan.call_count == 1
        assert result1 == result2
        assert len(result1) == 1
        assert result1[0]["name"] == "Test"

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
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
        service.scan(mode="both", min_roi=15.0, limit=20)

        # Wait for cache to expire
        time.sleep(1.1)

        # Second call - cache expired
        service.scan(mode="both", min_roi=15.0, limit=20)

        # Should have scanned twice
        assert mock_scanner.scan.call_count == 2

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
    def test_force_scan_bypasses_cache(self, mock_client_class, mock_scanner_class):
        """force_scan parameter always triggers scan regardless of cache state."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        mock_scanner.scan.return_value = [{"item_id": 3}]

        service = ScannerService(cache_ttl=300)

        # Normal call
        service.scan(mode="instant", min_roi=5.0, limit=5)

        # Force scan should bypass cache
        service.scan(mode="convergence", min_roi=10.0, limit=10, force_scan=True)

        # Should have scanned twice
        assert mock_scanner.scan.call_count == 2

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
    def test_cache_age_calculation(self, mock_client_class, mock_scanner_class):
        """Verify cache age is calculated correctly."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        mock_scanner.scan.return_value = []

        service = ScannerService(cache_ttl=300)

        # Scan
        service.scan(mode="both", min_roi=20.0, limit=50)

        # Check cache age immediately
        age1 = service.get_cache_age()
        assert age1 < 1.0

        # Wait a bit
        time.sleep(0.5)

        age2 = service.get_cache_age()
        assert age2 >= 0.5

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
    def test_get_cached_opportunities_returns_cache(self, mock_client_class, mock_scanner_class):
        """get_cached_opportunities returns cached data without triggering scan."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        mock_scanner.scan.return_value = [{"item_id": 4, "name": "Cached"}]

        service = ScannerService(cache_ttl=300)

        # First, populate cache
        service.scan(mode="instant", min_roi=10.0, limit=10)

        # Now get cached opportunities
        cached = service.get_cached_opportunities()

        # Should return same data
        assert len(cached) == 1
        assert cached[0]["name"] == "Cached"

        # Should not have triggered additional scan
        assert mock_scanner.scan.call_count == 1

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
    def test_cache_metadata_tracking(self, mock_client_class, mock_scanner_class):
        """Verify last_scan_time is tracked correctly."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner
        mock_scanner.scan.return_value = [{"item_id": 5}]

        service = ScannerService(cache_ttl=300)

        # Initially no scan
        assert service.last_scan_time == 0.0

        # Perform scan
        service.scan(mode="both", min_roi=10.0, limit=10)

        # Should have recorded scan time
        assert service.last_scan_time > 0.0
        scan_time = service.last_scan_time

        # Wait and scan again with force
        time.sleep(0.1)
        service.scan(mode="both", min_roi=10.0, limit=10, force_scan=True)

        # Scan time should have updated
        assert service.last_scan_time > scan_time


class TestCachedFiltering:
    """Test client-side filtering of cached opportunities."""

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
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

        # Scan to populate cache with mode="both"
        service.scan(mode="both", min_roi=5.0, limit=100, force_scan=True)

        # Request only instant opportunities
        results = service.scan(mode="instant", min_roi=5.0, limit=10)

        # Should only return items with "instant" key
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "Instant Item" in names
        assert "Both" in names
        assert "Conv Item" not in names

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
    def test_filter_by_mode_convergence(self, mock_client_class, mock_scanner_class):
        """Filter cached opportunities by mode=convergence."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner

        mock_scanner.scan.return_value = [
            {"item_id": 1, "name": "Instant Item", "instant": {"instant_roi_after_tax": 15.0}},
            {"item_id": 2, "name": "Conv Item", "convergence": {"upside_pct": 25.0}},
            {"item_id": 3, "name": "Both", "instant": {"instant_roi_after_tax": 10.0}, "convergence": {"upside_pct": 20.0}}
        ]

        service = ScannerService(cache_ttl=300)

        # Scan to populate cache
        service.scan(mode="both", min_roi=5.0, limit=100, force_scan=True)

        # Request only convergence opportunities
        results = service.scan(mode="convergence", min_roi=5.0, limit=10)

        # Should only return items with "convergence" key
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "Conv Item" in names
        assert "Both" in names
        assert "Instant Item" not in names

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
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
        service.scan(mode="both", min_roi=0.0, limit=100, force_scan=True)

        # Request min_roi >= 20
        results = service.scan(mode="both", min_roi=20.0, limit=10)

        # Should filter out Low ROI (5.0%)
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "High ROI" in names
        assert "Med ROI" in names
        assert "Low ROI" not in names

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
    def test_filter_by_mode_and_min_roi(self, mock_client_class, mock_scanner_class):
        """Filter by both mode and min_roi together."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner

        mock_scanner.scan.return_value = [
            {"item_id": 1, "name": "High Instant", "instant": {"instant_roi_after_tax": 50.0}},
            {"item_id": 2, "name": "Low Instant", "instant": {"instant_roi_after_tax": 5.0}},
            {"item_id": 3, "name": "High Conv", "convergence": {"upside_pct": 40.0}},
            {"item_id": 4, "name": "Both High", "instant": {"instant_roi_after_tax": 30.0}, "convergence": {"upside_pct": 35.0}}
        ]

        service = ScannerService(cache_ttl=300)
        service.scan(mode="both", min_roi=0.0, limit=100, force_scan=True)

        # Request instant mode with min_roi >= 20
        results = service.scan(mode="instant", min_roi=20.0, limit=10)

        # Should only return instant items with ROI >= 20
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "High Instant" in names
        assert "Both High" in names
        assert "Low Instant" not in names
        assert "High Conv" not in names

    @patch('osrs_flipper.server.scanner_service.ItemScanner')
    @patch('osrs_flipper.server.scanner_service.OSRSClient')
    def test_filter_respects_limit(self, mock_client_class, mock_scanner_class):
        """Filter respects the limit parameter."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_scanner = Mock()
        mock_scanner_class.return_value = mock_scanner

        # Many high ROI items
        mock_scanner.scan.return_value = [
            {"item_id": i, "name": f"Item {i}", "instant": {"instant_roi_after_tax": 50.0}}
            for i in range(1, 11)
        ]

        service = ScannerService(cache_ttl=300)
        service.scan(mode="instant", min_roi=0.0, limit=100, force_scan=True)

        # Request with limit=3
        results = service.scan(mode="instant", min_roi=5.0, limit=3)

        # Should return only 3 items
        assert len(results) == 3
