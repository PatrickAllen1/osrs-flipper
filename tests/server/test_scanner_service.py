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
        mock_scanner.scan.return_value = [{"item_id": 1, "name": "Test"}]

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
