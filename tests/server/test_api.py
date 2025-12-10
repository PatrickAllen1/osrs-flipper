"""Tests for FastAPI server endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch


class TestHealthEndpoint:
    """Test /api/health endpoint."""

    @patch('osrs_flipper.server.api.get_scanner_service')
    def test_health_check_returns_ok(self, mock_get_service):
        """Health endpoint returns status and cache info."""
        # Import here to allow patching before app creation
        from osrs_flipper.server.api import app

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


class TestStatusEndpoint:
    """Test /api/status endpoint."""

    @patch('osrs_flipper.server.api.get_scanner_service')
    def test_status_check_returns_ok(self, mock_get_service):
        """Status endpoint returns status and cache info."""
        from osrs_flipper.server.api import app

        # Mock scanner service
        mock_service = Mock()
        mock_service.get_cache_age.return_value = 60.0
        mock_service.last_scan_time = 9876543210.0
        mock_get_service.return_value = mock_service

        client = TestClient(app)
        response = client.get("/api/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["cache_age_seconds"] == 60.0
        assert data["last_scan_time"] == 9876543210.0


class TestOpportunitiesEndpoint:
    """Test /api/opportunities endpoint."""

    @patch('osrs_flipper.server.api.get_scanner_service')
    def test_opportunities_returns_cached_data(self, mock_get_service):
        """GET /api/opportunities returns opportunities from service."""
        from osrs_flipper.server.api import app

        mock_service = Mock()
        mock_service.scan.return_value = [
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
        mock_service.scan.assert_called_once_with(
            mode="instant",
            min_roi=10.0,
            limit=20  # default
        )

    @patch('osrs_flipper.server.api.get_scanner_service')
    def test_opportunities_respects_query_params(self, mock_get_service):
        """Query params are passed to scanner service."""
        from osrs_flipper.server.api import app

        mock_service = Mock()
        mock_service.scan.return_value = []
        mock_service.last_scan_time = 0.0
        mock_service.get_cache_age.return_value = 0.0
        mock_get_service.return_value = mock_service

        client = TestClient(app)
        response = client.get("/api/opportunities?mode=convergence&min_roi=25.5&limit=50")

        assert response.status_code == 200

        # Verify params passed correctly
        mock_service.scan.assert_called_once_with(
            mode="convergence",
            min_roi=25.5,
            limit=50
        )
