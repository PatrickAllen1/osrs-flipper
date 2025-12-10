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


class TestAnalyzeEndpoint:
    """Test /api/analyze/{item_id} endpoint."""

    @patch('osrs_flipper.server.api.analyze_single_item')
    def test_analyze_endpoint_returns_item_analysis(self, mock_analyze):
        """GET /api/analyze/{item_id} returns deep analysis."""
        from osrs_flipper.server.api import app

        mock_analyze.return_value = {
            "item_id": 123,
            "name": "Test Item",
            "instabuy": 100,
            "instasell": 110,
            "bsr": 1.5,
            "instant": {"instant_roi_after_tax": 10.5, "spread_pct": 10.0},
            "convergence": {"upside_pct": 20.0, "is_convergence": True}
        }

        client = TestClient(app)
        response = client.get("/api/analyze/123")

        assert response.status_code == 200
        data = response.json()
        assert data["item_id"] == 123
        assert data["name"] == "Test Item"
        assert "instant" in data
        assert "convergence" in data

        mock_analyze.assert_called_once_with(123)

    @patch('osrs_flipper.server.api.analyze_single_item')
    def test_analyze_endpoint_returns_404_for_invalid_item(self, mock_analyze):
        """GET /api/analyze/{item_id} returns 404 for invalid item."""
        from osrs_flipper.server.api import app

        mock_analyze.side_effect = ValueError("Item 999 not found in mapping")

        client = TestClient(app)
        response = client.get("/api/analyze/999")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestPortfolioAllocationEndpoint:
    """Test /api/portfolio/allocate endpoint."""

    @patch('osrs_flipper.server.api.get_scanner_service')
    @patch('osrs_flipper.server.api.SlotAllocator')
    def test_portfolio_allocation_endpoint(self, mock_allocator_class, mock_get_service):
        """GET /api/portfolio/allocate calculates allocation."""
        from osrs_flipper.server.api import app

        # Mock service
        mock_service = Mock()
        mock_service.scan.return_value = [
            {"item_id": 1, "name": "Item1", "current_price": 100, "score": 10.0},
            {"item_id": 2, "name": "Item2", "current_price": 200, "score": 8.0}
        ]
        mock_service.get_cache_age.return_value = 60.0
        mock_get_service.return_value = mock_service

        # Mock allocator
        mock_allocator = Mock()
        mock_allocator.allocate.return_value = [
            {"slot": 1, "name": "Item1", "buy_price": 100, "quantity": 10, "capital": 1000, "target_roi_pct": 15.0},
            {"slot": 2, "name": "Item2", "buy_price": 200, "quantity": 5, "capital": 1000, "target_roi_pct": 20.0}
        ]
        mock_allocator_class.return_value = mock_allocator

        client = TestClient(app)
        response = client.get("/api/portfolio/allocate?cash=10000&slots=8&strategy=balanced")

        assert response.status_code == 200
        data = response.json()
        assert "allocation" in data
        assert len(data["allocation"]) == 2
        assert data["allocation"][0]["name"] == "Item1"
        assert data["total_capital"] == 2000
        assert data["expected_profit"] == 350  # (1000 * 0.15) + (1000 * 0.20)
        assert data["strategy"] == "balanced"

        # Verify allocator called with correct params
        mock_allocator.allocate.assert_called_once()
        call_kwargs = mock_allocator.allocate.call_args.kwargs
        assert call_kwargs["cash"] == 10000
        assert call_kwargs["slots"] == 8

    @patch('osrs_flipper.server.api.get_scanner_service')
    def test_portfolio_allocation_with_cash_suffix(self, mock_get_service):
        """Accept cash amounts with suffixes like 100m, 1.5b."""
        from osrs_flipper.server.api import app

        mock_service = Mock()
        mock_service.scan.return_value = []
        mock_get_service.return_value = mock_service

        client = TestClient(app)
        response = client.get("/api/portfolio/allocate?cash=100m&slots=8")

        assert response.status_code == 200
        # Should parse 100m to 100,000,000

    @patch('osrs_flipper.server.api.get_scanner_service')
    @patch('osrs_flipper.server.api.SlotAllocator')
    def test_portfolio_allocation_no_opportunities(self, mock_allocator_class, mock_get_service):
        """Return empty allocation when no opportunities available."""
        from osrs_flipper.server.api import app

        mock_service = Mock()
        mock_service.scan.return_value = []
        mock_get_service.return_value = mock_service

        client = TestClient(app)
        response = client.get("/api/portfolio/allocate?cash=10000&slots=8")

        assert response.status_code == 200
        data = response.json()
        assert data["allocation"] == []
        assert data["total_capital"] == 0
        assert data["expected_profit"] == 0
        assert "message" in data

    @patch('osrs_flipper.server.api.get_scanner_service')
    @patch('osrs_flipper.server.api.SlotAllocator')
    def test_portfolio_allocation_uses_default_params(self, mock_allocator_class, mock_get_service):
        """Use default values for slots, strategy, rotations."""
        from osrs_flipper.server.api import app

        mock_service = Mock()
        mock_service.scan.return_value = [
            {"item_id": 1, "name": "Item1", "current_price": 100, "score": 10.0}
        ]
        mock_get_service.return_value = mock_service

        mock_allocator = Mock()
        mock_allocator.allocate.return_value = []
        mock_allocator_class.return_value = mock_allocator

        client = TestClient(app)
        response = client.get("/api/portfolio/allocate?cash=5000000")

        assert response.status_code == 200

        # Verify default strategy used
        mock_allocator_class.assert_called_once_with(strategy="balanced")

        # Verify default slots and rotations
        call_kwargs = mock_allocator.allocate.call_args.kwargs
        assert call_kwargs["slots"] == 8  # default
        assert call_kwargs["rotations"] == 3  # default

    def test_portfolio_allocation_validates_slots(self):
        """Reject invalid slot values."""
        from osrs_flipper.server.api import app

        client = TestClient(app)

        # Slots too low
        response = client.get("/api/portfolio/allocate?cash=10000&slots=0")
        assert response.status_code == 422

        # Slots too high
        response = client.get("/api/portfolio/allocate?cash=10000&slots=9")
        assert response.status_code == 422

    def test_portfolio_allocation_validates_strategy(self):
        """Reject invalid strategy values."""
        from osrs_flipper.server.api import app

        client = TestClient(app)
        response = client.get("/api/portfolio/allocate?cash=10000&strategy=invalid")
        assert response.status_code == 422
