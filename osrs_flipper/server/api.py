"""FastAPI server exposing flip scanner via REST API."""
import logging
from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .scanner_service import ScannerService
from .item_analyzer import analyze_single_item

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


@app.get("/api/status")
async def get_status():
    """Get server status and cache information.

    This is an alias for /api/health for compatibility.

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
    mode: str = Query(default="both", pattern="^(instant|convergence|both|oversold|oscillator|all)$"),
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
        PROCESS: service.scan(mode, min_roi, limit) → opportunities
        OUT: {
            "opportunities": List[Dict],
            "scan_time": float,
            "cache_age_seconds": float
        }
    """
    service = get_scanner_service()

    opportunities = service.scan(
        mode=mode,
        min_roi=min_roi,
        limit=limit
    )

    return {
        "opportunities": opportunities,
        "scan_time": service.last_scan_time,
        "cache_age_seconds": service.get_cache_age()
    }


@app.get("/api/analyze/{item_id}")
async def analyze_item(item_id: int):
    """Deep analysis on specific item.

    Performs comprehensive analysis including instant spread, convergence,
    and oversold detection (if historical data available).

    Args:
        item_id: Item ID to analyze

    Returns:
        Complete analysis with instant, convergence, and oversold data

    Raises:
        HTTPException: 404 if item not found or missing price data

    Data Flow:
        IN: item_id (path param)
        PROCESS: analyze_single_item(item_id) → all analyzers
        OUT: {
            item_id: int,
            name: str,
            instabuy: int,
            instasell: int,
            bsr: float,
            instant: {...},
            convergence: {...},
            oversold: {...} [optional]
        }
    """
    try:
        result = analyze_single_item(item_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
