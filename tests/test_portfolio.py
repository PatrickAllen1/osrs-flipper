# tests/test_portfolio.py
import pytest
from osrs_flipper.portfolio import PortfolioManager, PRESETS, ParetoFrontier


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


def test_pareto_frontier_calculates_roi_per_day():
    """Pareto frontier calculates time-adjusted ROI."""
    frontier = ParetoFrontier()

    items = [
        {"name": "Quick Flip", "roi_pct": 15, "expected_hold_days": 2},
        {"name": "Mid Hold", "roi_pct": 30, "expected_hold_days": 7},
        {"name": "Long Hold", "roi_pct": 50, "expected_hold_days": 14},
    ]

    scored = frontier.score_items(items)

    assert scored[0]["roi_per_day"] == 7.5  # 15% / 2 days
    assert abs(scored[2]["roi_per_day"] - 3.57) < 0.1  # 50% / 14 days


def test_pareto_frontier_identifies_efficient_items():
    """Pareto frontier identifies non-dominated items."""
    frontier = ParetoFrontier()

    items = [
        {"name": "Efficient", "roi_pct": 30, "expected_hold_days": 5},
        {"name": "Dominated", "roi_pct": 20, "expected_hold_days": 7},
        {"name": "Also Efficient", "roi_pct": 50, "expected_hold_days": 12},
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

    assert score["avg_roi_pct"] == 30
    assert score["avg_hold_days"] == 6.5
    assert "efficiency_score" in score
