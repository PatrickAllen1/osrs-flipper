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
