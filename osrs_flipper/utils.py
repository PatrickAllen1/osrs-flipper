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
