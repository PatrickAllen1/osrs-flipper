# tests/test_utils.py
import pytest
from osrs_flipper.utils import parse_cash


@pytest.mark.parametrize("input_str,expected", [
    ("120m", 120_000_000),
    ("120M", 120_000_000),
    ("1.5b", 1_500_000_000),
    ("1.5B", 1_500_000_000),
    ("50k", 50_000),
    ("50K", 50_000),
    ("1000000", 1_000_000),
    ("1,000,000", 1_000_000),
])
def test_parse_cash_handles_suffixes(input_str, expected):
    assert parse_cash(input_str) == expected


def test_parse_cash_raises_on_invalid():
    with pytest.raises(ValueError):
        parse_cash("invalid")
