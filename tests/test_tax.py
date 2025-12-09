"""Tests for GE tax calculations."""
import pytest


class TestTaxExemptRegistry:
    """Tests for the tax-exempt items registry."""

    def test_known_exempt_items(self):
        """Verify known tax-exempt items are recognized."""
        from osrs_flipper.tax import is_tax_exempt

        assert is_tax_exempt("Bronze arrow") is True
        assert is_tax_exempt("Mind rune") is True
        assert is_tax_exempt("Lobster") is True
        assert is_tax_exempt("Old school bond") is True

    def test_non_exempt_items(self):
        """Verify non-exempt items are correctly identified."""
        from osrs_flipper.tax import is_tax_exempt

        assert is_tax_exempt("Abyssal whip") is False
        assert is_tax_exempt("Dragon bones") is False
        assert is_tax_exempt("Twisted bow") is False

    def test_case_insensitive(self):
        """Verify case-insensitive matching for tax-exempt items."""
        from osrs_flipper.tax import is_tax_exempt

        assert is_tax_exempt("bronze arrow") is True
        assert is_tax_exempt("BRONZE ARROW") is True
        assert is_tax_exempt("Bronze Arrow") is True


class TestTaxCalculation:
    """Tests for GE tax calculation function."""

    def test_tax_on_regular_item(self):
        """Verify 2% tax is applied to regular items."""
        from osrs_flipper.tax import calculate_ge_tax

        # 1000 GP sale = 20 GP tax (2%)
        assert calculate_ge_tax(1000, "Abyssal whip") == 20

    def test_tax_exempt_item_no_tax(self):
        """Verify tax-exempt items pay no tax regardless of price."""
        from osrs_flipper.tax import calculate_ge_tax

        # Bronze arrow is tax-exempt
        assert calculate_ge_tax(1000, "Bronze arrow") == 0
        assert calculate_ge_tax(100_000_000, "Bronze arrow") == 0

    def test_under_threshold_no_tax(self):
        """Verify no tax is applied under 50 GP threshold."""
        from osrs_flipper.tax import calculate_ge_tax

        # 49 GP = 0 tax (under threshold)
        assert calculate_ge_tax(49, "Abyssal whip") == 0
        # 50 GP = 1 tax (at threshold, 2% of 50 = 1)
        assert calculate_ge_tax(50, "Abyssal whip") == 1

    def test_tax_cap_at_5m(self):
        """Verify tax is capped at 5M GP."""
        from osrs_flipper.tax import calculate_ge_tax

        # 500M sale = 5M tax (capped, not 10M)
        assert calculate_ge_tax(500_000_000, "Twisted bow") == 5_000_000


class TestNetProfit:
    """Tests for net profit calculation function."""

    def test_net_profit_with_tax(self):
        """Verify net profit calculation includes tax."""
        from osrs_flipper.tax import calculate_net_profit

        # buy 900, sell 1000, tax = 20, profit = 1000 - 20 - 900 = 80
        assert calculate_net_profit(900, 1000, "Abyssal whip") == 80

    def test_net_profit_exempt_item(self):
        """Verify tax-exempt items have no tax in profit calculation."""
        from osrs_flipper.tax import calculate_net_profit

        # Bronze arrow: no tax, profit = sell - buy
        # buy 900, sell 1000, profit = 1000 - 0 - 900 = 100
        assert calculate_net_profit(900, 1000, "Bronze arrow") == 100

    def test_net_profit_loss(self):
        """Verify net profit calculation correctly shows losses."""
        from osrs_flipper.tax import calculate_net_profit

        # buy 1000, sell 900, tax = 18, loss = 900 - 18 - 1000 = -118
        assert calculate_net_profit(1000, 900, "Abyssal whip") == -118
