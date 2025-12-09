"""GE tax calculations and tax-exempt items registry.

This module handles Grand Exchange tax calculations according to OSRS mechanics:
- 2% tax on sales over 50 GP
- Capped at 5,000,000 GP
- 45 items are tax-exempt (mostly low-level items and bonds)
"""

# GE Tax Constants
GE_TAX_RATE = 0.02
GE_TAX_THRESHOLD = 50
GE_TAX_CAP = 5_000_000

# Tax-Exempt Items (from OSRS Wiki)
# All items in this set are stored in lowercase for case-insensitive matching
TAX_EXEMPT_ITEMS: set[str] = {
    "ardougne teleport (tablet)",
    "bass",
    "bread",
    "bronze arrow",
    "bronze dart",
    "cake",
    "camelot teleport (tablet)",
    "chisel",
    "civitas illa fortis teleport (tablet)",
    "cooked chicken",
    "cooked meat",
    "energy potion",
    "falador teleport (tablet)",
    "games necklace",
    "gardening trowel",
    "glassblowing pipe",
    "hammer",
    "herring",
    "iron arrow",
    "iron dart",
    "kourend castle teleport (tablet)",
    "lobster",
    "lumbridge teleport (tablet)",
    "mackerel",
    "meat pie",
    "mind rune",
    "needle",
    "old school bond",
    "pestle and mortar",
    "pike",
    "rake",
    "ring of dueling",
    "salmon",
    "saw",
    "secateurs",
    "seed dibber",
    "shears",
    "shrimps",
    "spade",
    "steel arrow",
    "steel dart",
    "teleport to house (tablet)",
    "tuna",
    "varrock teleport (tablet)",
    "watering can",
}


def is_tax_exempt(item_name: str) -> bool:
    """Check if an item is exempt from GE tax.

    Args:
        item_name: The name of the item to check (case-insensitive)

    Returns:
        True if the item is tax-exempt, False otherwise

    Examples:
        >>> is_tax_exempt("Bronze arrow")
        True
        >>> is_tax_exempt("LOBSTER")
        True
        >>> is_tax_exempt("Abyssal whip")
        False
    """
    return item_name.lower() in TAX_EXEMPT_ITEMS


def calculate_ge_tax(sell_price: int, item_name: str) -> int:
    """Calculate the Grand Exchange tax for a sale.

    The GE tax is 2% of the sell price, with the following rules:
    - Tax-exempt items pay no tax
    - Sales under 50 GP pay no tax
    - Tax is capped at 5,000,000 GP

    Args:
        sell_price: The sale price in GP
        item_name: The name of the item (case-insensitive)

    Returns:
        The tax amount in GP (integer)

    Examples:
        >>> calculate_ge_tax(1000, "Abyssal whip")
        20
        >>> calculate_ge_tax(1000, "Bronze arrow")
        0
        >>> calculate_ge_tax(49, "Abyssal whip")
        0
        >>> calculate_ge_tax(500_000_000, "Twisted bow")
        5000000
    """
    # Tax-exempt items pay no tax
    if is_tax_exempt(item_name):
        return 0

    # Sales under threshold pay no tax
    if sell_price < GE_TAX_THRESHOLD:
        return 0

    # Calculate tax with cap
    tax = int(sell_price * GE_TAX_RATE)
    return min(tax, GE_TAX_CAP)


def calculate_net_profit(buy_price: int, sell_price: int, item_name: str) -> int:
    """Calculate net profit after GE tax.

    Net profit = sell_price - tax - buy_price

    Args:
        buy_price: The purchase price in GP
        sell_price: The sale price in GP
        item_name: The name of the item (case-insensitive)

    Returns:
        The net profit in GP (can be negative for losses)

    Examples:
        >>> calculate_net_profit(900, 1000, "Abyssal whip")
        80
        >>> calculate_net_profit(900, 1000, "Bronze arrow")
        100
        >>> calculate_net_profit(1000, 900, "Abyssal whip")
        -118
    """
    tax = calculate_ge_tax(sell_price, item_name)
    return sell_price - tax - buy_price
