"""Year-end tax planning analysis for the taxable brokerage account."""

import pandas as pd


_TAX_LOSS_PAIRS = {
    "AAPL": ["MSFT", "QQQ", "XLK"],
    "MSFT": ["AAPL", "QQQ", "XLK"],
    "GOOGL": ["META", "XLC"],
    "GOOG": ["META", "XLC"],
    "META": ["GOOGL", "XLC"],
    "AMZN": ["SHOP", "XLY"],
    "NVDA": ["SMH", "SOXX", "AMD"],
    "AMD": ["SMH", "SOXX", "NVDA"],
    "TSLA": ["RIVN", "LCID", "CARZ"],
    "INTC": ["SMH", "SOXX", "TXN"],
    "VTI": ["ITOT", "SWTSX", "SCHB"],
    "ITOT": ["VTI", "SWTSX", "SCHB"],
    "VOO": ["IVV", "SPY", "SPLG"],
    "IVV": ["VOO", "SPY", "SPLG"],
    "SPY": ["VOO", "IVV", "SPLG"],
    "FSKAX": ["VTI", "ITOT", "SWTSX"],
    "FNILX": ["VOO", "IVV", "FXAIX"],
    "FXAIX": ["VOO", "IVV", "FNILX"],
    "VXUS": ["IXUS", "IEFA", "FZILX"],
    "IXUS": ["VXUS", "IEFA"],
    "VEA": ["IEFA", "SPDW"],
    "IEFA": ["VEA", "SPDW"],
    "BND": ["AGG", "SCHZ"],
    "AGG": ["BND", "SCHZ"],
    "JPM": ["BAC", "XLF"],
    "BAC": ["JPM", "XLF"],
    "XOM": ["CVX", "XLE"],
    "CVX": ["XOM", "XLE"],
    "JNJ": ["PFE", "XLV"],
    "PFE": ["JNJ", "XLV"],
    "DIS": ["CMCSA", "XLC"],
}


def _estimate_tax_rates(annual_income: float) -> tuple[float, float]:
    if annual_income < 47150:
        st_rate = 0.12
    elif annual_income < 100525:
        st_rate = 0.22
    elif annual_income < 191950:
        st_rate = 0.24
    elif annual_income < 578125:
        st_rate = 0.32
    else:
        st_rate = 0.35

    lt_rate = 0.20 if annual_income > 500000 else 0.15
    return st_rate, lt_rate


def tax_loss_pairs(symbol: str) -> list[str]:
    """Return similar-but-not-identical alternatives to avoid wash sales."""
    return _TAX_LOSS_PAIRS.get(symbol.upper(), [])


def analyze_tax_situation(df: pd.DataFrame, annual_income: float = 150000) -> dict:
    """Analyze unrealized gains/losses in the taxable account."""
    taxable = df[df["account_type"] == "taxable"].copy()

    if taxable.empty:
        return {
            "unrealized_gains": 0,
            "unrealized_losses": 0,
            "net_position": 0,
            "estimated_short_term_rate": 0,
            "estimated_long_term_rate": 0,
            "harvestable_losses": [],
            "gain_management": [],
            "max_deductible_loss": 3000.0,
            "net_tax_impact": 0,
            "strategies": ["No taxable holdings found."],
        }

    st_rate, lt_rate = _estimate_tax_rates(annual_income)

    gains_mask = taxable["gain_loss_dollar"] > 0
    losses_mask = taxable["gain_loss_dollar"] < 0

    total_gains = float(taxable.loc[gains_mask, "gain_loss_dollar"].sum())
    total_losses = float(taxable.loc[losses_mask, "gain_loss_dollar"].sum())
    net_position = total_gains + total_losses

    harvestable = []
    for _, row in taxable[losses_mask].sort_values("gain_loss_dollar").iterrows():
        loss = abs(float(row["gain_loss_dollar"]))
        tax_savings = loss * st_rate
        alternatives = tax_loss_pairs(row["symbol"])
        alt_str = f" Buy a similar alternative ({', '.join(alternatives[:2])}) to maintain exposure while avoiding wash sale." if alternatives else ""
        harvestable.append({
            "symbol": row["symbol"],
            "loss": float(row["gain_loss_dollar"]),
            "current_value": float(row["current_value"]),
            "tax_savings": round(tax_savings, 2),
            "suggestion": (
                f"Sell {row['symbol']} to harvest ${loss:,.0f} loss — "
                f"this could save ~${tax_savings:,.0f} on your taxes.{alt_str}"
            ),
        })

    gain_mgmt = []
    for _, row in taxable[gains_mask].sort_values("gain_loss_dollar", ascending=False).iterrows():
        gain = float(row["gain_loss_dollar"])
        tax_owed = gain * lt_rate
        gain_mgmt.append({
            "symbol": row["symbol"],
            "gain": gain,
            "current_value": float(row["current_value"]),
            "tax_owed": round(tax_owed, 2),
            "suggestion": (
                f"{row['symbol']} has ${gain:,.0f} in gains. If you sell, you'll owe ~${tax_owed:,.0f} "
                f"in taxes (at the {lt_rate*100:.0f}% long-term rate). Consider holding unless you need "
                f"to rebalance — unrealized gains aren't taxed."
            ),
        })

    if net_position < 0:
        deductible = min(abs(net_position), 3000)
        net_tax_impact = -(deductible * st_rate)
    else:
        net_tax_impact = net_position * lt_rate

    strategies = _build_strategies(total_gains, total_losses, net_position, harvestable, st_rate)

    return {
        "unrealized_gains": round(total_gains, 2),
        "unrealized_losses": round(total_losses, 2),
        "net_position": round(net_position, 2),
        "estimated_short_term_rate": st_rate,
        "estimated_long_term_rate": lt_rate,
        "harvestable_losses": harvestable,
        "gain_management": gain_mgmt,
        "max_deductible_loss": 3000.0,
        "net_tax_impact": round(net_tax_impact, 2),
        "strategies": strategies,
    }


def _build_strategies(gains, losses, net, harvestable, st_rate):
    strategies = []

    if harvestable:
        total_saveable = sum(h["tax_savings"] for h in harvestable)
        strategies.append(
            f"Harvest your losses: selling your losing positions could save ~${total_saveable:,.0f} "
            f"in taxes this year. You can use losses to offset gains, plus deduct up to $3,000 "
            f"against ordinary income."
        )

    if gains > 0 and losses < 0:
        strategies.append(
            f"You have ${gains:,.0f} in gains and ${abs(losses):,.0f} in losses. "
            f"Harvesting losses before year-end offsets your gains and reduces your tax bill."
        )

    if net > 10000:
        strategies.append(
            f"Your net gains are ${net:,.0f}. If you don't need to sell, consider holding — "
            f"unrealized gains aren't taxed, and holding longer than a year qualifies for "
            f"the lower long-term rate."
        )

    strategies.append(
        "Consider doing any tax-loss harvesting before December 31. "
        "Remember the wash sale rule: you can't buy the same stock back within 30 days "
        "or the loss is disallowed."
    )

    if not strategies or len(strategies) < 2:
        strategies.append(
            "Review your portfolio in November/December each year to optimize tax outcomes."
        )

    return strategies
