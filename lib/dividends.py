import pandas as pd


_TAX_NOTES = {
    "taxable": (
        "Qualified dividends in a taxable account are taxed at the lower long-term "
        "capital gains rate (0%, 15%, or 20% depending on your income)."
    ),
    "tax_free": (
        "Dividends inside a Roth IRA or HSA grow and can be withdrawn completely "
        "tax-free — this is one of the best places to hold high-yielders."
    ),
    "tax_deferred": (
        "Dividends here aren't taxed now, but you'll pay ordinary income tax when "
        "you withdraw. Good for compounding; less ideal for qualified dividends."
    ),
}


def compute_dividend_income(df: pd.DataFrame, market_data: dict) -> dict:
    """Return annual/monthly dividend income totals and per-holding breakdown."""
    by_holding = []
    by_account: dict[str, float] = {}

    for _, row in df.iterrows():
        symbol = row.get("symbol", "")
        value = float(row.get("current_value", 0) or 0)
        account = row.get("account_name", "Unknown")
        account_type = row.get("account_type", "taxable")

        mdata = market_data.get(symbol, {})
        raw_yield = mdata.get("dividend_yield") or 0.0
        annual_div = value * raw_yield

        by_holding.append(
            {
                "symbol": symbol,
                "annual_dividend": round(annual_div, 2),
                "yield_pct": round(raw_yield * 100, 2),
                "account": account,
                "tax_note": _TAX_NOTES.get(account_type, _TAX_NOTES["taxable"]),
            }
        )

        by_account[account] = round(by_account.get(account, 0.0) + annual_div, 2)

    annual_income = sum(h["annual_dividend"] for h in by_holding)
    return {
        "annual_income": round(annual_income, 2),
        "monthly_income": round(annual_income / 12, 2),
        "by_holding": by_holding,
        "by_account": by_account,
    }


def project_dividend_growth(
    current_income: float,
    growth_rate: float = 0.05,
    years: list[int] = [1, 3, 5, 10],
) -> list[dict]:
    """Project dividend income at a constant annual growth rate."""
    return [
        {
            "year": y,
            "projected_income": round(current_income * (1 + growth_rate) ** y, 2),
        }
        for y in years
    ]
