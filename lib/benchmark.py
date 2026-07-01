import pandas as pd

_FUND_TYPES = {"index_fund", "etf", "target_date_fund"}


def _bucket(sub: pd.DataFrame) -> dict:
    cost = sub["cost_basis_total"].sum()
    value = sub["current_value"].sum()
    gain_dollar = value - cost
    gain_pct = (gain_dollar / cost * 100) if cost else 0.0
    return {
        "cost_basis": round(cost, 2),
        "current_value": round(value, 2),
        "gain_dollar": round(gain_dollar, 2),
        "gain_pct": round(gain_pct, 2),
        "count": len(sub),
    }


def compute_stock_picking_scorecard(df: pd.DataFrame) -> dict:
    """Compare the money-weighted return on individual stock picks against the
    index funds & ETFs already held, using cost basis you already have.

    This is not time-matched (stocks and funds were bought on different dates),
    so treat it as a directional gut-check, not a precise alpha calculation.
    """
    stocks = df[(df["holding_type"] == "individual_stock") & df["cost_basis_total"].notna()]
    funds = df[df["holding_type"].isin(_FUND_TYPES) & df["cost_basis_total"].notna()]

    stock_bucket = _bucket(stocks)
    fund_bucket = _bucket(funds)

    hypothetical_value = None
    dollar_vs_indexing = None
    beating_index = None
    if fund_bucket["cost_basis"]:
        hypothetical_value = round(stock_bucket["cost_basis"] * (1 + fund_bucket["gain_pct"] / 100), 2)
        dollar_vs_indexing = round(stock_bucket["current_value"] - hypothetical_value, 2)
        beating_index = dollar_vs_indexing > 0

    ranked = stocks.sort_values("gain_loss_dollar", ascending=False)
    winners = ranked[ranked["gain_loss_dollar"] > 0].head(5)
    losers = ranked[ranked["gain_loss_dollar"] <= 0].tail(5).sort_values("gain_loss_dollar")

    cols = ["symbol", "gain_loss_dollar", "gain_loss_pct"]
    return {
        "stocks": stock_bucket,
        "funds": fund_bucket,
        "hypothetical_value_if_indexed": hypothetical_value,
        "dollar_vs_indexing": dollar_vs_indexing,
        "beating_index": beating_index,
        "top_winners": winners[cols].to_dict("records"),
        "top_losers": losers[cols].to_dict("records"),
    }
