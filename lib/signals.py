import pandas as pd
from .classifier import is_fund


SIGNAL_EXPLANATIONS = {
    "near_52w_low": (
        "This stock is trading near its lowest price in the past year. "
        "This could mean it's undervalued (buying opportunity) or that something "
        "is fundamentally wrong. Check the news before acting."
    ),
    "near_52w_high": (
        "Trading near its 52-week high. This usually means strong momentum, "
        "but the easy gains may already be priced in. Not necessarily a sell signal — "
        "stocks at highs often keep climbing."
    ),
    "low_pe": (
        "The price-to-earnings ratio measures how much you pay per dollar of profit. "
        "A low P/E (under 15) suggests the stock is cheap relative to its earnings — "
        "but could also mean the market expects earnings to decline."
    ),
    "high_pe": (
        "A high P/E means the market is pricing in strong future growth. "
        "If that growth doesn't materialize, the stock could drop significantly."
    ),
    "earnings_growth": (
        "Wall Street expects this company's earnings to grow, "
        "which is why the forward P/E is lower than the trailing P/E. A positive sign."
    ),
    "high_dividend": (
        "This stock pays a meaningful dividend — real cash returned to you regularly. "
        "In a taxable account, qualified dividends are taxed at favorable rates (0-20%)."
    ),
    "tax_loss_harvest": (
        "This position is significantly down from your purchase price. You could sell it "
        "to 'harvest' the loss — deducting it against gains or up to $3,000 of ordinary income. "
        "Buy a similar (not identical) fund to maintain exposure."
    ),
    "fund_overlap": (
        "You hold multiple funds that track similar indexes. This isn't harmful, "
        "but you're not getting extra diversification — consider consolidating into "
        "the one with the lowest expense ratio."
    ),
    "fund_strong_trend": (
        "This fund's index is in a strong uptrend. Good confirmation that "
        "the asset class it covers is performing well."
    ),
    "fund_weak_trend": (
        "This fund's index is near its 52-week low, meaning the broad asset class "
        "it covers is underperforming. This is normal market rotation — not a reason "
        "to sell a core index fund, but a reason to be patient."
    ),
    "fund_core_holding": (
        "A low-cost index fund is a cornerstone of a diversified portfolio. "
        "These should generally be held long-term through all market conditions."
    ),
}


def compute_signals(df: pd.DataFrame, market_data: dict) -> pd.DataFrame:
    rows = []
    for _, pos in df.iterrows():
        sym = pos["symbol"]
        md = market_data.get(sym, {})
        if not md or md.get("price", 0) == 0:
            continue

        holding_type = pos.get("holding_type", "individual_stock")

        if is_fund(holding_type):
            row = _signals_for_fund(pos, md)
        else:
            row = _signals_for_stock(pos, md)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _signals_for_stock(pos, md):
    sym = pos["symbol"]
    price = md.get("price", pos["last_price"])
    low52 = md.get("fifty_two_week_low")
    high52 = md.get("fifty_two_week_high")
    pe = md.get("pe_ratio")
    fwd_pe = md.get("forward_pe")
    div_yield = md.get("dividend_yield")
    beta = md.get("beta")

    if low52 and high52 and high52 > low52:
        range_position = (price - low52) / (high52 - low52)
    else:
        range_position = None

    signals = []
    signal_types = []
    score = 0
    verdict = "Hold"

    if range_position is not None:
        if range_position < 0.3:
            signals.append(f"Near 52-week low (bottom {range_position*100:.0f}% of range)")
            signal_types.append("near_52w_low")
            score += 2
        elif range_position > 0.9:
            signals.append(f"Near 52-week high (top {(1-range_position)*100:.0f}% of range)")
            signal_types.append("near_52w_high")
            score -= 1

    if pe is not None:
        if pe < 15:
            signals.append(f"Low P/E ratio ({pe:.1f}) — stock looks cheap relative to earnings")
            signal_types.append("low_pe")
            score += 1
        elif pe > 40:
            signals.append(f"High P/E ratio ({pe:.1f}) — market expects strong growth")
            signal_types.append("high_pe")
            score -= 1

    if fwd_pe is not None and pe is not None and fwd_pe < pe:
        pct_decline = (pe - fwd_pe) / pe * 100
        signals.append(f"Earnings expected to grow {pct_decline:.0f}% (forward P/E: {fwd_pe:.1f})")
        signal_types.append("earnings_growth")
        score += 1

    if div_yield is not None and div_yield > 0.03:
        signals.append(f"Pays {div_yield*100:.1f}% dividend yield")
        signal_types.append("high_dividend")
        score += 1

    if pos["gain_loss_pct"] < -20 and "Taxable" in pos.get("account_name", ""):
        signals.append(f"Down {pos['gain_loss_pct']:.0f}% — tax-loss harvesting candidate")
        signal_types.append("tax_loss_harvest")

    if score >= 2:
        verdict = "Looks Attractive"
    elif score <= -1:
        verdict = "Caution"

    return _build_signal_row(pos, md, price, pe, fwd_pe, div_yield, low52, high52,
                             range_position, beta, signals, signal_types, score, verdict)


def _signals_for_fund(pos, md):
    sym = pos["symbol"]
    price = md.get("price", pos["last_price"])
    low52 = md.get("fifty_two_week_low")
    high52 = md.get("fifty_two_week_high")
    div_yield = md.get("dividend_yield")
    beta = md.get("beta")

    if low52 and high52 and high52 > low52:
        range_position = (price - low52) / (high52 - low52)
    else:
        range_position = None

    signals = []
    signal_types = []
    score = 0
    verdict = "Core Holding"

    type_label = pos.get("holding_type_label", "Fund")
    asset_class = pos.get("asset_class", "")

    signals.append(f"{type_label} tracking {asset_class} — hold as part of your core allocation")
    signal_types.append("fund_core_holding")

    if range_position is not None:
        if range_position > 0.8:
            signals.append(f"Index in strong uptrend (top {(1-range_position)*100:.0f}% of 52-week range)")
            signal_types.append("fund_strong_trend")
            score += 1
        elif range_position < 0.3:
            signals.append(f"Index near 52-week low ({range_position*100:.0f}% of range) — be patient, don't panic-sell")
            signal_types.append("fund_weak_trend")

    if div_yield is not None and div_yield > 0.02:
        signals.append(f"Distributes {div_yield*100:.1f}% yield")
        signal_types.append("high_dividend")
        score += 1

    from .classifier import OVERLAP_GROUPS
    for group_syms in OVERLAP_GROUPS.values():
        if sym in group_syms:
            signals.append(f"Overlaps with: {', '.join(sorted(group_syms - {sym}))}")
            signal_types.append("fund_overlap")
            break

    return _build_signal_row(pos, md, price, None, None, div_yield, low52, high52,
                             range_position, beta, signals, signal_types, score, verdict)


def _build_signal_row(pos, md, price, pe, fwd_pe, div_yield, low52, high52,
                      range_position, beta, signals, signal_types, score, verdict):
    return {
        "symbol": pos["symbol"],
        "description": pos["description"],
        "account_name": pos["account_name"],
        "holding_type": pos.get("holding_type", "individual_stock"),
        "holding_type_label": pos.get("holding_type_label", "Stock"),
        "price": price,
        "pe_ratio": pe,
        "forward_pe": fwd_pe,
        "dividend_yield": div_yield,
        "fifty_two_week_low": low52,
        "fifty_two_week_high": high52,
        "range_position": range_position,
        "beta": beta,
        "gain_loss_pct": pos["gain_loss_pct"],
        "current_value": pos["current_value"],
        "signals": signals,
        "signal_types": signal_types,
        "score": score,
        "verdict": verdict,
    }


def dca_suggestions(
    df: pd.DataFrame,
    drift: pd.DataFrame,
    signals_df: pd.DataFrame,
    monthly_budget: float,
) -> list[dict]:
    if monthly_budget <= 0:
        return []

    underweight = drift[drift["drift_pct"] < -0.01].copy()
    if underweight.empty:
        return []

    suggestions = []
    remaining = monthly_budget

    for _, row in underweight.sort_values("drift_pct").iterrows():
        cls = row["asset_class"]
        share = min(remaining, abs(row["drift_dollars"]) * 0.1)
        share = min(share, remaining)

        if share < 10:
            continue

        best_fund = _best_fund_for_class(cls, signals_df)
        suggestions.append({
            "asset_class": cls,
            "suggested_fund": best_fund["symbol"],
            "suggested_amount": round(share, 2),
            "reason": f"Underweight by {abs(row['drift_pct'])*100:.1f}%",
            "explanation": (
                f"Your {cls} allocation is {abs(row['drift_pct'])*100:.1f}% below target. "
                f"Putting ${share:,.0f}/mo into {best_fund['symbol']} gradually closes this gap "
                f"without needing to sell anything."
            ),
            "signals": best_fund.get("signals", []),
        })
        remaining -= share

    return suggestions


def _best_fund_for_class(cls: str, signals_df: pd.DataFrame) -> dict:
    preferred = {
        "US Equities": ["FSKAX", "FNILX", "VTI", "FXAIX"],
        "International": ["FSPSX", "FSGGX"],
        "Bonds": ["FXNAX"],
        "Alternatives": ["O", "SPYD"],
    }

    for sym in preferred.get(cls, []):
        match = signals_df[signals_df["symbol"] == sym]
        if len(match) > 0:
            row = match.iloc[0]
            return {"symbol": sym, "signals": row["signals"]}

    return {"symbol": preferred.get(cls, ["VTI"])[0], "signals": []}
