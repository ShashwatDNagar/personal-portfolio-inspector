import pandas as pd
from .classifier import is_fund, OVERLAP_GROUPS


def grade_holdings(df: pd.DataFrame, market_data: dict, total_value: float) -> pd.DataFrame:
    rows = []

    for _, pos in df.iterrows():
        sym = pos["symbol"]
        md = market_data.get(sym, {})
        if not md:
            continue

        holding_type = pos.get("holding_type", "individual_stock")
        if holding_type == "money_market":
            continue

        if is_fund(holding_type):
            row = _grade_fund(pos, md, total_value)
        else:
            row = _grade_stock(pos, md, total_value)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)


def _grade_stock(pos, md, total_value):
    scores = {}
    flags = []
    positives = []

    pe = md.get("pe_ratio")
    fwd_pe = md.get("forward_pe")
    if pe is not None:
        if pe < 0:
            scores["valuation"] = 0
            flags.append("Negative earnings — company is losing money")
        elif pe < 15:
            scores["valuation"] = 5
            positives.append(f"Attractively valued (P/E {pe:.0f})")
        elif pe < 25:
            scores["valuation"] = 4
            positives.append(f"Reasonably valued (P/E {pe:.0f})")
        elif pe < 40:
            scores["valuation"] = 2
            flags.append(f"Pricey (P/E {pe:.0f}) — priced for growth that must materialize")
        else:
            scores["valuation"] = 1
            flags.append(f"Very expensive (P/E {pe:.0f}) — high expectations baked in")
    else:
        scores["valuation"] = 2

    if fwd_pe is not None and pe is not None and fwd_pe < pe * 0.85:
        scores["valuation"] = min(scores.get("valuation", 3) + 1, 5)
        positives.append("Earnings growth expected — forward P/E improving")

    scores["momentum"] = _momentum_score(md, positives, flags)
    scores["return"] = _return_score(pos, positives, flags)
    scores["income"] = _income_score(md, positives)
    scores["concentration"] = _concentration_score(pos, total_value, flags)

    weights = {"valuation": 0.25, "momentum": 0.20, "return": 0.20, "income": 0.15, "concentration": 0.20}
    composite = sum(scores.get(k, 3) * w for k, w in weights.items())
    letter = _composite_to_grade(composite)

    return _build_row(pos, md, scores, composite, letter, positives, flags, total_value)


def _grade_fund(pos, md, total_value):
    scores = {}
    flags = []
    positives = []
    holding_type = pos.get("holding_type", "etf")
    type_label = pos.get("holding_type_label", "Fund")

    # Funds don't get single-stock valuation scoring — the P/E is an aggregate
    # and not meaningful the same way. Instead we score on role clarity.
    positives.append(f"{type_label} — provides diversified exposure to {pos.get('asset_class', 'the market')}")
    scores["valuation"] = 4  # Funds are inherently diversified, neutral-good by default

    # Momentum still somewhat relevant for funds (is the index trending up?)
    scores["momentum"] = _momentum_score(md, positives, flags, is_fund=True)

    # Return still relevant
    scores["return"] = _return_score(pos, positives, flags)

    # Income
    scores["income"] = _income_score(md, positives)

    # Concentration — large fund positions are actually fine (diversified), so we're lenient
    weight = pos["current_value"] / total_value if total_value > 0 else 0
    if weight > 0.30:
        scores["concentration"] = 3
        flags.append(f"Large position ({weight*100:.1f}%) but it's a diversified fund — less risky than a single stock at this weight")
    else:
        scores["concentration"] = 5

    # Fund overlap detection
    overlap_sym = _check_fund_overlap(pos["symbol"])
    if overlap_sym:
        flags.append(f"Overlaps significantly with {', '.join(overlap_sym)} — you may be double-counting exposure")

    weights = {"valuation": 0.15, "momentum": 0.20, "return": 0.25, "income": 0.15, "concentration": 0.25}
    composite = sum(scores.get(k, 3) * w for k, w in weights.items())
    letter = _composite_to_grade(composite)

    return _build_row(pos, md, scores, composite, letter, positives, flags, total_value)


def _momentum_score(md, positives, flags, is_fund=False):
    high52 = md.get("fifty_two_week_high")
    low52 = md.get("fifty_two_week_low")
    price = md.get("price", 0)

    if high52 and low52 and high52 > low52:
        range_pos = (price - low52) / (high52 - low52)
        if range_pos > 0.8:
            label = "index" if is_fund else "stock"
            positives.append(f"Strong trend — near 52-week high")
            return 5
        elif range_pos > 0.5:
            positives.append("Healthy trend — upper half of range")
            return 4
        elif range_pos > 0.3:
            return 3
        else:
            if is_fund:
                flags.append(f"Fund near 52-week low ({range_pos*100:.0f}% of range) — broad market weakness in this segment")
            else:
                flags.append(f"Weak momentum — near 52-week low ({range_pos*100:.0f}% of range)")
            return 1
    return 3


def _return_score(pos, positives, flags):
    gain_pct = pos["gain_loss_pct"]
    if gain_pct > 50:
        positives.append(f"Excellent return ({gain_pct:+.0f}%)")
        return 5
    elif gain_pct > 10:
        positives.append(f"Solid return ({gain_pct:+.0f}%)")
        return 4
    elif gain_pct > -5:
        return 3
    elif gain_pct > -20:
        flags.append(f"Underwater ({gain_pct:+.0f}%) — monitor for recovery or cut")
        return 2
    else:
        flags.append(f"Significant loss ({gain_pct:+.0f}%) — reassess thesis or harvest tax loss")
        return 1


def _income_score(md, positives):
    div_yield = md.get("dividend_yield") or 0
    if div_yield > 0.03:
        positives.append(f"Strong income producer ({div_yield*100:.1f}% yield)")
        return 5
    elif div_yield > 0.015:
        positives.append(f"Pays a dividend ({div_yield*100:.1f}% yield)")
        return 4
    elif div_yield > 0:
        return 3
    else:
        return 2


def _concentration_score(pos, total_value, flags):
    weight = pos["current_value"] / total_value if total_value > 0 else 0
    if weight > 0.15:
        flags.append(f"Very concentrated ({weight*100:.1f}% of portfolio) — a single stock shouldn't be this large")
        return 1
    elif weight > 0.10:
        flags.append(f"High weight ({weight*100:.1f}%) — a bad day here really hurts")
        return 2
    elif weight > 0.05:
        return 4
    else:
        return 5


def _check_fund_overlap(symbol: str) -> list[str]:
    for group_syms in OVERLAP_GROUPS.values():
        if symbol in group_syms:
            others = group_syms - {symbol}
            return sorted(others)
    return []


def _composite_to_grade(score: float) -> str:
    if score >= 4.5:
        return "A"
    elif score >= 3.8:
        return "B"
    elif score >= 3.0:
        return "C"
    elif score >= 2.2:
        return "D"
    else:
        return "F"


def _build_row(pos, md, scores, composite, letter, positives, flags, total_value):
    weight = pos["current_value"] / total_value if total_value > 0 else 0
    return {
        "symbol": pos["symbol"],
        "description": pos["description"],
        "account_name": pos["account_name"],
        "holding_type": pos.get("holding_type", "individual_stock"),
        "holding_type_label": pos.get("holding_type_label", "Stock"),
        "current_value": pos["current_value"],
        "gain_loss_pct": pos["gain_loss_pct"],
        "weight_pct": weight * 100,
        "valuation_score": scores.get("valuation", 3),
        "momentum_score": scores.get("momentum", 3),
        "return_score": scores.get("return", 3),
        "income_score": scores.get("income", 2),
        "concentration_score": scores.get("concentration", 3),
        "composite": composite,
        "grade": letter,
        "positives": positives,
        "flags": flags,
    }


GRADE_EXPLANATIONS = {
    "A": "Strong holding — well-valued, good momentum, manageable risk. Keep and possibly add.",
    "B": "Solid holding — no major concerns. Hold and monitor.",
    "C": "Mixed signals — some strengths, some weaknesses. Worth reviewing your original thesis.",
    "D": "Concerning — multiple red flags. Consider whether this still belongs in your portfolio.",
    "F": "Problem position — underwater, overvalued, or too concentrated. Action likely needed.",
}

DIMENSION_EXPLANATIONS = {
    "valuation_score": "For stocks: how cheap or expensive vs. earnings (P/E). For funds: neutral — a fund's P/E is an aggregate, not a buy/sell signal.",
    "momentum_score": "Price trend over the past year. 5 = near 52-week high, 1 = near 52-week low.",
    "return_score": "How much money you've made or lost on this position. 5 = great gains, 1 = big loss.",
    "income_score": "Dividend yield — how much cash it pays you. 5 = high yield, 1 = no dividend.",
    "concentration_score": "Portfolio weight risk. For stocks: 5 = small, 1 = dangerously large. For funds: more lenient since they're inherently diversified.",
}
