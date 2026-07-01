import pandas as pd
from .allocation import IDEAL_ASSET_LOCATION


ASSET_LOCATION_EXPLANATIONS = {
    ("Bonds", "tax_deferred"): (
        "Bond interest is taxed as ordinary income (your highest tax rate). "
        "Holding bonds in your 401(k) means you defer that tax until withdrawal, "
        "when you may be in a lower bracket."
    ),
    ("Bonds", "tax_free"): (
        "Bond interest taxed as ordinary income is completely eliminated in a Roth/HSA. "
        "Good if your tax-deferred space is full."
    ),
    ("International", "taxable"): (
        "International funds pay foreign taxes on dividends. In a taxable account, "
        "you can claim the Foreign Tax Credit on your US tax return to avoid double-taxation. "
        "This credit is lost inside retirement accounts."
    ),
    ("US Equities", "tax_free"): (
        "US stock growth is best in Roth/HSA — all gains are tax-free forever. "
        "Since you have decades of compounding ahead, sheltering the highest-growth assets "
        "here maximizes the tax benefit."
    ),
    ("US Equities", "taxable"): (
        "Long-term stock gains are taxed at favorable capital gains rates (0-20%), "
        "and you control when to sell. Acceptable location when tax-advantaged space is full."
    ),
    ("Alternatives", "tax_deferred"): (
        "REITs and alternatives often generate ordinary income (not qualified dividends). "
        "Tax-deferred accounts shield this from your current tax rate."
    ),
}


def get_location_explanation(asset_class: str, account_type: str) -> str:
    return ASSET_LOCATION_EXPLANATIONS.get(
        (asset_class, account_type),
        ""
    )


def generate_rebalancing_actions(
    df: pd.DataFrame,
    drift: pd.DataFrame,
    total_value: float,
    monthly_contributions: dict | None = None,
) -> list[dict]:
    actions = []
    monthly_contributions = monthly_contributions or {}

    overweight = drift[drift["drift_pct"] > 0.02].sort_values("drift_pct", ascending=False)
    underweight = drift[drift["drift_pct"] < -0.02].sort_values("drift_pct")

    total_monthly = sum(monthly_contributions.values())

    for _, uw in underweight.iterrows():
        cls = uw["asset_class"]
        needed = abs(uw["drift_dollars"])

        best_accounts = IDEAL_ASSET_LOCATION.get(cls, ["taxable"])
        location_reason = get_location_explanation(cls, best_accounts[0])

        if total_monthly > 0:
            underweight_sum = drift[drift["drift_pct"] < 0]["drift_pct"].abs().sum()
            monthly_share = (abs(uw["drift_pct"]) / underweight_sum) * total_monthly if underweight_sum > 0 else 0
            months_to_fix = needed / monthly_share if monthly_share > 0 else float("inf")
        else:
            monthly_share = 0
            months_to_fix = float("inf")

        suggestion = _pick_fund_for_class(cls, df)

        actions.append({
            "action": "BUY",
            "asset_class": cls,
            "amount": needed,
            "drift_pct": abs(uw["drift_pct"]),
            "suggested_fund": suggestion["symbol"],
            "suggested_fund_name": suggestion["name"],
            "preferred_account": best_accounts[0],
            "all_preferred_accounts": best_accounts,
            "location_reason": location_reason,
            "monthly_contribution_share": monthly_share,
            "months_to_correct_via_contributions": months_to_fix,
            "urgency": "high" if abs(uw["drift_pct"]) > 0.05 else "medium",
            "tax_notes": _tax_notes_buy(best_accounts[0], cls),
            "plain_explanation": _plain_buy_explanation(cls, needed, suggestion, best_accounts[0], months_to_fix, total_monthly),
        })

    for _, ow in overweight.iterrows():
        cls = ow["asset_class"]
        excess = ow["drift_dollars"]

        sell_candidates = _find_sell_candidates(df, cls)
        for cand in sell_candidates:
            cand["tax_notes"] = _tax_notes_sell(cand)
            cand["plain_explanation"] = _plain_sell_explanation(cand)

        actions.append({
            "action": "REDUCE",
            "asset_class": cls,
            "amount": excess,
            "drift_pct": ow["drift_pct"],
            "candidates": sell_candidates,
            "urgency": "high" if ow["drift_pct"] > 0.05 else "medium",
            "recommendation": _reduce_recommendation(sell_candidates, excess),
            "plain_explanation": _plain_reduce_explanation(cls, excess, sell_candidates),
        })

    return actions


def _plain_buy_explanation(cls, needed, suggestion, account_type, months_to_fix, total_monthly):
    acct = account_type.replace("_", " ").title()
    lines = [
        f"Your portfolio needs about ${needed:,.0f} more in {cls} to hit your target.",
        f"The simplest way: buy **{suggestion['symbol']}** in your **{acct}** account.",
    ]
    if months_to_fix < float("inf") and total_monthly > 0:
        lines.append(
            f"If you just redirect future contributions, this fixes itself in "
            f"about **{months_to_fix:.0f} months** — no selling required."
        )
    else:
        lines.append("Consider adding this to your regular contribution plan.")
    return "\n\n".join(lines)


def _plain_reduce_explanation(cls, excess, candidates):
    lines = [f"You have about ${excess:,.0f} more in {cls} than your target calls for."]

    tax_free_cands = [c for c in candidates if c["account_type"] != "taxable"]
    taxable_losers = [c for c in candidates if c["account_type"] == "taxable" and c["gain_loss_dollar"] <= 0]
    taxable_winners = [c for c in candidates if c["account_type"] == "taxable" and c["gain_loss_dollar"] > 0]

    if tax_free_cands:
        lines.append(
            "**Best approach:** Sell within your retirement accounts first — "
            "no tax consequences at all."
        )
    if taxable_losers:
        lines.append(
            "If you sell losers in your taxable account, you can use those losses "
            "to offset gains elsewhere (tax-loss harvesting)."
        )
    if taxable_winners and not tax_free_cands:
        lines.append(
            "Selling winners in your taxable account triggers capital gains tax. "
            "Consider whether the rebalancing benefit outweighs the tax cost, "
            "or just redirect new contributions away from this class."
        )

    return "\n\n".join(lines)


def _plain_sell_explanation(cand):
    if cand["account_type"] != "taxable":
        return "No tax impact — this is inside a retirement account."

    if cand["gain_loss_dollar"] <= 0:
        return (
            f"This position is down ${abs(cand['gain_loss_dollar']):,.0f}. "
            f"Selling locks in that loss, which you can use on your tax return "
            f"to offset up to ${min(abs(cand['gain_loss_dollar']), 3000):,.0f} of other income, "
            f"or unlimited capital gains."
        )

    return (
        f"This position is up ${cand['gain_loss_dollar']:,.0f} ({cand['gain_loss_pct']:.0f}%). "
        f"Selling triggers capital gains tax — likely 15% federal if held over a year, "
        f"or your ordinary income rate if under a year."
    )


def _pick_fund_for_class(cls: str, df: pd.DataFrame) -> dict:
    preferred = {
        "US Equities": ["FSKAX", "FNILX", "VTI", "FXAIX"],
        "International": ["FSPSX", "FSGGX"],
        "Bonds": ["FXNAX"],
        "Alternatives": ["O", "SPYD"],
        "Cash": ["SPAXX"],
    }

    for sym in preferred.get(cls, []):
        match = df[df["symbol"] == sym]
        if len(match) > 0:
            return {"symbol": sym, "name": match.iloc[0]["description"]}

    return {"symbol": preferred.get(cls, ["VTI"])[0], "name": ""}


def _find_sell_candidates(df: pd.DataFrame, broad_class: str) -> list[dict]:
    holdings = df[df["broad_class"] == broad_class].copy()
    candidates = []
    for _, row in holdings.iterrows():
        candidates.append({
            "symbol": row["symbol"],
            "account_name": row["account_name"],
            "account_type": row["account_type"],
            "current_value": row["current_value"],
            "gain_loss_dollar": row["gain_loss_dollar"],
            "gain_loss_pct": row["gain_loss_pct"],
            "cost_basis_total": row["cost_basis_total"],
        })
    return sorted(candidates, key=lambda x: _sell_priority(x))


def _sell_priority(cand: dict) -> tuple:
    acct_order = {"tax_deferred": 0, "tax_free": 1, "taxable": 2}
    acct_score = acct_order.get(cand["account_type"], 3)

    if cand["account_type"] == "taxable":
        if cand["gain_loss_dollar"] <= 0:
            gain_score = -1
        else:
            gain_score = cand["gain_loss_pct"]
    else:
        gain_score = 0

    return (acct_score, gain_score)


def _tax_notes_buy(account_type: str, cls: str) -> str:
    notes = []
    if account_type == "taxable" and cls == "International":
        notes.append("Foreign Tax Credit: you can deduct foreign taxes paid by this fund on your US return")
    if account_type == "tax_free":
        notes.append("All growth is tax-free forever — ideal for high-growth assets")
    if account_type == "tax_deferred":
        notes.append("Growth is tax-deferred; withdrawals taxed as ordinary income in retirement")
    return "; ".join(notes) if notes else ""


def _tax_notes_sell(cand: dict) -> str:
    if cand["account_type"] != "taxable":
        return "No tax impact — tax-advantaged account"

    if cand["gain_loss_dollar"] <= 0:
        return f"Tax-loss harvesting: ${abs(cand['gain_loss_dollar']):,.0f} loss offsets gains on your return"

    return f"Taxable gain of ${cand['gain_loss_dollar']:,.0f} ({cand['gain_loss_pct']:.1f}%)"


def _reduce_recommendation(candidates: list[dict], target_amount: float) -> str:
    tax_free = [c for c in candidates if c["account_type"] in ("tax_free", "tax_deferred")]
    tax_free_total = sum(c["current_value"] for c in tax_free)

    if tax_free_total >= target_amount:
        return "Can rebalance entirely within tax-advantaged accounts — no tax impact"

    losers = [c for c in candidates if c["account_type"] == "taxable" and c["gain_loss_dollar"] <= 0]
    loser_total = sum(c["current_value"] for c in losers)

    if tax_free_total + loser_total >= target_amount:
        return "Sell in tax-advantaged accounts first, then harvest losses in taxable"

    return "Will require selling some winners in taxable — review tax impact before proceeding"


def contribution_allocation(
    monthly_contributions: dict,
    drift: pd.DataFrame,
    total_value: float,
) -> list[dict]:
    underweight = drift[drift["drift_pct"] < -0.01].copy()
    if underweight.empty:
        return []

    total_underweight = underweight["drift_pct"].abs().sum()
    recs = []

    for acct, amount in monthly_contributions.items():
        if amount <= 0:
            continue

        acct_type = {
            "401k": "tax_deferred",
            "hsa": "tax_free",
            "roth_ira": "tax_free",
            "taxable": "taxable",
        }.get(acct, "taxable")

        for _, row in underweight.iterrows():
            cls = row["asset_class"]
            share = (abs(row["drift_pct"]) / total_underweight) * amount
            ideal_accts = IDEAL_ASSET_LOCATION.get(cls, ["taxable"])

            if acct_type in ideal_accts[:2]:
                reason = get_location_explanation(cls, acct_type)
                recs.append({
                    "account": acct,
                    "asset_class": cls,
                    "suggested_amount": share,
                    "alignment": "good" if acct_type == ideal_accts[0] else "acceptable",
                    "reason": reason,
                })

    return recs
