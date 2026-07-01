import pandas as pd


ALLOCATION_PROFILES = {
    "aggressive": {
        "label": "Aggressive Growth",
        "description": "Maximize long-term growth. Higher volatility but historically higher returns over 10+ year periods.",
        "allocation": {
            "US Equities": 0.70,
            "International": 0.22,
            "Bonds": 0.05,
            "Alternatives": 0.02,
            "Cash": 0.01,
        },
        "who": "You — 10+ years to FI, high income, can stomach 30-40% drawdowns.",
    },
    "moderate_aggressive": {
        "label": "Growth-Oriented",
        "description": "Strong growth with a small cushion. Slightly less volatile while still capturing most upside.",
        "allocation": {
            "US Equities": 0.60,
            "International": 0.20,
            "Bonds": 0.12,
            "Alternatives": 0.05,
            "Cash": 0.03,
        },
        "who": "5-10 years to goal, or if big drawdowns would cause you to panic-sell.",
    },
    "moderate": {
        "label": "Balanced",
        "description": "Classic balanced portfolio. Smoother ride but lower expected returns.",
        "allocation": {
            "US Equities": 0.45,
            "International": 0.15,
            "Bonds": 0.25,
            "Alternatives": 0.10,
            "Cash": 0.05,
        },
        "who": "Nearing FI, or prioritizing stability over growth.",
    },
}

ASSET_CLASS_EXPLANATIONS = {
    "US Equities": (
        "US stocks — the engine of long-term growth. Historically ~10% annual returns. "
        "Includes large-cap (S&P 500), total market, and individual stocks. "
        "Higher allocation = more growth potential but bigger swings."
    ),
    "International": (
        "Stocks outside the US — Europe, Asia, emerging markets. "
        "Provides diversification: when US markets struggle, international may hold up (and vice versa). "
        "Also captures growth in faster-growing economies."
    ),
    "Bonds": (
        "Loans to governments/companies that pay interest. Much lower returns than stocks "
        "but they cushion your portfolio during crashes. A 2022-style stock drop of 25% "
        "might only be 15% with bonds in the mix."
    ),
    "Alternatives": (
        "REITs (real estate), commodities, dividend stocks. "
        "Behave differently from stocks and bonds, adding another layer of diversification. "
        "Small allocation goes a long way."
    ),
    "Cash": (
        "Money market funds, savings. Zero growth but zero risk. "
        "Keep enough for opportunistic buying during dips, but too much is a drag on returns."
    ),
}


def suggest_target_allocation(age: int, fi_target_age: int) -> dict:
    years_to_fi = max(fi_target_age - age, 1)

    if years_to_fi > 10:
        return ALLOCATION_PROFILES["aggressive"]["allocation"].copy()
    elif years_to_fi > 5:
        return ALLOCATION_PROFILES["moderate_aggressive"]["allocation"].copy()
    else:
        return ALLOCATION_PROFILES["moderate"]["allocation"].copy()


def get_profile_for_user(age: int, fi_target_age: int) -> str:
    years_to_fi = max(fi_target_age - age, 1)
    if years_to_fi > 10:
        return "aggressive"
    elif years_to_fi > 5:
        return "moderate_aggressive"
    else:
        return "moderate"


IDEAL_ASSET_LOCATION = {
    "Bonds": ["tax_deferred", "tax_free", "taxable"],
    "REITs": ["tax_deferred", "tax_free", "taxable"],
    "Alternatives": ["tax_deferred", "tax_free", "taxable"],
    "International": ["taxable", "tax_free", "tax_deferred"],
    "US Equities": ["tax_free", "taxable", "tax_deferred"],
    "Cash": ["taxable", "tax_deferred", "tax_free"],
}

LOCATION_EXPLANATIONS = {
    "Bonds": "Bond interest is taxed at your highest rate → shelter it in tax-deferred accounts",
    "Alternatives": "REITs/alternatives generate ordinary income → best in tax-deferred accounts",
    "International": "Foreign tax credit only works in taxable accounts → keep international there",
    "US Equities": "Growth assets benefit most from tax-free compounding → prioritize Roth/HSA",
    "Cash": "No growth to shelter → location doesn't matter much",
}


def compute_allocation(df: pd.DataFrame) -> pd.DataFrame:
    total = df["current_value"].sum()
    if total == 0:
        return pd.DataFrame()

    by_class = df.groupby("broad_class")["current_value"].sum().reset_index()
    by_class.columns = ["asset_class", "current_value"]
    by_class["current_pct"] = by_class["current_value"] / total
    return by_class.sort_values("current_pct", ascending=False).reset_index(drop=True)


def compute_drift(current_alloc: pd.DataFrame, target: dict) -> pd.DataFrame:
    total_value = current_alloc["current_value"].sum()
    all_classes = set(current_alloc["asset_class"].tolist()) | set(target.keys())

    rows = []
    for cls in sorted(all_classes):
        current_row = current_alloc[current_alloc["asset_class"] == cls]
        current_val = current_row["current_value"].values[0] if len(current_row) > 0 else 0
        current_pct = current_val / total_value if total_value > 0 else 0
        target_pct = target.get(cls, 0)
        drift = current_pct - target_pct
        drift_dollars = drift * total_value

        status = "On Target"
        if drift > 0.03:
            status = "Overweight"
        elif drift < -0.03:
            status = "Underweight"

        rows.append({
            "asset_class": cls,
            "current_value": current_val,
            "current_pct": current_pct,
            "target_pct": target_pct,
            "drift_pct": drift,
            "drift_dollars": drift_dollars,
            "status": status,
        })

    return pd.DataFrame(rows).sort_values("drift_pct", ascending=False).reset_index(drop=True)


def compute_account_allocation(df: pd.DataFrame) -> pd.DataFrame:
    by_acct_class = df.groupby(["account_name", "account_type", "broad_class"])["current_value"].sum().reset_index()
    return by_acct_class
