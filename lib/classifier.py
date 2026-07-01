import pandas as pd

FUND_CLASSIFICATIONS = {
    "FNILX": "US Large Cap",
    "FSKAX": "US Total Market",
    "FXAIX": "US Large Cap",
    "FSPSX": "International Developed",
    "FSGGX": "International",
    "SPYG": "US Large Cap Growth",
    "QQQ": "US Large Cap Growth",
    "VTI": "US Total Market",
    "SPYD": "US High Dividend",
    "FNSFX": "Target Date",
}

KNOWN_FUNDS = {
    "FNILX": "index_fund",
    "FSKAX": "index_fund",
    "FXAIX": "index_fund",
    "FSPSX": "index_fund",
    "FSGGX": "index_fund",
    "FNSFX": "target_date_fund",
    "FSKAX": "index_fund",
    "FDRXX": "money_market",
    "SPAXX": "money_market",
}

KNOWN_ETFS = {"SPYG", "QQQ", "VTI", "SPYD"}

TARGET_DATE_DECOMPOSITION = {
    "FNSFX": {
        "US Large Cap": 0.42,
        "US Mid/Small Cap": 0.12,
        "International Developed": 0.26,
        "Emerging Markets": 0.10,
        "Bonds": 0.08,
        "Cash": 0.02,
    }
}

SECTOR_TO_CLASS = {
    "Technology": "US Tech",
    "Communication Services": "US Tech",
    "Consumer Cyclical": "US Growth",
    "Healthcare": "US Healthcare",
    "Financial Services": "US Financials",
    "Energy": "US Energy",
    "Industrials": "US Industrials",
    "Consumer Defensive": "US Defensive",
    "Real Estate": "REITs",
    "Utilities": "US Defensive",
    "Basic Materials": "US Industrials",
}

BROAD_CLASS_MAP = {
    "US Large Cap": "US Equities",
    "US Large Cap Growth": "US Equities",
    "US Total Market": "US Equities",
    "US Mid/Small Cap": "US Equities",
    "US High Dividend": "US Equities",
    "US Tech": "US Equities",
    "US Growth": "US Equities",
    "US Healthcare": "US Equities",
    "US Financials": "US Equities",
    "US Energy": "US Equities",
    "US Industrials": "US Equities",
    "US Defensive": "US Equities",
    "International Developed": "International",
    "International": "International",
    "Emerging Markets": "International",
    "Bonds": "Bonds",
    "REITs": "Alternatives",
    "Cash": "Cash",
    "Target Date": "Target Date",
}

OVERLAP_GROUPS = {
    "us_broad": {"FNILX", "FSKAX", "VTI", "FXAIX", "SPYG", "QQQ"},
    "international": {"FSPSX", "FSGGX"},
}

HOLDING_TYPE_LABELS = {
    "individual_stock": "Individual Stock",
    "etf": "ETF",
    "index_fund": "Index Fund",
    "target_date_fund": "Target Date Fund",
    "money_market": "Money Market",
}


def _classify_holding_type(sym: str, market_data: dict) -> str:
    if sym in KNOWN_FUNDS:
        return KNOWN_FUNDS[sym]
    if sym in KNOWN_ETFS:
        return "etf"

    md = market_data.get(sym, {})
    quote_type = md.get("quote_type", "")

    if quote_type == "ETF":
        return "etf"
    if quote_type == "MUTUALFUND":
        return "index_fund"

    return "individual_stock"


def classify_holdings(df, market_data: dict):
    df = df.copy()
    classes = []
    broad_classes = []
    holding_types = []

    for _, row in df.iterrows():
        sym = row["symbol"]

        holding_type = _classify_holding_type(sym, market_data)
        holding_types.append(holding_type)

        if sym in FUND_CLASSIFICATIONS:
            cls = FUND_CLASSIFICATIONS[sym]
        else:
            md = market_data.get(sym, {})
            sector = md.get("sector", "")
            quote_type = md.get("quote_type", "")
            if quote_type == "ETF":
                cls = "US Large Cap"
            elif sector and sector in SECTOR_TO_CLASS:
                cls = SECTOR_TO_CLASS[sector]
            else:
                cls = "US Equities"

        classes.append(cls)
        broad_classes.append(BROAD_CLASS_MAP.get(cls, "US Equities"))

    df["asset_class"] = classes
    df["broad_class"] = broad_classes
    df["holding_type"] = holding_types
    df["holding_type_label"] = df["holding_type"].map(HOLDING_TYPE_LABELS).fillna("Stock")
    return df


def is_fund(holding_type: str) -> bool:
    return holding_type in ("etf", "index_fund", "target_date_fund", "money_market")


def decompose_target_date(df):
    rows = []
    for _, row in df.iterrows():
        if row["symbol"] in TARGET_DATE_DECOMPOSITION:
            decomp = TARGET_DATE_DECOMPOSITION[row["symbol"]]
            for cls, pct in decomp.items():
                new_row = row.copy()
                new_row["asset_class"] = cls
                new_row["broad_class"] = BROAD_CLASS_MAP.get(cls, "US Equities")
                new_row["current_value"] = row["current_value"] * pct
                new_row["cost_basis_total"] = row["cost_basis_total"] * pct
                new_row["symbol"] = f"{row['symbol']} ({cls})"
                rows.append(new_row)
        else:
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)
