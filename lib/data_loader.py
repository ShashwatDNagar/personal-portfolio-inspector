import pandas as pd
import re
import shutil
from datetime import datetime
from pathlib import Path


ACCOUNT_TYPE_MAP = {
    "Taxable Brokerage": "taxable",
    "ROTH IRA": "tax_free",
    "Traditional IRA": "tax_deferred",
    "HSA": "tax_free",
}

# Fallback for account names that don't match exactly above — Fidelity 401(k)/403(b)
# nicknames vary per employer (e.g. "401(K) - ACME"), so match on keyword instead of
# hardcoding any one employer's plan name.
ACCOUNT_TYPE_KEYWORDS = [
    ("ROTH", "tax_free"),
    ("HSA", "tax_free"),
    ("401", "tax_deferred"),
    ("403", "tax_deferred"),
    ("IRA", "tax_deferred"),
    ("BROKERAGE", "taxable"),
]

ACCOUNT_TYPE_LABELS = {
    "taxable": "Taxable",
    "tax_free": "Tax-Free",
    "tax_deferred": "Tax-Deferred",
}


def _classify_account_type(account_name: str) -> str:
    if account_name in ACCOUNT_TYPE_MAP:
        return ACCOUNT_TYPE_MAP[account_name]
    upper = account_name.upper()
    for keyword, acc_type in ACCOUNT_TYPE_KEYWORDS:
        if keyword in upper:
            return acc_type
    return "unknown"


def _parse_currency(val):
    if pd.isna(val) or val == "":
        return 0.0
    s = str(val).strip()
    s = s.replace("$", "").replace(",", "").replace("%", "")
    s = re.sub(r"[()]", lambda m: "-" if m.group() == "(" else "", s)
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def standardize_csv(source_path: Path, data_dir: Path) -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    standard_name = f"positions_{today}.csv"
    dest = data_dir / standard_name

    if source_path.resolve() != dest.resolve():
        shutil.copy2(source_path, dest)

    return dest


def save_uploaded_csv(uploaded_file, data_dir: Path) -> Path:
    data_dir.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    dest = data_dir / f"positions_{today}.csv"
    dest.write_bytes(uploaded_file.getvalue())
    return dest


def load_fidelity_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df = df[df["Symbol"].notna()].copy()
    df = df[~df["Symbol"].str.contains(r"\*\*", na=False)]
    df = df[~df["Description"].str.contains("Pending activity", case=False, na=False)]

    df["symbol"] = df["Symbol"].str.strip()
    df["description"] = df["Description"].str.strip()
    df["account_name"] = df["Account Name"].str.strip()
    df["account_type"] = df["account_name"].apply(_classify_account_type)
    df["account_type_label"] = df["account_type"].map(ACCOUNT_TYPE_LABELS).fillna("Unknown")
    df["quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
    df["last_price"] = df["Last Price"].apply(_parse_currency)
    df["current_value"] = df["Current Value"].apply(_parse_currency)
    df["cost_basis_total"] = df["Cost Basis Total"].apply(_parse_currency)
    df["avg_cost_basis"] = df["Average Cost Basis"].apply(_parse_currency)
    df["gain_loss_dollar"] = df["Total Gain/Loss Dollar"].apply(_parse_currency)
    df["gain_loss_pct"] = df["Total Gain/Loss Percent"].apply(_parse_currency)

    keep = [
        "account_name", "account_type", "account_type_label", "symbol", "description",
        "quantity", "last_price", "current_value",
        "cost_basis_total", "avg_cost_basis",
        "gain_loss_dollar", "gain_loss_pct",
    ]
    return df[keep].reset_index(drop=True)
