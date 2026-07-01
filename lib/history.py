import json
from datetime import datetime
from pathlib import Path
import pandas as pd

SNAPSHOT_DIR = Path(__file__).parent.parent / "snapshots"


def save_snapshot(df: pd.DataFrame, allocation: pd.DataFrame):
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "total_value": float(df["current_value"].sum()),
        "allocation": {
            row["asset_class"]: {
                "value": float(row["current_value"]),
                "pct": float(row["current_pct"]),
            }
            for _, row in allocation.iterrows()
        },
        "positions": [
            {
                "symbol": row["symbol"],
                "account": row["account_name"],
                "value": float(row["current_value"]),
                "cost_basis": float(row["cost_basis_total"]),
                "gain_loss_pct": float(row["gain_loss_pct"]),
            }
            for _, row in df.iterrows()
        ],
        "accounts": {
            acct: float(group["current_value"].sum())
            for acct, group in df.groupby("account_name")
        },
    }

    path = SNAPSHOT_DIR / f"snapshot_{ts}.json"
    path.write_text(json.dumps(snapshot, indent=2))
    return path


def load_snapshots() -> list[dict]:
    if not SNAPSHOT_DIR.exists():
        return []

    snapshots = []
    for f in sorted(SNAPSHOT_DIR.glob("snapshot_*.json")):
        try:
            snapshots.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, IOError):
            continue

    return snapshots


def snapshots_to_df(snapshots: list[dict]) -> pd.DataFrame:
    if not snapshots:
        return pd.DataFrame()

    rows = []
    for snap in snapshots:
        row = {
            "date": snap["timestamp"][:10],
            "total_value": snap["total_value"],
        }
        for cls, data in snap.get("allocation", {}).items():
            row[f"{cls}_pct"] = data["pct"]
            row[f"{cls}_value"] = data["value"]
        rows.append(row)

    return pd.DataFrame(rows)
