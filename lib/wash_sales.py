"""Track potential wash sale violations by comparing portfolio snapshots."""

import json
from pathlib import Path
from datetime import datetime, timedelta


def load_recent_sells(snapshots_dir: str = "snapshots") -> list[dict]:
    """Return positions that disappeared between the two most recent snapshots."""
    snap_dir = Path(snapshots_dir)
    if not snap_dir.exists():
        return []

    snap_files = sorted(snap_dir.glob("snapshot_*.json"))
    if len(snap_files) < 2:
        return []

    with open(snap_files[-2]) as f:
        older = json.load(f)
    with open(snap_files[-1]) as f:
        newer = json.load(f)

    older_positions = {p["symbol"]: p for p in older.get("positions", [])}
    newer_symbols = {p["symbol"] for p in newer.get("positions", [])}

    sells = []
    for sym, pos in older_positions.items():
        if sym not in newer_symbols:
            gain = pos.get("gain_loss_dollar", 0)
            sells.append({
                "symbol": sym,
                "approximate_date": newer.get("timestamp", "")[:10],
                "was_loss": gain < 0,
            })

    return sells


def detect_wash_sale_risk(
    buy_recommendations: list[str],
    snapshots_dir: str = "snapshots",
) -> list[dict]:
    """Flag buy recommendations that could trigger wash sale rules."""
    recent_sells = load_recent_sells(snapshots_dir)
    if not recent_sells:
        return []

    loss_sells = [s for s in recent_sells if s["was_loss"]]
    if not loss_sells:
        return []

    buy_set = {s.upper() for s in buy_recommendations}
    risks = []

    for sell in loss_sells:
        if sell["symbol"].upper() not in buy_set:
            continue

        try:
            sell_date = datetime.strptime(sell["approximate_date"], "%Y-%m-%d")
        except (ValueError, TypeError):
            sell_date = datetime.now()

        days_since = (datetime.now() - sell_date).days
        window_end = sell_date + timedelta(days=30)

        if days_since > 30:
            continue

        risks.append({
            "symbol": sell["symbol"],
            "sold_date": sell["approximate_date"],
            "days_since_sale": days_since,
            "wash_sale_window_ends": window_end.strftime("%Y-%m-%d"),
            "warning": (
                f"You sold {sell['symbol']} at a loss ~{days_since} days ago. "
                f"Buying it back within 30 days means you can't deduct that loss on your taxes. "
                f"Wait until {window_end.strftime('%B %d')} or buy a similar-but-not-identical "
                f"stock instead."
            ),
        })

    return risks
