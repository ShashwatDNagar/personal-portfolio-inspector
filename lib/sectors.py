import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLC": "Communication Services",
}

THEME_ETFS = {
    "ARKK": "Disruptive Innovation",
    "ICLN": "Clean Energy",
    "SOXX": "Semiconductors",
    "IBB": "Biotech",
    "XBI": "Biotech (Small Cap)",
    "HACK": "Cybersecurity",
    "BOTZ": "Robotics & AI",
    "TAN": "Solar",
    "KWEB": "China Internet",
    "GDX": "Gold Miners",
}


def fetch_sector_performance() -> pd.DataFrame:
    all_syms = list(SECTOR_ETFS.keys()) + list(THEME_ETFS.keys())
    end = datetime.now()
    start = end - timedelta(days=365)

    data = yf.download(all_syms, start=start, end=end, progress=False)

    if data.empty:
        return pd.DataFrame()

    close = data["Close"] if "Close" in data.columns else data[("Close",)]

    rows = []
    for sym in all_syms:
        try:
            if sym not in close.columns:
                continue
            prices = close[sym].dropna()
            if len(prices) < 5:
                continue

            current = prices.iloc[-1]

            def _return_over(days):
                if len(prices) <= days:
                    return None
                return (current / prices.iloc[-min(days, len(prices))] - 1) * 100

            ret_1w = _return_over(5)
            ret_1m = _return_over(21)
            ret_3m = _return_over(63)
            ret_6m = _return_over(126)
            ret_ytd = None
            year_start = prices[prices.index >= f"{end.year}-01-01"]
            if not year_start.empty:
                ret_ytd = (current / year_start.iloc[0] - 1) * 100

            is_sector = sym in SECTOR_ETFS
            label = SECTOR_ETFS.get(sym) or THEME_ETFS.get(sym, sym)

            rows.append({
                "symbol": sym,
                "name": label,
                "type": "Sector" if is_sector else "Theme",
                "price": current,
                "1 Week": ret_1w,
                "1 Month": ret_1m,
                "3 Months": ret_3m,
                "6 Months": ret_6m,
                "YTD": ret_ytd,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def compute_rotation_signal(perf_df: pd.DataFrame) -> list[dict]:
    if perf_df.empty:
        return []

    sectors = perf_df[perf_df["type"] == "Sector"].copy()
    if sectors.empty:
        return []

    signals = []

    # Momentum leaders (strong across all timeframes)
    for _, row in sectors.iterrows():
        short_term = row.get("1 Week", 0) or 0
        medium_term = row.get("1 Month", 0) or 0
        long_term = row.get("3 Months", 0) or 0

        if short_term > 1 and medium_term > 3 and long_term > 5:
            signals.append({
                "sector": row["name"],
                "signal": "Strong Momentum",
                "description": (
                    f"{row['name']} is leading across all timeframes "
                    f"(+{short_term:.1f}% this week, +{medium_term:.1f}% this month, "
                    f"+{long_term:.1f}% over 3 months). "
                    f"Money is flowing into this sector consistently."
                ),
                "action": f"Consider adding {row['name'].lower()} exposure if underweight.",
                "strength": "strong",
            })
        elif short_term > 2 and medium_term < 0:
            signals.append({
                "sector": row["name"],
                "signal": "Potential Reversal",
                "description": (
                    f"{row['name']} was down this month ({medium_term:+.1f}%) but bouncing "
                    f"this week (+{short_term:.1f}%). Could be a reversal or a dead cat bounce."
                ),
                "action": "Watch for confirmation before adding. Wait for a second strong week.",
                "strength": "watch",
            })
        elif short_term < -2 and long_term > 5:
            signals.append({
                "sector": row["name"],
                "signal": "Pullback in Uptrend",
                "description": (
                    f"{row['name']} has a strong 3-month trend (+{long_term:.1f}%) but is "
                    f"pulling back this week ({short_term:+.1f}%). This is often a buying opportunity."
                ),
                "action": f"If you're looking to add {row['name'].lower()} stocks, this dip could be attractive.",
                "strength": "opportunity",
            })
        elif medium_term < -5 and long_term < -5:
            signals.append({
                "sector": row["name"],
                "signal": "Downtrend",
                "description": (
                    f"{row['name']} is weak across timeframes "
                    f"({medium_term:+.1f}% this month, {long_term:+.1f}% over 3 months). "
                    f"Money is rotating out."
                ),
                "action": "Avoid adding here. If you hold positions, consider whether your thesis still holds.",
                "strength": "avoid",
            })

    # Theme momentum
    themes = perf_df[perf_df["type"] == "Theme"].copy()
    for _, row in themes.iterrows():
        ret_3m = row.get("3 Months", 0) or 0
        ret_1m = row.get("1 Month", 0) or 0
        if ret_3m > 15:
            signals.append({
                "sector": row["name"],
                "signal": "Hot Theme",
                "description": (
                    f"The {row['name']} theme is surging (+{ret_3m:.0f}% in 3 months). "
                    f"This could represent a secular trend worth exposure to."
                ),
                "action": f"Research {row['name']} ETF ({row['symbol']}) or leading stocks in this theme.",
                "strength": "strong",
            })
        elif ret_3m < -15:
            signals.append({
                "sector": row["name"],
                "signal": "Cold Theme",
                "description": (
                    f"{row['name']} is out of favor ({ret_3m:+.0f}% in 3 months). "
                    f"Could be a contrarian opportunity if the fundamentals haven't changed."
                ),
                "action": "Only consider if you have conviction in the long-term thesis.",
                "strength": "watch",
            })

    return signals
