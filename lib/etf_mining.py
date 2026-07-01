"""Fetch top holdings of growth/thematic ETFs to build a dynamic screener universe."""

import yfinance as yf
import pandas as pd

DEFAULT_ETFS = ["ARKK", "VBK", "IWO", "MTUM", "VUG", "VONG", "IWF", "QQQM"]


def mine_etf_holdings(
    etf_symbols: list[str] | None = None,
    min_overlap: int = 1,
    progress_callback=None,
) -> dict:
    """Return tickers held by growth ETFs, with overlap counts."""
    etfs = etf_symbols or DEFAULT_ETFS
    holding_map: dict[str, list[str]] = {}
    etf_metadata: dict[str, dict] = {}

    for i, etf_sym in enumerate(etfs):
        try:
            t = yf.Ticker(etf_sym)
            try:
                holdings_df = t.funds_data.top_holdings
            except Exception:
                holdings_df = None

            if holdings_df is None or holdings_df.empty:
                continue

            tickers = []
            for idx in holdings_df.index:
                sym = str(idx).strip().upper()
                if sym and len(sym) <= 5 and sym.isalpha():
                    tickers.append(sym)

            etf_metadata[etf_sym] = {
                "name": (t.info or {}).get("shortName", etf_sym),
                "count": len(tickers),
            }

            for sym in tickers:
                if sym not in holding_map:
                    holding_map[sym] = []
                holding_map[sym].append(etf_sym)

        except Exception:
            continue

        if progress_callback:
            progress_callback((i + 1) / len(etfs))

    details = {
        sym: {"held_by": etfs_list, "overlap_count": len(etfs_list)}
        for sym, etfs_list in holding_map.items()
        if len(etfs_list) >= min_overlap
    }

    tickers = sorted(details.keys(), key=lambda s: -details[s]["overlap_count"])

    return {
        "tickers": tickers,
        "details": details,
        "etf_metadata": etf_metadata,
    }


def get_etf_universe(min_overlap: int = 1) -> list[str]:
    """Return a flat list of tickers held by growth ETFs."""
    return mine_etf_holdings(min_overlap=min_overlap)["tickers"]
