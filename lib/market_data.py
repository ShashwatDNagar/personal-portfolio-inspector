import yfinance as yf
import pandas as pd
from functools import lru_cache


@lru_cache(maxsize=1)
def _fetch_batch(symbols_tuple: tuple[str, ...]) -> dict:
    results = {}
    tickers = yf.Tickers(" ".join(symbols_tuple))
    for sym in symbols_tuple:
        try:
            t = tickers.tickers.get(sym)
            if t is None:
                continue
            info = t.info or {}
            fast_info = t.fast_info if hasattr(t, "fast_info") else {}

            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or (fast_info.get("last_price") if hasattr(fast_info, "get") else getattr(fast_info, "last_price", None))
                or 0
            )

            # yfinance's `dividendYield` is reported as a percentage number (e.g. 2.61
            # meaning 2.61%), not a decimal fraction — normalize here since every
            # consumer of this field (display formatting, screener thresholds) expects
            # a fraction like 0.0261.
            raw_dividend_yield = info.get("dividendYield")
            dividend_yield = raw_dividend_yield / 100 if raw_dividend_yield is not None else None

            results[sym] = {
                "price": price,
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "dividend_yield": dividend_yield,
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "market_cap": info.get("marketCap"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "short_name": info.get("shortName", ""),
                "quote_type": info.get("quoteType", ""),
                "beta": info.get("beta"),
            }
        except Exception:
            results[sym] = {"price": 0}
    return results


def fetch_market_data(symbols: list[str]) -> dict:
    unique = sorted(set(s for s in symbols if s))
    return _fetch_batch(tuple(unique))


def update_prices(df: pd.DataFrame, market: dict) -> pd.DataFrame:
    df = df.copy()
    for idx, row in df.iterrows():
        sym_data = market.get(row["symbol"], {})
        live_price = sym_data.get("price", 0)
        if live_price and live_price > 0:
            df.at[idx, "last_price"] = live_price
            df.at[idx, "current_value"] = row["quantity"] * live_price
            if row["cost_basis_total"] > 0:
                df.at[idx, "gain_loss_dollar"] = (row["quantity"] * live_price) - row["cost_basis_total"]
                df.at[idx, "gain_loss_pct"] = (df.at[idx, "gain_loss_dollar"] / row["cost_basis_total"]) * 100
    return df
