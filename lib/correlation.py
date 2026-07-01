import pandas as pd
import yfinance as yf


def compute_correlation_matrix(symbols: list[str], period: str = "1y") -> pd.DataFrame:
    """Download daily closes and return a symbol×symbol correlation matrix."""
    raw = yf.download(symbols, period=period, auto_adjust=True, progress=False)["Close"]

    # yfinance returns a Series (not DataFrame) when only one symbol is given
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=symbols[0])

    # Keep only columns that actually have data
    raw = raw.dropna(axis=1, how="all")

    returns = raw.pct_change().dropna(how="all")
    return returns.corr()


def find_correlated_clusters(
    corr_matrix: pd.DataFrame, threshold: float = 0.7
) -> list[dict]:
    """Return pairs whose absolute correlation exceeds threshold."""
    results = []
    symbols = list(corr_matrix.columns)

    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i + 1 :]:
            if sym1 not in corr_matrix.index or sym2 not in corr_matrix.columns:
                continue
            corr_val = corr_matrix.loc[sym1, sym2]
            if pd.isna(corr_val):
                continue
            if abs(corr_val) >= threshold:
                pct = round(abs(corr_val) * 100)
                direction = "move together" if corr_val > 0 else "move in opposite directions"
                warning = (
                    f"{sym1} and {sym2} {direction} {pct}% of the time — "
                    f"owning both gives less diversification than you think."
                )
                results.append(
                    {
                        "pair": (sym1, sym2),
                        "correlation": round(float(corr_val), 4),
                        "warning": warning,
                    }
                )

    results.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return results
