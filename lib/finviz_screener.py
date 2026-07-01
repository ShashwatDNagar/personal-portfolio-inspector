"""Live stock discovery via Finviz screener. Replaces hardcoded universes with real-time screens."""

from finvizfinance.screener.overview import Overview
import pandas as pd


def _run_finviz_screen(filters: dict) -> list[str]:
    try:
        foverview = Overview()
        foverview.set_filter(filters_dict=filters)
        result = foverview.screener_view()
        if result is not None and not result.empty:
            return result["Ticker"].tolist()
    except Exception:
        pass
    return []


def discover_high_growth(max_results: int = 40) -> list[str]:
    """Small/mid-cap stocks with strong revenue growth and healthy margins."""
    return _run_finviz_screen({
        "Market Cap.": "Small ($300mln to $2bln)",
        "Sales growthqtr over qtr": "Over 25%",
        "Gross Margin": "Over 50%",
        "Average Volume": "Over 200K",
    })[:max_results] + _run_finviz_screen({
        "Market Cap.": "Mid ($2bln to $10bln)",
        "Sales growthqtr over qtr": "Over 25%",
        "Gross Margin": "Over 50%",
        "Average Volume": "Over 200K",
    })[:max_results]


def discover_value_dividend(max_results: int = 30) -> list[str]:
    """Cheap stocks with solid dividends and manageable debt."""
    return _run_finviz_screen({
        "P/E": "Under 20",
        "Dividend Yield": "Over 3%",
        "Payout Ratio": "Under 70%",
        "Debt/Equity": "Under 1",
        "Average Volume": "Over 200K",
    })[:max_results]


def discover_oversold_quality(max_results: int = 30) -> list[str]:
    """Quality stocks with RSI below 40 — potential pullback buys."""
    return _run_finviz_screen({
        "RSI (14)": "Oversold (40)",
        "Return on Equity": "Over 15%",
        "Average Volume": "Over 200K",
        "Market Cap.": "+Mid (over $2bln)",
    })[:max_results]


def discover_momentum_breakout(max_results: int = 30) -> list[str]:
    """Stocks hitting new highs with strong fundamentals."""
    return _run_finviz_screen({
        "20-Day Simple Moving Average": "Price above SMA20",
        "50-Day Simple Moving Average": "Price above SMA50",
        "New 52-Week High/Low": "New High",
        "EPS growththis year": "Over 20%",
        "Average Volume": "Over 500K",
    })[:max_results]


FINVIZ_SCREENS = {
    "high_growth": {
        "label": "High-Growth Small/Mid-Cap",
        "description": "Revenue growing >25%, gross margins >50%, market cap $300M–$10B",
        "func": discover_high_growth,
    },
    "value_dividend": {
        "label": "Value Dividend",
        "description": "P/E <20, dividend >3%, low debt, sustainable payout",
        "func": discover_value_dividend,
    },
    "oversold_quality": {
        "label": "Oversold Quality",
        "description": "RSI <40, ROE >15% — quality names on sale",
        "func": discover_oversold_quality,
    },
    "momentum_breakout": {
        "label": "Momentum Breakout",
        "description": "New 52-week highs with >20% EPS growth and strong moving averages",
        "func": discover_momentum_breakout,
    },
}


def run_live_discovery(screen_keys: list[str] | None = None, progress_callback=None) -> dict[str, list[str]]:
    """Run selected Finviz screens and return {screen_key: [tickers]}."""
    keys = screen_keys or list(FINVIZ_SCREENS.keys())
    results = {}
    for i, key in enumerate(keys):
        screen = FINVIZ_SCREENS.get(key)
        if screen:
            results[key] = screen["func"]()
        if progress_callback:
            progress_callback((i + 1) / len(keys))
    return results
