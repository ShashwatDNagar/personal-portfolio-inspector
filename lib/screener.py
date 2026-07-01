import yfinance as yf
import pandas as pd
import numpy as np

DEFAULT_UNIVERSE = [
    # Large-cap stalwarts
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B", "JPM", "V",
    "JNJ", "UNH", "PG", "MA", "HD", "DIS", "ADBE", "CRM", "NFLX", "PYPL",
    "INTC", "AMD", "QCOM", "TXN", "AVGO", "MU", "AMAT", "LRCX", "KLAC", "MRVL",
    "PFE", "ABBV", "TMO", "DHR", "BMY", "GILD", "AMGN", "VRTX", "REGN", "ISRG",
    "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "C", "USB", "PNC",
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "OXY", "VLO", "MPC", "PSX",
    "CAT", "DE", "UNP", "HON", "GE", "MMM", "RTX", "LMT", "BA", "UPS",
    "KO", "PEP", "WMT", "COST", "MCD", "SBUX", "NKE", "TGT", "LOW", "TJX",
    "NEE", "DUK", "SO", "AEP", "D", "SRE",
    "AMT", "PLD", "CCI", "O", "SPG", "WELL",
    "T", "VZ", "TMUS", "CMCSA", "CHTR",
    "LLY", "MRK", "ABT", "MDT", "SYK", "ZTS",
    "ASML", "TSM", "SHOP", "SQ", "SNOW", "DDOG", "NET", "CRWD", "ZS", "PANW",
]

# Small/mid-cap growth stocks with asymmetric upside potential
HIGH_UPSIDE_UNIVERSE = [
    # AI / Software (small-mid cap)
    "PLTR", "AI", "BBAI", "SOUN", "UPST", "BRZE", "CWAN",
    "IONQ", "RGTI", "QUBT",
    # Biotech / Genomics (binary outcomes = asymmetric)
    "CRSP", "BEAM", "NTLA", "EDIT", "VERV", "RXRX", "PRME",
    "DNLI", "FATE", "ARCT", "IOVA",
    # Fintech / Payments
    "SOFI", "AFRM", "HOOD", "NU", "BILL", "TOST",
    # Cybersecurity
    "S", "QLYS", "TENB", "RPD",
    # Clean energy / EV
    "ENPH", "SEDG", "RUN", "RIVN", "LCID", "QS",
    # Space / Defense tech
    "RKLB", "ASTS", "LUNR", "ASTR",
    # Consumer / eCommerce
    "HIMS", "DUOL", "CELH", "MNST",
    # Semiconductors (smaller)
    "ALAB", "WOLF", "ACLS", "ONTO",
    # Infrastructure / Cloud
    "CFLT", "MDB", "ESTC", "GTLB",
    # Misc high-growth
    "AXON", "TMDX", "CAVA", "CART", "ARM",
]


def compute_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean().iloc[-1]
    avg_loss = loss.rolling(window=period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def fetch_screener_data(symbols: list[str], progress_callback=None) -> pd.DataFrame:
    rows = []
    total = len(symbols)

    batch_size = 20
    for batch_start in range(0, total, batch_size):
        batch = symbols[batch_start:batch_start + batch_size]
        tickers = yf.Tickers(" ".join(batch))

        for sym in batch:
            try:
                t = tickers.tickers.get(sym)
                if t is None:
                    continue
                info = t.info or {}

                hist = t.history(period="3mo")
                rsi = compute_rsi(hist["Close"]) if not hist.empty else 50.0

                price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                high52 = info.get("fiftyTwoWeekHigh") or 0
                low52 = info.get("fiftyTwoWeekLow") or 0
                market_cap = info.get("marketCap") or 0

                if high52 > low52 > 0:
                    range_pos = (price - low52) / (high52 - low52)
                else:
                    range_pos = 0.5

                eps_growth = info.get("earningsQuarterlyGrowth")
                rev_growth = info.get("revenueGrowth")

                rows.append({
                    "symbol": sym,
                    "name": info.get("shortName", sym),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "price": price,
                    "market_cap": market_cap,
                    "market_cap_category": _categorize_cap(market_cap),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "pb_ratio": info.get("priceToBook"),
                    "ps_ratio": info.get("priceToSalesTrailing12Months"),
                    # yfinance reports this as a percentage number (2.61 = 2.61%), not a
                    # fraction — normalize to match the 0.03/0.015-style thresholds below.
                    "dividend_yield": (info.get("dividendYield") or 0) / 100,
                    "payout_ratio": info.get("payoutRatio"),
                    "eps_growth_qoq": eps_growth,
                    "revenue_growth": rev_growth,
                    "revenue_per_share": info.get("revenuePerShare"),
                    "gross_margins": info.get("grossMargins"),
                    "profit_margin": info.get("profitMargins"),
                    "roe": info.get("returnOnEquity"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "institutional_pct": info.get("heldPercentInstitutions"),
                    "insider_pct": info.get("heldPercentInsiders"),
                    "beta": info.get("beta"),
                    "avg_volume": info.get("averageVolume", 0),
                    "fifty_two_week_high": high52,
                    "fifty_two_week_low": low52,
                    "range_position": range_pos,
                    "rsi_14": rsi,
                    "near_52w_high": range_pos > 0.9,
                    "short_pct_float": info.get("shortPercentOfFloat"),
                    "total_revenue": info.get("totalRevenue"),
                    "free_cashflow": info.get("freeCashflow"),
                })
            except Exception:
                continue

        if progress_callback:
            progress_callback(min(batch_start + batch_size, total) / total)

    result = pd.DataFrame(rows)
    numeric_cols = [
        "price", "market_cap", "pe_ratio", "forward_pe", "peg_ratio", "pb_ratio",
        "ps_ratio", "dividend_yield", "payout_ratio", "eps_growth_qoq",
        "revenue_growth", "revenue_per_share", "gross_margins", "profit_margin",
        "roe", "debt_to_equity", "institutional_pct", "insider_pct", "beta",
        "avg_volume", "fifty_two_week_high", "fifty_two_week_low", "range_position",
        "rsi_14", "short_pct_float", "total_revenue", "free_cashflow",
    ]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def _categorize_cap(cap):
    if cap <= 0:
        return "Unknown"
    elif cap < 300e6:
        return "Micro-Cap"
    elif cap < 2e9:
        return "Small-Cap"
    elif cap < 10e9:
        return "Mid-Cap"
    elif cap < 200e9:
        return "Large-Cap"
    else:
        return "Mega-Cap"


def _safe_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def score_canslim(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scores = []

    for _, row in df.iterrows():
        score = 0
        reasons = []

        if _safe_float(row["eps_growth_qoq"]) is not None and row["eps_growth_qoq"] > 0.20:
            score += 2
            reasons.append(f"Strong quarterly earnings growth ({row['eps_growth_qoq']*100:.0f}%)")
        elif row["eps_growth_qoq"] is not None and row["eps_growth_qoq"] > 0.10:
            score += 1
            reasons.append(f"Solid quarterly earnings growth ({row['eps_growth_qoq']*100:.0f}%)")

        if row["revenue_growth"] is not None and row["revenue_growth"] > 0.15:
            score += 2
            reasons.append(f"Strong revenue growth ({row['revenue_growth']*100:.0f}%)")
        elif row["revenue_growth"] is not None and row["revenue_growth"] > 0.05:
            score += 1
            reasons.append(f"Positive revenue growth ({row['revenue_growth']*100:.0f}%)")

        if row["near_52w_high"]:
            score += 1
            reasons.append("Trading near 52-week high (momentum)")

        if row["institutional_pct"] is not None and 0.3 < row["institutional_pct"] < 0.8:
            score += 1
            reasons.append(f"Healthy institutional ownership ({row['institutional_pct']*100:.0f}%)")

        if row["roe"] is not None and row["roe"] > 0.20:
            score += 1
            reasons.append(f"Strong return on equity ({row['roe']*100:.0f}%)")

        fpe = _safe_float(row["forward_pe"])
        tpe = _safe_float(row["pe_ratio"])
        if fpe is not None and tpe is not None and fpe < tpe * 0.85:
            score += 1
            reasons.append("Earnings expected to accelerate")

        scores.append({"canslim_score": score, "canslim_reasons": reasons})

    score_df = pd.DataFrame(scores)
    df["canslim_score"] = score_df["canslim_score"]
    df["canslim_reasons"] = score_df["canslim_reasons"]
    return df


def score_value_dividend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scores = []

    for _, row in df.iterrows():
        score = 0
        reasons = []

        if row["pe_ratio"] is not None and 0 < row["pe_ratio"] < 20:
            score += 2
            reasons.append(f"Low P/E ({row['pe_ratio']:.1f}) — stock is cheap relative to earnings")
        elif row["pe_ratio"] is not None and 0 < row["pe_ratio"] < 25:
            score += 1
            reasons.append(f"Reasonable P/E ({row['pe_ratio']:.1f})")

        if row["pb_ratio"] is not None and 0 < row["pb_ratio"] < 2:
            score += 1
            reasons.append(f"Low price-to-book ({row['pb_ratio']:.1f}) — trading near asset value")

        if row["dividend_yield"] > 0.03:
            score += 2
            reasons.append(f"High dividend yield ({row['dividend_yield']*100:.1f}%)")
        elif row["dividend_yield"] > 0.015:
            score += 1
            reasons.append(f"Decent dividend yield ({row['dividend_yield']*100:.1f}%)")

        if row["payout_ratio"] is not None and 0.2 < row["payout_ratio"] < 0.7:
            score += 1
            reasons.append(f"Sustainable payout ratio ({row['payout_ratio']*100:.0f}%) — room to grow dividend")

        if row["profit_margin"] is not None and row["profit_margin"] > 0.15:
            score += 1
            reasons.append(f"Strong profit margin ({row['profit_margin']*100:.0f}%)")

        if row["debt_to_equity"] is not None and row["debt_to_equity"] < 100:
            score += 1
            reasons.append("Manageable debt levels")

        scores.append({"value_div_score": score, "value_div_reasons": reasons})

    score_df = pd.DataFrame(scores)
    df["value_div_score"] = score_df["value_div_score"]
    df["value_div_reasons"] = score_df["value_div_reasons"]
    return df


def score_pullback(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scores = []

    for _, row in df.iterrows():
        score = 0
        reasons = []

        is_quality = (
            (row["roe"] is not None and row["roe"] > 0.12)
            or (row["profit_margin"] is not None and row["profit_margin"] > 0.10)
        )

        if is_quality and row["rsi_14"] <= 35:
            score += 3
            reasons.append(f"Quality stock with RSI at {row['rsi_14']:.0f} — deeply oversold, strong bounce candidate")
        elif is_quality and row["rsi_14"] <= 40:
            score += 2
            reasons.append(f"Quality stock with RSI at {row['rsi_14']:.0f} — oversold territory")
        elif row["rsi_14"] <= 30:
            score += 1
            reasons.append(f"RSI at {row['rsi_14']:.0f} — oversold but check fundamentals")

        if row["range_position"] < 0.3 and is_quality:
            score += 1
            reasons.append(f"Near 52-week low ({row['range_position']*100:.0f}% of range) — potential deep value")

        fpe2 = _safe_float(row["forward_pe"])
        tpe2 = _safe_float(row["pe_ratio"])
        if fpe2 is not None and tpe2 is not None and fpe2 < tpe2:
            score += 1
            reasons.append("Earnings still expected to grow despite pullback — market may be overreacting")

        if row["dividend_yield"] > 0.02 and row["rsi_14"] <= 45:
            score += 1
            reasons.append(f"Collecting {row['dividend_yield']*100:.1f}% yield while you wait for recovery")

        scores.append({"pullback_score": score, "pullback_reasons": reasons})

    score_df = pd.DataFrame(scores)
    df["pullback_score"] = score_df["pullback_score"]
    df["pullback_reasons"] = score_df["pullback_reasons"]
    return df


def score_high_upside(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scores = []

    for _, row in df.iterrows():
        score = 0
        reasons = []
        risk_flags = []

        cap = row["market_cap"]
        cap_cat = row["market_cap_category"]

        # Smaller companies have more room to grow
        if cap_cat == "Micro-Cap":
            score += 2
            reasons.append(f"Micro-cap (${cap/1e6:.0f}M) — small base means a single contract or product can move the needle dramatically")
            risk_flags.append("Very volatile — position size conservatively")
        elif cap_cat == "Small-Cap":
            score += 2
            reasons.append(f"Small-cap (${cap/1e9:.1f}B) — still early enough for 5-10x potential if thesis plays out")
            risk_flags.append("Higher volatility than large-caps")
        elif cap_cat == "Mid-Cap":
            score += 1
            reasons.append(f"Mid-cap (${cap/1e9:.0f}B) — proven enough to survive, small enough to double or triple")

        # High revenue growth = expanding market
        if row["revenue_growth"] is not None and row["revenue_growth"] > 0.40:
            score += 3
            reasons.append(f"Revenue growing {row['revenue_growth']*100:.0f}% — company is capturing market share rapidly")
        elif row["revenue_growth"] is not None and row["revenue_growth"] > 0.25:
            score += 2
            reasons.append(f"Revenue growing {row['revenue_growth']*100:.0f}% — strong top-line expansion")
        elif row["revenue_growth"] is not None and row["revenue_growth"] > 0.15:
            score += 1
            reasons.append(f"Revenue growing {row['revenue_growth']*100:.0f}%")

        # High gross margins = scalable business model
        if row["gross_margins"] is not None and row["gross_margins"] > 0.70:
            score += 2
            reasons.append(f"Gross margins of {row['gross_margins']*100:.0f}% — software-like scalability, each new dollar of revenue is mostly profit")
        elif row["gross_margins"] is not None and row["gross_margins"] > 0.50:
            score += 1
            reasons.append(f"Healthy gross margins ({row['gross_margins']*100:.0f}%)")

        # Low institutional ownership = under-discovered
        if row["institutional_pct"] is not None and row["institutional_pct"] < 0.40:
            score += 1
            reasons.append(f"Only {row['institutional_pct']*100:.0f}% institutional — under the radar of big funds. When they start buying, it pushes the price up")

        # High insider ownership = skin in the game
        if row["insider_pct"] is not None and row["insider_pct"] > 0.10:
            score += 1
            reasons.append(f"Insiders own {row['insider_pct']*100:.0f}% — management has serious skin in the game")

        # Not yet profitable but burning toward it (classic growth pattern)
        if row["profit_margin"] is not None and row["profit_margin"] < 0 and row["revenue_growth"] is not None and row["revenue_growth"] > 0.20:
            reasons.append("Not yet profitable but growing fast — classic high-risk/high-reward growth profile")
            risk_flags.append("Pre-profit company — could dilute shares or run out of runway")
        elif row["profit_margin"] is not None and row["profit_margin"] > 0 and row["revenue_growth"] is not None and row["revenue_growth"] > 0.20:
            score += 1
            reasons.append("Profitable AND growing fast — rare combination, reduces risk significantly")

        # Price-to-sales for pre-profit companies
        if row["ps_ratio"] is not None and row["ps_ratio"] < 5 and row["revenue_growth"] is not None and row["revenue_growth"] > 0.20:
            score += 1
            reasons.append(f"Price-to-sales of {row['ps_ratio']:.1f}x with {row['revenue_growth']*100:.0f}% growth — reasonable for a high-growth company")

        # High beta = amplified moves
        if row["beta"] is not None and row["beta"] > 1.5:
            reasons.append(f"Beta of {row['beta']:.1f} — moves {row['beta']:.1f}x the market. Amplified gains AND losses")

        # Short interest (potential squeeze catalyst)
        if row["short_pct_float"] is not None and row["short_pct_float"] > 0.10:
            score += 1
            reasons.append(f"High short interest ({row['short_pct_float']*100:.0f}% of float) — if the thesis works, shorts covering adds fuel to the rally")

        scores.append({
            "upside_score": score,
            "upside_reasons": reasons,
            "upside_risks": risk_flags,
        })

    score_df = pd.DataFrame(scores)
    df["upside_score"] = score_df["upside_score"]
    df["upside_reasons"] = score_df["upside_reasons"]
    df["upside_risks"] = score_df["upside_risks"]
    return df


def run_all_screens(df: pd.DataFrame) -> pd.DataFrame:
    df = score_canslim(df)
    df = score_value_dividend(df)
    df = score_pullback(df)
    df = score_high_upside(df)
    df["composite_score"] = df["canslim_score"] + df["value_div_score"] + df["pullback_score"]
    return df
