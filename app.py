import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# Local-only, gitignored file for personal inputs (age, income, contributions).
# Never pre-populated by anyone but the user's own sidebar entries — nothing
# personal is ever written into tracked source files.
USER_CONFIG_PATH = Path("local_config.json")


def _load_user_config() -> dict:
    if USER_CONFIG_PATH.exists():
        try:
            return json.loads(USER_CONFIG_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_user_config(config: dict) -> None:
    USER_CONFIG_PATH.write_text(json.dumps(config, indent=2))

from lib.data_loader import load_fidelity_csv, standardize_csv, save_uploaded_csv
from lib.market_data import fetch_market_data, update_prices
from lib.classifier import classify_holdings, decompose_target_date
from lib.allocation import (
    suggest_target_allocation,
    compute_allocation,
    compute_drift,
    compute_account_allocation,
    ALLOCATION_PROFILES,
    ASSET_CLASS_EXPLANATIONS,
    LOCATION_EXPLANATIONS,
    get_profile_for_user,
)
from lib.rebalancer import generate_rebalancing_actions, contribution_allocation, get_location_explanation
from lib.signals import compute_signals, dca_suggestions, SIGNAL_EXPLANATIONS
from lib.history import save_snapshot, load_snapshots, snapshots_to_df
from lib.health import grade_holdings, GRADE_EXPLANATIONS, DIMENSION_EXPLANATIONS
from lib.classifier import is_fund as _is_fund_check
from lib.screener import DEFAULT_UNIVERSE, HIGH_UPSIDE_UNIVERSE, fetch_screener_data, run_all_screens
from lib.sectors import fetch_sector_performance, compute_rotation_signal, SECTOR_ETFS, THEME_ETFS
from lib.correlation import compute_correlation_matrix, find_correlated_clusters
from lib.dividends import compute_dividend_income, project_dividend_growth
from lib.fi_projection import monte_carlo_projection, probability_of_fi, coast_fi, compute_fi_number
from lib.tax_planning import analyze_tax_situation, tax_loss_pairs
from lib.wash_sales import detect_wash_sale_risk
from lib.finviz_screener import FINVIZ_SCREENS, run_live_discovery
from lib.etf_mining import mine_etf_holdings

st.set_page_config(page_title="Portfolio Dashboard", layout="wide", page_icon="🏛")

SECTOR_THEME_TO_ETF = {name: ticker for ticker, name in {**SECTOR_ETFS, **THEME_ETFS}.items()}


def _get_etf_holdings_cached(etf_ticker):
    """Mine and cache a single ETF's top holdings for the session (cheap: one fetch per ETF)."""
    cache_key = f"etf_holdings_cache_{etf_ticker}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = mine_etf_holdings([etf_ticker], min_overlap=1)
    return st.session_state[cache_key]


def _theme_exposure(theme_name, holdings_df):
    """Real exposure to a theme = value of holdings that overlap the theme ETF's top holdings.

    Themes (Semiconductors, Cybersecurity, etc.) aren't GICS sectors, so they can't be
    matched via Yahoo's `sector` field the way "Technology" or "Healthcare" can — a stock's
    GICS sector says nothing about which thematic ETF it'd show up in.
    """
    etf_ticker = SECTOR_THEME_TO_ETF.get(theme_name)
    if not etf_ticker:
        return 0.0, []
    holdings = set(_get_etf_holdings_cached(etf_ticker)["tickers"])
    matched = holdings_df[holdings_df["symbol"].isin(holdings)]
    return matched["current_value"].sum(), matched["symbol"].tolist()


def _render_sector_discovery(sector_name, owned_symbols):
    """Mine the sector/theme ETF's top holdings and quick-screen them, inline."""
    etf_ticker = SECTOR_THEME_TO_ETF.get(sector_name)
    if not etf_ticker:
        return

    result_key = f"discovery_{sector_name}"
    if st.button(f"Discover {sector_name} stocks →", key=f"discover_btn_{sector_name}"):
        with st.spinner(f"Mining {etf_ticker} holdings and screening..."):
            etf_data = _get_etf_holdings_cached(etf_ticker)
            tickers = etf_data["tickers"][:15]
            screener_df = fetch_screener_data(tickers) if tickers else pd.DataFrame()
            if not screener_df.empty:
                screener_df = run_all_screens(screener_df)
                screener_df["total_score"] = (
                    screener_df["canslim_score"] + screener_df["value_div_score"]
                    + screener_df["pullback_score"] + screener_df["upside_score"]
                )
                st.session_state[result_key] = screener_df.nlargest(5, "total_score")
                st.session_state["etf_universe"] = etf_data
            else:
                st.session_state[result_key] = pd.DataFrame()

    if result_key in st.session_state:
        result_df = st.session_state[result_key]
        if result_df.empty:
            st.caption(f"No screenable holdings found for {etf_ticker} right now.")
        else:
            for _, row in result_df.iterrows():
                own_badge = " *(you own this)*" if row["symbol"] in owned_symbols else ""
                st.markdown(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;→ **{row['symbol']}** — {row['name']} — "
                    f"${row['price']:,.2f}{own_badge}"
                )
            st.caption(
                f"Mined from {etf_ticker}'s top holdings. For full reasoning and DCA suggestions, "
                f"visit **Find Stocks → Stock Universe → ETF Holdings**."
            )


def _years_to_fi(current: float, annual_contrib: float, target: float, rate: float = 0.08) -> float:
    if current >= target:
        return 0
    years = 0
    balance = current
    while balance < target and years < 50:
        balance = balance * (1 + rate) + annual_contrib
        years += 1
    return years


def _build_summary_data(sig_df, include_pe=True):
    rows = []
    for _, row in sig_df.iterrows():
        range_str = f"{row['range_position']*100:.0f}%" if row["range_position"] is not None else ""
        entry = {
            "Ticker": row["symbol"],
            "Type": row.get("holding_type_label", "Stock"),
            "Account": row["account_name"],
            "Price": row["price"],
            "Market Value": row["current_value"],
            "Your Return": row["gain_loss_pct"],
            "52w Position": range_str,
            "Verdict": row["verdict"],
        }
        if include_pe:
            entry["P/E Ratio"] = row["pe_ratio"]
            entry["Fwd P/E"] = row["forward_pe"]
        entry["Div Yield"] = row["dividend_yield"]
        rows.append(entry)
    return pd.DataFrame(rows)


def _style_stock_summary(styler):
    styler.format({
        "Price": "${:,.2f}",
        "Market Value": "${:,.0f}",
        "Your Return": "{:+.1f}%",
        "P/E Ratio": lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "Fwd P/E": lambda v: f"{v:.1f}" if pd.notna(v) else "—",
        "Div Yield": lambda v: f"{v*100:.1f}%" if pd.notna(v) and v > 0 else "—",
    })
    styler.map(
        lambda v: f"color: #7FBF8F; font-weight: 600"
        if isinstance(v, (int, float)) and v > 0
        else (f"color: #C4746A; font-weight: 600"
              if isinstance(v, (int, float)) and v < 0 else ""),
        subset=["Your Return"],
    )
    styler.map(
        lambda v: f"background: rgba(127, 191, 143, 0.18); color: #7FBF8F; font-weight: 600"
        if v == "Looks Attractive"
        else (f"background: rgba(196, 116, 106, 0.18); color: #C4746A; font-weight: 600"
              if v == "Caution"
              else f"background: rgba(201, 162, 75, 0.18); font-weight: 500"),
        subset=["Verdict"],
    )
    return styler


def _style_fund_summary(styler):
    styler.format({
        "Price": "${:,.2f}",
        "Market Value": "${:,.0f}",
        "Your Return": "{:+.1f}%",
        "Div Yield": lambda v: f"{v*100:.1f}%" if pd.notna(v) and v > 0 else "—",
    })
    styler.map(
        lambda v: f"color: #7FBF8F; font-weight: 600"
        if isinstance(v, (int, float)) and v > 0
        else (f"color: #C4746A; font-weight: 600"
              if isinstance(v, (int, float)) and v < 0 else ""),
        subset=["Your Return"],
    )
    styler.map(
        lambda v: f"background: rgba(111, 145, 166, 0.18); color: #6F91A6; font-weight: 600"
        if v == "Core Holding"
        else f"background: rgba(201, 162, 75, 0.18); font-weight: 500",
        subset=["Verdict"],
    )
    return styler


# ── Color Palette (Vault / Private Bank) ──────────────────────────────────────

COLORS = {
    "vibrant": ["#B08D57", "#7FBF8F", "#C4746A", "#6F91A6", "#A67F9E", "#8FA66F", "#C9A24B", "#5E7A6E"],
    "status_green": "#7FBF8F",
    "status_red": "#C4746A",
    "status_yellow": "#C9A24B",
    "status_blue": "#6F91A6",
    "bg_green": "rgba(127, 191, 143, 0.18)",
    "bg_red": "rgba(196, 116, 106, 0.18)",
    "bg_yellow": "rgba(201, 162, 75, 0.18)",
    "ink": "#E9E4D8",
    "gold": "#B08D57",
    "panel": "#17281F",
}

# Colorblind-safe diverging scale (blue = underperforming, gold = outperforming).
# Blue/orange hues stay distinguishable across all common forms of color blindness,
# unlike red/green — used only for the Sectors tab heatmaps per user feedback.
SECTOR_HEATMAP_COLORSCALE = [
    [0, "#4E7188"],
    [0.35, "#233440"],
    [0.5, "#1A2E24"],
    [0.65, "#3D3320"],
    [1, "#C9A24B"],
]

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=13, color="#E9E4D8", family="Public Sans, sans-serif"),
    margin=dict(t=30, b=30, l=30, r=30),
    hoverlabel=dict(
        bgcolor="#1D3226",
        font_size=14,
        font_family="Public Sans, sans-serif",
        font_color="#E9E4D8",
        bordercolor="#B08D57",
    ),
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .vault-masthead {
        display: flex; justify-content: space-between; align-items: baseline;
        padding: 4px 0 18px 0; margin-bottom: 8px;
        border-bottom: 1px solid #33443A;
    }
    .vault-masthead .vault-name {
        font-family: "Fraunces", serif; font-size: 1.9rem; font-weight: 600;
        letter-spacing: 0.02em; color: #E9E4D8;
    }
    .vault-masthead .vault-tag {
        font-family: "IBM Plex Mono", monospace; font-size: 0.75rem;
        letter-spacing: 0.16em; text-transform: uppercase; color: #B08D57;
    }
    .stMetric > div {
        padding: 14px 18px; border-radius: 4px;
        background: linear-gradient(135deg, #B08D5712, #7FBF8F0a);
        border: 1px solid #33443A;
    }
    [data-testid="stMetricValue"] { font-family: "IBM Plex Mono", monospace; }
    .stExpander { border: 1px solid #33443A; border-radius: 4px; }
    div[data-testid="stDataFrame"] { border-radius: 4px; }
    .signal-card { padding: 16px; border-radius: 4px; margin: 8px 0; border: 1px solid #33443A; }
    .signal-good { background: linear-gradient(135deg, #7FBF8F14, #7FBF8F05); border-left: 3px solid #7FBF8F; }
    .signal-caution { background: linear-gradient(135deg, #C4746A14, #C4746A05); border-left: 3px solid #C4746A; }
    .signal-neutral { background: linear-gradient(135deg, #6F91A614, #6F91A605); border-left: 3px solid #6F91A6; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="vault-masthead"><span class="vault-name">Portfolio</span>'
    '<span class="vault-tag">Private &middot; FI Tracker</span></div>',
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Portfolio Dashboard")

data_dir = Path("Data")
data_dir.mkdir(exist_ok=True)
csv_files = sorted(data_dir.glob("positions_*.csv"))

uploaded = st.sidebar.file_uploader("Import Fidelity CSV", type="csv")
if uploaded:
    saved_path = save_uploaded_csv(uploaded, data_dir)
    csv_files = sorted(data_dir.glob("positions_*.csv"))
    st.sidebar.success(f"Saved as {saved_path.name}")

if csv_files:
    selected_csv = st.sidebar.selectbox(
        "Select snapshot",
        csv_files,
        index=len(csv_files) - 1,
        format_func=lambda p: p.stem.replace("positions_", "Positions — "),
    )
elif not uploaded:
    old_csvs = sorted(data_dir.glob("*.csv"))
    if old_csvs:
        selected_csv = old_csvs[-1]
        new_path = standardize_csv(selected_csv, data_dir)
        csv_files = sorted(data_dir.glob("positions_*.csv"))
        selected_csv = csv_files[-1] if csv_files else old_csvs[-1]
    else:
        st.info("Upload a Fidelity CSV export to get started. Go to Fidelity → Positions → Download.")
        st.stop()
else:
    selected_csv = None

user_config = _load_user_config()
if not USER_CONFIG_PATH.exists():
    st.sidebar.info("First run: fill in the fields below once — they'll be remembered locally (never committed to git).")

st.sidebar.divider()
st.sidebar.subheader("About You")
age = st.sidebar.number_input("Current age", value=user_config.get("age", 30), min_value=18, max_value=80)
fi_target_age = st.sidebar.number_input(
    "Target FI age", value=max(user_config.get("fi_target_age", 45), age + 1), min_value=age + 1, max_value=80
)
annual_income = st.sidebar.number_input("Annual gross income ($)", value=user_config.get("annual_income", 0), step=5000)

st.sidebar.divider()
st.sidebar.subheader("Monthly Contributions")
contrib_401k = st.sidebar.number_input("401(k)", value=user_config.get("contrib_401k", 0), step=100)
contrib_hsa = st.sidebar.number_input("HSA", value=user_config.get("contrib_hsa", 0), step=50)
contrib_roth = st.sidebar.number_input("Roth IRA (÷12)", value=user_config.get("contrib_roth", 0), step=50,
                                       help="Annual max $7,000 ÷ 12 ≈ $583/mo")
contrib_taxable = st.sidebar.number_input("Taxable brokerage", value=user_config.get("contrib_taxable", 0), step=100)

st.sidebar.divider()
st.sidebar.subheader("Expenses (for FI calc)")
monthly_expenses = st.sidebar.number_input(
    "Monthly expenses ($)", value=user_config.get("monthly_expenses", 0), step=500,
    help="If set, FI target = annual expenses ÷ 4% withdrawal rate. Leave at 0 to use 60% of income.",
)

new_user_config = {
    "age": age,
    "fi_target_age": fi_target_age,
    "annual_income": annual_income,
    "contrib_401k": contrib_401k,
    "contrib_hsa": contrib_hsa,
    "contrib_roth": contrib_roth,
    "contrib_taxable": contrib_taxable,
    "monthly_expenses": monthly_expenses,
}
if new_user_config != user_config:
    _save_user_config(new_user_config)

monthly_contributions = {
    "401k": contrib_401k,
    "hsa": contrib_hsa,
    "roth_ira": contrib_roth,
    "taxable": contrib_taxable,
}
total_monthly = sum(monthly_contributions.values())

# ── Load Data ────────────────────────────────────────────────────────────────

source = uploaded if uploaded and not selected_csv else selected_csv
if source is None:
    st.info("Upload a Fidelity CSV export to get started.")
    st.stop()

df_raw = load_fidelity_csv(source)
symbols = df_raw["symbol"].unique().tolist()

with st.spinner("Fetching live prices from Yahoo Finance..."):
    market_data = fetch_market_data(symbols)

df = update_prices(df_raw, market_data)
df = classify_holdings(df, market_data)
df_decomposed = decompose_target_date(df)

total_value = df["current_value"].sum()
target_alloc = suggest_target_allocation(age, fi_target_age)
current_alloc = compute_allocation(df_decomposed)
drift_df = compute_drift(current_alloc, target_alloc)

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_today, tab_overview, tab_rebalance, tab_health, tab_screener, tab_sectors, tab_research, tab_history = st.tabs(
    ["Today's Actions", "Overview", "Rebalancing", "Health Check", "Find Stocks", "Sectors", "Research", "History"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: TODAY'S ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_today:
    st.header("What Should I Do Today?")
    st.caption("A prioritized list of actions based on your current portfolio vs. your targets.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio Value", f"${total_value:,.0f}")
    col2.metric("Monthly Investing", f"${total_monthly:,.0f}")
    savings_rate = (total_monthly * 12 / annual_income * 100) if annual_income > 0 else 0
    col3.metric("Savings Rate", f"{savings_rate:.0f}%",
                help="What % of gross income you're investing annually")
    total_gain = df["gain_loss_dollar"].sum()
    col4.metric("Total Gain/Loss", f"${total_gain:,.0f}",
                delta=f"{total_gain/df['cost_basis_total'].sum()*100:.1f}%" if df["cost_basis_total"].sum() > 0 else None)

    fi_number = compute_fi_number(
        monthly_expenses=monthly_expenses if monthly_expenses > 0 else None,
        annual_income=annual_income,
    )
    fi_progress = total_value / fi_number * 100 if fi_number > 0 else 0
    fi_basis = (
        f"FI target = ${monthly_expenses:,.0f}/mo expenses × 12 ÷ 4% withdrawal rate"
        if monthly_expenses > 0
        else "FI target = 60% of income × 25 (the '4% rule'). Set monthly expenses in the sidebar for a more accurate number"
    )
    st.progress(min(fi_progress / 100, 1.0),
                text=f"FI Progress: ${total_value:,.0f} / ${fi_number:,.0f} target ({fi_progress:.1f}%)")
    st.caption(f"{fi_basis}. "
               f"At ${total_monthly:,.0f}/mo with ~8% returns, you could reach this in ~{_years_to_fi(total_value, total_monthly * 12, fi_number):.0f} years.")

    st.divider()

    actions = generate_rebalancing_actions(df_decomposed, drift_df, total_value, monthly_contributions)
    signals_df = compute_signals(df, market_data)
    dca = dca_suggestions(df_decomposed, drift_df, signals_df, contrib_taxable)

    high_actions = [a for a in actions if a.get("urgency") == "high"]
    medium_actions = [a for a in actions if a.get("urgency") == "medium"]

    if not high_actions and not medium_actions:
        st.success("Your portfolio is well-balanced! No urgent actions needed — keep contributing as usual.")
    else:
        st.subheader("Priority Actions")
        for i, action in enumerate(high_actions + medium_actions):
            if action["urgency"] == "high":
                icon, color = "🔴", "red"
            else:
                icon, color = "🟡", "orange"

            if action["action"] == "BUY":
                acct_label = action["preferred_account"].replace("_", " ").title()
                with st.expander(
                    f"{icon} Buy more {action['asset_class']} — ${action['amount']:,.0f} below target",
                    expanded=(i < 2),
                ):
                    st.markdown(action["plain_explanation"])
                    if action["tax_notes"]:
                        st.info(f"💡 **Tax note:** {action['tax_notes']}")
                    if action["location_reason"]:
                        st.caption(f"Why this account? {action['location_reason']}")
            else:
                with st.expander(
                    f"{icon} Trim {action['asset_class']} — ${action['amount']:,.0f} above target",
                    expanded=(i < 2),
                ):
                    st.markdown(action["plain_explanation"])
                    if action.get("recommendation"):
                        st.info(f"💡 **Strategy:** {action['recommendation']}")

    st.divider()
    st.subheader("Where to Put New Money")
    st.caption("Based on which asset classes are underweight and which accounts are best for each.")

    contrib_recs = contribution_allocation(monthly_contributions, drift_df, total_value)
    if contrib_recs:
        for rec in contrib_recs:
            alignment_icon = "✅" if rec["alignment"] == "good" else "⚠️"
            acct_display = rec["account"].replace("_", " ").upper()
            st.markdown(
                f"{alignment_icon} **{acct_display}**: "
                f"${rec['suggested_amount']:,.0f}/mo → **{rec['asset_class']}**"
            )
            if rec.get("reason"):
                st.caption(rec["reason"])
    else:
        st.info("Portfolio is balanced — distribute contributions evenly across asset classes.")

    if dca:
        st.divider()
        st.subheader("DCA Plan for Taxable Account")
        st.caption(
            "Dollar-cost averaging: invest a fixed amount regularly regardless of price. "
            "This removes the stress of timing the market."
        )
        for sug in dca:
            with st.expander(f"**{sug['suggested_fund']}** — ${sug['suggested_amount']:,.0f}/mo"):
                st.markdown(sug["explanation"])

    losers = signals_df[signals_df["gain_loss_pct"] < -10].copy()
    losers = losers[losers["account_name"].str.contains("Taxable", case=False)]
    if not losers.empty:
        st.divider()
        st.subheader("Tax-Loss Harvesting Opportunities")
        st.caption(
            "These positions are down significantly. Selling them lets you deduct the loss "
            "on your taxes — up to $3,000/year against income, or unlimited against capital gains. "
            "Buy a similar (not identical) fund within 30 days to maintain your market exposure."
        )
        for _, row in losers.iterrows():
            alternatives = tax_loss_pairs(row["symbol"])
            alt_str = f" Consider swapping to {', '.join(alternatives[:2])}." if alternatives else ""
            st.markdown(
                f"• **{row['symbol']}** — down **{row['gain_loss_pct']:.0f}%** — "
                f"potential tax savings of ~${abs(row['current_value'] * row['gain_loss_pct']/100 * 0.22):,.0f}"
                f"{alt_str}"
            )

    # ── Coast FI ──
    st.divider()
    st.subheader("Coast FI Check")
    st.caption(
        "Coast FI = the amount you'd need today so that, with zero future contributions, "
        "your portfolio would grow to your FI target by your target age."
    )
    coast = coast_fi(total_value, fi_number, age, fi_target_age)
    if coast["have_enough_to_coast"]:
        st.success(coast["description"])
    else:
        st.info(coast["description"])
    c1, c2 = st.columns(2)
    c1.metric("Coast FI Number", f"${coast['coast_fi_number']:,.0f}")
    gap_label = "Surplus" if coast["surplus_or_deficit"] >= 0 else "Gap"
    c2.metric(gap_label, f"${coast['surplus_or_deficit']:+,.0f}")

    # ── Monte Carlo Projection ──
    st.divider()
    st.subheader("FI Probability (Monte Carlo)")
    st.caption(
        "Instead of assuming a flat 8% return every year, this simulates 1,000 possible futures "
        "using realistic market volatility. It shows the range of outcomes you might experience."
    )
    mc = monte_carlo_projection(
        current_value=total_value,
        annual_contribution=total_monthly * 12,
        years=fi_target_age - age,
    )
    fi_prob = probability_of_fi(mc, fi_number, current_age=age)

    if fi_prob["probability"] >= 0.8:
        st.success(fi_prob["description"])
    elif fi_prob["probability"] >= 0.5:
        st.info(fi_prob["description"])
    else:
        st.warning(fi_prob["description"])

    import numpy as np
    fig_mc = go.Figure()
    year_labels = [age + y for y in mc["years"]]
    fig_mc.add_trace(go.Scatter(
        x=year_labels, y=mc["p10"], mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig_mc.add_trace(go.Scatter(
        x=year_labels, y=mc["p90"], mode="lines", fill="tonexty",
        fillcolor="rgba(108, 92, 231, 0.15)", line=dict(width=0),
        name="10th–90th percentile",
    ))
    fig_mc.add_trace(go.Scatter(
        x=year_labels, y=mc["p25"], mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig_mc.add_trace(go.Scatter(
        x=year_labels, y=mc["p75"], mode="lines", fill="tonexty",
        fillcolor="rgba(108, 92, 231, 0.25)", line=dict(width=0),
        name="25th–75th percentile",
    ))
    fig_mc.add_trace(go.Scatter(
        x=year_labels, y=mc["median"], mode="lines",
        line=dict(color="#B08D57", width=3), name="Median path",
        hovertemplate="Age %{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig_mc.add_hline(
        y=fi_number, line_dash="dash", line_color="#7FBF8F",
        annotation_text=f"FI Target (${fi_number:,.0f})",
    )
    fig_mc.update_layout(
        xaxis_title="Age", yaxis_title="Portfolio Value ($)",
        **PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.header("Portfolio Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What You Own")
        fig = px.pie(
            current_alloc,
            values="current_value",
            names="asset_class",
            hole=0.45,
            color_discrete_sequence=COLORS["vibrant"],
        )
        fig.update_traces(
            textposition="inside",
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Value: $%{value:,.0f}<br>Weight: %{percent}<extra></extra>",
        )
        fig.update_layout(**PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("What You're Targeting")
        target_df = pd.DataFrame([
            {"Asset Class": k, "Target Weight": v}
            for k, v in target_alloc.items()
        ])
        fig = px.pie(
            target_df,
            values="Target Weight",
            names="Asset Class",
            hole=0.45,
            color_discrete_sequence=COLORS["vibrant"],
        )
        fig.update_traces(
            textposition="inside",
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>Target: %{percent}<extra></extra>",
        )
        fig.update_layout(**PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("How Far Off Are You?")
    st.caption("Green bars = your current weight. Red bars = your target. Gaps = action items.")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="You Have",
        x=drift_df["asset_class"],
        y=drift_df["current_pct"] * 100,
        marker_color="#B08D57",
        hovertemplate="<b>%{x}</b><br>Current: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Target",
        x=drift_df["asset_class"],
        y=drift_df["target_pct"] * 100,
        marker_color="#5E7A6E",
        hovertemplate="<b>%{x}</b><br>Target: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(barmode="group", yaxis_title="% of Portfolio", **PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("By Account")
    st.caption("How your money is spread across accounts and asset classes.")
    acct_alloc = compute_account_allocation(df_decomposed)
    fig = px.sunburst(
        acct_alloc,
        path=["account_name", "broad_class"],
        values="current_value",
        color_discrete_sequence=COLORS["vibrant"],
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.0f}<br>Weight: %{percentRoot:.1%} of total<extra></extra>",
    )
    fig.update_layout(**PLOTLY_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True)

    # ── Dividend Income Forecast ──
    st.subheader("Dividend Income")
    st.caption("How much passive income your portfolio generates, and what it could grow to.")

    div_data = compute_dividend_income(df, market_data)
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Annual Dividend Income", f"${div_data['annual_income']:,.0f}")
    dc2.metric("Monthly Dividend Income", f"${div_data['monthly_income']:,.0f}")
    div_yield_overall = (div_data["annual_income"] / total_value * 100) if total_value > 0 else 0
    dc3.metric("Portfolio Yield", f"{div_yield_overall:.2f}%")

    div_projection = project_dividend_growth(div_data["annual_income"])
    if div_data["annual_income"] > 0:
        proj_df = pd.DataFrame(div_projection)
        fig_div = go.Figure()
        fig_div.add_trace(go.Bar(
            x=[f"Year {r['year']}" for r in div_projection],
            y=[r["projected_income"] for r in div_projection],
            marker_color=COLORS["vibrant"][:len(div_projection)],
            hovertemplate="<b>%{x}</b><br>Projected: $%{y:,.0f}/yr<extra></extra>",
        ))
        fig_div.add_hline(
            y=div_data["annual_income"], line_dash="dash", line_color="#999",
            annotation_text=f"Current: ${div_data['annual_income']:,.0f}/yr",
        )
        fig_div.update_layout(yaxis_title="Annual Dividend Income ($)", **PLOTLY_TEMPLATE)
        st.plotly_chart(fig_div, use_container_width=True)

        with st.expander("Dividend breakdown by holding"):
            div_holders = [h for h in div_data["by_holding"] if h["annual_dividend"] > 0]
            if div_holders:
                div_display = pd.DataFrame(div_holders)
                div_display.columns = ["Ticker", "Annual Dividend", "Yield %", "Account", "Tax Note"]
                st.dataframe(div_display, use_container_width=True, hide_index=True)
    else:
        st.info("Your portfolio doesn't generate significant dividend income.")

    st.divider()

    st.subheader("All Holdings")
    display_df = df[[
        "account_name", "symbol", "description", "holding_type_label", "quantity",
        "last_price", "current_value", "cost_basis_total",
        "gain_loss_dollar", "gain_loss_pct",
        "asset_class",
    ]].copy()
    display_df.columns = [
        "Account", "Ticker", "Name", "Type", "Shares",
        "Price", "Market Value", "Cost Basis",
        "Gain/Loss", "Return %",
        "Category",
    ]
    display_df = display_df.sort_values("Market Value", ascending=False)

    def _style_holdings(styler):
        styler.format({
            "Price": "${:,.2f}",
            "Market Value": "${:,.0f}",
            "Cost Basis": "${:,.0f}",
            "Gain/Loss": "${:+,.0f}",
            "Return %": "{:+.1f}%",
            "Shares": "{:.2f}",
        })
        styler.map(
            lambda v: f"color: {COLORS['status_green']}; font-weight: 600"
            if isinstance(v, (int, float)) and v > 0
            else (f"color: {COLORS['status_red']}; font-weight: 600"
                  if isinstance(v, (int, float)) and v < 0 else ""),
            subset=["Gain/Loss", "Return %"],
        )
        return styler

    st.dataframe(
        display_df.style.pipe(_style_holdings),
        use_container_width=True,
        height=600,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: REBALANCING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_rebalance:
    st.header("Rebalancing Plan")
    st.markdown(
        "Rebalancing means adjusting your portfolio back to your target mix. "
        "Over time, winners grow and losers shrink, which drifts you away from the risk level you chose. "
        "**The goal isn't to chase returns — it's to maintain the level of risk you're comfortable with.**"
    )

    st.divider()

    # ── Target allocation with education ──
    st.subheader("Step 1: Choose Your Target Allocation")

    profile_key = get_profile_for_user(age, fi_target_age)
    years_to_fi = fi_target_age - age

    st.markdown(
        f"Based on your age (**{age}**) and FI target (**age {fi_target_age}**, "
        f"**{years_to_fi} years away**), I'm suggesting the **{ALLOCATION_PROFILES[profile_key]['label']}** profile."
    )

    profile_tabs = st.tabs([p["label"] for p in ALLOCATION_PROFILES.values()])
    for tab, (key, profile) in zip(profile_tabs, ALLOCATION_PROFILES.items()):
        with tab:
            is_recommended = key == profile_key
            if is_recommended:
                st.success(f"✨ **Recommended for you.** {profile['who']}")
            else:
                st.caption(profile["who"])
            st.markdown(profile["description"])

            cols = st.columns(len(profile["allocation"]))
            for col, (cls, pct) in zip(cols, profile["allocation"].items()):
                col.metric(cls, f"{pct*100:.0f}%")

    st.divider()
    st.subheader("Customize Your Target")
    st.caption(
        "Start from the recommended profile and adjust. "
        "Increasing stocks = more growth potential but bigger drawdowns. "
        "Increasing bonds = smoother ride but lower long-term returns."
    )

    edited_target = {}
    cols = st.columns(len(target_alloc))
    for i, (cls, pct) in enumerate(target_alloc.items()):
        with cols[i]:
            edited_target[cls] = st.number_input(
                cls,
                value=int(pct * 100),
                min_value=0,
                max_value=100,
                step=1,
                key=f"target_{cls}",
                help=ASSET_CLASS_EXPLANATIONS.get(cls, ""),
            ) / 100

    total_target = sum(edited_target.values())
    if abs(total_target - 1.0) > 0.001:
        st.warning(f"Your percentages add up to **{total_target*100:.0f}%** — they should total 100%.")
    else:
        st.success("Allocation totals 100%.")

    if edited_target != target_alloc:
        drift_df = compute_drift(current_alloc, edited_target)
        used_target = edited_target
    else:
        used_target = target_alloc

    # ── What each asset class means ──
    with st.expander("What does each asset class mean?"):
        for cls, explanation in ASSET_CLASS_EXPLANATIONS.items():
            st.markdown(f"**{cls}:** {explanation}")

    st.divider()

    # ── Drift analysis ──
    st.subheader("Step 2: See Where You Stand")
    st.caption("How your current portfolio compares to the target you set above.")

    drift_display = drift_df.copy()
    drift_display.columns = [
        "Asset Class", "Current Value", "You Have", "Target", "Drift", "Drift ($)", "Status",
    ]

    def _style_drift(styler):
        styler.format({
            "Current Value": "${:,.0f}",
            "You Have": "{:.1%}",
            "Target": "{:.1%}",
            "Drift": "{:+.1%}",
            "Drift ($)": "${:+,.0f}",
        })
        styler.map(
            lambda v: f"background: {COLORS['bg_red']}; color: {COLORS['status_red']}; font-weight: 600"
            if v == "Overweight"
            else (f"background: {COLORS['bg_green']}; color: {COLORS['status_green']}; font-weight: 600"
                  if v == "Underweight"
                  else f"background: {COLORS['bg_yellow']}; color: #666; font-weight: 500"),
            subset=["Status"],
        )
        return styler

    st.dataframe(drift_display.style.pipe(_style_drift), use_container_width=True, hide_index=True)

    st.divider()

    # ── Actionable recommendations ──
    st.subheader("Step 3: What to Do About It")
    st.caption(
        "I'll always suggest using new contributions first (cheapest, no tax consequences). "
        "Selling is a last resort and I'll flag the tax impact."
    )

    actions = generate_rebalancing_actions(
        df_decomposed, drift_df, total_value, monthly_contributions
    )

    if not actions:
        st.success("Your portfolio is within tolerance of your target. No action needed!")
    else:
        for action in actions:
            if action["action"] == "BUY":
                with st.expander(
                    f"📈 **Buy {action['asset_class']}** — {action['drift_pct']*100:.1f}% below target (${action['amount']:,.0f})",
                    expanded=True,
                ):
                    st.markdown(action["plain_explanation"])

                    col1, col2 = st.columns(2)
                    col1.metric("Amount Needed", f"${action['amount']:,.0f}")
                    if action["months_to_correct_via_contributions"] < float("inf"):
                        col2.metric("Months to Fix (via contributions)", f"{action['months_to_correct_via_contributions']:.0f}")

                    if action["tax_notes"]:
                        st.info(f"💡 {action['tax_notes']}")
            else:
                with st.expander(
                    f"📉 **Trim {action['asset_class']}** — {action['drift_pct']*100:.1f}% above target (${action['amount']:,.0f})",
                    expanded=True,
                ):
                    st.markdown(action["plain_explanation"])

                    if action.get("recommendation"):
                        st.info(f"💡 {action['recommendation']}")

                    if action.get("candidates"):
                        st.markdown("**If you do need to sell, here's the priority order:**")
                        for j, cand in enumerate(action["candidates"][:5], 1):
                            st.markdown(
                                f"{j}. **{cand['symbol']}** in {cand['account_name']} "
                                f"(${cand['current_value']:,.0f})"
                            )
                            st.caption(cand.get("plain_explanation", cand["tax_notes"]))

    st.divider()

    # ── Asset location ──
    st.subheader("Step 4: Are Your Assets in the Right Accounts?")
    st.markdown(
        "Different asset types are taxed differently. Putting each asset in the right "
        "account type can save you thousands over your investing lifetime. "
        "This is called **asset location** (not allocation)."
    )

    for cls, explanation in LOCATION_EXPLANATIONS.items():
        st.caption(f"**{cls}:** {explanation}")

    st.markdown("")

    acct_alloc = compute_account_allocation(df_decomposed)
    location_data = []
    for _, row in acct_alloc.iterrows():
        from lib.allocation import IDEAL_ASSET_LOCATION
        ideal = IDEAL_ASSET_LOCATION.get(row["broad_class"], [])
        if ideal:
            if row["account_type"] == ideal[0]:
                placement = "Optimal"
            elif row["account_type"] in ideal[:2]:
                placement = "Good"
            else:
                placement = "Suboptimal"
        else:
            placement = "N/A"

        location_data.append({
            "Account": row["account_name"],
            "Asset Class": row["broad_class"],
            "Value": row["current_value"],
            "Placement": placement,
            "Why": get_location_explanation(row["broad_class"], row["account_type"]),
        })

    loc_df = pd.DataFrame(location_data)

    def _style_location(styler):
        styler.format({"Value": "${:,.0f}"})
        styler.map(
            lambda v: f"background: {COLORS['bg_green']}; color: {COLORS['status_green']}; font-weight: 600"
            if v == "Optimal"
            else (f"background: {COLORS['bg_yellow']}; color: {COLORS['status_yellow']}; font-weight: 600"
                  if v == "Good"
                  else (f"background: {COLORS['bg_red']}; color: {COLORS['status_red']}; font-weight: 600"
                        if v == "Suboptimal" else "")),
            subset=["Placement"],
        )
        return styler

    st.dataframe(loc_df.style.pipe(_style_location), use_container_width=True, hide_index=True)

    # ── Tax Planning ──
    st.divider()
    st.subheader("Step 5: Year-End Tax Planning")
    st.markdown(
        "Strategic tax management can save you thousands. This section analyzes "
        "your **taxable account** for opportunities to reduce your tax bill."
    )

    tax_analysis = analyze_tax_situation(df, annual_income)

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Unrealized Gains", f"${tax_analysis['unrealized_gains']:,.0f}")
    tc2.metric("Unrealized Losses", f"${tax_analysis['unrealized_losses']:,.0f}")
    tc3.metric("Net Position", f"${tax_analysis['net_position']:+,.0f}")

    if tax_analysis["harvestable_losses"]:
        st.markdown("**Loss Harvesting Candidates** (sorted by largest potential savings):")
        for h in tax_analysis["harvestable_losses"][:5]:
            with st.expander(f"📉 {h['symbol']} — ${abs(h['loss']):,.0f} loss → ~${h['tax_savings']:,.0f} tax savings"):
                st.markdown(h["suggestion"])

    if tax_analysis["gain_management"]:
        with st.expander(f"📈 Positions with gains ({len(tax_analysis['gain_management'])} holdings)"):
            for g in tax_analysis["gain_management"][:5]:
                st.markdown(f"**{g['symbol']}**: {g['suggestion']}")

    st.markdown("**Tax Strategies:**")
    for strategy in tax_analysis["strategies"]:
        st.markdown(f"• {strategy}")

    # ── Wash Sale Warnings ──
    wash_risks = detect_wash_sale_risk(
        [a["asset_class"] for a in actions if a["action"] == "BUY"] +
        [s for s in df["symbol"].unique()]
    )
    if wash_risks:
        st.divider()
        st.subheader("⚠️ Wash Sale Warnings")
        for risk in wash_risks:
            st.warning(risk["warning"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

with tab_health:
    st.header("Portfolio Health Check")
    st.markdown(
        "Every holding gets a letter grade (A–F) based on five dimensions: "
        "**valuation**, **momentum**, **your return**, **income**, and **concentration risk**. "
        "This helps you spot which positions need attention and which are doing their job."
    )

    with st.expander("How does the grading work?"):
        for dim, explanation in DIMENSION_EXPLANATIONS.items():
            label = dim.replace("_score", "").replace("_", " ").title()
            st.markdown(f"**{label}:** {explanation}")
        st.divider()
        for grade, explanation in GRADE_EXPLANATIONS.items():
            st.markdown(f"**{grade}:** {explanation}")

    health_df = grade_holdings(df, market_data, total_value)

    if not health_df.empty:
        # Grade distribution
        st.subheader("Grade Distribution")
        grade_counts = health_df["grade"].value_counts().reindex(["A", "B", "C", "D", "F"], fill_value=0)
        grade_colors = {"A": "#7FBF8F", "B": "#6F91A6", "C": "#C9A24B", "D": "#B8825A", "F": "#C4746A"}

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=grade_counts.index.tolist(),
            y=grade_counts.values.tolist(),
            marker_color=[grade_colors.get(g, "#999") for g in grade_counts.index],
            hovertemplate="<b>Grade %{x}</b><br>%{y} holdings<extra></extra>",
        ))
        fig.update_layout(
            yaxis_title="Number of Holdings",
            xaxis_title="Grade",
            **PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.subheader("All Holdings Graded")
        health_display = health_df[[
            "symbol", "holding_type_label", "account_name", "current_value", "gain_loss_pct",
            "weight_pct", "valuation_score", "momentum_score", "return_score",
            "income_score", "concentration_score", "grade",
        ]].copy()
        health_display.columns = [
            "Ticker", "Type", "Account", "Market Value", "Your Return %",
            "Portfolio Weight %", "Valuation", "Momentum", "Return",
            "Income", "Concentration", "Grade",
        ]

        def _style_health(styler):
            styler.format({
                "Market Value": "${:,.0f}",
                "Your Return %": "{:+.1f}%",
                "Portfolio Weight %": "{:.1f}%",
            })
            score_cols = ["Valuation", "Momentum", "Return", "Income", "Concentration"]
            for col in score_cols:
                styler.map(
                    lambda v: f"background: {COLORS['bg_green']}; font-weight: 600"
                    if isinstance(v, (int, float)) and v >= 4
                    else (f"background: {COLORS['bg_red']}; font-weight: 600"
                          if isinstance(v, (int, float)) and v <= 2
                          else ""),
                    subset=[col],
                )
            styler.map(
                lambda v: f"background: #7FBF8F; color: white; font-weight: 700; font-size: 1.1em"
                if v == "A"
                else (f"background: #6F91A6; color: white; font-weight: 700; font-size: 1.1em"
                      if v == "B"
                      else (f"background: #C9A24B; color: white; font-weight: 700; font-size: 1.1em"
                            if v == "C"
                            else (f"background: #B8825A; color: white; font-weight: 700; font-size: 1.1em"
                                  if v == "D"
                                  else (f"background: #C4746A; color: white; font-weight: 700; font-size: 1.1em"
                                        if v == "F" else "")))),
                subset=["Grade"],
            )
            return styler

        st.dataframe(health_display.style.pipe(_style_health), use_container_width=True, hide_index=True, height=500)

        # Detailed cards for flagged holdings
        flagged = health_df[health_df["flags"].apply(len) > 0]
        if not flagged.empty:
            st.divider()
            st.subheader("Holdings That Need Attention")
            st.caption("These positions have one or more red flags. Expand to see details and what to consider.")

            for _, row in flagged.iterrows():
                grade_icon = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"}.get(row["grade"], "⚪")
                with st.expander(f"{grade_icon} **{row['symbol']}** — Grade {row['grade']} — ${row['current_value']:,.0f}"):
                    if row["positives"]:
                        st.markdown("**Strengths:**")
                        for p in row["positives"]:
                            st.markdown(f"- ✅ {p}")
                    if row["flags"]:
                        st.markdown("**Concerns:**")
                        for f in row["flags"]:
                            st.markdown(f"- ⚠️ {f}")

        # Top holdings spotlight
        stars = health_df[health_df["grade"].isin(["A", "B"]) & (health_df["positives"].apply(len) > 0)]
        if not stars.empty:
            st.divider()
            st.subheader("Your Strongest Holdings")
            st.caption("These are performing well across multiple dimensions.")
            for _, row in stars.head(5).iterrows():
                with st.expander(f"🟢 **{row['symbol']}** — Grade {row['grade']} — ${row['current_value']:,.0f}"):
                    for p in row["positives"]:
                        st.markdown(f"- ✅ {p}")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: FIND STOCKS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_screener:
    st.header("Find New Stocks to Buy")
    st.markdown(
        "Screen stocks using four strategies — including a **High Upside** screener "
        "designed to find smaller, faster-growing companies with outsized return potential."
    )

    with st.expander("What are the four strategies?"):
        st.markdown("""
**CANSLIM (Growth)** — William O'Neil's method for finding high-growth stocks:
- **C**urrent quarterly earnings growth (>20% ideal)
- **A**nnual revenue growth (>15% ideal)
- **N**ew highs — stock at or near 52-week high (momentum)
- **S**upply/demand — healthy institutional ownership (30-80%)
- **L**eader — high return on equity (>20%)
- Forward earnings improving

**Value Dividend** — Finding cheap, income-producing stocks:
- Low P/E ratio (<20) — paying less per dollar of earnings
- Low price-to-book (<2) — paying near asset value
- Dividend yield above 1.5-3%
- Sustainable payout ratio (20-70%)
- Strong profit margins and manageable debt

**Pullback** — Quality stocks that are temporarily oversold:
- RSI ≤ 40 (momentum indicator showing oversold conditions)
- Stock near 52-week low despite solid fundamentals
- Earnings still expected to grow (forward P/E < trailing)
- Dividend income while you wait for recovery

**High Upside (Non-Linear)** — Finding smaller companies that could 3-10x:
- Small or mid-cap ($300M–$10B) — big enough to be real, small enough to multiply
- Revenue growing >25% — capturing market share in a growing space
- High gross margins (>50%) — scalable business, each new dollar is mostly profit
- Low institutional ownership (<40%) — not yet "discovered" by big funds
- High insider ownership — management believes in the company
- Reasonable price-to-sales for growth rate
""")

    with st.expander("How should I think about non-linear gains?"):
        st.markdown("""
**The core idea:** Most of your portfolio (80-90%) should be in boring index funds that compound reliably.
But a small slice (10-20%) can go into higher-risk, higher-reward positions where you're not looking for
8% annual returns — you're looking for 3-10x over several years.

**Why this works mathematically:** If you put $200/mo into 5 high-conviction small positions ($40 each),
and most go nowhere but one hits 5x, that single winner can return more than years of index fund contributions.
The key is that your downside is capped (you lose $40/mo on the losers) but upside is uncapped.

**Strategies for monthly non-linear investing:**

1. **The Barbell** — Put 85% of contributions into index funds, 15% into 3-5 high-conviction small/mid-caps.
   Rebalance the "moonshot" bucket quarterly. Cut losers that break your thesis, let winners run.

2. **Small-Cap DCA** — Pick 3-5 high-upside stocks and dollar-cost average $25-50/mo into each.
   The DCA smooths out volatility (these stocks swing wildly). Review quarterly.

3. **Theme Concentration** — When you identify a secular trend early (AI, genomics, space),
   concentrate your moonshot bucket there. The rising tide lifts all boats in the theme.

4. **Biotech Binary Bets** — Biotech stocks often double or go to zero based on FDA approvals.
   Very small positions ($100-200 each) across several pre-catalyst biotechs.
   Most will lose, but the winners can 5-20x.

**Risk management rules:**
- Never put more than 2-3% of your portfolio in a single moonshot
- Use your taxable account (not retirement) so you can harvest losses
- Set a thesis for each position — if the thesis breaks, sell regardless of price
- Don't check these daily — review monthly at most
""")


    # ── Live Discovery via Finviz ──
    st.subheader("Live Stock Discovery")
    st.caption(
        "Discover new stocks in real time using Finviz screeners. "
        "These aren't hardcoded lists — they query live market data to find stocks matching each strategy."
    )

    discovery_cols = st.columns(len(FINVIZ_SCREENS))
    selected_screens = []
    for col, (key, screen) in zip(discovery_cols, FINVIZ_SCREENS.items()):
        with col:
            if st.checkbox(screen["label"], value=True, key=f"fvz_{key}"):
                selected_screens.append(key)
            st.caption(screen["description"])

    if selected_screens and st.button("🌐 Discover New Stocks", type="secondary"):
        discovery_progress = st.progress(0, text="Scanning Finviz...")
        discovered = run_live_discovery(
            selected_screens,
            progress_callback=lambda p: discovery_progress.progress(p, f"Scanning Finviz... {p*100:.0f}%"),
        )
        discovery_progress.empty()

        all_discovered = []
        for key, tickers in discovered.items():
            label = FINVIZ_SCREENS[key]["label"]
            if tickers:
                st.markdown(f"**{label}:** {len(tickers)} stocks found — {', '.join(tickers[:15])}{'...' if len(tickers) > 15 else ''}")
                all_discovered.extend(tickers)
            else:
                st.caption(f"{label}: no matches right now")

        all_discovered = list(dict.fromkeys(all_discovered))
        if all_discovered:
            st.session_state["discovered_tickers"] = all_discovered
            st.success(f"Found {len(all_discovered)} unique stocks. Select 'Live Discovery' below to screen them.")

    # ── ETF Holdings Mining ──
    with st.expander("🔬 Mine ETF Holdings"):
        st.caption(
            "See what professional growth fund managers hold. "
            "Stocks held by multiple growth ETFs are worth investigating."
        )
        if st.button("Mine ETF Holdings", key="mine_etfs"):
            etf_progress = st.progress(0, text="Fetching ETF holdings...")
            etf_data = mine_etf_holdings(
                progress_callback=lambda p: etf_progress.progress(p, f"Fetching ETF holdings... {p*100:.0f}%"),
            )
            etf_progress.empty()

            if etf_data["tickers"]:
                st.session_state["etf_universe"] = etf_data
                multi_held = {s: d for s, d in etf_data["details"].items() if d["overlap_count"] >= 2}
                if multi_held:
                    st.markdown(f"**Held by 2+ ETFs** ({len(multi_held)} stocks):")
                    for sym, detail in sorted(multi_held.items(), key=lambda x: -x[1]["overlap_count"])[:20]:
                        st.markdown(f"• **{sym}** — held by {', '.join(detail['held_by'])}")
                st.success(f"Found {len(etf_data['tickers'])} unique holdings. Select 'ETF Holdings' below to screen them.")
            else:
                st.warning("Couldn't fetch ETF holdings. yfinance may not support this for all ETFs.")

    st.divider()

    # Universe selection
    st.subheader("Stock Universe")
    universe_options = [
        "Blue-chip universe (~110 large-caps)",
        "High-upside universe (~65 small/mid-caps)",
        "Both (~175 stocks)",
        "Custom tickers",
    ]
    if "discovered_tickers" in st.session_state:
        universe_options.insert(0, f"Live Discovery ({len(st.session_state['discovered_tickers'])} stocks)")
    if "etf_universe" in st.session_state:
        universe_options.insert(0, f"ETF Holdings ({len(st.session_state['etf_universe']['tickers'])} stocks)")

    universe_option = st.radio(
        "Which stocks to screen?",
        universe_options,
        horizontal=True,
    )

    if universe_option == "Custom tickers":
        custom_input = st.text_area(
            "Enter tickers separated by commas",
            placeholder="e.g., AAPL, MSFT, GOOGL, AMZN",
        )
        universe = [s.strip().upper() for s in custom_input.split(",") if s.strip()] if custom_input else []
    elif universe_option == "High-upside universe (~65 small/mid-caps)":
        universe = HIGH_UPSIDE_UNIVERSE
    elif universe_option == "Both (~175 stocks)":
        universe = list(dict.fromkeys(DEFAULT_UNIVERSE + HIGH_UPSIDE_UNIVERSE))
    elif universe_option.startswith("Live Discovery"):
        universe = st.session_state.get("discovered_tickers", [])
    elif universe_option.startswith("ETF Holdings"):
        universe = st.session_state.get("etf_universe", {}).get("tickers", [])
    else:
        universe = DEFAULT_UNIVERSE

    if universe:
        if st.button("🔍 Run Screener", type="primary"):
            progress_bar = st.progress(0, text="Scanning stocks...")

            def update_progress(pct):
                progress_bar.progress(pct, text=f"Scanning stocks... {pct*100:.0f}%")

            with st.spinner(""):
                screener_df = fetch_screener_data(universe, progress_callback=update_progress)
                if not screener_df.empty:
                    screener_df = run_all_screens(screener_df)
                    st.session_state["screener_results"] = screener_df

            progress_bar.empty()

        if "screener_results" in st.session_state:
            screener_df = st.session_state["screener_results"]

            # Already own filter
            owned_symbols = set(df["symbol"].unique())
            screener_df["you_own"] = screener_df["symbol"].isin(owned_symbols)

            st.divider()

            # ── CANSLIM Leaders ──
            st.subheader("🚀 Growth Leaders (CANSLIM)")
            st.caption(
                "Stocks with strong earnings growth, revenue momentum, and institutional backing. "
                "Best for: aggressive growth in your Roth IRA or taxable account."
            )
            canslim_top = screener_df.nlargest(10, "canslim_score")
            for _, row in canslim_top.iterrows():
                own_badge = " *(you own this)*" if row["you_own"] else ""
                score_bar = "🟢" * row["canslim_score"] + "⚪" * (8 - row["canslim_score"])
                with st.expander(f"{score_bar} **{row['symbol']}** — {row['name']}{own_badge}"):
                    cols = st.columns(4)
                    cols[0].metric("Price", f"${row['price']:,.2f}")
                    cols[1].metric("P/E", f"{row['pe_ratio']:.1f}" if pd.notna(row['pe_ratio']) else "N/A")
                    cols[2].metric("Sector", row["sector"])
                    cols[3].metric("Mkt Cap", f"${row['market_cap']/1e9:.0f}B" if row['market_cap'] > 0 else "N/A")
                    for reason in row["canslim_reasons"]:
                        st.markdown(f"- ✅ {reason}")

            st.divider()

            # ── Value Dividend ──
            st.subheader("💰 Value & Dividend Picks")
            st.caption(
                "Stocks that are cheap relative to earnings, pay meaningful dividends, and have solid fundamentals. "
                "Best for: steady income in your taxable account (qualified dividends get favorable tax rates)."
            )
            value_top = screener_df.nlargest(10, "value_div_score")
            for _, row in value_top.iterrows():
                own_badge = " *(you own this)*" if row["you_own"] else ""
                score_bar = "🟢" * row["value_div_score"] + "⚪" * (8 - row["value_div_score"])
                with st.expander(f"{score_bar} **{row['symbol']}** — {row['name']}{own_badge}"):
                    cols = st.columns(4)
                    cols[0].metric("Price", f"${row['price']:,.2f}")
                    cols[1].metric("P/E", f"{row['pe_ratio']:.1f}" if pd.notna(row['pe_ratio']) else "N/A")
                    cols[2].metric("Div Yield", f"{row['dividend_yield']*100:.1f}%" if row['dividend_yield'] else "N/A")
                    cols[3].metric("Payout", f"{row['payout_ratio']*100:.0f}%" if pd.notna(row['payout_ratio']) else "N/A")
                    for reason in row["value_div_reasons"]:
                        st.markdown(f"- ✅ {reason}")

            st.divider()

            # ── Pullback ──
            st.subheader("📉 Pullback Opportunities")
            st.caption(
                "Quality stocks that are temporarily oversold — buying the dip on fundamentally strong companies. "
                "Best for: adding to existing positions or opening new ones at a discount."
            )
            pullback_top = screener_df[screener_df["pullback_score"] > 0].nlargest(10, "pullback_score")
            if pullback_top.empty:
                st.info("No significant pullback opportunities found in the current universe. The market may be broadly strong right now.")
            else:
                for _, row in pullback_top.iterrows():
                    own_badge = " *(you own this)*" if row["you_own"] else ""
                    score_bar = "🟢" * row["pullback_score"] + "⚪" * (6 - row["pullback_score"])
                    with st.expander(f"{score_bar} **{row['symbol']}** — {row['name']}{own_badge}"):
                        cols = st.columns(4)
                        cols[0].metric("Price", f"${row['price']:,.2f}")
                        cols[1].metric("RSI (14)", f"{row['rsi_14']:.0f}")
                        cols[2].metric("52w Range", f"{row['range_position']*100:.0f}%")
                        cols[3].metric("Sector", row["sector"])
                        for reason in row["pullback_reasons"]:
                            st.markdown(f"- ✅ {reason}")

            st.divider()

            # ── High Upside ──
            st.subheader("🚀 High Upside / Non-Linear Picks")
            st.caption(
                "Smaller, faster-growing companies that could deliver 3-10x returns over several years. "
                "**These are high-risk** — most won't pan out, but the winners can be transformative. "
                "Position size accordingly (1-3% of portfolio each, max)."
            )
            upside_top = screener_df[screener_df["upside_score"] > 0].nlargest(15, "upside_score")
            if upside_top.empty:
                st.info("No high-upside candidates found in this universe. Try the 'High-upside universe' option above.")
            else:
                for _, row in upside_top.iterrows():
                    own_badge = " *(you own this)*" if row["you_own"] else ""
                    cap_label = row["market_cap_category"]
                    score_bar = "🟢" * min(row["upside_score"], 10) + "⚪" * max(10 - row["upside_score"], 0)
                    with st.expander(
                        f"{score_bar} **{row['symbol']}** — {row['name']} — {cap_label}{own_badge}"
                    ):
                        cols = st.columns(5)
                        cols[0].metric("Price", f"${row['price']:,.2f}")
                        cols[1].metric("Market Cap", f"${row['market_cap']/1e9:.1f}B" if row['market_cap'] > 1e9 else f"${row['market_cap']/1e6:.0f}M")
                        cols[2].metric("Revenue Growth", f"{row['revenue_growth']*100:.0f}%" if pd.notna(row['revenue_growth']) else "N/A")
                        cols[3].metric("Gross Margin", f"{row['gross_margins']*100:.0f}%" if pd.notna(row['gross_margins']) else "N/A")
                        cols[4].metric("Inst. Ownership", f"{row['institutional_pct']*100:.0f}%" if pd.notna(row['institutional_pct']) else "N/A")

                        st.markdown("**Why this could work:**")
                        for reason in row["upside_reasons"]:
                            st.markdown(f"- ✅ {reason}")

                        if row["upside_risks"]:
                            st.markdown("**Risks to know:**")
                            for risk in row["upside_risks"]:
                                st.markdown(f"- ⚠️ {risk}")

                        # Suggested monthly DCA
                        if row["price"] > 0:
                            shares_per_50 = 50 / row["price"]
                            st.caption(
                                f"💡 At $50/mo you'd accumulate {shares_per_50:.1f} shares/mo. "
                                f"If this stock 5x'd, that monthly $50 would become ${50 * 5:,.0f}/mo worth."
                            )

            st.divider()

            # ── Composite top picks ──
            st.subheader("⭐ Top Composite Picks")
            st.caption(
                "Stocks that score well across multiple strategies — the best of the best. "
                "A high composite score means a stock looks good from multiple angles, not just one."
            )
            composite_top = screener_df.nlargest(10, "composite_score")
            composite_display = composite_top[[
                "symbol", "name", "sector", "price", "pe_ratio", "dividend_yield",
                "rsi_14", "canslim_score", "value_div_score", "pullback_score", "composite_score",
            ]].copy()
            composite_display.columns = [
                "Ticker", "Name", "Sector", "Price", "P/E",
                "Div Yield", "RSI", "Growth", "Value", "Pullback", "Total",
            ]
            composite_display["you_own"] = composite_top["you_own"].values

            def _style_composite(styler):
                styler.format({
                    "Price": "${:,.2f}",
                    "P/E": lambda v: f"{v:.1f}" if pd.notna(v) else "—",
                    "Div Yield": lambda v: f"{v*100:.1f}%" if pd.notna(v) and v > 0 else "—",
                    "RSI": "{:.0f}",
                })
                for col in ["Growth", "Value", "Pullback", "Total"]:
                    styler.map(
                        lambda v: f"background: {COLORS['bg_green']}; font-weight: 600"
                        if isinstance(v, (int, float)) and v >= 4
                        else (f"background: {COLORS['bg_yellow']}"
                              if isinstance(v, (int, float)) and v >= 2 else ""),
                        subset=[col],
                    )
                return styler

            st.dataframe(composite_display.style.pipe(_style_composite), use_container_width=True, hide_index=True)
    else:
        st.info("Enter some tickers above to screen them.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: SECTORS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_sectors:
    st.header("Sector & Theme Analysis")
    st.markdown(
        "Understanding where money is flowing helps you align new purchases with market momentum. "
        "**Sectors** represent broad parts of the economy. **Themes** are narrower trends (AI, clean energy, etc.)."
    )

    if "sector_data" not in st.session_state:
        if st.button("📊 Load Sector Data", type="primary"):
            with st.spinner("Fetching sector and theme ETF data..."):
                st.session_state["sector_data"] = fetch_sector_performance()

    if "sector_data" in st.session_state:
        perf_df = st.session_state["sector_data"]

        if not perf_df.empty:
            # ── Sector Heatmap ──
            st.subheader("Sector Performance Heatmap")
            st.caption(
                "Each cell shows the return for that sector over that time period. "
                "Gold = outperforming. Blue = underperforming. "
                "Look for sectors that are gold across all timeframes (strong momentum)."
            )

            sector_perf = perf_df[perf_df["type"] == "Sector"].copy()
            if not sector_perf.empty:
                heatmap_data = sector_perf.set_index("name")[["1 Week", "1 Month", "3 Months", "6 Months", "YTD"]]
                heatmap_data = heatmap_data.sort_values("3 Months", ascending=False)

                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data.values,
                    x=heatmap_data.columns.tolist(),
                    y=heatmap_data.index.tolist(),
                    colorscale=SECTOR_HEATMAP_COLORSCALE,
                    zmid=0,
                    text=[[f"{v:+.1f}%" if pd.notna(v) else "—" for v in row] for row in heatmap_data.values],
                    texttemplate="%{text}",
                    textfont={"size": 13, "color": "#E9E4D8"},
                    hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
                    colorbar=dict(title="Return %"),
                ))
                fig.update_layout(
                    height=450,
                    yaxis=dict(autorange="reversed"),
                    **PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Theme Performance ──
            theme_perf = perf_df[perf_df["type"] == "Theme"].copy()
            if not theme_perf.empty:
                st.subheader("Theme Performance")
                st.caption(
                    "Thematic ETFs track specific trends cutting across sectors. "
                    "A surging theme could signal where the next big opportunities are."
                )

                theme_data = theme_perf.set_index("name")[["1 Week", "1 Month", "3 Months", "6 Months", "YTD"]]
                theme_data = theme_data.sort_values("3 Months", ascending=False)

                fig = go.Figure(data=go.Heatmap(
                    z=theme_data.values,
                    x=theme_data.columns.tolist(),
                    y=theme_data.index.tolist(),
                    colorscale=SECTOR_HEATMAP_COLORSCALE,
                    zmid=0,
                    text=[[f"{v:+.1f}%" if pd.notna(v) else "—" for v in row] for row in theme_data.values],
                    texttemplate="%{text}",
                    textfont={"size": 13, "color": "#E9E4D8"},
                    hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
                    colorbar=dict(title="Return %"),
                ))
                fig.update_layout(
                    height=400,
                    yaxis=dict(autorange="reversed"),
                    **PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig, use_container_width=True)

            # ── Rotation signals ──
            signals = compute_rotation_signal(perf_df)
            if signals:
                st.divider()
                st.subheader("What the Rotation Tells You")
                st.caption("Actionable takeaways based on where money is flowing.")

                for sig in signals:
                    strength_icon = {
                        "strong": "🟢",
                        "opportunity": "🔵",
                        "watch": "🟡",
                        "avoid": "🔴",
                    }.get(sig["strength"], "⚪")

                    with st.expander(f"{strength_icon} **{sig['sector']}** — {sig['signal']}"):
                        st.markdown(sig["description"])
                        st.info(f"💡 **Action:** {sig['action']}")

            # ── Your exposure ──
            st.divider()
            st.subheader("Your Sector Exposure")
            st.caption("How your portfolio maps to sectors, so you can see gaps and overlaps.")

            your_sectors = {}
            for _, row in df.iterrows():
                md = market_data.get(row["symbol"], {})
                sector = md.get("sector", "Unknown")
                your_sectors[sector] = your_sectors.get(sector, 0) + row["current_value"]

            if your_sectors:
                sector_df = pd.DataFrame([
                    {"Sector": k, "Your Exposure": v, "% of Portfolio": v / total_value * 100}
                    for k, v in sorted(your_sectors.items(), key=lambda x: -x[1])
                ])
                fig = px.bar(
                    sector_df,
                    x="Sector",
                    y="% of Portfolio",
                    color_discrete_sequence=COLORS["vibrant"],
                )
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>Weight: %{y:.1f}%<extra></extra>",
                )
                fig.update_layout(
                    yaxis_title="% of Portfolio",
                    xaxis_title="",
                    **PLOTLY_TEMPLATE,
                )
                st.plotly_chart(fig, use_container_width=True)

                hot_sectors = [s["sector"] for s in signals if s["strength"] == "strong"]
                cold_sectors = [s["sector"] for s in signals if s["strength"] == "avoid"]

                if hot_sectors or cold_sectors:
                    st.markdown("**Cross-referencing with momentum:**")
                    owned_symbols = set(df["symbol"].unique())

                    def _exposure_for(name):
                        if name in THEME_ETFS.values():
                            return _theme_exposure(name, df)
                        return your_sectors.get(name, 0), []

                    for sector in hot_sectors:
                        exposure, matched = _exposure_for(sector)
                        if exposure > 0:
                            detail = f" ({', '.join(matched)})" if matched else ""
                            st.markdown(f"- ✅ You have **${exposure:,.0f}** in {sector}{detail}, which has strong momentum")
                        else:
                            st.markdown(f"- 💡 {sector} has strong momentum but you have **no exposure** — worth researching")
                            _render_sector_discovery(sector, owned_symbols)
                    for sector in cold_sectors:
                        exposure, matched = _exposure_for(sector)
                        if exposure > 0:
                            detail = f" ({', '.join(matched)})" if matched else ""
                            st.markdown(f"- ⚠️ You have **${exposure:,.0f}** in {sector}{detail}, which is losing momentum — monitor closely")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════

with tab_research:
    st.header("Research & Signals")
    st.markdown(
        "This tab analyzes each holding using publicly available data. "
        "**No single signal is a buy/sell recommendation** — use them together with your own judgment."
    )

    signals_df = compute_signals(df, market_data)

    stock_signals = signals_df[signals_df["holding_type"].apply(lambda t: not _is_fund_check(t))].copy()
    fund_signals = signals_df[signals_df["holding_type"].apply(_is_fund_check)].copy()

    # ── Individual Stocks ──
    st.subheader("Individual Stocks")
    st.caption("Valuation, momentum, and income signals for your individual stock picks.")

    if not stock_signals.empty:
        stock_summary = _build_summary_data(stock_signals, include_pe=True)
        st.dataframe(stock_summary.style.pipe(_style_stock_summary), use_container_width=True, hide_index=True, height=400)
    else:
        st.info("No individual stocks in your portfolio.")

    # ── Funds & ETFs ──
    st.subheader("Funds & ETFs")
    st.caption(
        "Index funds and ETFs are graded differently — P/E ratios are aggregates of hundreds of stocks "
        "and aren't meaningful buy/sell signals. Instead, we look at trend, yield, and overlap."
    )

    if not fund_signals.empty:
        fund_summary = _build_summary_data(fund_signals, include_pe=False)
        st.dataframe(fund_summary.style.pipe(_style_fund_summary), use_container_width=True, hide_index=True)

    # ── Column explainer ──
    with st.expander("What do these columns mean?"):
        st.markdown("""
- **P/E Ratio (Trailing):** Price ÷ last 12 months' earnings. Lower = cheaper relative to current profits. S&P 500 average is ~20-25.
- **Forward P/E:** Price ÷ *expected* next-year earnings. If lower than trailing P/E, analysts expect earnings to grow.
- **Dividend Yield:** Annual dividend ÷ price. Above 3% is notable. Funds like SPYD focus specifically on this.
- **52-Week Position:** Where the price sits in its past-year range. 0% = at the yearly low, 100% = at the yearly high.
- **Verdict:** A rough composite score. "Looks Attractive" = multiple positive signals. "Caution" = stretched valuation or momentum.
""")

    st.divider()

    # ── Detailed signals ──
    st.subheader("Detailed Stock Signals")
    st.caption("Click any stock to see specific signals and what they mean for you.")

    stock_with_signals = stock_signals[stock_signals["signals"].apply(len) > 0]
    stock_no_signals = stock_signals[stock_signals["signals"].apply(len) == 0]

    for _, row in stock_with_signals.iterrows():
        verdict_icon = {"Looks Attractive": "🟢", "Caution": "🔴", "Hold": "🔵"}.get(row["verdict"], "⚪")
        with st.expander(
            f"{verdict_icon} **{row['symbol']}** — {row['verdict']} — ${row['current_value']:,.0f}"
        ):
            cols = st.columns(5)
            cols[0].metric("Price", f"${row['price']:,.2f}")
            cols[1].metric("P/E", f"{row['pe_ratio']:.1f}" if pd.notna(row["pe_ratio"]) else "N/A")
            cols[2].metric("Fwd P/E", f"{row['forward_pe']:.1f}" if pd.notna(row["forward_pe"]) else "N/A")
            cols[3].metric("Div Yield",
                           f"{row['dividend_yield']*100:.1f}%" if row['dividend_yield'] else "N/A")
            if row["range_position"] is not None:
                cols[4].metric("52w Range Position", f"{row['range_position']*100:.0f}%")

            for sig, sig_type in zip(row["signals"], row["signal_types"]):
                st.markdown(f"**→ {sig}**")
                explanation = SIGNAL_EXPLANATIONS.get(sig_type, "")
                if explanation:
                    st.caption(explanation)

    if not stock_no_signals.empty:
        with st.expander(f"ℹ️ {len(stock_no_signals)} stocks with no notable signals"):
            st.caption("These are trading in normal ranges with no standout valuation or momentum signals.")
            for _, row in stock_no_signals.iterrows():
                st.markdown(f"• {row['symbol']} — ${row['current_value']:,.0f}")

    st.divider()

    st.subheader("Fund & ETF Notes")
    st.caption("Core funds are evaluated on trend, yield, and overlap — not single-stock valuation metrics.")

    for _, row in fund_signals.iterrows():
        with st.expander(f"🔵 **{row['symbol']}** — {row['verdict']} — ${row['current_value']:,.0f}"):
            cols = st.columns(4)
            cols[0].metric("Price", f"${row['price']:,.2f}")
            cols[1].metric("Your Return", f"{row['gain_loss_pct']:+.1f}%")
            cols[2].metric("Div Yield",
                           f"{row['dividend_yield']*100:.1f}%" if row['dividend_yield'] else "N/A")
            if row["range_position"] is not None:
                cols[3].metric("Index Trend", f"{row['range_position']*100:.0f}% of 52w range")

            for sig, sig_type in zip(row["signals"], row["signal_types"]):
                st.markdown(f"**→ {sig}**")
                explanation = SIGNAL_EXPLANATIONS.get(sig_type, "")
                if explanation:
                    st.caption(explanation)

    st.divider()

    # ── 52-week chart ──
    st.subheader("Where Is Each Stock in Its 52-Week Range?")
    st.caption(
        "Individual stocks only (fund P/E and range positions are aggregates and less actionable). "
        "Green = near yearly low (potential value). Red = near yearly high (momentum, but possibly stretched)."
    )

    range_data = stock_signals[stock_signals["range_position"].notna()].copy()
    if not range_data.empty:
        range_data = range_data.sort_values("range_position")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=range_data["symbol"],
            y=range_data["range_position"] * 100,
            marker_color=[
                "#7FBF8F" if v < 0.3 else "#C4746A" if v > 0.8 else "#6F91A6"
                for v in range_data["range_position"]
            ],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Position: %{y:.0f}% of 52-week range<br>"
                "<extra></extra>"
            ),
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="#999", annotation_text="Midpoint")
        fig.update_layout(yaxis_title="Position in 52-Week Range (%)", **PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Concentration ──
    st.subheader("Concentration Risk")
    st.caption(
        "If any single holding is more than 10% of your portfolio, a bad day for that stock "
        "really hurts. Diversification = not putting all eggs in one basket."
    )

    top_holdings = df.nlargest(10, "current_value")[["symbol", "current_value", "account_name"]].copy()
    top_holdings["pct_of_portfolio"] = top_holdings["current_value"] / total_value * 100
    fig = px.bar(
        top_holdings,
        x="symbol",
        y="pct_of_portfolio",
        color="account_name",
        color_discrete_sequence=COLORS["vibrant"],
    )
    fig.add_hline(y=10, line_dash="dash", line_color=COLORS["status_red"],
                  annotation_text="10% concentration threshold")
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Weight: %{y:.1f}%<br>Account: %{fullData.name}<extra></extra>",
    )
    fig.update_layout(
        yaxis_title="% of Total Portfolio",
        xaxis_title="",
        legend_title="Account",
        **PLOTLY_TEMPLATE,
    )
    st.plotly_chart(fig, use_container_width=True)

    over_10 = top_holdings[top_holdings["pct_of_portfolio"] > 10]
    if not over_10.empty:
        st.warning(
            f"**{', '.join(over_10['symbol'].tolist())}** "
            f"{'is' if len(over_10) == 1 else 'are'} above 10% — "
            f"consider whether you're comfortable with that concentration."
        )

    # ── Correlation Matrix ──
    st.divider()
    st.subheader("Correlation Matrix")
    st.caption(
        "This shows how closely your holdings move together. High correlation (deep gold) means "
        "less diversification than you think — a bad day for one is a bad day for both."
    )

    stock_symbols = stock_signals["symbol"].tolist() if not stock_signals.empty else []
    if len(stock_symbols) >= 2:
        if "corr_matrix" not in st.session_state:
            if st.button("📊 Compute Correlations", type="primary"):
                with st.spinner("Downloading 1 year of price data..."):
                    corr_matrix = compute_correlation_matrix(stock_symbols)
                    st.session_state["corr_matrix"] = corr_matrix

        if "corr_matrix" in st.session_state:
            corr_matrix = st.session_state["corr_matrix"]
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale=[
                    [0, "#C4746A"],
                    [0.5, "#1A2E24"],
                    [1, "#B08D57"],
                ],
                zmid=0,
                text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                texttemplate="%{text}",
                textfont={"size": 10, "color": "#E9E4D8"},
                hovertemplate="<b>%{x} × %{y}</b><br>Correlation: %{text}<extra></extra>",
                colorbar=dict(title="Correlation"),
            ))
            fig_corr.update_layout(
                height=max(400, len(stock_symbols) * 25),
                yaxis=dict(autorange="reversed"),
                **PLOTLY_TEMPLATE,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            clusters = find_correlated_clusters(corr_matrix, threshold=0.7)
            if clusters:
                st.markdown("**Highly Correlated Pairs** (>0.7 — hidden concentration risk):")
                for cluster in clusters[:10]:
                    st.markdown(f"• ⚠️ {cluster['warning']}")
            else:
                st.success("No highly correlated pairs found — your individual stocks are well diversified.")
    else:
        st.info("Need at least 2 individual stocks to compute correlations.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB: HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.header("Historical Tracking")
    st.caption(
        "Each time you run the dashboard, save a snapshot to track how your "
        "portfolio and allocation change over time."
    )

    if st.button("📸 Save Today's Snapshot", type="primary"):
        path = save_snapshot(df, current_alloc)
        st.success(f"Snapshot saved: {path.name}")

    snapshots = load_snapshots()

    if not snapshots:
        st.info(
            "No snapshots yet. Click the button above to save your first one. "
            "Over time, you'll see trends in your portfolio value and allocation drift."
        )
    else:
        hist_df = snapshots_to_df(snapshots)

        st.subheader("Portfolio Value Over Time")
        if len(hist_df) > 1:
            fig = px.line(
                hist_df, x="date", y="total_value",
                markers=True,
                color_discrete_sequence=["#B08D57"],
            )
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>Value: $%{y:,.0f}<extra></extra>",
                line=dict(width=3),
            )
            fig.update_layout(yaxis_title="Total Value ($)", xaxis_title="", **PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            col1.metric("Current Value", f"${hist_df['total_value'].iloc[-1]:,.0f}")
            col2.metric("Snapshot Date", hist_df['date'].iloc[-1])

        alloc_cols = [c for c in hist_df.columns if c.endswith("_pct")]
        if alloc_cols and len(hist_df) > 1:
            st.subheader("Allocation Drift Over Time")
            st.caption("Watch how your mix shifts as markets move. Big shifts = time to rebalance.")
            alloc_hist = hist_df[["date"] + alloc_cols].copy()
            alloc_hist.columns = ["date"] + [c.replace("_pct", "") for c in alloc_cols]
            alloc_melted = alloc_hist.melt(id_vars="date", var_name="Asset Class", value_name="Weight")
            alloc_melted["Weight"] *= 100
            fig = px.area(
                alloc_melted, x="date", y="Weight", color="Asset Class",
                color_discrete_sequence=COLORS["vibrant"],
            )
            fig.update_traces(
                hovertemplate="<b>%{fullData.name}</b><br>Date: %{x}<br>Weight: %{y:.1f}%<extra></extra>",
            )
            fig.update_layout(yaxis_title="Allocation %", xaxis_title="", **PLOTLY_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Saved Snapshots")
        for snap in reversed(snapshots[-5:]):
            with st.expander(f"📸 {snap['timestamp'][:10]} — ${snap['total_value']:,.0f}"):
                cols = st.columns(len(snap["allocation"]))
                for col, (cls, data) in zip(cols, snap["allocation"].items()):
                    col.metric(cls, f"{data['pct']*100:.1f}%", f"${data['value']:,.0f}")

# ── Sidebar footer ───────────────────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.markdown(f"**Portfolio: ${total_value:,.0f}**")
st.sidebar.caption("🔒 All data stays on your machine. Only outbound calls: Yahoo Finance for live prices.")


