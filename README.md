# Portfolio Dashboard

A personal Streamlit dashboard for tracking a portfolio toward financial
independence — allocation drift, tax-aware rebalancing, health grading,
stock screening, sector/theme rotation, correlation risk, dividend
forecasting, tax planning, and historical snapshots.

Everything runs locally. The only outbound network calls are to Yahoo
Finance (live prices) and, optionally, Finviz (live screener discovery).

## Requirements

- [pixi](https://pixi.sh) for environment management
- A **Fidelity** brokerage account (see below — this only works with
  Fidelity's export format)

## Setup

```bash
pixi install
pixi run streamlit run app.py --server.headless true
```

On first run, the sidebar will ask for your age, income, and monthly
contributions. These are saved locally to `local_config.json` (gitignored)
so you only have to enter them once — they're never written into any
tracked file.

## Getting your data in

**This only works with Fidelity's Positions export.** Other brokerages
use different column layouts and won't parse correctly.

1. In Fidelity: **Positions → Download**.
2. **Open the downloaded CSV and delete the first column** (`Account
   Number`) before uploading it here. The app never reads that column —
   it identifies accounts by name (e.g. "Roth IRA"), not account number —
   so removing it costs you nothing functionally, and keeps a sensitive
   identifier out of any file this app touches.
3. Upload the trimmed CSV via the sidebar uploader, or drop it in `Data/`
   — it'll be copied and renamed to `positions_YYYY-MM-DD.csv`.

Account names are mapped to tax treatment (taxable / tax-deferred /
tax-free) in `lib/data_loader.py`. Common exact names (e.g. "ROTH IRA",
"HSA") are matched directly; anything else falls back to a keyword match
(e.g. any account name containing "401" is treated as tax-deferred), so
most Fidelity account nicknames — including employer-specific 401(k)
names — should classify correctly without editing anything.

## How to read the dashboard

The app is organized into eight tabs, roughly ordered from "what do I do
right now" to "how has this played out over time."

### Today's Actions
The starting point for each visit. **Priority Actions** ranks what needs
attention most (biggest drift, harvestable losses, etc.). **Where to Put
New Money** tells you which account/asset class your next contribution
should go to, so you rebalance passively instead of selling anything.
**DCA Plan** breaks that into a per-holding dollar plan for your taxable
account. **Tax-Loss Harvesting** flags positions at a loss worth
realizing. **Coast FI Check** and **FI Probability (Monte Carlo)** answer
"am I actually on track" — the Monte Carlo run shows a range of outcomes
(not just one fixed-return projection), so look at the probability, not
just the median line.

### Overview
A snapshot of what you hold vs. what you're targeting — pie charts side
by side, a drift bar chart ("How Far Off Are You?"), a per-account
breakdown, projected dividend income, and the full holdings table. Use
this to sanity-check the Today tab's recommendations against the whole
picture.

### Rebalancing
A guided, five-step walkthrough: pick or customize a target allocation,
see your drift, get specific buy/sell actions to close it, check whether
your holdings sit in the *right kind* of account (e.g. bonds in
tax-deferred, growth stocks in Roth), and — closer to year-end — a tax
planning section with wash-sale warnings. This is the "why" behind the
Today tab's actions, spelled out step by step.

### Health Check
Every holding gets an A–F grade. Individual stocks are graded on
valuation, momentum, returns, income, and concentration; funds/ETFs are
graded more leniently (an index fund's aggregate P/E isn't a meaningful
signal the way a single stock's is) and get overlap detection instead
(e.g. flagging that two funds both hold the same 500 stocks). Look at
**Holdings That Need Attention** first — that's the actionable part.

### Find Stocks
Screens for *new* stocks to consider, across four strategies (CANSLIM
growth, Value/Dividend, Pullback, and High Upside/non-linear small-caps).
**Live Stock Discovery** queries Finviz in real time rather than a
hardcoded list. You can also mine the top holdings of growth ETFs
(ARKK, VBK, etc.) as a universe to screen. Results are tagged if you
already own them, so you can tell "new idea" apart from "add to what I
have."

### Sectors
Where money is rotating, and whether your portfolio is positioned for
it. The heatmaps use a colorblind-safe gold/blue scale (gold =
outperforming, blue = underperforming) rather than red/green. **What the
Rotation Tells You** turns the raw returns into plain-English signals
(strong momentum, pullback-in-uptrend, downtrend, etc.), and **Your
Sector Exposure** cross-references those signals against what you
actually own — including a "Discover \[theme\] stocks →" button next to
any theme you have zero exposure to, so you can act on a gap immediately
instead of just noting it.

### Research
The deep-dive tab: per-holding signals (split into Individual Stocks vs.
Funds/ETFs, since the two need different lenses), 52-week range
positioning, concentration risk, and a correlation matrix. The
correlation matrix is the one to watch if you hold several stocks in the
same industry (e.g. semiconductors) — high correlation means less real
diversification than the position count suggests, even if no single
holding looks concentrated on its own.

### History
Save a snapshot any time (button at the top), and this tab charts your
total value and allocation drift over time. Nothing here without
snapshots — it's only useful if you save one periodically (e.g. monthly).

## Keeping your data private

`.gitignore` already excludes the things that matter:

- `Data/*.csv` — your actual holdings
- `snapshots/*.json` — historical portfolio snapshots
- `local_config.json` — your age/income/contribution inputs
- `.streamlit/secrets.toml` — reserved for local secrets, if you add any

None of these are required for the app to run for someone else — they're
generated locally the first time you use it.
