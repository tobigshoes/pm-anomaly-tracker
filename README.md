# ⚡ PM Anomaly + Whale Tracker

Real-time anomaly detection and whale monitoring for **Kalshi** and **Polymarket**.

## Features

**Anomaly Tracker**
- Fetches top 60 markets by volume from both platforms every refresh
- Detects volume spikes via rolling z-scores (Notable / Strong / Extreme)
- Detects rapid price moves ≥ 5%
- Alert log with timestamped history

**Whale Tracker (Polymarket)**
- Pulls top traders by all-time P&L from Polymarket's public leaderboard
- Monitors their recent trades and flags positions above your size threshold (default $5K)
- 🐳 MEGA WHALE tag for trades above $25K
- Live leaderboard with P&L, volume, and profile links
- Deduplication by transaction hash — no repeat alerts

> Kalshi does not expose public trader data so whale tracking is Polymarket-only.
> Kalshi large trades are still caught indirectly via the anomaly volume z-score.

## Deploy on Streamlit Cloud (free)

1. Create a GitHub repo and upload `app.py`, `requirements.txt`, `README.md`
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. Click **New app** → select your repo → main file: `app.py` → **Deploy**
4. Bookmark the permanent URL on your phone

## How to use

| Button | What it does |
|--------|-------------|
| 🔄 Refresh | Fetches fresh market data from Kalshi + Polymarket |
| 🐳 Scan Whales | Loads leaderboard + scans recent trades for all tracked wallets |
| Auto-refresh toggle | Polls both automatically every 60s |

**Workflow:**
1. Hit 🔄 Refresh 3+ times to build a volume baseline (anomaly signals need history)
2. Hit 🐳 Scan Whales to load the top traders and their latest moves
3. Set a min trade size in the sidebar (default $5K) — lower it if you want more signals
4. Check the **Whale Feed** tab for large trades and **Anomalies** for market-level spikes

## Signal levels

| Signal | Condition |
|--------|-----------|
| 🔴 Extreme | Volume z-score ≥ 4.0 |
| 🟠 Strong  | Volume z-score ≥ 3.0 |
| 🟡 Notable | Volume z-score ≥ 2.0 or price Δ ≥ 5% |
| 🐳 Whale   | Top trader trade ≥ min size threshold |
| 🐳🐳 Mega  | Trade ≥ $25,000 |

## Kalshi API key (optional)

Public market data works without a key.
kalshi.com → Settings → API → Generate key → paste in sidebar.
