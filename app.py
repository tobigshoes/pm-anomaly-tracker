"""
Prediction Market Anomaly + Whale Tracker — v3 (debugged)
Monitors Kalshi and Polymarket for unusual trading activity and top-trader moves.
Deploy free at streamlit.io/cloud
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import json
from collections import deque

st.set_page_config(
    page_title="PM Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
Z_NOTABLE            = 2.0
Z_STRONG             = 3.0
Z_EXTREME            = 4.0
PRICE_MOVE_THRESHOLD = 0.05
VOLUME_WINDOW        = 20
REFRESH_SECONDS      = 60
WHALE_MIN_TRADE_USD  = 5000
WHALE_TOP_N          = 50
MEGA_WHALE_USD       = 25000

DATA_API   = "https://data-api.polymarket.com"
GAMMA_API  = "https://gamma-api.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family:'Syne',sans-serif; background:#0a0c10; color:#e2e8f0; }
.main,.stApp { background:#0a0c10; }
h1,h2,h3 { font-family:'Syne',sans-serif; font-weight:800; }

.anomaly-normal  { background:#0f1420; border-left:3px solid #1e2535; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; }
.anomaly-notable { background:#1a1a00; border-left:3px solid #d4a017; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; }
.anomaly-strong  { background:#1a0d00; border-left:3px solid #e05c00; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; }
.anomaly-extreme { background:#1a0000; border-left:3px solid #e02020; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; animation:pulse-red 2s infinite; }
@keyframes pulse-red { 0%{border-left-color:#e02020} 50%{border-left-color:#ff6060} 100%{border-left-color:#e02020} }

.whale-card     { background:#060d1a; border-left:3px solid #3b82f6; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; }
.whale-card-big { background:#0d0820; border-left:3px solid #a855f7; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; animation:pulse-purple 2s infinite; }
@keyframes pulse-purple { 0%{border-left-color:#a855f7} 50%{border-left-color:#d8b4fe} 100%{border-left-color:#a855f7} }

.lb-row    { background:#0d1117; border:1px solid #1e2535; border-radius:6px; padding:8px 12px; margin:3px 0; font-family:'Space Mono',monospace; font-size:0.78rem; }
.debug-box { background:#0a0a0a; border:1px solid #2a2a2a; border-radius:6px; padding:10px 12px; font-family:monospace; font-size:0.72rem; color:#6b7280; white-space:pre-wrap; word-break:break-all; margin:6px 0; }

.badge-kalshi     { background:#0d3b2e; color:#2dd4aa; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; }
.badge-polymarket { background:#0e1f3d; color:#60a5fa; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; }
.badge-buy  { background:#0d3b1a; color:#4ade80; border-radius:4px; padding:2px 7px; font-size:0.7rem; font-weight:700; }
.badge-sell { background:#3b0d0d; color:#f87171; border-radius:4px; padding:2px 7px; font-size:0.7rem; font-weight:700; }
.badge-extreme { color:#f87171; font-weight:700; }
.badge-strong  { color:#fb923c; font-weight:700; }
.badge-notable { color:#fbbf24; font-weight:700; }
.badge-normal  { color:#6b7280; }

.signal-bar { background:linear-gradient(90deg,#1e293b,#0f172a); border:1px solid #1e2535; border-radius:4px; padding:6px 12px; margin:2px 0; font-family:'Space Mono',monospace; font-size:0.78rem; color:#94a3b8; }
div[data-testid="stMetric"] { background:#111520; border:1px solid #1e2535; border-radius:8px; padding:12px; }
.stTabs [data-baseweb="tab-list"]  { background:#0a0c10; border-bottom:1px solid #1e2535; }
.stTabs [data-baseweb="tab"]       { color:#64748b; font-family:'Space Mono',monospace; font-size:0.82rem; }
.stTabs [aria-selected="true"]     { color:#e2e8f0 !important; border-bottom-color:#60a5fa !important; }
hr { border-color:#1e2535; }
.error-box { background:#1a0a00; border:1px solid #7c2d12; border-radius:6px; padding:10px 14px; color:#fca5a5; font-family:'Space Mono',monospace; font-size:0.8rem; margin:4px 0; }
.info-box  { background:#0a1420; border:1px solid #1e3a5f; border-radius:6px; padding:10px 14px; color:#93c5fd; font-family:'Space Mono',monospace; font-size:0.8rem; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE  (all keys declared upfront)
# ─────────────────────────────────────────────
_defaults = {
    "kalshi_history":  {},
    "poly_history":    {},
    "alerts":          [],
    "last_refresh":    None,
    "next_refresh":    None,     # epoch for auto-refresh countdown
    "kalshi_error":    None,
    "poly_error":      None,
    "kalshi_markets":  [],
    "poly_markets":    [],
    "kalshi_api_key":  "",
    "auto_refresh":    False,
    # whale
    "whale_wallets":   [],
    "whale_trades":    [],
    "seen_tx":         [],       # list not set — avoids serialization issues
    "whale_last_scan": None,
    "whale_error":     None,
    "whale_min_usd":   WHALE_MIN_TRADE_USD,
    "whale_top_n":     WHALE_TOP_N,
    # debug
    "debug_log":       [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def dlog(msg: str):
    """Append to debug log (last 40 entries)."""
    entry = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    st.session_state.debug_log = ([entry] + st.session_state.debug_log)[:40]

# ─────────────────────────────────────────────
# ANOMALY FETCHERS
# ─────────────────────────────────────────────
def fetch_kalshi_markets(api_key: str = "") -> list[dict]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.get(f"{KALSHI_API}/markets", headers=headers,
                     params={"status":"open","limit":100}, timeout=15)
    r.raise_for_status()
    markets = r.json().get("markets", [])
    dlog(f"Kalshi: got {len(markets)} markets")
    results = []
    for m in markets:
        yes_bid = m.get("yes_bid") or 0
        yes_ask = m.get("yes_ask") or 0
        mid = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else (yes_bid or yes_ask)
        results.append({
            "id":       m.get("ticker", ""),
            "title":    (m.get("title") or "")[:80],
            "volume":   float(m.get("volume") or 0),
            "open_int": float(m.get("open_interest") or 0),
            "price":    round(mid / 100, 4) if mid else 0,
            "platform": "Kalshi",
            "url":      f"https://kalshi.com/markets/{m.get('event_ticker','')}/{m.get('ticker','')}",
        })
    results.sort(key=lambda x: x["volume"], reverse=True)
    return results[:60]


def fetch_polymarket_markets() -> list[dict]:
    r = requests.get(f"{GAMMA_API}/markets",
                     params={"active":"true","closed":"false","limit":100,
                             "_order":"volume24hr","_sort":"desc"}, timeout=15)
    r.raise_for_status()
    raw = r.json()
    markets = raw if isinstance(raw, list) else raw.get("markets", [])
    dlog(f"Polymarket: got {len(markets)} markets")
    results = []
    for m in markets:
        prices_raw = m.get("outcomePrices", "[]")
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            price = float(prices[0]) if prices else 0
        except Exception:
            price = 0
        results.append({
            "id":       str(m.get("id", "")),
            "title":    (m.get("question") or m.get("title") or "")[:80],
            "volume":   float(m.get("volume24hr") or 0),
            "open_int": float(m.get("liquidity") or 0),
            "price":    round(price, 4),
            "platform": "Polymarket",
            "url":      f"https://polymarket.com/event/{m['slug']}" if m.get("slug") else "",
        })
    results.sort(key=lambda x: x["volume"], reverse=True)
    return results[:60]

# ─────────────────────────────────────────────
# ANOMALY ENGINE
# ─────────────────────────────────────────────
def update_history(history: dict, markets: list[dict]):
    for m in markets:
        mid = m["id"]
        if mid not in history:
            history[mid] = deque(maxlen=VOLUME_WINDOW)
        history[mid].append({"volume": m["volume"], "price": m["price"]})


def compute_anomalies(markets: list[dict], history: dict) -> list[dict]:
    enriched = []
    for m in markets:
        hist      = list(history.get(m["id"], []))
        vol_z     = 0.0
        price_chg = 0.0
        signal    = "normal"
        reasons   = []
        if len(hist) >= 3:
            vols  = [h["volume"] for h in hist[:-1]]
            mu    = np.mean(vols)
            sigma = np.std(vols)
            if sigma > 0:
                vol_z = (m["volume"] - mu) / sigma
            if len(hist) >= 2 and hist[-2]["price"] > 0:
                price_chg = abs(m["price"] - hist[-2]["price"]) / hist[-2]["price"]
            if   vol_z >= Z_EXTREME: signal = "extreme"; reasons.append(f"Vol z={vol_z:.1f} ⚡")
            elif vol_z >= Z_STRONG:  signal = "strong";  reasons.append(f"Vol z={vol_z:.1f}")
            elif vol_z >= Z_NOTABLE: signal = "notable"; reasons.append(f"Vol z={vol_z:.1f}")
            if price_chg >= PRICE_MOVE_THRESHOLD:
                reasons.append(f"Price Δ={price_chg*100:.1f}%")
                if signal == "normal": signal = "notable"
        enriched.append({
            **m,
            "vol_z":     round(vol_z, 2),
            "price_chg": round(price_chg * 100, 2),
            "signal":    signal,
            "reasons":   ", ".join(reasons) if reasons else "—",
            "snapshots": len(hist),
        })
    order = {"extreme":0,"strong":1,"notable":2,"normal":3}
    enriched.sort(key=lambda x: (order[x["signal"]], -abs(x["vol_z"])))
    return enriched


def push_alert(market: dict):
    entry = {
        "ts":       datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
        "platform": market["platform"],
        "title":    market["title"],
        "signal":   market["signal"],
        "reasons":  market["reasons"],
        "vol_z":    market["vol_z"],
        "price":    market["price"],
        "url":      market.get("url", ""),
    }
    st.session_state.alerts.insert(0, entry)
    if len(st.session_state.alerts) > 100:
        st.session_state.alerts = st.session_state.alerts[:100]

# ─────────────────────────────────────────────
# WHALE TRACKER
# ─────────────────────────────────────────────
def fetch_whale_leaderboard(top_n: int) -> list[dict]:
    """
    Polymarket Data API leaderboard.
    Correct param is `period` (not `window`), and `startDate` style periods.
    Falls back through multiple variants for resilience.
    """
    endpoints_to_try = [
        # Confirmed working variants (period-based, not window-based)
        (f"{DATA_API}/leaderboard",  {"limit": top_n, "period": "all",     "sortBy": "PROFIT"}),
        (f"{DATA_API}/leaderboard",  {"limit": top_n, "period": "allTime", "sortBy": "PROFIT"}),
        (f"{DATA_API}/leaderboard",  {"limit": top_n, "period": "month",   "sortBy": "PROFIT"}),
        (f"{DATA_API}/leaderboard",  {"limit": top_n}),
        # Gamma API also has a leaderboard endpoint
        (f"{GAMMA_API}/leaderboard", {"limit": top_n, "sortBy": "PROFIT"}),
        (f"{GAMMA_API}/leaderboard", {"limit": top_n}),
    ]

    data = None
    for url, params in endpoints_to_try:
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            dlog(f"Leaderboard hit: {url} params={params} → {type(data).__name__} len={len(data) if isinstance(data,list) else '?'}")
            break
        except Exception as e:
            dlog(f"Leaderboard attempt failed: {url} → {e}")
            continue

    if data is None:
        raise RuntimeError("All leaderboard endpoints failed — check debug log")

    # Normalise response shape: may be list or {"data":[...]} or {"leaderboard":[...]}
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = (data.get("data")
                or data.get("leaderboard")
                or data.get("results")
                or [])
    else:
        rows = []

    dlog(f"Leaderboard rows extracted: {len(rows)}")

    wallets = []
    for row in rows:
        # Polymarket uses proxyWallet for the address
        addr = (row.get("proxyWallet")
                or row.get("proxy_wallet")
                or row.get("address")
                or row.get("wallet")
                or "")
        if not addr:
            continue
        name = (row.get("name")
                or row.get("pseudonym")
                or row.get("username")
                or addr[:10] + "…")
        pnl    = float(row.get("pnl") or row.get("profit") or row.get("cashPnl") or 0)
        volume = float(row.get("volume") or row.get("tradingVolume") or 0)
        wallets.append({
            "address": addr,
            "name":    name,
            "pnl":     pnl,
            "volume":  volume,
            "rank":    len(wallets) + 1,
        })

    dlog(f"Wallets parsed: {len(wallets)}")
    return wallets[:top_n]


def fetch_whale_activity(wallet: dict, min_usd: float, seen: list) -> list[dict]:
    """
    Fetch recent trades for one wallet.
    Uses /trades endpoint (confirmed) with /activity as fallback.
    size = tokens, price = USDC/token, USD value = size * price.
    """
    trades = []
    # Try /trades first (confirmed endpoint per cheatsheet), then /activity as fallback
    for endpoint, params in [
        (f"{DATA_API}/trades",   {"user": wallet["address"], "limit": 25}),
        (f"{DATA_API}/activity", {"user": wallet["address"], "limit": 25, "type": "TRADE"}),
    ]:
        try:
            r = requests.get(endpoint, params=params, timeout=10)
            r.raise_for_status()
            raw = r.json()
            trades = raw if isinstance(raw, list) else raw.get("data", raw.get("activity", []))
            if trades:
                dlog(f"Trades from {endpoint.split('/')[-1]} for {wallet['address'][:8]}: {len(trades)}")
                break
        except Exception as e:
            dlog(f"Trade fetch failed ({endpoint.split('/')[-1]}) {wallet['address'][:8]}: {e}")
            continue

    if not trades:
        return []

    new_trades = []
    for t in trades:
        tx    = t.get("transactionHash") or t.get("id") or ""
        side  = (t.get("side") or "").upper()
        size  = float(t.get("size") or 0)       # tokens
        price = float(t.get("price") or 0)      # USDC per token
        usd   = size * price                     # approximate USDC value

        if tx in seen:
            continue
        if usd < min_usd:
            continue

        seen.append(tx)
        # Keep seen list from growing unbounded
        if len(seen) > 5000:
            del seen[:1000]

        ts_raw = t.get("timestamp")
        try:
            ts_str = datetime.fromtimestamp(int(ts_raw), tz=timezone.utc).strftime("%H:%M UTC") if ts_raw else "—"
        except Exception:
            ts_str = "—"

        new_trades.append({
            "ts":       ts_str,
            "platform": "Polymarket",
            "wallet":   wallet["address"],
            "name":     wallet["name"],
            "rank":     wallet["rank"],
            "side":     side,
            "size_usd": round(usd, 2),
            "size_tok": round(size, 2),
            "price":    price,
            "title":    (t.get("title") or "")[:72],
            "outcome":  t.get("outcome") or "",
            "url":      f"https://polymarket.com/event/{t['slug']}" if t.get("slug") else "",
            "tx":       tx,
        })

    return new_trades


def scan_whale_trades(wallets: list[dict], min_usd: float) -> list[dict]:
    seen  = st.session_state.seen_tx
    found = []
    for w in wallets:
        found.extend(fetch_whale_activity(w, min_usd, seen))
    found.sort(key=lambda x: x["size_usd"], reverse=True)
    dlog(f"Whale scan: {len(found)} new trades above ${min_usd:,.0f} across {len(wallets)} wallets")
    return found

# ─────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────
SIGNAL_EMOJI = {"extreme":"🔴","strong":"🟠","notable":"🟡","normal":"⚪"}

def badge(platform):
    cls = "badge-kalshi" if platform == "Kalshi" else "badge-polymarket"
    return f'<span class="{cls}">{platform}</span>'

def sig_badge(signal):
    labels = {"extreme":"EXTREME","strong":"STRONG","notable":"NOTABLE","normal":"normal"}
    return f'<span class="badge-{signal}">{SIGNAL_EMOJI[signal]} {labels[signal]}</span>'

def side_badge(side):
    if side == "BUY":  return '<span class="badge-buy">▲ BUY</span>'
    if side == "SELL": return '<span class="badge-sell">▼ SELL</span>'
    return f'<span style="color:#6b7280">{side or "?"}</span>'

def render_market_row(m: dict):
    chg   = f'+{m["price_chg"]:.1f}%' if m["price_chg"] else ""
    snap  = f"({m['snapshots']} snaps — need 3)" if m["snapshots"] < 3 else (f"({m['snapshots']} snaps)" if m["snapshots"] < 6 else "")
    link  = f'&nbsp;<a href="{m["url"]}" target="_blank" style="color:#60a5fa;font-size:0.73rem;">↗</a>' if m.get("url") else ""
    chg_s = f'<span style="color:#fb923c">{chg}</span>' if chg else ""
    st.markdown(f"""
    <div class="anomaly-{m['signal']}">
      {badge(m['platform'])} &nbsp;{sig_badge(m['signal'])} &nbsp;<strong>{m['title']}</strong>{link}<br>
      <span style="color:#64748b;font-size:0.75rem;">
        Vol:${m['volume']:,.0f} &nbsp;|&nbsp; {m['price']:.3f} {chg_s}
        &nbsp;|&nbsp; z={m['vol_z']:+.2f} &nbsp;|&nbsp; {m['reasons']} &nbsp;{snap}
      </span>
    </div>""", unsafe_allow_html=True)

def render_alert_row(a: dict):
    link = f'&nbsp;<a href="{a["url"]}" target="_blank" style="color:#60a5fa;font-size:0.73rem;">↗</a>' if a.get("url") else ""
    st.markdown(f"""
    <div class="signal-bar">
      <span style="color:#475569">{a['ts']}</span> &nbsp;
      {badge(a['platform'])} &nbsp;{sig_badge(a['signal'])} &nbsp;
      <strong>{a['title'][:60]}</strong>{link}<br>
      <span style="color:#64748b;font-size:0.73rem;">{a['reasons']} · price={a['price']:.3f} · z={a['vol_z']:+.2f}</span>
    </div>""", unsafe_allow_html=True)

def render_whale_trade(t: dict):
    is_big = t["size_usd"] >= MEGA_WHALE_USD
    css    = "whale-card-big" if is_big else "whale-card"
    mega   = "🐳 MEGA &nbsp;" if is_big else ""
    link   = f'&nbsp;<a href="{t["url"]}" target="_blank" style="color:#60a5fa;font-size:0.73rem;">↗ market</a>' if t.get("url") else ""
    prof   = f'<a href="https://polymarket.com/profile/{t["wallet"]}" target="_blank" style="color:#a78bfa;font-size:0.7rem;">↗ profile</a>'
    st.markdown(f"""
    <div class="{css}">
      {mega}<span style="color:#c4b5fd;font-weight:700;">#{t['rank']} {t['name']}</span>
      &nbsp;{side_badge(t['side'])}&nbsp;
      <strong style="color:#f0f6fc;">${t['size_usd']:,.0f}</strong>{link}<br>
      <span style="color:#475569;font-size:0.75rem;">{t['title'] or '(market unknown)'}</span>
      &nbsp;<span style="color:#6b7280;font-size:0.72rem;">{t['outcome']}</span><br>
      <span style="color:#334155;font-size:0.7rem;">price={t['price']:.3f} · {t['ts']} · {prof}</span>
    </div>""", unsafe_allow_html=True)

def render_leaderboard_row(w: dict, i: int):
    pnl_col = "#4ade80" if w["pnl"] >= 0 else "#f87171"
    vol_s   = f"${w['volume']:,.0f}" if w["volume"] else "—"
    st.markdown(f"""
    <div class="lb-row">
      <span style="color:#475569">#{i+1}</span> &nbsp;
      <strong style="color:#c4b5fd">{w['name']}</strong> &nbsp;
      <span style="color:{pnl_col};font-weight:700">${w['pnl']:,.0f} P&L</span> &nbsp;
      <span style="color:#4a5568;font-size:0.72rem">vol:{vol_s}</span> &nbsp;
      <a href="https://polymarket.com/profile/{w['address']}" target="_blank"
         style="color:#60a5fa;font-size:0.7rem">↗ profile</a>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ PM Tracker")
    st.markdown("---")

    st.markdown("### Kalshi API Key *(optional)*")
    st.caption("Public data works without a key. Adds higher rate limits.")
    kalshi_key = st.text_input("API Key", type="password",
                               value=st.session_state.kalshi_api_key, key="key_input")
    st.session_state.kalshi_api_key = kalshi_key

    st.markdown("---")
    st.markdown("### Anomaly Thresholds")
    z_notable = st.slider("Notable z-score", 1.0, 5.0, Z_NOTABLE, 0.5)
    z_strong  = st.slider("Strong z-score",  1.0, 6.0, Z_STRONG,  0.5)
    z_extreme = st.slider("Extreme z-score", 2.0, 8.0, Z_EXTREME, 0.5)
    price_thr = st.slider("Price move %",    1,   25,  int(PRICE_MOVE_THRESHOLD * 100), 1)

    st.markdown("---")
    st.markdown("### 🐳 Whale Settings")
    whale_min = st.number_input("Min trade size ($)", value=int(st.session_state.whale_min_usd), step=1000, min_value=100)
    whale_top = st.slider("# top traders to watch", 10, 100, int(st.session_state.whale_top_n), 10)
    st.session_state.whale_min_usd = whale_min
    st.session_state.whale_top_n   = whale_top
    st.caption("Tracks Polymarket top traders by all-time P&L.\nKalshi trader data is not publicly available.")

    st.markdown("---")
    st.markdown("### Filters")
    platform_filter = st.multiselect("Platforms", ["Kalshi","Polymarket"], default=["Kalshi","Polymarket"])
    signal_filter   = st.multiselect("Show signals", ["extreme","strong","notable","normal"],
                                     default=["extreme","strong","notable"])
    min_volume      = st.number_input("Min 24h volume ($)", value=0, step=1000)

    st.markdown("---")
    auto_refresh = st.toggle("Auto-refresh (60s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

    c1, c2 = st.columns(2)
    refresh_btn = c1.button("🔄 Refresh",       use_container_width=True)
    whale_btn   = c2.button("🐳 Scan Whales",   use_container_width=True)

    st.markdown("---")
    if st.button("🗑 Clear All Logs", use_container_width=True):
        st.session_state.alerts      = []
        st.session_state.whale_trades= []
        st.session_state.seen_tx     = []
        st.session_state.debug_log   = []
        st.rerun()

    st.caption(f"Anomaly: {st.session_state.last_refresh or '—'}")
    st.caption(f"Whales:  {st.session_state.whale_last_scan or '—'}")

# Apply sidebar threshold values
Z_NOTABLE            = z_notable
Z_STRONG             = z_strong
Z_EXTREME            = z_extreme
PRICE_MOVE_THRESHOLD = price_thr / 100

# ─────────────────────────────────────────────
# AUTO-REFRESH (timestamp-based, no sleep())
# ─────────────────────────────────────────────
now_epoch = time.time()
if auto_refresh and st.session_state.next_refresh and now_epoch >= st.session_state.next_refresh:
    refresh_btn = True   # trigger the block below

# ─────────────────────────────────────────────
# ANOMALY FETCH + SCORE
# ─────────────────────────────────────────────
do_first_load = st.session_state.last_refresh is None

if refresh_btn or do_first_load:
    with st.spinner("Fetching market data…"):
        try:
            raw = fetch_kalshi_markets(st.session_state.kalshi_api_key)
            update_history(st.session_state.kalshi_history, raw)
            st.session_state.kalshi_markets = compute_anomalies(raw, st.session_state.kalshi_history)
            st.session_state.kalshi_error   = None
        except Exception as e:
            st.session_state.kalshi_error = str(e)
            dlog(f"Kalshi fetch error: {e}")

        try:
            raw = fetch_polymarket_markets()
            update_history(st.session_state.poly_history, raw)
            st.session_state.poly_markets = compute_anomalies(raw, st.session_state.poly_history)
            st.session_state.poly_error   = None
        except Exception as e:
            st.session_state.poly_error = str(e)
            dlog(f"Polymarket fetch error: {e}")

    for m in [*st.session_state.kalshi_markets, *st.session_state.poly_markets]:
        if m["signal"] in ("extreme","strong","notable") and m["snapshots"] >= 3:
            push_alert(m)

    st.session_state.last_refresh  = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    st.session_state.next_refresh  = time.time() + REFRESH_SECONDS

# ─────────────────────────────────────────────
# WHALE SCAN
# ─────────────────────────────────────────────
if whale_btn:
    with st.spinner(f"Loading top {st.session_state.whale_top_n} traders from Polymarket leaderboard…"):
        try:
            wallets = fetch_whale_leaderboard(st.session_state.whale_top_n)
            st.session_state.whale_wallets = wallets
            st.session_state.whale_error   = None
        except Exception as e:
            st.session_state.whale_error = str(e)
            dlog(f"Leaderboard error: {e}")
            wallets = st.session_state.whale_wallets  # fall back to cached

    if st.session_state.whale_wallets:
        with st.spinner(f"Scanning recent trades for {len(st.session_state.whale_wallets)} wallets…"):
            try:
                new_trades = scan_whale_trades(
                    st.session_state.whale_wallets,
                    st.session_state.whale_min_usd,
                )
                if new_trades:
                    st.session_state.whale_trades = (new_trades + st.session_state.whale_trades)[:200]
                st.session_state.whale_last_scan = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            except Exception as e:
                st.session_state.whale_error = f"Trade scan error: {e}"
                dlog(f"Whale scan error: {e}")

# ─────────────────────────────────────────────
# SCHEDULE NEXT AUTO-REFRESH
# ─────────────────────────────────────────────
if auto_refresh:
    remaining = max(0, int((st.session_state.next_refresh or time.time()) - time.time()))
    st.sidebar.caption(f"Next refresh in {remaining}s")
    # Streamlit will re-run on user interaction; for true auto we use a short sleep + rerun
    if remaining <= 2:
        time.sleep(2)
        st.rerun()

# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;margin-bottom:0;letter-spacing:-0.02em;">
  ⚡ Prediction Market Tracker
</h1>
<p style="color:#475569;font-family:'Space Mono',monospace;font-size:0.8rem;margin-top:4px;">
  Anomaly detection &amp; whale tracking · Kalshi + Polymarket
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────
all_markets = [*st.session_state.kalshi_markets, *st.session_state.poly_markets]
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("🔴 Extreme",      sum(1 for m in all_markets if m["signal"]=="extreme"))
c2.metric("🟠 Strong",       sum(1 for m in all_markets if m["signal"]=="strong"))
c3.metric("🟡 Notable",      sum(1 for m in all_markets if m["signal"]=="notable"))
c4.metric("Markets tracked", len(all_markets))
c5.metric("🐳 Whale trades", len(st.session_state.whale_trades))
c6.metric("Whales tracked",  len(st.session_state.whale_wallets))
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_all, tab_kalshi, tab_poly, tab_whale, tab_lb, tab_alerts, tab_raw, tab_debug = st.tabs([
    "🔍 Anomalies", "🟢 Kalshi", "🔵 Polymarket",
    "🐳 Whale Feed", "🏆 Leaderboard", "🔔 Alerts", "📊 Raw Data", "🛠 Debug",
])

def filter_markets(markets):
    return [m for m in markets
            if m["platform"] in platform_filter
            and m["signal"]   in signal_filter
            and m["volume"]   >= min_volume]

# ── Anomalies ────────────────────────────────
with tab_all:
    if st.session_state.kalshi_error:
        st.markdown(f'<div class="error-box">⚠ Kalshi: {st.session_state.kalshi_error}</div>', unsafe_allow_html=True)
    if st.session_state.poly_error:
        st.markdown(f'<div class="error-box">⚠ Polymarket: {st.session_state.poly_error}</div>', unsafe_allow_html=True)

    anomalies = [m for m in filter_markets(all_markets) if m["signal"] != "normal"]
    if not all_markets:
        st.info("No markets loaded yet. Hit **🔄 Refresh** in the sidebar.")
    elif not anomalies:
        st.info("No anomalies detected above current thresholds. "
                "Need ≥3 refreshes per market to build a baseline.")
    else:
        st.markdown(f"**{len(anomalies)} anomaly signal(s)** — sorted by severity")
        for m in anomalies:
            render_market_row(m)

# ── Kalshi ───────────────────────────────────
with tab_kalshi:
    if st.session_state.kalshi_error:
        st.markdown(f'<div class="error-box">⚠ {st.session_state.kalshi_error}</div>', unsafe_allow_html=True)
    km = filter_markets(st.session_state.kalshi_markets)
    if km:
        st.markdown(f"**{len(km)} markets** (sorted by signal → z-score)")
        for m in km: render_market_row(m)
    else:
        st.info("No Kalshi data. Hit 🔄 Refresh.")

# ── Polymarket ───────────────────────────────
with tab_poly:
    if st.session_state.poly_error:
        st.markdown(f'<div class="error-box">⚠ {st.session_state.poly_error}</div>', unsafe_allow_html=True)
    pm = filter_markets(st.session_state.poly_markets)
    if pm:
        st.markdown(f"**{len(pm)} markets** (sorted by signal → z-score)")
        for m in pm: render_market_row(m)
    else:
        st.info("No Polymarket data. Hit 🔄 Refresh.")

# ── Whale Feed ───────────────────────────────
with tab_whale:
    st.markdown("**Whale Tracker** monitors top Polymarket traders (by all-time P&L) "
                "and surfaces trades above your size threshold.")
    st.caption(f"Min trade: ${st.session_state.whale_min_usd:,.0f} · "
               f"Tracking: {len(st.session_state.whale_wallets)} wallets · "
               f"Last scan: {st.session_state.whale_last_scan or '—'}")

    if st.session_state.whale_error:
        st.markdown(f'<div class="error-box">⚠ {st.session_state.whale_error}</div>', unsafe_allow_html=True)

    if not st.session_state.whale_wallets:
        st.info("Hit **🐳 Scan Whales** in the sidebar to load the leaderboard and scan for trades.")
    elif not st.session_state.whale_trades:
        st.info(f"Wallets loaded ({len(st.session_state.whale_wallets)} whales). "
                f"No trades above ${st.session_state.whale_min_usd:,.0f} found yet. "
                f"Try lowering the min trade size in the sidebar, or scan again.")
    else:
        size_opts = {"All": 0, "$1K+": 1000, "$5K+": 5000, "$10K+": 10000,
                     "$25K+": 25000, "$50K+": 50000, "$100K+": 100000}
        size_choice = st.selectbox("Show trades above:", list(size_opts.keys()), index=2)
        cutoff   = size_opts[size_choice]
        filtered = [t for t in st.session_state.whale_trades if t["size_usd"] >= cutoff]
        st.markdown(f"**{len(filtered)} trade(s)** above {size_choice}")
        for t in filtered:
            render_whale_trade(t)

    st.markdown("---")
    st.caption("ℹ Kalshi does not expose public trader data. "
               "Kalshi whale activity is best detected via anomaly z-scores.")

# ── Leaderboard ──────────────────────────────
with tab_lb:
    st.markdown("### 🏆 Polymarket Trader Leaderboard")
    st.caption("Source: Polymarket Data API — public, no auth required. "
               "Hit **🐳 Scan Whales** to load.")
    if not st.session_state.whale_wallets:
        st.info("Hit **🐳 Scan Whales** in the sidebar to populate the leaderboard.")
    else:
        search = st.text_input("Search name or address", "")
        rows   = st.session_state.whale_wallets
        if search:
            q    = search.lower()
            rows = [w for w in rows if q in w["name"].lower() or q in w["address"].lower()]
        st.markdown(f"**{len(rows)} traders**")
        for i, w in enumerate(rows):
            render_leaderboard_row(w, i)

# ── Alerts ───────────────────────────────────
with tab_alerts:
    if not st.session_state.alerts:
        st.info("No anomaly alerts yet. Alerts appear after a market has ≥3 refresh snapshots.")
    else:
        st.markdown(f"**{len(st.session_state.alerts)} alert(s)** — most recent first")
        for a in st.session_state.alerts:
            render_alert_row(a)

# ── Raw Data ─────────────────────────────────
with tab_raw:
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown("#### Kalshi Markets")
        if st.session_state.kalshi_markets:
            df = pd.DataFrame(st.session_state.kalshi_markets)[
                ["title","price","volume","vol_z","price_chg","signal","snapshots"]]
            st.dataframe(df, use_container_width=True, height=280)
        else:
            st.info("No data.")
        st.markdown("#### Whale Trades")
        if st.session_state.whale_trades:
            df = pd.DataFrame(st.session_state.whale_trades)[
                ["ts","name","rank","side","size_usd","price","title","outcome"]]
            st.dataframe(df, use_container_width=True, height=280)
        else:
            st.info("No whale data.")
    with c_b:
        st.markdown("#### Polymarket Markets")
        if st.session_state.poly_markets:
            df = pd.DataFrame(st.session_state.poly_markets)[
                ["title","price","volume","vol_z","price_chg","signal","snapshots"]]
            st.dataframe(df, use_container_width=True, height=280)
        else:
            st.info("No data.")
        st.markdown("#### Tracked Wallets")
        if st.session_state.whale_wallets:
            df = pd.DataFrame(st.session_state.whale_wallets)[
                ["rank","name","pnl","volume","address"]]
            st.dataframe(df, use_container_width=True, height=280)
        else:
            st.info("No whale data.")

# ── Debug ────────────────────────────────────
with tab_debug:
    st.markdown("### 🛠 Debug Log")
    st.caption("Internal log of API calls, response shapes, and errors. "
               "Share this tab's output if something isn't working.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**API Status**")
        st.write({
            "kalshi_markets":   len(st.session_state.kalshi_markets),
            "poly_markets":     len(st.session_state.poly_markets),
            "kalshi_error":     st.session_state.kalshi_error,
            "poly_error":       st.session_state.poly_error,
            "whale_wallets":    len(st.session_state.whale_wallets),
            "whale_trades":     len(st.session_state.whale_trades),
            "whale_error":      st.session_state.whale_error,
            "seen_tx_count":    len(st.session_state.seen_tx),
            "last_refresh":     st.session_state.last_refresh,
            "whale_last_scan":  st.session_state.whale_last_scan,
        })
    with col2:
        st.markdown("**Event Log**")
        if st.session_state.debug_log:
            st.code("\n".join(st.session_state.debug_log), language=None)
        else:
            st.info("No log entries yet. Hit Refresh or Scan Whales.")

    if st.button("Clear debug log"):
        st.session_state.debug_log = []
        st.rerun()
