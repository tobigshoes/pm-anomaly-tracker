"""
Prediction Market Anomaly + Whale Tracker
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
# THRESHOLDS
# ─────────────────────────────────────────────
Z_NOTABLE            = 2.0
Z_STRONG             = 3.0
Z_EXTREME            = 4.0
PRICE_MOVE_THRESHOLD = 0.05
VOLUME_WINDOW        = 20
REFRESH_SECONDS      = 60

WHALE_MIN_TRADE_USD  = 5000    # flag trades above this size
WHALE_TOP_N          = 50      # how many leaderboard wallets to track
DATA_API             = "https://data-api.polymarket.com"
GAMMA_API            = "https://gamma-api.polymarket.com"
KALSHI_BASE          = "https://api.elections.kalshi.com/trade-api/v2"

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: #0a0c10; color: #e2e8f0; }
.main, .stApp { background-color: #0a0c10; }
h1, h2, h3 { font-family: 'Syne', sans-serif; font-weight: 800; }

.anomaly-normal  { background:#0f1420; border-left:3px solid #1e2535; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; }
.anomaly-notable { background:#1a1a00; border-left:3px solid #d4a017; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; }
.anomaly-strong  { background:#1a0d00; border-left:3px solid #e05c00; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; }
.anomaly-extreme { background:#1a0000; border-left:3px solid #e02020; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; animation:pulse-red 2s infinite; }

@keyframes pulse-red { 0%{border-left-color:#e02020} 50%{border-left-color:#ff6060} 100%{border-left-color:#e02020} }

.whale-card { background:#060d1a; border-left:3px solid #3b82f6; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; }
.whale-card-big { background:#080f20; border-left:3px solid #a855f7; border-radius:6px; padding:10px 14px; margin:4px 0; font-family:'Space Mono',monospace; font-size:0.82rem; animation:pulse-purple 2s infinite; }
@keyframes pulse-purple { 0%{border-left-color:#a855f7} 50%{border-left-color:#d8b4fe} 100%{border-left-color:#a855f7} }

.leaderboard-row { background:#0d1117; border:1px solid #1e2535; border-radius:6px; padding:8px 12px; margin:3px 0; font-family:'Space Mono',monospace; font-size:0.78rem; }
.badge-kalshi    { background:#0d3b2e; color:#2dd4aa; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; letter-spacing:0.05em; }
.badge-polymarket{ background:#0e1f3d; color:#60a5fa; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:700; letter-spacing:0.05em; }
.badge-buy  { background:#0d3b1a; color:#4ade80; border-radius:4px; padding:2px 7px; font-size:0.7rem; font-weight:700; }
.badge-sell { background:#3b0d0d; color:#f87171; border-radius:4px; padding:2px 7px; font-size:0.7rem; font-weight:700; }
.badge-extreme { color:#f87171; font-weight:700; }
.badge-strong  { color:#fb923c; font-weight:700; }
.badge-notable { color:#fbbf24; font-weight:700; }
.badge-normal  { color:#6b7280; }

.signal-bar { background:linear-gradient(90deg,#1e293b,#0f172a); border:1px solid #1e2535; border-radius:4px; padding:6px 12px; margin:2px 0; font-family:'Space Mono',monospace; font-size:0.78rem; color:#94a3b8; }
div[data-testid="stMetric"] { background:#111520; border:1px solid #1e2535; border-radius:8px; padding:12px; }
.stTabs [data-baseweb="tab-list"] { background-color:#0a0c10; border-bottom:1px solid #1e2535; }
.stTabs [data-baseweb="tab"] { color:#64748b; font-family:'Space Mono',monospace; font-size:0.82rem; }
.stTabs [aria-selected="true"] { color:#e2e8f0 !important; border-bottom-color:#60a5fa !important; }
hr { border-color:#1e2535; }
.error-box { background:#1a0a00; border:1px solid #7c2d12; border-radius:6px; padding:10px 14px; color:#fca5a5; font-family:'Space Mono',monospace; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("kalshi_history", {}), ("poly_history", {}), ("alerts", []),
    ("last_refresh", None), ("kalshi_error", None), ("poly_error", None),
    ("kalshi_markets", []), ("poly_markets", []),
    ("kalshi_api_key", ""), ("auto_refresh", False),
    # whale tracker state
    ("whale_wallets", []),          # list of {address, name, pnl, volume}
    ("whale_trades", []),           # recent big trades log
    ("seen_tx_hashes", set()),      # dedup set
    ("whale_last_scan", None),
    ("whale_error", None),
    ("whale_min_usd", WHALE_MIN_TRADE_USD),
    ("whale_top_n", WHALE_TOP_N),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# ANOMALY FETCHERS
# ─────────────────────────────────────────────
def fetch_kalshi_markets(api_key: str = "") -> list[dict]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.get(f"{KALSHI_BASE}/markets", headers=headers,
                     params={"status": "open", "limit": 100}, timeout=12)
    r.raise_for_status()
    markets = r.json().get("markets", [])
    results = []
    for m in markets:
        yes_bid = m.get("yes_bid", 0) or 0
        yes_ask = m.get("yes_ask", 0) or 0
        mid     = (yes_bid + yes_ask) / 2 if (yes_bid and yes_ask) else (yes_bid or yes_ask)
        results.append({
            "id":       m.get("ticker", ""),
            "title":    m.get("title", "")[:80],
            "volume":   m.get("volume", 0) or 0,
            "open_int": m.get("open_interest", 0) or 0,
            "price":    round(mid / 100, 4) if mid else 0,
            "platform": "Kalshi",
            "url":      f"https://kalshi.com/markets/{m.get('event_ticker','')}/{m.get('ticker','')}",
        })
    results.sort(key=lambda x: x["volume"], reverse=True)
    return results[:60]


def fetch_polymarket_markets() -> list[dict]:
    r = requests.get(f"{GAMMA_API}/markets",
                     params={"active":"true","closed":"false","limit":100,
                             "_order":"volume24hr","_sort":"desc"}, timeout=12)
    r.raise_for_status()
    markets = r.json()
    results = []
    for m in markets:
        prices_raw = m.get("outcomePrices", "[]")
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            price  = float(prices[0]) if prices else 0
        except Exception:
            price = 0
        results.append({
            "id":       str(m.get("id", "")),
            "title":    (m.get("question") or m.get("title",""))[:80],
            "volume":   float(m.get("volume24hr", 0) or 0),
            "open_int": float(m.get("liquidity", 0) or 0),
            "price":    round(price, 4),
            "platform": "Polymarket",
            "url":      f"https://polymarket.com/event/{m.get('slug','')}" if m.get("slug") else "",
        })
    results.sort(key=lambda x: x["volume"], reverse=True)
    return results[:60]

# ─────────────────────────────────────────────
# ANOMALY ENGINE
# ─────────────────────────────────────────────
def update_history(history: dict, markets: list[dict], window: int = VOLUME_WINDOW):
    for m in markets:
        mid = m["id"]
        if mid not in history:
            history[mid] = deque(maxlen=window)
        history[mid].append({"volume": m["volume"], "price": m["price"],
                              "ts": datetime.now(timezone.utc).isoformat()})

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
            if   vol_z >= Z_EXTREME: signal = "extreme"; reasons.append(f"Volume z={vol_z:.1f} (EXTREME)")
            elif vol_z >= Z_STRONG:  signal = "strong";  reasons.append(f"Volume z={vol_z:.1f} (strong)")
            elif vol_z >= Z_NOTABLE: signal = "notable"; reasons.append(f"Volume z={vol_z:.1f}")
            if price_chg >= PRICE_MOVE_THRESHOLD:
                reasons.append(f"Price Δ={price_chg*100:.1f}%")
                if signal == "normal": signal = "notable"
        enriched.append({**m, "vol_z": round(vol_z,2), "price_chg": round(price_chg*100,2),
                         "signal": signal, "reasons": ", ".join(reasons) if reasons else "—",
                         "snapshots": len(hist)})
    order = {"extreme":0,"strong":1,"notable":2,"normal":3}
    enriched.sort(key=lambda x: (order[x["signal"]], -abs(x["vol_z"])))
    return enriched

def push_alert(market: dict):
    entry = {"ts": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
             "platform": market["platform"], "title": market["title"],
             "signal": market["signal"], "reasons": market["reasons"],
             "vol_z": market["vol_z"], "price": market["price"],
             "url": market.get("url","")}
    st.session_state.alerts.insert(0, entry)
    if len(st.session_state.alerts) > 100:
        st.session_state.alerts = st.session_state.alerts[:100]

# ─────────────────────────────────────────────
# WHALE TRACKER — POLYMARKET
# ─────────────────────────────────────────────

def fetch_whale_leaderboard(top_n: int = 50) -> list[dict]:
    """
    Pull top traders from Polymarket's public leaderboard (Data API).
    Returns list of {address, name, pnl, volume}.
    """
    # Try profit leaderboard (all-time) — best proxy for "top 1%"
    r = requests.get(
        f"{DATA_API}/leaderboard",
        params={"limit": top_n, "window": "all", "sortBy": "PROFIT"},
        timeout=12,
    )
    r.raise_for_status()
    data = r.json()
    wallets = []
    rows = data if isinstance(data, list) else data.get("data", data.get("leaderboard", []))
    for row in rows:
        addr = row.get("proxyWallet") or row.get("address") or row.get("wallet","")
        if not addr:
            continue
        wallets.append({
            "address": addr,
            "name":    row.get("name") or row.get("pseudonym") or addr[:10]+"…",
            "pnl":     float(row.get("pnl") or row.get("profit") or 0),
            "volume":  float(row.get("volume") or 0),
            "rank":    len(wallets) + 1,
        })
    return wallets


def fetch_whale_activity(wallet: dict, min_usd: float, seen: set) -> list[dict]:
    """
    Fetch recent trades for a single wallet, return new ones above min_usd.
    """
    try:
        r = requests.get(
            f"{DATA_API}/activity",
            params={"user": wallet["address"], "limit": 20, "type": "TRADE"},
            timeout=8,
        )
        r.raise_for_status()
        trades = r.json()
        if isinstance(trades, dict):
            trades = trades.get("data", trades.get("activity", []))
    except Exception:
        return []

    new_trades = []
    for t in trades:
        tx   = t.get("transactionHash","") or t.get("id","")
        side = (t.get("side") or "").upper()
        size = float(t.get("size") or 0)
        price= float(t.get("price") or 0)
        usd  = size * price  # approximate USD value

        if tx in seen:
            continue
        if usd < min_usd:
            continue

        seen.add(tx)
        new_trades.append({
            "ts":       datetime.fromtimestamp(
                            int(t.get("timestamp",0)), tz=timezone.utc
                        ).strftime("%H:%M:%S UTC") if t.get("timestamp") else
                        datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            "platform": "Polymarket",
            "wallet":   wallet["address"],
            "name":     wallet["name"],
            "rank":     wallet["rank"],
            "side":     side,
            "size_usd": usd,
            "size_tok": size,
            "price":    price,
            "title":    (t.get("title") or "")[:72],
            "outcome":  t.get("outcome",""),
            "url":      f"https://polymarket.com/event/{t.get('slug','')}" if t.get("slug") else "",
            "tx":       tx,
        })
    return new_trades


def scan_whale_trades(wallets: list[dict], min_usd: float) -> list[dict]:
    """Poll recent activity for all tracked wallets."""
    seen   = st.session_state.seen_tx_hashes
    found  = []
    for w in wallets[:st.session_state.whale_top_n]:
        found.extend(fetch_whale_activity(w, min_usd, seen))
    found.sort(key=lambda x: x["size_usd"], reverse=True)
    return found


# ─────────────────────────────────────────────
# RENDER HELPERS — ANOMALY
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
    return f'<span style="color:#6b7280">{side}</span>'

def render_market_row(m: dict):
    chg_fmt  = f'+{m["price_chg"]:.1f}%' if m["price_chg"] else ""
    snap_note = f"({m['snapshots']} snapshots)" if m["snapshots"] < 5 else ""
    url_part  = f'&nbsp;<a href="{m["url"]}" target="_blank" style="color:#60a5fa;font-size:0.75rem;">↗ open</a>' if m.get("url") else ""
    st.markdown(f"""
    <div class="anomaly-{m['signal']}">
      {badge(m['platform'])} &nbsp; {sig_badge(m['signal'])} &nbsp;
      <strong>{m['title']}</strong>{url_part}<br>
      <span style="color:#64748b;font-size:0.75rem;">
        Vol: ${m['volume']:,.0f} &nbsp;|&nbsp; Price: {m['price']:.3f} &nbsp;
        {'<span style="color:#fb923c">'+chg_fmt+'</span>' if chg_fmt else ''}
        &nbsp;|&nbsp; z={m['vol_z']:+.2f} &nbsp;|&nbsp; {m['reasons']} &nbsp;{snap_note}
      </span>
    </div>""", unsafe_allow_html=True)

def render_alert_row(a: dict):
    url_part = f'&nbsp;<a href="{a["url"]}" target="_blank" style="color:#60a5fa;font-size:0.75rem;">↗</a>' if a.get("url") else ""
    st.markdown(f"""
    <div class="signal-bar">
      <span style="color:#475569">{a['ts']}</span> &nbsp;
      {badge(a['platform'])} &nbsp; {sig_badge(a['signal'])} &nbsp;
      <strong>{a['title'][:60]}</strong>{url_part}<br>
      <span style="color:#64748b;font-size:0.75rem;">{a['reasons']} &nbsp;|&nbsp; price={a['price']:.3f} &nbsp;|&nbsp; z={a['vol_z']:+.2f}</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RENDER HELPERS — WHALE
# ─────────────────────────────────────────────
def render_whale_trade(t: dict):
    usd_fmt  = f"${t['size_usd']:,.0f}"
    is_big   = t["size_usd"] >= 25000
    css      = "whale-card-big" if is_big else "whale-card"
    big_tag  = "🐳 MEGA WHALE &nbsp;" if is_big else ""
    url_part = f'&nbsp;<a href="{t["url"]}" target="_blank" style="color:#60a5fa;font-size:0.75rem;">↗ open</a>' if t.get("url") else ""
    poly_url = f'<a href="https://polymarket.com/profile/{t["wallet"]}" target="_blank" style="color:#60a5fa;font-size:0.7rem;">↗ profile</a>'
    st.markdown(f"""
    <div class="{css}">
      {big_tag}<span style="color:#a78bfa;font-weight:700;">#{t['rank']} {t['name']}</span>
      &nbsp; {side_badge(t['side'])} &nbsp;
      <strong style="color:#e2e8f0;">{usd_fmt}</strong>{url_part}<br>
      <span style="font-size:0.75rem;color:#475569;">{t['title'] or '(unknown market)'}</span>
      &nbsp;<span style="color:#6b7280;font-size:0.72rem;">{t['outcome']}</span><br>
      <span style="color:#334155;font-size:0.7rem;">
        price={t['price']:.3f} &nbsp;|&nbsp; {t['ts']} &nbsp;|&nbsp; {poly_url}
      </span>
    </div>""", unsafe_allow_html=True)

def render_leaderboard_row(w: dict, i: int):
    pnl_color = "#4ade80" if w["pnl"] >= 0 else "#f87171"
    pnl_fmt   = f"${w['pnl']:,.0f}"
    vol_fmt   = f"${w['volume']:,.0f}" if w["volume"] else "—"
    poly_url  = f'https://polymarket.com/profile/{w["address"]}'
    st.markdown(f"""
    <div class="leaderboard-row">
      <span style="color:#475569;font-size:0.72rem;">#{i+1}</span> &nbsp;
      <strong style="color:#c4b5fd;">{w['name']}</strong> &nbsp;
      <span style="color:{pnl_color};font-weight:700;">{pnl_fmt} P&L</span> &nbsp;
      <span style="color:#4a5568;font-size:0.72rem;">vol: {vol_fmt}</span> &nbsp;
      <a href="{poly_url}" target="_blank" style="color:#60a5fa;font-size:0.7rem;">↗ profile</a>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ PM Tracker")
    st.markdown("---")
    st.markdown("### Kalshi Auth *(optional)*")
    st.caption("Public data works without a key.")
    kalshi_key = st.text_input("API Key", type="password",
                               value=st.session_state.kalshi_api_key, key="key_input")
    st.session_state.kalshi_api_key = kalshi_key

    st.markdown("---")
    st.markdown("### Anomaly Thresholds")
    z_notable = st.slider("Notable z-score", 1.0, 5.0, Z_NOTABLE, 0.5)
    z_strong  = st.slider("Strong z-score",  1.0, 6.0, Z_STRONG,  0.5)
    z_extreme = st.slider("Extreme z-score", 2.0, 8.0, Z_EXTREME, 0.5)
    price_thr = st.slider("Price move %",    1,   25,  int(PRICE_MOVE_THRESHOLD*100), 1)

    st.markdown("---")
    st.markdown("### Whale Tracker")
    whale_min = st.number_input("Min trade size ($)", value=st.session_state.whale_min_usd, step=1000)
    whale_top = st.slider("# of top traders to watch", 10, 100, st.session_state.whale_top_n, 10)
    st.session_state.whale_min_usd = whale_min
    st.session_state.whale_top_n   = whale_top
    st.caption("Tracks Polymarket top traders by all-time P&L. Kalshi trader data is not public.")

    st.markdown("---")
    st.markdown("### Filters")
    platform_filter = st.multiselect("Platforms", ["Kalshi","Polymarket"], default=["Kalshi","Polymarket"])
    signal_filter   = st.multiselect("Show signals", ["extreme","strong","notable","normal"],
                                     default=["extreme","strong","notable"])
    min_volume = st.number_input("Min 24h volume ($)", value=0, step=1000)

    st.markdown("---")
    auto_refresh = st.toggle("Auto-refresh (60s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

    col1, col2 = st.columns(2)
    refresh_btn = col1.button("🔄 Refresh", use_container_width=True)
    whale_btn   = col2.button("🐳 Scan Whales", use_container_width=True)

    st.markdown("---")
    if st.button("🗑 Clear All Logs", use_container_width=True):
        st.session_state.alerts      = []
        st.session_state.whale_trades= []
        st.session_state.seen_tx_hashes = set()
        st.rerun()

    st.caption(f"Anomaly last refresh: {st.session_state.last_refresh or '—'}")
    st.caption(f"Whale last scan: {st.session_state.whale_last_scan or '—'}")

Z_NOTABLE            = z_notable
Z_STRONG             = z_strong
Z_EXTREME            = z_extreme
PRICE_MOVE_THRESHOLD = price_thr / 100

# ─────────────────────────────────────────────
# ANOMALY FETCH + SCORE
# ─────────────────────────────────────────────
if refresh_btn or st.session_state.last_refresh is None:
    with st.spinner("Fetching markets…"):
        try:
            raw = fetch_kalshi_markets(st.session_state.kalshi_api_key)
            update_history(st.session_state.kalshi_history, raw)
            st.session_state.kalshi_markets = compute_anomalies(raw, st.session_state.kalshi_history)
            st.session_state.kalshi_error   = None
        except Exception as e:
            st.session_state.kalshi_error = str(e)

        try:
            raw = fetch_polymarket_markets()
            update_history(st.session_state.poly_history, raw)
            st.session_state.poly_markets = compute_anomalies(raw, st.session_state.poly_history)
            st.session_state.poly_error   = None
        except Exception as e:
            st.session_state.poly_error = str(e)

    for m in [*st.session_state.kalshi_markets, *st.session_state.poly_markets]:
        if m["signal"] in ("extreme","strong","notable") and m["snapshots"] >= 3:
            push_alert(m)
    st.session_state.last_refresh = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

# ─────────────────────────────────────────────
# WHALE SCAN
# ─────────────────────────────────────────────
if whale_btn or (st.session_state.whale_btn_auto if hasattr(st.session_state,"whale_btn_auto") else False):
    with st.spinner(f"Loading top {st.session_state.whale_top_n} traders…"):
        try:
            wallets = fetch_whale_leaderboard(st.session_state.whale_top_n)
            st.session_state.whale_wallets = wallets
            st.session_state.whale_error   = None
        except Exception as e:
            st.session_state.whale_error = f"Leaderboard fetch failed: {e}"
            wallets = st.session_state.whale_wallets  # use cached

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
                st.session_state.whale_error = f"Trade scan failed: {e}"

if st.session_state.auto_refresh:
    time.sleep(REFRESH_SECONDS)
    st.rerun()

# ─────────────────────────────────────────────
# HEADER
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
tab_all, tab_kalshi, tab_poly, tab_whale, tab_leaderboard, tab_alerts, tab_raw = st.tabs([
    "🔍 Anomalies", "🟢 Kalshi", "🔵 Polymarket",
    "🐳 Whale Feed", "🏆 Leaderboard", "🔔 Alert Log", "📊 Raw Data"
])

def filter_markets(markets):
    return [m for m in markets
            if m["platform"] in platform_filter
            and m["signal"] in signal_filter
            and m["volume"] >= min_volume]

# ── Anomalies ────────────────────────────────
with tab_all:
    anomalies = [m for m in filter_markets(all_markets) if m["signal"] != "normal"]
    if not all_markets:
        st.info("No markets loaded yet. Hit **🔄 Refresh** in the sidebar.")
    elif not anomalies:
        st.info("No anomalies detected. Markets look calm — or lower thresholds / wait for ≥3 snapshots.")
    else:
        st.markdown(f"**{len(anomalies)} anomaly signal(s)** detected")
        for m in anomalies: render_market_row(m)

# ── Kalshi ───────────────────────────────────
with tab_kalshi:
    if st.session_state.kalshi_error:
        st.markdown(f'<div class="error-box">⚠ {st.session_state.kalshi_error}</div>', unsafe_allow_html=True)
    km = filter_markets(st.session_state.kalshi_markets)
    if km:
        st.markdown(f"**{len(km)} markets**")
        for m in km: render_market_row(m)
    else:
        st.info("No Kalshi markets. Hit Refresh.")

# ── Polymarket ───────────────────────────────
with tab_poly:
    if st.session_state.poly_error:
        st.markdown(f'<div class="error-box">⚠ {st.session_state.poly_error}</div>', unsafe_allow_html=True)
    pm = filter_markets(st.session_state.poly_markets)
    if pm:
        st.markdown(f"**{len(pm)} markets**")
        for m in pm: render_market_row(m)
    else:
        st.info("No Polymarket markets. Hit Refresh.")

# ── Whale Feed ───────────────────────────────
with tab_whale:
    st.markdown("""
    **Whale Tracker** monitors the top Polymarket traders by all-time P&L and alerts you
    when they make large trades. Hit **🐳 Scan Whales** in the sidebar to load wallets and
    pull their latest activity.
    """)

    if st.session_state.whale_error:
        st.markdown(f'<div class="error-box">⚠ {st.session_state.whale_error}</div>', unsafe_allow_html=True)

    if not st.session_state.whale_wallets:
        st.info("No whale data yet. Hit **🐳 Scan Whales** in the sidebar to start.")
    elif not st.session_state.whale_trades:
        st.info(f"Wallets loaded ({len(st.session_state.whale_wallets)} whales). "
                f"No trades found above ${st.session_state.whale_min_usd:,.0f} yet. "
                f"Try lowering the min trade size or scan again.")
    else:
        st.markdown(f"**{len(st.session_state.whale_trades)} whale trade(s)** logged · "
                    f"min size: ${st.session_state.whale_min_usd:,.0f} · "
                    f"tracking {len(st.session_state.whale_wallets)} wallets")

        # Size filter
        size_filter = st.selectbox("Show trades above:", ["$5K","$10K","$25K","$50K","$100K"], index=0)
        size_map = {"$5K":5000,"$10K":10000,"$25K":25000,"$50K":50000,"$100K":100000}
        cutoff   = size_map[size_filter]

        filtered = [t for t in st.session_state.whale_trades if t["size_usd"] >= cutoff]
        st.markdown(f"*{len(filtered)} trade(s) above {size_filter}*")
        for t in filtered:
            render_whale_trade(t)

    st.markdown("---")
    st.caption("ℹ Kalshi does not expose public trader data. Whale tracking is Polymarket-only.")

# ── Leaderboard ──────────────────────────────
with tab_leaderboard:
    st.markdown("### 🏆 Top Polymarket Traders (All-Time P&L)")
    st.caption("Source: Polymarket Data API — fully public, updated every few minutes.")

    if not st.session_state.whale_wallets:
        st.info("Hit **🐳 Scan Whales** in the sidebar to load the leaderboard.")
    else:
        search = st.text_input("Search by name or address", "")
        wallets_to_show = st.session_state.whale_wallets
        if search:
            q = search.lower()
            wallets_to_show = [w for w in wallets_to_show
                               if q in w["name"].lower() or q in w["address"].lower()]
        st.markdown(f"**{len(wallets_to_show)} traders**")
        for i, w in enumerate(wallets_to_show):
            render_leaderboard_row(w, i)

# ── Alert Log ────────────────────────────────
with tab_alerts:
    if not st.session_state.alerts:
        st.info("No anomaly alerts yet. Alerts are recorded after ≥3 snapshots per market.")
    else:
        st.markdown(f"**{len(st.session_state.alerts)} alert(s)** — most recent first")
        for a in st.session_state.alerts: render_alert_row(a)

# ── Raw Data ─────────────────────────────────
with tab_raw:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Kalshi Markets")
        if st.session_state.kalshi_markets:
            st.dataframe(pd.DataFrame(st.session_state.kalshi_markets)
                         [["title","price","volume","vol_z","price_chg","signal","snapshots"]],
                         use_container_width=True, height=300)
        else: st.info("No data.")

        st.markdown("#### Whale Trades")
        if st.session_state.whale_trades:
            df = pd.DataFrame(st.session_state.whale_trades)[
                ["ts","name","rank","side","size_usd","price","title","outcome"]]
            st.dataframe(df, use_container_width=True, height=300)
        else: st.info("No whale data.")

    with col_b:
        st.markdown("#### Polymarket Markets")
        if st.session_state.poly_markets:
            st.dataframe(pd.DataFrame(st.session_state.poly_markets)
                         [["title","price","volume","vol_z","price_chg","signal","snapshots"]],
                         use_container_width=True, height=300)
        else: st.info("No data.")

        st.markdown("#### Tracked Whales")
        if st.session_state.whale_wallets:
            st.dataframe(pd.DataFrame(st.session_state.whale_wallets)
                         [["rank","name","pnl","volume","address"]],
                         use_container_width=True, height=300)
        else: st.info("No whale data.")
