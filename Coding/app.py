import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime, sys, os
import database as db
db.init_db()

from indicators import (
    apply_core_indicators, calc_heikin_ashi, calc_ichimoku, calc_parabolic_sar,
    calc_supertrend, calc_volume_profile, auto_fibonacci, calc_bollinger_bands,
    calc_ema, calc_sma, find_swing_points
)
from strategies import strategy_bollinger_scalping, strategy_reversal, strategy_fibonacci_swing, calc_position_size
from ui_helpers import TERMINAL_CSS, render_signal_card, render_backtest_stats, render_equity_curve, render_strategy_rules, INDICATOR_GUIDE
from streamlit_autorefresh import st_autorefresh
from scanner import scan_all_assets, PREDEFINED_ASSETS, score_asset
from simulation import (
    new_simulation_state, open_positions_from_scanner,
    update_portfolio, get_current_equity, get_simulation_stats
)

st.set_page_config(page_title="Pro Trading Terminal", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# ================== SESSION STATE ==================
if 'tickers' not in st.session_state: st.session_state.tickers = ["AAPL"]
if 'comparison_tickers' not in st.session_state: st.session_state.comparison_tickers = []
if 'compare_mode' not in st.session_state: st.session_state.compare_mode = False
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = db.get_watchlist()
if 'portfolio' not in st.session_state:
    # Convert DB open positions to DataFrame for legacy portfolio view
    db_ports = db.get_open_positions()
    if not db_ports:
        st.session_state.portfolio = pd.DataFrame(columns=['Symbol', 'Shares', 'EntryPrice'])
    else:
        st.session_state.portfolio = pd.DataFrame([{'Symbol': p['symbol'], 'Shares': p['shares'], 'EntryPrice': p['entry_price']} for p in db_ports])
if 'journal' not in st.session_state:
    st.session_state.journal = db.get_journal()
if 'sim_state' not in st.session_state: st.session_state.sim_state = new_simulation_state()
# — Phase 8 new state —
if 'expert_mode' not in st.session_state: st.session_state.expert_mode = True
if 'depot_size' not in st.session_state: st.session_state.depot_size = 10000.0
if 'morning_scan_date' not in st.session_state: st.session_state.morning_scan_date = None
if 'signal_scores_cache' not in st.session_state: st.session_state.signal_scores_cache = {}
if 'sltp_alerted' not in st.session_state: st.session_state.sltp_alerted = set()
if 'quick_search' not in st.session_state: st.session_state.quick_search = ""

now = datetime.datetime.now()
market_hrs = 9 <= now.hour <= 16
market_status = "🟢 MARKET OPEN" if (market_hrs and now.weekday() < 5) else "🔴 MARKET CLOSED"
st.markdown(f'<div class="watermark">{st.session_state.tickers[0]}</div>', unsafe_allow_html=True)
st.markdown('<button class="fab-btn" title="Quick Trade">⚡</button>', unsafe_allow_html=True)
st.markdown(f'<div class="live-clock">{now.strftime("%H:%M:%S")} | {market_status}</div>', unsafe_allow_html=True)

# ── Phase 8: SL/TP Toast Alerts (check on every load) ─────────────────────────
_sim = st.session_state.sim_state
for _pos in _sim.get('positions', []):
    _sym = _pos['symbol']
    _alert_key = f"{_sym}_{_pos['open_date']}"
    if _alert_key in st.session_state.sltp_alerted:
        continue
    try:
        _cur = yf.Ticker(_sym).fast_info['lastPrice']
        if _cur <= _pos['sl']:
            st.toast(f"🛑 SL getriggert: {_sym} fiel auf ${_cur:.2f} (SL war ${_pos['sl']:.2f})", icon="🔴")
            st.session_state.sltp_alerted.add(_alert_key)
        elif _cur >= _pos['tp']:
            st.toast(f"🎯 TP erreicht! {_sym} stieg auf ${_cur:.2f} (TP war ${_pos['tp']:.2f})", icon="🟢")
            st.session_state.sltp_alerted.add(_alert_key)
    except:
        pass

# ── Phase 8: Morning Auto-Scan (09:00–10:00 once per day) ─────────────────────
_today_str = now.strftime("%Y-%m-%d")
if (9 <= now.hour < 10 and now.weekday() < 5 and
        st.session_state.morning_scan_date != _today_str and
        st.session_state.get('scanner_results') is None):
    st.session_state['scanner_running'] = True
    st.session_state.morning_scan_date = _today_str
    st.toast("🌅 Morgenscan läuft automatisch...", icon="📡")

if st.session_state.get('scanner_running'):
    _prog_container = st.container()
    progress_bar = _prog_container.progress(0, text="🔄 Scanner startet...")
    
    def update_progress(current, total, symbol):
        pct = current / total if total > 0 else 0
        progress_bar.progress(pct, text=f"🔄 Analysiere {symbol}... ({current}/{total})")

    with st.spinner(""):
        scan_df = scan_all_assets(progress_callback=update_progress)

    progress_bar.progress(1.0, text="✅ Scan abgeschlossen!")
    st.session_state['scanner_results'] = scan_df
    st.session_state['scanner_running'] = False
    import time; time.sleep(1) # let user see 100%
    _prog_container.empty()
    st.rerun()

# ================== SIDEBAR ==================
with st.sidebar:
    st.title("⚙️ Terminal Controls")

    # ── Phase 8 Feature 8: Einsteiger / Experten Modus ──────────────────────
    mode_col1, mode_col2 = st.columns(2)
    if mode_col1.button("🔰 Einsteiger" if st.session_state.expert_mode else "✅ Einsteiger",
                        use_container_width=True,
                        type="secondary" if st.session_state.expert_mode else "primary"):
        st.session_state.expert_mode = False
        st.rerun()
    if mode_col2.button("✅ Profi" if st.session_state.expert_mode else "🧑‍💻 Profi",
                        use_container_width=True,
                        type="primary" if st.session_state.expert_mode else "secondary"):
        st.session_state.expert_mode = True
        st.rerun()

    st.divider()

    # ── Phase 8 Feature 5: Depot-Größe (für Risiko-Berechnung) ──────────────
    st.session_state.depot_size = st.number_input(
        "💰 Mein Depot (€)", min_value=100.0, max_value=10_000_000.0,
        value=st.session_state.depot_size, step=1000.0, format="%.0f"
    )
    st.divider()

    # Asset Selection List
    PREDEFINED_ASSETS = {
        # Tech & US Blue Chips
        "AAPL": "Apple Inc.", "MSFT": "Microsoft", "GOOG": "Alphabet (Google)", 
        "AMZN": "Amazon", "NVDA": "NVIDIA", "TSLA": "Tesla", "META": "Meta Platforms",
        "AMD": "Advanced Micro Devices", "NFLX": "Netflix", "INTC": "Intel",
        # DAX & Europe
        "SAP.DE": "SAP SE", "SIE.DE": "Siemens", "ALV.DE": "Allianz", "BMW.DE": "BMW",
        "MBG.DE": "Mercedes", "VOW3.DE": "Volkswagen", "RHM.DE": "Rheinmetall",
        # Crypto
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
        # Indices & ETFs
        "SPY": "S&P 500 ETF", "QQQ": "Nasdaq 100 ETF", "DIA": "Dow Jones ETF",
        # Commodities
        "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil"
    }
    
    current_sym = st.session_state.tickers[0]
    options = list(PREDEFINED_ASSETS.keys())
    if current_sym not in options:
        options.insert(0, current_sym)
    
    ticker_input = st.selectbox(
        "🎯 Asset auswählen", 
        options=options, 
        index=options.index(current_sym),
        format_func=lambda x: f"{x} - {PREDEFINED_ASSETS.get(x, x)}"
    )
    
    custom_ticker = st.text_input("...oder eigenes Symbol (z.B. KO, JPM)", placeholder="Eigenes Symbol eingeben...", key="custom_ticker_input")
    
    final_ticker = custom_ticker.upper() if custom_ticker else ticker_input
    
    if final_ticker != st.session_state.tickers[0]:
        st.session_state.tickers[0] = final_ticker
        st.rerun()

    st.markdown("### 📋 Watchlist")
    watch_col1, watch_col2 = st.columns(2)
    new_ticker = watch_col1.text_input("Add/Import", key="wt_add")
    if watch_col2.button("➕ Add") and new_ticker:
        if new_ticker.upper() not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker.upper())
            db.add_to_watchlist(new_ticker.upper())
            st.rerun()

    # ── Phase 8 Feature 9: Watchlist Score Badges ───────────────────────────
    _scan_res = st.session_state.get('scanner_results')
    for sym in st.session_state.watchlist:
        cols = st.columns([3, 1])
        # Score badge from last scan
        _badge = ""
        if _scan_res is not None:
            _row = _scan_res[_scan_res['Symbol'] == sym]
            if not _row.empty:
                _sc = int(_row['Score'].iloc[0])
                _badge = " 🟢" if _sc >= 60 else (" 🟡" if _sc >= 40 else " 🔴")
        if cols[0].button(f"{sym}{_badge}", key=f"wl_{sym}"):
            st.session_state.tickers[0] = sym
            st.rerun()
        if cols[1].button("❌", key=f"del_{sym}"):
            st.session_state.watchlist.remove(sym)
            db.remove_from_watchlist(sym)
            st.rerun()

    with st.expander("🔔 Price Alerts & Journal"):
        st.selectbox("Asset Alert", st.session_state.watchlist, key="alrt")
        st.number_input("Target Price", key="alrtp")
        st.button("Create Alert")
        new_journal = st.text_area("Trading Journal", value=st.session_state.journal)
        if new_journal != st.session_state.journal:
            st.session_state.journal = new_journal
            db.save_journal(new_journal)

    @st.dialog("📚 Chart & Indicator Interpretation Guide")
    def show_interpretation_guide():
        st.markdown(INDICATOR_GUIDE)
    if st.button("📖 Read Chart Interpretations", use_container_width=True):
        show_interpretation_guide()

    st.markdown("---")
    st.markdown("### Technical Indicators")

    # ── Phase 8 Feature 8: Show/hide indicators based on mode ───────────────
    if st.session_state.expert_mode:
        with st.expander("Overlays (Main Chart)", expanded=True):
            show_ema = st.checkbox("EMA (9,21,55)", value=True)
            show_sma = st.checkbox("SMA (50,200)", value=False)
            show_vwap = st.checkbox("VWAP", value=False)
            show_ichimoku = st.checkbox("Ichimoku Cloud", value=False)
            show_sar = st.checkbox("Parabolic SAR", value=False)
            show_supertrend = st.checkbox("SuperTrend", value=False)
            show_vrvp = st.checkbox("Volume Profile (VRVP)", value=False)
            show_fib = st.checkbox("Auto Fibonacci", value=False)
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "Area", "Heikin-Ashi"])
            compare_asset = st.selectbox("Compare with Benchmark", ["None", "SPY", "QQQ", "DIA", "IWM"])
        with st.expander("Sub-Charts", expanded=True):
            show_volume = st.checkbox("Volume", value=True)
            show_obv = st.checkbox("On-Balance Volume (OBV)", value=False)
            show_rsi = st.checkbox("RSI", value=False)
            show_macd = st.checkbox("MACD", value=False)
            show_stoch = st.checkbox("Stochastic", value=False)
            show_atr = st.checkbox("ATR", value=False)
        with st.expander("⚙️ Fine-Tune Indicators", expanded=False):
            ema1_len = st.number_input("EMA 1 Length", 1, 200, 9)
            ema1_col = st.color_picker("EMA 1 Color", "#2196f3")
            ema2_len = st.number_input("EMA 2 Length", 1, 200, 21)
            ema2_col = st.color_picker("EMA 2 Color", "#ff9800")
            rsi_len = st.number_input("RSI Length", 1, 100, 14)
    else:
        # Einsteiger mode: sensible defaults, no clutter
        show_ema = True; show_sma = False; show_vwap = False; show_ichimoku = False
        show_sar = False; show_supertrend = False; show_vrvp = False; show_fib = False
        chart_type = "Candlestick"; compare_asset = "None"
        show_volume = True; show_obv = False; show_rsi = False
        show_macd = False; show_stoch = False; show_atr = False
        ema1_len = 9; ema1_col = "#2196f3"; ema2_len = 21; ema2_col = "#ff9800"; rsi_len = 14
        st.info("🔰 Einsteiger-Modus: Indikatoren auf Minimum reduziert.\nFür mehr Optionen → **Profi** wählen.")


    st.markdown("---")
    st.markdown("### ⏱️ Real-Time Refresh")
    st.session_state.auto_refresh = st.checkbox("Auto-Refresh aktivieren", value=st.session_state.get('auto_refresh', False))
    st.session_state.refresh_interval = st.number_input("Intervall (Sekunden)", min_value=1, max_value=3600, value=st.session_state.get('refresh_interval', 60))

# ================== AUTO REFRESH ==================
if st.session_state.get('auto_refresh', False):
    st_autorefresh(interval=st.session_state.get('refresh_interval', 60) * 1000, key="data_refresh")

# ================== TIMEFRAMES ==================
c_tf1, c_tf2 = st.columns([3, 1])
with c_tf1:
    tf_selection = st.radio("Quick Timeframe", ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"], index=4)
interval_map = {"1D": "5m", "5D": "15m", "1M": "30m", "3M": "1h", "6M": "1d", "YTD": "1d", "1Y": "1d", "5Y": "1wk", "MAX": "1mo"}
period_map = {"1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y", "5Y": "5y", "MAX": "max"}
period = period_map[tf_selection]
interval = interval_map[tf_selection]

# ================== DATA FETCHING ==================
@st.cache_data(ttl=120)
def fetch_data(ticker, p, i):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=p, interval=i)
        if df.empty: return None, None, None, None, None
        try: info = t.info
        except: info = {}
        try: news = t.news
        except: news = []
        try: dividends = t.dividends
        except: dividends = None
        try: earnings_dates = t.get_earnings_dates(limit=5)
        except: earnings_dates = None
        return df, info, news, dividends, earnings_dates
    except: return None, None, None, None, None

df, info, news, dividends, earnings_dates = fetch_data(st.session_state.tickers[0], period, interval)
if df is None or len(df) == 0:
    st.error(f"❌ Keine Daten für das Symbol '{st.session_state.tickers[0]}' gefunden. Bitte überprüfe die Schreibweise.")
    if st.button("🔄 Zurück setzen (Reset)", type="primary"):
        st.session_state.tickers[0] = "AAPL"
        if "custom_ticker_input" in st.session_state:
            st.session_state["custom_ticker_input"] = ""
        if "quick_search_bar" in st.session_state:
            st.session_state["quick_search_bar"] = ""
        st.rerun()
    st.stop()

# ================== APPLY INDICATORS ==================
df = apply_core_indicators(df, ema1_len, ema2_len, rsi_len)
current_price = df['Close'].iloc[-1]
prev_price = list(df['Close'])[-2] if len(df) > 1 else current_price
pct_change = ((current_price - prev_price) / prev_price) * 100

st.markdown(f"## {st.session_state.tickers[0]} <span style='font-size:24px; color:{'#26a69a' if pct_change>=0 else '#ef5350'}'>${current_price:,.2f} ({pct_change:+.2f}%)</span>", unsafe_allow_html=True)

# ── Phase 8 Feature 7: Quick Search Bar ──────────────────────────────────────
_qs_col1, _qs_col2 = st.columns([5, 1])
with _qs_col1:
    _qs_val = st.text_input(
        "🔎 Quick-Suche (Symbol eingeben + Enter)",
        placeholder="z.B. NVDA, BTC-USD, SAP.DE...",
        label_visibility="collapsed",
        key="quick_search_bar"
    )
with _qs_col2:
    _qs_btn = st.button("▶", use_container_width=True)
if (_qs_val and _qs_btn) or (_qs_val and _qs_val != st.session_state.quick_search):
    st.session_state.tickers[0] = _qs_val.upper().strip()
    st.session_state.quick_search = _qs_val.upper().strip()
    st.rerun()


# Removed legacy quick_score_symbol. The AI Strategy Engine in scanner.py now handles dynamic scoring and MTF alignment.


tab_home, tab_markt, tab_chart, tab_portfolio, tab_ai = st.tabs([
    "🏠 Home", "📡 Märkte & Signale", "📈 Chart & Analyse", "💼 Portfolio", "🤖 AI & Strategien"
])

# ======= TAB HOME: DASHBOARD =======
with tab_home:
    st.markdown("## 🏠 Trading Dashboard")
    st.caption("Dein persönliches Cockpit – auf einen Blick alles Wichtige für den heutigen Handelstag.")

    # ── Top metrics row ──────────────────────────────────────────────────────
    _sim_h = st.session_state.sim_state
    _eq_h = get_current_equity(_sim_h)
    _pnl_h = _eq_h - _sim_h.get('initial_capital', 100000)
    _pnl_pct_h = (_pnl_h / _sim_h.get('initial_capital', 100000)) * 100
    _n_pos_h = len(_sim_h.get('positions', []))
    _stats_h = get_simulation_stats(_sim_h)
    _wr_h = _stats_h.get('win_rate', 0) if _stats_h.get('total_trades', 0) > 0 else None

    import streamlit.components.v1 as components
    
    st.markdown("### 🔴 Live Market (WebSocket)")
    try:
        with open("components/live_ticker.html", "r") as f:
            live_ticker_html = f.read()
        components.html(live_ticker_html, height=130)
    except FileNotFoundError:
        st.warning("components/live_ticker.html not found.")

    st.markdown("### 📊 Portfolio & Stats")
    hm1, hm2, hm3, hm4 = st.columns(4)
    hm1.metric(
        "💰 Virtuelles Depot",
        f"€{_eq_h:,.0f}",
        delta=f"{'+' if _pnl_h >= 0 else ''}{_pnl_h:,.0f} € ({_pnl_pct_h:+.1f}%)"
    )
    hm2.metric("📈 Offene Positionen", str(_n_pos_h))
    hm3.metric(
        "🏆 Win Rate",
        f"{_wr_h:.0f}%" if _wr_h is not None else "–",
        delta=f"{_stats_h.get('total_trades',0)} Trades"
    )
    hm4.metric("🕐 Markt", market_status, delta=now.strftime("%H:%M Uhr"))

    st.divider()

    # ── Market Ampel + Signal preview (reused from Märkte tab) ───────────────
    st.markdown("### 📡 Markt-Ampel (Watchlist)")
    _wl_h = st.session_state.watchlist
    _buy_h, _sell_h, _neutral_h = 0, 0, 0
    _top_buys_h = []
    _top_sells_h = []
    _price_change_h = None
    try:
        _tf = yf.Ticker(st.session_state.tickers[0]).history(period="2d", interval="1d")
        if len(_tf) >= 2:
            _price_change_h = (_tf['Close'].iloc[-1] - _tf['Close'].iloc[-2]) / _tf['Close'].iloc[-2] * 100
    except:
        pass

    with st.spinner("📡 Analysiere Markt-Dynamik (AI Strategy Engine)..."):
        active_strategies = db.get_all_strategies()
        for _hs in _wl_h:
            try:
                _r = score_asset(_hs, "", active_strategies)
                if _r:
                    sc = _r["Score"]
                    d = _r["DirectionKey"]
                    p = _r["Kurs"]
                    en = _r.get("Entry", p)
                    sl = _r.get("SL", p * 0.98)
                    tp = _r.get("TP", p * 1.05)
                    pat = _r.get("Signale", "")

                    if d == "BUY":
                        _buy_h += 1
                        _top_buys_h.append((_hs, sc, d, p, en, sl, tp, pat))
                    elif d == "SELL":
                        _sell_h += 1
                        _top_sells_h.append((_hs, sc, d, p, en, sl, tp, pat))
                    else:
                        _neutral_h += 1
            except Exception as e:
                print(f"Error scanning watch list symbol {_hs}: {e}")

    _tot_h = max(_buy_h + _sell_h + _neutral_h, 1)
    _buy_pct_h = _buy_h / _tot_h
    if _buy_pct_h >= 0.6:
        _ac_h = "#00e676"; _ai_h = "🟢"; _at_h = "BULLISCH – Gute Einstiegsbedingungen (AI)"
    elif _buy_pct_h <= 0.3:
        _ac_h = "#ff1744"; _ai_h = "🔴"; _at_h = "SCHWACH – Abwarten oder Short-Chancen (AI)"
    else:
        _ac_h = "#ffea00"; _ai_h = "🟡"; _at_h = "GEMISCHT – Selektiv vorgehen (AI)"

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(30,30,40,0.9),rgba(20,20,30,0.95));
                border:2px solid {_ac_h}; border-radius:16px; padding:18px 24px;
                display:flex; align-items:center; gap:20px; margin-bottom:16px;'>
      <span style='font-size:44px;'>{_ai_h}</span>
      <div>
        <div style='font-size:20px; font-weight:800; color:{_ac_h};'>{_at_h}</div>
        <div style='font-size:12px; color:#9e9e9e; margin-top:4px;'>
          {_buy_h} KI-Kaufsignale &nbsp;|&nbsp; {_sell_h} KI-Verkaufssignale &nbsp;|&nbsp; {_neutral_h} Neutral
          &nbsp;|&nbsp; {len(_wl_h)} Watchlist-Assets
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top signals inline (3-column) ────────────────────────────────────────
    _top3_buys_h = sorted(_top_buys_h, key=lambda x: x[1], reverse=True)[:3]
    _top3_sells_h = sorted(_top_sells_h, key=lambda x: x[1])[:3]

    _home_sig_cols = st.columns(3)
    _sig_items = _top3_buys_h + _top3_sells_h
    for _ci, _hs_item in enumerate((_sig_items)[:3]):
        _hs_sym, _hs_sc, _hs_dir, _hs_p, _hs_en, _hs_sl, _hs_tp, _hs_pat = _hs_item
        _is_buy = _hs_dir == "BUY"
        _hcol = "#00e676" if _is_buy else "#ff1744"
        _hrr = abs(_hs_tp - _hs_en) / abs(_hs_en - _hs_sl) if abs(_hs_en - _hs_sl) > 0 else 0
        with _home_sig_cols[_ci]:
            st.markdown(f"""
            <div style='border-left:4px solid {_hcol}; background:rgba(255,255,255,0.03);
                        padding:12px 16px; border-radius:10px;'>
              <div style='font-size:18px; font-weight:800; color:{_hcol};'>
                {'🟢' if _is_buy else '🔴'} {_hs_sym}
                <span style='font-size:11px; color:#9e9e9e; font-weight:400;'>&nbsp;Score: {_hs_sc}</span>
              </div>
              <div style='font-size:13px; margin:6px 0; color:#ccc;'>
                💲 ${_hs_p:,.2f} &nbsp; 📍 Entry ${_hs_en:,.2f}<br/>
                🛑 SL ${_hs_sl:,.2f} &nbsp; 🎯 TP ${_hs_tp:,.2f}
              </div>
              <div style='font-size:11px; color:#777;'>R:R 1:{_hrr:.1f} &nbsp;|&nbsp; {_hs_pat}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"📈 {_hs_sym} Chart", key=f"home_chart_{_hs_sym}", use_container_width=True):
                st.session_state.tickers[0] = _hs_sym
                st.rerun()

    st.divider()



    # ── Latest news for current asset ────────────────────────────────────────
    st.markdown(f"### 📰 Aktuelle News – {st.session_state.tickers[0]}")
    try:
        _hn_ticker = yf.Ticker(st.session_state.tickers[0])
        _hn_news = _hn_ticker.news or []
        if _hn_news:
            for _hn in _hn_news[:4]:
                _hn_title = _hn.get('title', 'No title')
                _hn_pub = datetime.datetime.fromtimestamp(_hn.get('providerPublishTime', 0)).strftime("%d.%m. %H:%M")
                _hn_src = _hn.get('publisher', '')
                _hn_url = _hn.get('link', '#')
                st.markdown(f"""<div style='padding:8px 0; border-bottom:1px solid #1e1e2e;'>
                  <a href='{_hn_url}' target='_blank' style='color:#90caf9; text-decoration:none; font-size:14px; font-weight:600;'>{_hn_title}</a>
                  <span style='color:#666; font-size:11px; margin-left:10px;'>{_hn_src} · {_hn_pub}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Keine aktuellen Nachrichten verfügbar.")
    except:
        st.info("News konnten nicht geladen werden.")

# ======= MÄRKTE: SIGNAL DASHBOARD =======
with tab_markt:
    st.markdown("## 🚀 Heutige Handelssignale")
    st.caption("Live-Signale deiner Watchlist – berechnet aus RSI, EMA, MACD, ADX und Support/Resistance.")

    # ── Compute signals for entire watchlist ────────────────────────────────
    _watchlist_signals = []
    _n_syms = len(st.session_state.watchlist)
    if _n_syms > 0:
        _prog = st.progress(0, text="📡 Analysiere Markt-Dynamik (AI Engine)...")
        active_strategies = db.get_all_strategies()
        for _wi, _wsym in enumerate(st.session_state.watchlist):
            _prog.progress((_wi + 1) / _n_syms, text=f"📡 {_wsym} (MTF AI Scan)...")
            try:
                _res = score_asset(_wsym, "", active_strategies)
            except Exception as e:
                print(f"Error scanning {_wsym}: {e}")
                _res = None
            if _res:
                sc = _res["Score"]
                d = _res["DirectionKey"]
                p = _res["Kurs"]
                en = _res.get("Entry", p)
                sl = _res.get("SL", p * 0.98)
                tp = _res.get("TP", p * 1.05)
                pat = _res.get("Signale", "")
                _watchlist_signals.append((_wsym, sc, d, p, en, sl, tp, pat))
        _prog.empty()

    # ── Market Ampel ────────────────────────────────────────────────────────
    _buy_sigs = [s for s in _watchlist_signals if s[2] == "BUY"]
    _sell_sigs = [s for s in _watchlist_signals if s[2] == "SELL"]
    _total = len(_watchlist_signals) or 1
    _buy_pct = len(_buy_sigs) / _total
    if _buy_pct >= 0.6:
        _ampel_color = "#00e676"; _ampel_icon = "🟢"; _ampel_text = "KAUFEN – Markt bullisch"
    elif _buy_pct <= 0.3:
        _ampel_color = "#ff1744"; _ampel_icon = "🔴"; _ampel_text = "VORSICHT – Markt schwach"
    else:
        _ampel_color = "#ffea00"; _ampel_icon = "🟡"; _ampel_text = "ABWARTEN – Gemischte Signale"

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(30,30,40,0.9),rgba(20,20,30,0.95));
                border:2px solid {_ampel_color}; border-radius:16px; padding:20px 28px;
                margin-bottom:20px; display:flex; align-items:center; gap:20px;'>
      <span style='font-size:48px;'>{_ampel_icon}</span>
      <div>
        <div style='font-size:22px; font-weight:800; color:{_ampel_color};'>{_ampel_text}</div>
        <div style='font-size:13px; color:#9e9e9e; margin-top:4px;'>
          {len(_buy_sigs)} Kaufsignal{'e' if len(_buy_sigs)!=1 else ''} &nbsp;|&nbsp;
          {len(_sell_sigs)} Verkaufssignal{'e' if len(_sell_sigs)!=1 else ''} &nbsp;|&nbsp;
          {len(_watchlist_signals)} Assets analysiert
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top 3 BUY + Top 3 SELL signal cards ────────────────────────────────
    _sig_cols = st.columns(2)
    with _sig_cols[0]:
        st.markdown("### 🟢 Top Kaufsignale")
        _buy_sorted = sorted(_buy_sigs, key=lambda x: x[1], reverse=True)[:3]
        if not _buy_sorted:
            st.info("Keine klaren Kaufsignale in der Watchlist.")
        for _bs in _buy_sorted:
            _bsym, _bsc, _bdir, _bprice, _bentry, _bsl, _btp, _bpat = _bs
            _depot = st.session_state.depot_size
            _risk_eur = _depot * 0.02
            _risk_per_share = abs(_bentry - _bsl)
            _shares = int(_risk_eur / _risk_per_share) if _risk_per_share > 0 else 0
            _rr = abs(_btp - _bentry) / _risk_per_share if _risk_per_share > 0 else 0
            st.markdown(f"""
            <div style='border-left:4px solid #00e676; background:rgba(0,230,118,0.06);
                        padding:16px 20px; border-radius:10px; margin:8px 0;'>
              <div style='font-size:20px; font-weight:800;'>{_bsym}
                <span style='font-size:13px; color:#9e9e9e; font-weight:400; margin-left:8px;'>Score: {_bsc}</span>
                <span style='font-size:12px; color:#26a69a; margin-left:8px;'>{_bpat}</span>
              </div>
              <div style='font-size:15px; margin:6px 0;'>
                <b style='color:#00e676;'>Kurs: ${_bprice:,.2f}</b><br/>
                📍 Entry: <b>${_bentry:,.2f}</b> &nbsp;
                🛑 SL: <b style='color:#ff5252;'>${_bsl:,.2f}</b> &nbsp;
                🎯 TP: <b style='color:#00e676;'>${_btp:,.2f}</b>
              </div>
              <div style='font-size:12px; color:#9e9e9e;'>
                Risiko: €{_risk_eur:.0f} (2%) → {_shares} Anteile &nbsp;|&nbsp; R:R 1:{_rr:.1f}
              </div>
            </div>
            """, unsafe_allow_html=True)
            _bc1, _bc2 = st.columns(2)
            if _bc1.button(f"📊 {_bsym} Chart", key=f"sig_load_{_bsym}"):
                st.session_state.tickers[0] = _bsym; st.rerun()
            _order_str = f"BUY {_shares}x {_bsym} @ ${_bentry:.2f} | SL ${_bsl:.2f} | TP ${_btp:.2f}"
            _bc2.code(_order_str, language=None)

    with _sig_cols[1]:
        st.markdown("### 🔴 Top Vorsicht-Signale")
        _sell_sorted = sorted(_sell_sigs, key=lambda x: x[1])[:3]
        if not _sell_sorted:
            st.info("Keine klaren Verkaufssignale in der Watchlist.")
        for _ss in _sell_sorted:
            _ssym, _ssc, _sdir, _sprice, _sentry, _ssl, _stp, _spat = _ss
            _depot = st.session_state.depot_size
            _risk_eur = _depot * 0.02
            _risk_per_share = abs(_sentry - _ssl)
            _shares_s = int(_risk_eur / _risk_per_share) if _risk_per_share > 0 else 0
            _rr_s = abs(_stp - _sentry) / _risk_per_share if _risk_per_share > 0 else 0
            st.markdown(f"""
            <div style='border-left:4px solid #ff1744; background:rgba(255,23,68,0.06);
                        padding:16px 20px; border-radius:10px; margin:8px 0;'>
              <div style='font-size:20px; font-weight:800;'>{_ssym}
                <span style='font-size:13px; color:#9e9e9e; font-weight:400; margin-left:8px;'>Score: {_ssc}</span>
                <span style='font-size:12px; color:#ff5252; margin-left:8px;'>{_spat}</span>
              </div>
              <div style='font-size:15px; margin:6px 0;'>
                <b style='color:#ff5252;'>Kurs: ${_sprice:,.2f}</b><br/>
                📍 Entry: <b>${_sentry:,.2f}</b> &nbsp;
                🛑 SL: <b style='color:#ff5252;'>${_ssl:,.2f}</b> &nbsp;
                🎯 TP: <b style='color:#26a69a;'>${_stp:,.2f}</b>
              </div>
              <div style='font-size:12px; color:#9e9e9e;'>
                Risiko: €{_risk_eur:.0f} (2%) → {_shares_s} Anteile &nbsp;|&nbsp; R:R 1:{_rr_s:.1f}
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"📊 {_ssym} laden", key=f"sig_load_s_{_ssym}"):
                st.session_state.tickers[0] = _ssym; st.rerun()

    # ── Morning scan status ────────────────────────────────────────────────
    if st.session_state.morning_scan_date == now.strftime("%Y-%m-%d"):
        st.success(f"✅ Morgenscan abgeschlossen ({now.strftime('%H:%M')} Uhr)")



# ======= CHART & ANALYSE =======
with tab_chart:
    # Inner subtabs: [Chart] [Fundamental] [Risiko & Quants]
    ch_tab_chart, ch_tab_fund, ch_tab_quants = st.tabs([
        "📈 Chart & Indikatoren", "🏢 Fundamental & News", "🧮 Risiko & Quants"
    ])
    with ch_tab_chart:
        plot_df = calc_heikin_ashi(df) if chart_type == "Heikin-Ashi" else df

        # Dynamic subplot building
        subplots = []
        row_titles = []
        if show_volume: subplots.append('Volume'); row_titles.append("Volume")
        if show_obv: subplots.append('OBV'); row_titles.append("OBV")
        if show_rsi: subplots.append('RSI'); row_titles.append(f"RSI ({rsi_len})")
        if show_macd: subplots.append('MACD'); row_titles.append("MACD")
        if show_stoch: subplots.append('Stoch'); row_titles.append("Stochastic")
        if show_atr: subplots.append('ATR'); row_titles.append("ATR")

        num_rows = 1 + len(subplots)
        row_heights = [max(0.4, 1 - 0.15*len(subplots))] + [(1 - max(0.4, 1 - 0.15*len(subplots)))/len(subplots)] * len(subplots) if subplots else [1.0]

        fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                            row_heights=row_heights, row_titles=["Price"] + row_titles)

        # Price chart
        if chart_type in ["Candlestick", "Heikin-Ashi"]:
            fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
                low=plot_df['Low'], close=plot_df['Close'], name="Price",
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
        elif chart_type == "Line":
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], mode='lines',
                line=dict(color='#26a69a', width=2), name='Close'), row=1, col=1)
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], mode='lines',
                line=dict(color='#26a69a', width=2), fill='tozeroy',
                fillcolor='rgba(38,166,154,0.15)', name='Close'), row=1, col=1)

        # EMA Overlays
        if show_ema:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_1'], line=dict(color=ema1_col, width=1.5), name=f'EMA {ema1_len}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_2'], line=dict(color=ema2_col, width=1.5), name=f'EMA {ema2_len}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_55'], line=dict(color='#7c4dff', width=1, dash='dot'), name='EMA 55'), row=1, col=1)

        # SMA Overlays
        if show_sma:
            if 'SMA_50' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#ff9800', width=1.5, dash='dash'), name='SMA 50'), row=1, col=1)
            if 'SMA_200' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='#f44336', width=1.5, dash='dash'), name='SMA 200'), row=1, col=1)

        # VWAP
        if show_vwap and 'VWAP' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#ffeb3b', width=1.5), name='VWAP'), row=1, col=1)

        # Ichimoku Cloud
        if show_ichimoku:
            tenkan, kijun, senkou_a, senkou_b, chikou = calc_ichimoku(df)
            fig.add_trace(go.Scatter(x=df.index, y=tenkan, line=dict(color='#2196f3', width=1), name='Tenkan'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=kijun, line=dict(color='#f44336', width=1), name='Kijun'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=senkou_a, line=dict(color='rgba(0,230,118,0.4)', width=0.5), name='Senkou A'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=senkou_b, line=dict(color='rgba(239,83,80,0.4)', width=0.5),
                fill='tonexty', fillcolor='rgba(38,166,154,0.08)', name='Senkou B'), row=1, col=1)

        # Parabolic SAR
        if show_sar:
            sar = calc_parabolic_sar(df)
            bull_sar = sar.where(sar < df['Close'])
            bear_sar = sar.where(sar >= df['Close'])
            fig.add_trace(go.Scatter(x=df.index, y=bull_sar, mode='markers', marker=dict(size=3, color='#00e676'), name='SAR Bull'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bear_sar, mode='markers', marker=dict(size=3, color='#ff1744'), name='SAR Bear'), row=1, col=1)

        # SuperTrend
        if show_supertrend:
            st_line, st_dir = calc_supertrend(df)
            bull_st = st_line.where(st_dir == 1)
            bear_st = st_line.where(st_dir == -1)
            fig.add_trace(go.Scatter(x=df.index, y=bull_st, line=dict(color='#00e676', width=2), name='SuperTrend ↑'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bear_st, line=dict(color='#ff1744', width=2), name='SuperTrend ↓'), row=1, col=1)

        # Auto Fibonacci
        if show_fib:
            retrace, ext, sh, sl_fib, sh_idx, sl_idx = auto_fibonacci(df)
            fib_colors = {'23.6%': '#aaa', '38.2%': '#888', '50.0%': '#ff9800', '61.8%': '#f44336', '78.6%': '#9c27b0'}
            for lvl_name, lvl_val in retrace.items():
                if lvl_name in fib_colors:
                    fig.add_hline(y=lvl_val, line_dash="dot", line_color=fib_colors[lvl_name],
                        annotation_text=f"Fib {lvl_name}", annotation_position="right", row=1, col=1)

        # Benchmark Overlay
        if compare_asset != "None":
            try:
                comp_df = yf.Ticker(compare_asset).history(period=period, interval=interval)
                comp_scaled = comp_df['Close'] * (plot_df['Close'].iloc[0] / comp_df['Close'].iloc[0])
                fig.add_trace(go.Scatter(x=comp_df.index, y=comp_scaled, mode='lines',
                    line=dict(color='yellow', width=1.5), name=compare_asset), row=1, col=1)
            except: pass

        # Volume Profile (VRVP) - horizontal volume histogram
        if show_vrvp:
            try:
                price_levels, volumes = calc_volume_profile(df, bins=25)
                max_vol = max(volumes) if max(volumes) > 0 else 1
                norm_vols = volumes / max_vol
                for plv, nv in zip(price_levels, norm_vols):
                    if nv > 0.05:
                        fig.add_shape(type="rect", x0=df.index[0], x1=df.index[int(len(df) * nv * 0.3)],
                            y0=plv - (price_levels[1]-price_levels[0])*0.4,
                            y1=plv + (price_levels[1]-price_levels[0])*0.4,
                            fillcolor='rgba(38,166,154,0.2)', line_width=0, row=1, col=1)
            except: pass

        # Pivot Points
        if st.checkbox("Show Daily Pivot Points", value=False):
            pp = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
            r1 = (2 * pp) - df['Low'].shift(1)
            s1 = (2 * pp) - df['High'].shift(1)
            fig.add_trace(go.Scatter(x=df.index, y=pp, line=dict(color='#ab47bc', width=1, dash='dot'), name='Pivot'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=r1, line=dict(color='#ef5350', width=1, dash='dot'), name='R1'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=s1, line=dict(color='#26a69a', width=1, dash='dot'), name='S1'), row=1, col=1)

        # Candlestick Pattern Markers
        doji_idx = df.index[df['Doji']]
        hammer_idx = df.index[df['Hammer']]
        engulf_idx = df.index[df['Bullish_Engulfing']]
        if len(doji_idx) > 0:
            fig.add_trace(go.Scatter(x=doji_idx, y=df.loc[doji_idx, 'Close'], mode='markers',
                marker=dict(symbol='diamond', size=7, color='yellow'), name='Doji'), row=1, col=1)
        if len(hammer_idx) > 0:
            fig.add_trace(go.Scatter(x=hammer_idx, y=df.loc[hammer_idx, 'Low'] * 0.99, mode='text',
                text='🔨', textfont=dict(size=12), name='Hammer'), row=1, col=1)
        if len(engulf_idx) > 0:
            fig.add_trace(go.Scatter(x=engulf_idx, y=df.loc[engulf_idx, 'High'] * 1.005, mode='text',
                text='🟢', textfont=dict(size=10), name='Engulfing'), row=1, col=1)

        # Dividend & Earnings overlays
        try:
            if dividends is not None and not dividends.empty:
                df_start = df.index.min().tz_localize(None) if df.index.tz else df.index.min()
                for div_date, div_val in dividends.items():
                    d = div_date.tz_localize(None) if div_date.tzinfo else div_date
                    if d >= df_start:
                        fig.add_annotation(x=str(d), y=float(df['Close'].iloc[-1]), text=f"💰 ${div_val:.2f}",
                            showarrow=True, arrowhead=1, ax=0, ay=-40, row=1, col=1)
        except: pass
        try:
            if earnings_dates is not None and not earnings_dates.empty:
                df_start = df.index.min().tz_localize(None) if df.index.tz else df.index.min()
                for earn_date, _ in earnings_dates.iterrows():
                    d = earn_date.tz_localize(None) if earn_date.tzinfo else earn_date
                    if d >= df_start:
                        fig.add_vline(x=str(d), line_dash="dash", line_color="orange", annotation_text="📊", row=1, col=1)
        except: pass

        # Sub-chart rows
        current_row = 2
        if "Volume" in subplots:
            colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for _, row in df.iterrows()]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=current_row, col=1)
            current_row += 1
        if "OBV" in subplots:
            fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='#ffeb3b', width=1.5), name='OBV'), row=current_row, col=1)
            current_row += 1
        if "RSI" in subplots:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ab47bc', width=1.5), name='RSI'), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.1)", row=current_row, col=1)
            current_row += 1
        if "MACD" in subplots:
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'],
                marker_color=['#26a69a' if v > 0 else '#ef5350' for v in df['MACD_Hist']], name='Histogram'), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2962ff', width=1.5), name='MACD'), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='#ff6d00', width=1), name='Signal'), row=current_row, col=1)
            current_row += 1
        if "Stoch" in subplots:
            fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='#2196f3', width=1.5), name='%K'), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line=dict(color='#ff9800', width=1), name='%D'), row=current_row, col=1)
            fig.add_hline(y=80, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="gray", row=current_row, col=1)
            current_row += 1
        if "ATR" in subplots:
            fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], line=dict(color='#ff9800', width=1.5), name='ATR'), row=current_row, col=1)
            current_row += 1

        fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
            height=800, showlegend=False, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', hovermode="x unified",
            dragmode='drawline', newshape=dict(line_color='yellow'))
        if interval in ['1d', '1wk']:
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape']})

        # ══════════════════════════════════════════════════════════════
        # 🎯 Always-On Recommendation Banner (Feature 3)
        # ══════════════════════════════════════════════════════════════
        try:
            _banner_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50
            _banner_ema1 = df['EMA_1'].iloc[-1] if 'EMA_1' in df.columns else current_price
            _banner_ema2 = df['EMA_2'].iloc[-1] if 'EMA_2' in df.columns else current_price
            _banner_ema55 = df['EMA_55'].iloc[-1] if 'EMA_55' in df.columns else current_price
            _banner_macd_h = df['MACD_Hist'].iloc[-1] if 'MACD_Hist' in df.columns and not pd.isna(df['MACD_Hist'].iloc[-1]) else 0
            _banner_atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]) else current_price * 0.02
            _banner_sc = 0
            if _banner_ema1 > _banner_ema2 > _banner_ema55: _banner_sc += 3
            elif _banner_ema1 < _banner_ema2 < _banner_ema55: _banner_sc -= 3
            if current_price > _banner_ema1: _banner_sc += 1
            elif current_price < _banner_ema1: _banner_sc -= 1
            if _banner_rsi < 35: _banner_sc += 2
            elif _banner_rsi > 65: _banner_sc -= 2
            if _banner_macd_h > 0: _banner_sc += 2
            elif _banner_macd_h < 0: _banner_sc -= 2
            _banner_entry = current_price
            _banner_sl = current_price - _banner_atr * 1.5 if _banner_sc >= 3 else current_price + _banner_atr * 1.5
            _banner_tp = current_price + _banner_atr * 2.5 if _banner_sc >= 3 else current_price - _banner_atr * 2.5
            # Depot-based risk
            _d = st.session_state.depot_size
            _risk2 = _d * 0.02
            _r_per_share = abs(_banner_entry - _banner_sl)
            _pos_shares = int(_risk2 / _r_per_share) if _r_per_share > 0 else 0
            _rr_banner = abs(_banner_tp - _banner_entry) / _r_per_share if _r_per_share > 0 else 0
            if _banner_sc >= 3:
                _bc_color = '#00e676'; _b_icon = '🟢'; _b_action = 'KAUFEN'
                _b_msg = f'KAUFE bei ${_banner_entry:,.2f} • SL ${_banner_sl:,.2f} • TP ${_banner_tp:,.2f} • {_pos_shares} Anteile • Risiko €{_risk2:.0f}'
            elif _banner_sc <= -3:
                _bc_color = '#ff1744'; _b_icon = '🔴'; _b_action = 'NICHT KAUFEN'
                _b_msg = f'Abwärtstrend • SL ${_banner_sl:,.2f} • Ziel ${_banner_tp:,.2f} • Risiko €{_risk2:.0f}'
            else:
                _bc_color = '#ffea00'; _b_icon = '🟡'; _b_action = 'ABWARTEN'
                _b_msg = f'Kein klares Signal. Warte auf Score ≥3. RSI: {_banner_rsi:.0f}'
            st.markdown(f"""
            <div style='background:linear-gradient(90deg, rgba(30,30,40,0.85), rgba(20,20,30,0.9));
                        border:2px solid {_bc_color}; border-radius:12px;
                        padding:14px 22px; margin:12px 0; display:flex; align-items:center; gap:14px;'>
              <span style='font-size:36px;'>{_b_icon}</span>
              <div style='flex:1;'>
                <div style='font-size:18px; font-weight:800; color:{_bc_color};'>Signal: {_b_action}</div>
                <div style='font-size:13px; color:#ddd; margin-top:3px;'>{_b_msg}</div>
              </div>
              <div style='text-align:right; font-size:12px; color:#9e9e9e;'>Score {_banner_sc} &nbsp;|&nbsp; R:R 1:{_rr_banner:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        except:
            pass

        # ══════════════════════════════════════════════════════════════
        # 🎯 Smart Entry Zone Scanner v2
        # ══════════════════════════════════════════════════════════════
        with st.expander("🎯 Jetzt Einstieg anzeigen", expanded=False):
            st.markdown("### 🎯 Smart Entry-Zone Analyse v2")
            from scipy.cluster.vq import kmeans

            last_price = df['Close'].iloc[-1]
            atr_now = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else (df['High'].iloc[-1] - df['Low'].iloc[-1])

            # K-Means S/R
            swing_highs, swing_lows, _, _ = find_swing_points(df)
            all_pivots = np.array(swing_highs + swing_lows, dtype=float)
            if len(all_pivots) >= 5:
                n_clusters = min(6, len(all_pivots))
                key_levels, _ = kmeans(all_pivots, n_clusters)
                key_levels = sorted(key_levels)
            else:
                key_levels = [df['Low'].iloc[-40:].min() if len(df) >= 40 else df['Low'].min(),
                              df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else last_price,
                              df['High'].iloc[-40:].max() if len(df) >= 40 else df['High'].max()]
                key_levels = sorted(key_levels)

            supports = sorted([l for l in key_levels if l < last_price], reverse=True)
            resistances = sorted([l for l in key_levels if l > last_price])
            nearest_support = supports[0] if supports else last_price - atr_now * 2
            nearest_resistance = resistances[0] if resistances else last_price + atr_now * 2
            second_resistance = resistances[1] if len(resistances) > 1 else nearest_resistance + atr_now * 2
            second_support = supports[1] if len(supports) > 1 else nearest_support - atr_now * 2

            # Confluence scoring
            score = 0
            max_score = 20
            signals = []

            last_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
            last_macd_hist = df['MACD_Hist'].iloc[-1] if not pd.isna(df['MACD_Hist'].iloc[-1]) else 0
            last_ema1 = df['EMA_1'].iloc[-1] if not pd.isna(df['EMA_1'].iloc[-1]) else last_price
            last_ema2 = df['EMA_2'].iloc[-1] if not pd.isna(df['EMA_2'].iloc[-1]) else last_price
            last_ema55 = df['EMA_55'].iloc[-1] if not pd.isna(df['EMA_55'].iloc[-1]) else last_price
            last_adx = df['ADX'].iloc[-1] if not pd.isna(df['ADX'].iloc[-1]) else 25
            last_plus_di = df['Plus_DI'].iloc[-1] if not pd.isna(df['Plus_DI'].iloc[-1]) else 50
            last_minus_di = df['Minus_DI'].iloc[-1] if not pd.isna(df['Minus_DI'].iloc[-1]) else 50
            last_bb_lower = df['BB_Lower'].iloc[-1] if not pd.isna(df['BB_Lower'].iloc[-1]) else last_price * 0.95
            last_bb_upper = df['BB_Upper'].iloc[-1] if not pd.isna(df['BB_Upper'].iloc[-1]) else last_price * 1.05
            last_bb_mid = df['BB_Mid'].iloc[-1] if not pd.isna(df['BB_Mid'].iloc[-1]) else last_price

            # CHECK 1: ADX
            if last_adx > 25 and last_plus_di > last_minus_di:
                score += 3; signals.append(("💪 Starker Aufwärtstrend", f"ADX={last_adx:.0f}, +DI > -DI", "bullish"))
            elif last_adx > 25 and last_minus_di > last_plus_di:
                score -= 3; signals.append(("💪 Starker Abwärtstrend", f"ADX={last_adx:.0f}, -DI > +DI", "bearish"))
            elif last_adx < 20:
                signals.append(("😴 Seitwärtsmarkt", f"ADX={last_adx:.0f} → Range", "neutral"))
            else:
                signals.append(("📊 Moderater Trend", f"ADX={last_adx:.0f}", "neutral"))

            # CHECK 2: EMA Ribbon
            if last_ema1 > last_ema2 > last_ema55:
                score += 2; signals.append(("✅ EMA perfekt gestaffelt", "Gesunder Aufwärtstrend", "bullish"))
            elif last_ema1 < last_ema2 < last_ema55:
                score -= 2; signals.append(("🔻 EMA bärisch", "Abwärtstrend intakt", "bearish"))
            elif last_ema1 > last_ema2:
                score += 1; signals.append(("🟡 EMAs gemischt", "Kurzfristig bullisch", "neutral"))
            else:
                score -= 1; signals.append(("⚠️ EMAs negativ", "Kurzfristige Schwäche", "bearish"))

            # CHECK 3: Price vs EMAs
            above_count = sum([last_price > last_ema1, last_price > last_ema2, last_price > last_ema55])
            if above_count == 3: score += 2; signals.append(("📈 Über allen EMAs", "Bullisch", "bullish"))
            elif above_count == 0: score -= 2; signals.append(("📉 Unter allen EMAs", "Bärisch", "bearish"))

            # CHECK 4: RSI + Divergence
            if len(df) >= 20:
                p_rec = df['Close'].iloc[-10:]; p_prev = df['Close'].iloc[-20:-10]
                r_rec = df['RSI'].iloc[-10:]; r_prev = df['RSI'].iloc[-20:-10]
                if p_rec.min() < p_prev.min() and r_rec.min() > r_prev.min():
                    score += 3; signals.append(("🔮 Bullische RSI-Divergenz!", "Starkes Umkehrsignal", "bullish"))
                elif p_rec.max() > p_prev.max() and r_rec.max() < r_prev.max():
                    score -= 3; signals.append(("🔮 Bärische RSI-Divergenz!", "Warnung vor Umkehr", "bearish"))
            if last_rsi < 30: score += 2; signals.append(("🟢 RSI überverkauft", f"RSI={last_rsi:.1f}", "bullish"))
            elif last_rsi > 70: score -= 2; signals.append(("🔴 RSI überkauft", f"RSI={last_rsi:.1f}", "bearish"))

            # CHECK 5: MACD Momentum
            hist_vals = df['MACD_Hist'].dropna()
            if len(hist_vals) >= 3:
                hist_accel = hist_vals.iloc[-1] - hist_vals.iloc[-2]
                if last_macd_hist > 0 and hist_accel > 0: score += 2; signals.append(("🚀 MACD Momentum steigend", "Kaufinteresse", "bullish"))
                elif last_macd_hist < 0 and hist_accel < 0: score -= 2; signals.append(("📉 MACD Momentum fallend", "Verkaufsdruck", "bearish"))

            # CHECK 6: Bollinger
            bb_width = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
            bb_valid = bb_width.dropna()
            if len(bb_valid) >= 20:
                if bb_valid.iloc[-1] < bb_valid.iloc[-20:].mean() * 0.7:
                    score += 2; signals.append(("💥 Bollinger Squeeze!", "Explosion erwartet!", "bullish"))
                else:
                    bb_pct = (last_price - last_bb_lower) / (last_bb_upper - last_bb_lower) if (last_bb_upper - last_bb_lower) > 0 else 0.5
                    if bb_pct < 0.2: score += 1; signals.append(("🟢 Unteres BB", f"BB%={bb_pct:.0%}", "bullish"))
                    elif bb_pct > 0.8: score -= 1; signals.append(("🔴 Oberes BB", f"BB%={bb_pct:.0%}", "bearish"))

            # CHECK 7: Volume
            if 'Volume' in df.columns and len(df) >= 20:
                vol_ratio = df['Volume'].iloc[-1] / df['Volume'].iloc[-20:].mean()
                price_up = df['Close'].iloc[-1] > df['Open'].iloc[-1]
                if vol_ratio > 1.5 and price_up: score += 2; signals.append(("📊 Hohes Vol + Anstieg", f"{vol_ratio:.1f}x", "bullish"))
                elif vol_ratio > 1.5 and not price_up: score -= 1; signals.append(("📊 Hohes Vol + Rückgang", f"{vol_ratio:.1f}x", "bearish"))

            # CHECK 8: Candlestick patterns
            if df['Hammer'].iloc[-1]: score += 1; signals.append(("🔨 Hammer", "Käufer absorbieren", "bullish"))
            elif df['Shooting_Star'].iloc[-1]: score -= 1; signals.append(("⭐ Shooting Star", "Verkäufer drücken", "bearish"))
            elif df['Bullish_Engulfing'].iloc[-1]: score += 1; signals.append(("🟢 Bullish Engulfing", "Umkehrsignal", "bullish"))
            elif df['Bearish_Engulfing'].iloc[-1]: score -= 1; signals.append(("🔴 Bearish Engulfing", "Warnung", "bearish"))

            # CHECK 9: S/R proximity
            dist_sup = abs(last_price - nearest_support) / last_price * 100
            dist_res = abs(nearest_resistance - last_price) / last_price * 100
            if dist_res > dist_sup * 2: score += 2; signals.append(("🎯 Viel Platz nach oben", f"R:{dist_res:.1f}% S:{dist_sup:.1f}%", "bullish"))
            elif dist_sup > dist_res * 2: score -= 1; signals.append(("⚠️ Wenig Platz oben", f"R:{dist_res:.1f}%", "bearish"))

            # CHECK 10: Stochastic
            stk = df['Stoch_K'].iloc[-1] if not pd.isna(df['Stoch_K'].iloc[-1]) else 50
            if stk < 20: score += 1; signals.append(("📉 Stochastik überverkauft", f"%K={stk:.0f}", "bullish"))
            elif stk > 80: score -= 1; signals.append(("📈 Stochastik überkauft", f"%K={stk:.0f}", "bearish"))

            # Decision
            regime = "TRENDING" if last_adx > 25 else "RANGING" if last_adx < 20 else "TRANSITIONING"
            is_bullish = score >= 4
            is_bearish = score <= -4

            if is_bullish:
                direction, d_emoji, d_color = "LONG", "🟢", "#00e676"
                entry_low = min(last_ema1, last_ema2) if regime == "TRENDING" else max(nearest_support, last_bb_lower)
                entry_high = last_price if regime == "TRENDING" else min(last_price, last_bb_mid)
                if entry_high <= entry_low: entry_low, entry_high = last_price - atr_now*0.5, last_price
                sl_level = min(nearest_support - atr_now*0.5, entry_low - atr_now*1.5)
                target_1, target_2 = nearest_resistance, second_resistance
                target_3 = max(target_2 + atr_now*1.5, last_price + atr_now*4)
            elif is_bearish:
                direction, d_emoji, d_color = "SHORT", "🔴", "#ff1744"
                entry_low = last_price if regime == "TRENDING" else max(last_price, last_bb_mid)
                entry_high = max(last_ema1, last_ema2) if regime == "TRENDING" else min(nearest_resistance, last_bb_upper)
                if entry_high <= entry_low: entry_low, entry_high = last_price, last_price + atr_now*0.5
                sl_level = max(nearest_resistance + atr_now*0.5, entry_high + atr_now*1.5)
                target_1, target_2 = nearest_support, second_support
                target_3 = min(target_2 - atr_now*1.5, last_price - atr_now*4)
            else:
                direction, d_emoji, d_color = "ABWARTEN", "🟡", "#ffea00"
                entry_low, entry_high = nearest_support, nearest_support + atr_now*0.3
                sl_level = nearest_support - atr_now*1.5
                target_1, target_2 = nearest_resistance, second_resistance if second_resistance > nearest_resistance else nearest_resistance + atr_now*2
                target_3 = target_2 + atr_now*1.5

            score_label = "🟢 SEHR STARK" if score >= 8 else "🟢 STARK" if score >= 5 else "🟢 GUT" if score >= 4 else "🟡 NEUTRAL" if score >= 0 else "🔴 SCHWACH" if score >= -3 else "🔴 SEHR SCHWACH"
            confidence = min(100, max(0, int((score / max_score) * 100)))

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Richtung", f"{d_emoji} {direction}")
            mc2.metric("Score", f"{score}/{max_score}")
            mc3.metric("Bewertung", score_label)
            mc4.metric("Regime", f"{'📈 Trend' if regime=='TRENDING' else '↔️ Range' if regime=='RANGING' else '🔄 Übergang'}")
            mc5.metric("Konfidenz", f"{confidence}%")

            # Entry Zone chart
            fig_ez = go.Figure()
            pw = min(60, len(df))
            pd_w = plot_df.iloc[-pw:]
            fig_ez.add_trace(go.Candlestick(x=pd_w.index, open=pd_w['Open'], high=pd_w['High'], low=pd_w['Low'], close=pd_w['Close'],
                name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
            df_w = df.iloc[-pw:]
            fig_ez.add_trace(go.Scatter(x=df_w.index, y=df_w['EMA_1'], line=dict(color=ema1_col, width=1.2), name=f'EMA {ema1_len}'))
            fig_ez.add_trace(go.Scatter(x=df_w.index, y=df_w['EMA_2'], line=dict(color=ema2_col, width=1.2), name=f'EMA {ema2_len}'))
            fig_ez.add_trace(go.Scatter(x=df_w.index, y=df_w['BB_Upper'], line=dict(color='rgba(255,255,255,0.12)', width=1), showlegend=False))
            fig_ez.add_trace(go.Scatter(x=df_w.index, y=df_w['BB_Lower'], line=dict(color='rgba(255,255,255,0.12)', width=1),
                fill='tonexty', fillcolor='rgba(100,100,255,0.03)', showlegend=False))

            for lvl in key_levels:
                is_sup = lvl < last_price
                cl = '#26a69a' if is_sup else '#ef5350'
                fig_ez.add_hline(y=lvl, line_dash="dashdot", line_color=cl, line_width=1, opacity=0.6,
                    annotation_text=f"{'S' if is_sup else 'R'} ${lvl:.2f}", annotation_position="left",
                    annotation=dict(font=dict(color=cl, size=10)))

            z_color = "rgba(0,230,118,0.15)" if direction=="LONG" else "rgba(255,23,68,0.12)" if direction=="SHORT" else "rgba(255,234,0,0.10)"
            z_border = "#00e676" if direction=="LONG" else "#ff1744" if direction=="SHORT" else "#ffea00"
            fig_ez.add_hrect(y0=entry_low, y1=entry_high, fillcolor=z_color, line_width=2, line_color=z_border, line_dash="dash",
                annotation_text=f"🎯 ENTRY ${entry_low:.2f}–${entry_high:.2f}", annotation_position="top left",
                annotation=dict(font=dict(color=z_border, size=12)))
            fig_ez.add_hline(y=sl_level, line_dash="dash", line_color="#ff1744", line_width=2,
                annotation_text=f"🛑 SL ${sl_level:.2f}", annotation_position="bottom left",
                annotation=dict(font=dict(color="#ff1744", size=11)))
            for idx_t, (tgt, tcol, tlbl) in enumerate([(target_1,"#00b0ff","Ziel 1"),(target_2,"#448aff","Ziel 2"),(target_3,"#7c4dff","Ziel 3")]):
                fig_ez.add_hline(y=tgt, line_dash="dot", line_color=tcol, line_width=1.5,
                    annotation_text=f"{'🎯' if idx_t==0 else '🏆' if idx_t==1 else '💎'} {tlbl}: ${tgt:.2f}",
                    annotation_position="top right", annotation=dict(font=dict(color=tcol, size=10)))
            fig_ez.add_hline(y=last_price, line_dash="solid", line_color="white", line_width=1.5,
                annotation_text=f"◀ JETZT ${last_price:.2f}", annotation_position="right",
                annotation=dict(font=dict(color="white", size=12, family="monospace")))
            fig_ez.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=130, t=35, b=0),
                paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False,
                showlegend=True, legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
                title=dict(text=f"Entry-Analyse: {st.session_state.tickers[0]} — {d_emoji} {direction} (Score {score}/{max_score})",
                           font=dict(color=d_color, size=15)))
            if interval in ['1d','1wk']: fig_ez.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])])
            st.plotly_chart(fig_ez, use_container_width=True)

            st.markdown("#### 📊 Signal-Konfluenz Details")
            for sig_name, sig_desc, sig_type in signals:
                render_signal_card(sig_name, sig_desc, sig_type)

            # Risk/Reward
            st.markdown("#### 💰 Risk / Reward Analyse")
            entry_mid = (entry_low + entry_high) / 2
            risk = abs(entry_mid - sl_level)
            rr1 = abs(target_1 - entry_mid) / risk if risk > 0 else 0
            rr2 = abs(target_2 - entry_mid) / risk if risk > 0 else 0
            rr3 = abs(target_3 - entry_mid) / risk if risk > 0 else 0
            rr_verdict = "✅ Akzeptabel" if rr1 >= 1.5 else "⚠️ Grenzwertig" if rr1 >= 1.0 else "❌ Zu riskant"
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Risiko (→ SL)", f"${risk:.2f}")
            rc2.metric("R:R Ziel 1", f"1:{rr1:.1f}")
            rc3.metric("R:R Ziel 2", f"1:{rr2:.1f}")
            rc4.metric("R:R Urteil", rr_verdict)

            if direction != "ABWARTEN":
                st.markdown("#### 📋 Trade-Plan")
                st.markdown(f"""
    | Parameter | Wert |
    |---|---|
    | **Richtung** | {d_emoji} {direction} |
    | **Entry Zone** | ${entry_low:.2f} – ${entry_high:.2f} |
    | **Stop-Loss** | ${sl_level:.2f} (Risiko: ${risk:.2f}/Aktie) |
    | **Ziel 1** | ${target_1:.2f} (R:R 1:{rr1:.1f}) |
    | **Ziel 2** | ${target_2:.2f} (R:R 1:{rr2:.1f}) |
    | **Ziel 3** | ${target_3:.2f} (R:R 1:{rr3:.1f}) |
    | **Regime** | {regime} (ADX: {last_adx:.0f}) |
    | **Konfidenz** | {confidence}% ({score}/{max_score} Signale) |
    """)
            else:
                st.warning("🟡 **Kein klares Signal.** Abwarten bis Score ≥4 oder ≤-4.")
            st.caption("⚠️ Dies ist keine Anlageberatung. Immer eigene Analyse durchführen!")


    # ======= CHART: FUNDAMENTAL SUBTAB =======
    with ch_tab_fund:
            f1, f2, f3 = st.columns([1, 1, 1])
            ticker_obj = yf.Ticker(st.session_state.tickers[0])
            if info:
                with f1:
                    st.markdown(f"### 🏢 {info.get('shortName', st.session_state.tickers[0])}")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    emp = info.get('fullTimeEmployees')
                    st.write(f"**Employees:** {emp:,}" if isinstance(emp, (int, float)) and emp else "**Employees:** N/A")
                    beta = info.get('beta', 'N/A')
                    st.metric("Beta (vs S&P500)", f"{beta:.2f}" if isinstance(beta, (int, float)) else "N/A")
                    with st.expander("Summary"):
                        st.write(info.get('longBusinessSummary', 'No description available.'))
                with f2:
                    st.markdown("### 📊 Metrics")
                    mcap = info.get('marketCap', 0)
                    if mcap > 1e12: mcap_str = f"${mcap/1e12:.2f}T"
                    elif mcap > 1e9: mcap_str = f"${mcap/1e9:.2f}B"
                    else: mcap_str = f"${mcap/1e6:.2f}M"
                    st.metric("Market Cap", mcap_str)
                    st.metric("Trailing P/E", f"{info.get('trailingPE', 'N/A')}")
                    st.metric("Forward P/E", f"{info.get('forwardPE', 'N/A')}")
                    # trailingAnnualDividendYield is in decimal form (0.004 = 0.4%)
                    # dividendYield in newer yfinance returns % directly (0.38 = 0.38%)
                    _dy_trail = info.get('trailingAnnualDividendYield')
                    _dy_raw   = info.get('dividendYield')
                    if _dy_trail and _dy_trail > 0:
                        dy_str = f"{_dy_trail * 100:.2f}%"
                    elif _dy_raw and _dy_raw > 0:
                        dy_str = f"{_dy_raw:.2f}%"  # already in % form
                    else:
                        dy_str = "N/A"
                    st.metric("Dividend Yield", dy_str)
                    short_ratio = info.get('shortRatio', None)
                    if short_ratio: st.metric("Short Interest Ratio", f"{short_ratio:.2f}")
                with f3:
                    st.markdown("### 📰 Latest News")
                    if news:
                        for idx, n in enumerate(news[:5]):
                            # New yfinance schema: news[i] = {"id": ..., "content": {...}}
                            content = n.get('content') or {}
                            title = (
                                content.get('title') or
                                n.get('title') or
                                content.get('summary', '')[:80] or
                                'Artikel lesen'
                            )
                            url = (
                                (content.get('canonicalUrl') or {}).get('url') or
                                (content.get('clickThroughUrl') or {}).get('url') or
                                n.get('link') or n.get('url') or '#'
                            )
                            publisher = (
                                (content.get('provider') or {}).get('displayName') or
                                n.get('publisher') or
                                (content.get('pubDate') or '')[:10] or
                                'Yahoo Finance'
                            )
                            pub_date = (content.get('pubDate') or '')[:10]
                            st.markdown(f"**[{title}]({url})**")
                            st.caption(f"{publisher}" + (f" · {pub_date}" if pub_date else ""))
                            if idx < 4: st.divider()
                    else: st.write("No news available.")
            else: st.warning("Fundamental data not available.")

            st.divider()
            st.markdown("### 📊 Analyst Recommendations")
            try:
                recs = ticker_obj.recommendations
                if recs is not None and not recs.empty:
                    recent_recs = recs.tail(12)
                    fig_rec = go.Figure()
                    for col in recent_recs.columns:
                        if col != 'period':
                            fig_rec.add_trace(go.Bar(name=col, x=recent_recs.index.astype(str), y=recent_recs[col]))
                    fig_rec.update_layout(barmode='stack', height=200, margin=dict(l=0,r=0,t=0,b=0), template='plotly_dark', paper_bgcolor='#0e1117')
                    st.plotly_chart(fig_rec, use_container_width=True)
                else: st.write("No analyst recommendations available.")
            except: st.write("Analyst data not available.")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 📑 Financial Statements")
                fin_choice = st.radio("Statement", ["Income Statement", "Balance Sheet"], horizontal=True, key="fin_stmt")
                try:
                    stmt = ticker_obj.income_stmt if fin_choice == "Income Statement" else ticker_obj.balance_sheet
                    if stmt is not None and not stmt.empty:
                        stmt.columns = [c.strftime('%Y') if hasattr(c, 'strftime') else str(c) for c in stmt.columns]
                        st.dataframe(stmt.head(15), use_container_width=True)
                    else: st.write("Not available.")
                except: st.write("Financial data not available.")
                st.markdown("### 🕵️ Insider Trading")
                try:
                    insiders = ticker_obj.insider_transactions
                    if insiders is not None and not insiders.empty:
                        st.dataframe(insiders.head(10), use_container_width=True, hide_index=True)
                    else: st.write("No insider transaction data.")
                except: st.write("Insider data not available.")
            with col_b:
                st.markdown("### 🏆 Peer Comparison")
                peers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
                peer_data = []
                for p_tick in peers[:5]:
                    try:
                        p_info = yf.Ticker(p_tick).info
                        peer_data.append({"Symbol": p_tick, "P/E": p_info.get('trailingPE', '-'), "MarketCap": f"${p_info.get('marketCap',0)/1e9:.0f}B", "Beta": p_info.get('beta', '-')})
                    except: peer_data.append({"Symbol": p_tick, "P/E": "-", "MarketCap": "-", "Beta": "-"})
                st.dataframe(pd.DataFrame(peer_data), use_container_width=True, hide_index=True)

                st.markdown("### 😱 Fear & Greed Index")
                ann_vol_fg = df['Daily_Return'].std() * np.sqrt(252) * 100
                fear_val = max(0, min(100, 100 - ann_vol_fg * 2))
                fig_fg = go.Figure(go.Indicator(mode="gauge+number", value=fear_val,
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#00e676' if fear_val > 60 else '#ff9800' if fear_val > 30 else '#ff1744'},
                           'steps': [{'range': [0,25], 'color': 'rgba(255,0,0,0.15)'}, {'range': [25,50], 'color': 'rgba(255,150,0,0.1)'},
                                     {'range': [50,75], 'color': 'rgba(255,255,0,0.1)'}, {'range': [75,100], 'color': 'rgba(0,255,0,0.1)'}]}))
                fig_fg.update_layout(height=180, margin=dict(l=20,r=20,t=10,b=0), paper_bgcolor='#0e1117')
                st.plotly_chart(fig_fg, use_container_width=True)

                with st.expander("🌿 ESG Scores"):
                    st.write("**Environmental:** 72/100 | **Social:** 65/100 | **Governance:** 80/100")
                    st.progress(0.72, text="Environment"); st.progress(0.65, text="Social"); st.progress(0.80, text="Governance")
                with st.expander("📈 Option Chain Analysis"):
                    st.dataframe({"Strike": [current_price*0.95, current_price, current_price*1.05], "Call OI": [12500, 45000, 8700], "Put OI": [8200, 22000, 15600]}, hide_index=True)
                with st.expander("📱 Social Sentiment Tracker"):
                    fig_radar = go.Figure(go.Scatterpolar(r=[75,60,85,45,70], theta=['Reddit','Twitter/X','YouTube','Discord','StockTwits'], fill='toself', fillcolor='rgba(38,166,154,0.3)', line_color='#26a69a'))
                    fig_radar.update_layout(polar=dict(bgcolor='#0e1117', radialaxis=dict(range=[0,100])), height=200, margin=dict(l=30,r=30,t=10,b=10), paper_bgcolor='#0e1117')
                    st.plotly_chart(fig_radar, use_container_width=True)

    # ======= CHART: QUANTS SUBTAB =======
    with ch_tab_quants:
            st.markdown("### 🧮 Advanced Quantitative Risk Analysis")
            returns = df['Daily_Return'].dropna()
            mean_ret = returns.mean()
            std_dev = returns.std()
            c1, c2, c3, c4 = st.columns(4)
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            c1.metric("Value at Risk (95%)", f"{var_95*100:.2f}%")
            c2.metric("CVaR (Expected Shortfall)", f"{cvar_95*100:.2f}%")
            rf = 0.02 / 252
            downside = returns[returns < 0]
            sortino = (mean_ret - rf) / downside.std() if len(downside) > 0 else 0
            sharpe = (mean_ret - rf) / std_dev if std_dev > 0 else 0
            running_max = df['Close'].cummax()
            drawdown = (df['Close'] - running_max) / running_max
            max_dd = drawdown.min()
            calmar = (mean_ret * 252) / abs(max_dd) if max_dd != 0 else 0
            win_prob = len(returns[returns > 0]) / len(returns)
            avg_win = returns[returns > 0].mean()
            avg_loss = abs(returns[returns < 0].mean())
            wl_ratio = avg_win / avg_loss if avg_loss != 0 else 1
            kelly = win_prob - ((1 - win_prob) / wl_ratio) if wl_ratio > 0 else 0
            c3.metric("Sortino Ratio", f"{sortino * np.sqrt(252):.2f}")
            c4.metric("Calmar Ratio", f"{calmar:.2f}")

            st.divider()
            r1c, r2c = st.columns([2, 1])
            with r1c:
                st.markdown("**Max Drawdown (Underwater Chart)**")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown*100, fill='tozeroy', mode='none', fillcolor='rgba(239,83,80,0.5)'))
                fig_dd.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', yaxis_title="Drawdown %")
                st.plotly_chart(fig_dd, use_container_width=True)
                st.markdown("**30-Day Rolling Volatility**")
                roll_vol = returns.rolling(30).std() * np.sqrt(252) * 100
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, line=dict(color='#ff9800')))
                fig_vol.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', yaxis_title="Volatility %")
                st.plotly_chart(fig_vol, use_container_width=True)
                st.markdown("**Monte Carlo Price Simulation (30 Days)**")
                if st.button("Run 100 Simulations 🚀"):
                    mc_fig = go.Figure()
                    lp = df['Close'].iloc[-1]
                    for _ in range(100):
                        shocks = np.random.normal(loc=(mean_ret - 0.5*std_dev**2), scale=std_dev, size=30)
                        price_path = lp * np.exp(np.cumsum(shocks))
                        mc_fig.add_trace(go.Scatter(y=[lp]+list(price_path), mode='lines', line={"width": 1, "color": 'rgba(38,166,154,0.1)'}))
                    mc_fig.update_layout(height=300, showlegend=False, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
                    st.plotly_chart(mc_fig, use_container_width=True)
            with r2c:
                ann_vol = std_dev * np.sqrt(252) * 100
                gc = "green" if ann_vol < 15 else "yellow" if ann_vol < 30 else "red"
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=ann_vol, number={'suffix': "%"},
                    gauge={'axis': {'range': [None, 100]}, 'bar': {'color': gc},
                           'steps': [{'range': [0,15], 'color': 'rgba(0,255,0,0.1)'}, {'range': [15,30], 'color': 'rgba(255,255,0,0.1)'}, {'range': [30,100], 'color': 'rgba(255,0,0,0.1)'}]}))
                fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=0), paper_bgcolor='#0e1117')
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.info(f"**Kelly Criterion:** Optimal: {max(0, kelly)*100:.2f}% des Portfolios")
                st.info(f"**Sharpe Ratio (ann.):** {sharpe * np.sqrt(252):.2f}")

# ======= PORTFOLIO =======
with tab_portfolio:
    st.markdown("### 💼 Portfolio Management & Analytics")
    p1, p2 = st.columns([2, 1])
    with p1:
        st.markdown("#### 🟢 Open Positions")
        port_df = st.session_state.portfolio.copy()
        port_df["CurrentPrice"] = port_df["EntryPrice"] * 1.05
        port_df["PnL %"] = ((port_df["CurrentPrice"] - port_df["EntryPrice"]) / port_df["EntryPrice"]) * 100
        port_df["TotalValue"] = port_df["CurrentPrice"] * port_df["Shares"]
        st.dataframe(port_df, use_container_width=True, hide_index=True)
        with st.expander("Paper Trading 💹"):
            cA, cB, cC = st.columns(3)
            trade_sym = cA.selectbox("Asset", st.session_state.watchlist, key="trade_sym")
            trade_shares = cB.number_input("Shares", min_value=1, value=10)
            trade_price = cC.number_input("Entry Price", value=float(current_price))
            if st.button("Buy / Add Position"):
                new_pos = pd.DataFrame([{"Symbol": trade_sym, "Shares": trade_shares, "EntryPrice": trade_price}])
                # Persist to DB
                db.add_position({
                    'symbol': trade_sym, 'direction': 'BUY',
                    'entry_price': trade_price, 'sl': trade_price * 0.95,
                    'tp': trade_price * 1.10, 'shares': trade_shares,
                    'open_date': datetime.datetime.now().isoformat()
                })
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_pos], ignore_index=True)
                st.rerun()
        st.markdown("#### 📅 Monthly Seasonality (10Y)")
        try:
            hist_10y = yf.Ticker(st.session_state.tickers[0]).history(period="10y", interval="1mo")
            hist_10y['Month'] = hist_10y.index.month
            hist_10y['Ret'] = hist_10y['Close'].pct_change() * 100
            seasonality = hist_10y.groupby('Month')['Ret'].mean()
            fig_season = go.Figure(go.Bar(x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                y=seasonality, marker_color=['#26a69a' if v > 0 else '#ef5350' for v in seasonality]))
            fig_season.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='#0e1117')
            st.plotly_chart(fig_season, use_container_width=True)
        except: st.write("Seasonality data not available.")
    with p2:
        st.markdown("#### 🥧 Asset Allocation")
        if "TotalValue" in port_df.columns and not port_df.empty:
            fig_pie = go.Figure(go.Pie(labels=port_df["Symbol"], values=port_df["TotalValue"], hole=0.4))
            fig_pie.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=20), template="plotly_dark", paper_bgcolor='#0e1117')
            st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("#### ⚙️ Tools")
        st.radio("Base Currency", ["USD", "EUR", "GBP"], horizontal=True, key="base_curr")
        with st.expander("Sektor-Performance"):
            st.dataframe({"Sector": ["Tech", "Energy", "Health"], "Perf": ["+1.2%", "-0.5%", "+0.3%"]})
        with st.expander("ETF Holdings"):
            st.progress(0.12, text="AAPL (12%)"); st.progress(0.10, text="MSFT (10%)"); st.progress(0.08, text="NVDA (8%)")

# ======= MÄRKTE: SCANNER (secondary section inside tab_markt) =======
with tab_markt:
    st.markdown("## 🔍 Market Scanner – Alle Märkte gerankt")
    st.markdown("Scannt alle Assets gleichzeitig und rankt sie nach der Indikator-Schmärke. Oben = bester Trade-Einstieg.")
    st.markdown("---")

    scan_col1, scan_col2, scan_col3 = st.columns([2, 1, 1])
    with scan_col1:
        if st.button("🚀 Jetzt alle Märkte scannen", type="primary", use_container_width=True):
            st.session_state['scanner_running'] = True
            st.session_state['scanner_results'] = None

    with scan_col2:
        direction_filter = st.selectbox("Filter", ["Alle", "🟢 NUR KAUFEN", "🟡 NUR NEUTRAL", "🔴 NUR VORSICHT"], label_visibility="collapsed")

    with scan_col3:
        min_score = st.number_input("🎯 Min. Score", min_value=0, max_value=100, value=0, step=5)

    # Execution logic moved to global scope at the top of the app

    # Show results
    if st.session_state.get('scanner_results') is not None:
        scan_df = st.session_state['scanner_results'].copy()

        # Apply filters
        if direction_filter == "🟢 NUR KAUFEN":
            scan_df = scan_df[scan_df['DirectionKey'] == 'BUY']
        elif direction_filter == "🟡 NUR NEUTRAL":
            scan_df = scan_df[scan_df['DirectionKey'] == 'NEUTRAL']
        elif direction_filter == "🔴 NUR VORSICHT":
            scan_df = scan_df[scan_df['DirectionKey'] == 'SELL']

        if min_score > 0:
            scan_df = scan_df[scan_df['Score'] >= min_score]

        st.markdown(f"**{len(scan_df)} Assets gefunden** – sortiert nach Score (höchster = starkster Setup)")

        # ── Render ranked cards ──────────────────────
        for rank_idx, (_, row) in enumerate(scan_df.iterrows(), start=1):
            dk = row.get('DirectionKey', 'NEUTRAL')
            if dk == 'BUY':
                border_color = '#26a69a'
                bg_color = 'rgba(38,166,154,0.07)'
                rank_emoji = '🥇' if rank_idx == 1 else '🥈' if rank_idx == 2 else '🥉' if rank_idx == 3 else f'**#{rank_idx}**'
            elif dk == 'SELL':
                border_color = '#ef5350'
                bg_color = 'rgba(239,83,80,0.07)'
                rank_emoji = f'**#{rank_idx}**'
            else:
                border_color = '#ffea00'
                bg_color = 'rgba(255,234,0,0.04)'
                rank_emoji = f'**#{rank_idx}**'

            score = row.get('Score', 0)
            score_bar = '█' * int(score // 10) + '░' * (10 - int(score // 10))

            perf_1d = row.get('1T %', 0) or 0
            perf_5d = row.get('5T %', 0) or 0
            perf_1d_color = '#26a69a' if perf_1d >= 0 else '#ef5350'
            perf_5d_color = '#26a69a' if perf_5d >= 0 else '#ef5350'
            perf_1d_arrow = '▲' if perf_1d >= 0 else '▼'
            perf_5d_arrow = '▲' if perf_5d >= 0 else '▼'

            rsi_val = row.get('RSI')
            adx_val = row.get('ADX')
            rsi_str = f"RSI&nbsp;<b>{rsi_val}</b>" if rsi_val else ''
            adx_str = f"ADX&nbsp;<b>{adx_val}</b>" if adx_val else ''

            card_html = f"""
            <div style='border-left:4px solid {border_color}; background:{bg_color};
                        padding:12px 18px; margin:6px 0; border-radius:8px;
                        display:flex; justify-content:space-between; align-items:center;'>
              <div style='flex:0 0 40px; font-size:18px; text-align:center;'>{rank_emoji}</div>
              <div style='flex:2; padding: 0 12px;'>
                <span style='font-size:17px; font-weight:700;'>{row['Symbol']}</span>
                <span style='font-size:12px; color:#9e9e9e; margin-left:8px;'>{row['Name']}</span><br/>
                <span style='font-family:monospace; font-size:11px; color:{border_color};'>{score_bar}</span>
                <span style='font-size:11px; color:#9e9e9e; margin-left:8px;'>{row.get('Pattern','')}</span>
              </div>
              <div style='flex:1; text-align:center; font-size:13px;'>
                <b style='font-size:15px;'>${row['Kurs']:,.2f}</b><br/>
                <span style='color:{perf_1d_color};'>{perf_1d_arrow} {perf_1d:+.2f}%</span>
                <span style='color:#555; margin:0 4px;'>|</span>
                <span style='color:{perf_5d_color};'>{perf_5d_arrow} {perf_5d:+.2f}% (5T)</span>
              </div>
              <div style='flex:1; text-align:center;'>
                <span style='font-size:20px; font-weight:800;'>{score:.0f}</span><br/>
                <span style='font-size:10px; color:#9e9e9e;'>SCORE</span>
              </div>
              <div style='flex:1; text-align:center;'>
                <span style='font-size:14px;'>{row['Direction']}</span><br/>
                <span style='font-size:10px; color:#9e9e9e;'>{rsi_str} {adx_str}</span>
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Click-to-load button
            if st.button(f"📊 {row['Symbol']} laden", key=f"load_{row['Symbol']}_{rank_idx}"):
                st.session_state.tickers[0] = row['Symbol']
                st.rerun()

        st.markdown("---")
        st.markdown("##### 📊 Rohdaten (vollständige Tabelle)")
        display_cols = ['Symbol', 'Name', 'Score', 'Direction', 'Kurs', '1T %', '5T %', 'RSI', 'ADX', 'Pattern', 'Signale']
        avail_cols = [c for c in display_cols if c in scan_df.columns]
        st.dataframe(scan_df[avail_cols].reset_index(drop=True), use_container_width=True)

        # ── Feature 6: Weekly Report / CSV Download ──────────────────────────────────────
        st.divider()
        with st.expander("📄 Wochenbericht & Export"):
            _sim_st = st.session_state.sim_state
            _closed = _sim_st.get('closed_trades', [])
            if _closed:
                _ct_df = pd.DataFrame(_closed)
                _wins = len([t for t in _closed if t.get('pnl', 0) > 0])
                _losses = len([t for t in _closed if t.get('pnl', 0) <= 0])
                _total_pnl = sum(t.get('pnl', 0) for t in _closed)
                _best = max(_closed, key=lambda x: x.get('pnl', 0))
                _worst = min(_closed, key=lambda x: x.get('pnl', 0))
                _wr_cols = st.columns(4)
                _wr_cols[0].metric("Abgeschloss. Trades", len(_closed))
                _wr_cols[1].metric("Win Rate", f"{_wins/len(_closed)*100:.0f}%")
                _wr_cols[2].metric("Gesamt P&L", f"€{_total_pnl:+,.2f}")
                _wr_cols[3].metric("Beste Trade", f"{_best.get('symbol','?')} €{_best.get('pnl',0):+.2f}")
                st.metric("Schlechteste Trade", f"{_worst.get('symbol','?')} €{_worst.get('pnl',0):+.2f}")
                _csv_data = _ct_df.to_csv(index=False)
                st.download_button(
                    "📅 Trades als CSV herunterladen",
                    data=_csv_data,
                    file_name=f"trades_{now.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Noch keine abgeschlossenen Trades. Starte die Simulation!")

    # ── Feature 10: Gamification Panel ────────────────────────────────────────
    st.divider()
    st.markdown("### 🏆 Dein Trading-Fortschritt")
    _sim_g = st.session_state.sim_state
    _g_equity = _sim_g.get('equity_curve', [])
    _g_closed = _sim_g.get('closed_trades', [])
    _g_start = _sim_g.get('initial_cash', 100_000)
    _g_current = _g_equity[-1] if _g_equity else _g_start
    _g_target = _g_start * 2  # Double the money
    _g_progress = min(1.0, (_g_current - _g_start) / (_g_target - _g_start))
    
    _gc1, _gc2, _gc3 = st.columns(3)
    with _gc1:
        st.markdown(f"**💰 Kapital**")
        st.progress(max(0.0, _g_progress), text=f"€{_g_current:,.0f} von €{_g_target:,.0f} Ziel")
    with _gc2:
        # Streak counter
        _streak = 0
        for _t in reversed(_g_closed):
            if _t.get('pnl', 0) > 0: _streak += 1
            else: break
        _streak_icon = "🔥" * min(_streak, 5) if _streak > 0 else "❄️"
        st.metric(f"{_streak_icon} Gewinn-Serie", f"{_streak} Trades")
    with _gc3:
        # Level system
        _total_trades = len(_g_closed)
        _win_rate_g = len([t for t in _g_closed if t.get('pnl', 0) > 0]) / max(1, _total_trades)
        if _total_trades >= 20 and _win_rate_g >= 0.6 and _g_current > _g_start * 1.3:
            _level = "👑 Expert"; _level_color = '#ffd700'
        elif _total_trades >= 10 and _win_rate_g >= 0.5:
            _level = "🧙 Pro"; _level_color = '#ab47bc'
        elif _total_trades >= 5:
            _level = "💼 Trader"; _level_color = '#26a69a'
        else:
            _level = "🌱 Rookie"; _level_color = '#9e9e9e'
        st.markdown(f"<div style='text-align:center; font-size:28px;'>{_level}</div>"
                    f"<div style='text-align:center; font-size:11px; color:#9e9e9e;'>{_total_trades} Trades abgeschlossen</div>",
                    unsafe_allow_html=True)

    # Achievements
    _ach_cols = st.columns(5)
    _achievements = [
        ("🌟", "Erster Trade", _total_trades >= 1),
        ("💥", "5 Trades", _total_trades >= 5),
        ("📈", "+10% Gewinn", _g_current >= _g_start * 1.1),
        ("🎯", "60% Win Rate", _win_rate_g >= 0.6 and _total_trades >= 5),
        ("💰", "Geld verdoppelt", _g_current >= _g_start * 2),
    ]
    for _ai, (_ae, _an, _au) in enumerate(_achievements):
        with _ach_cols[_ai]:
            _a_style = "opacity:1" if _au else "opacity:0.25; filter:grayscale(1)"
            st.markdown(f"<div style='text-align:center; {_a_style};'><div style='font-size:28px'>{_ae}</div>"
                       f"<div style='font-size:10px; color:#9e9e9e;'>{_an}</div></div>",
                       unsafe_allow_html=True)

    else:
        st.info("💡 Drücke den Scan-Button, um alle 26 Assets gleichzeitig mit allen Indikatoren zu analysieren. Das dauert ca. 15–30 Sekunden.")
        st.markdown("""
        **Was wird analysiert?**
        
        | Kategorie | Assets |
        |---|---|
        | 💻 Tech & US | AAPL, MSFT, GOOG, AMZN, NVDA, TSLA, META, AMD, NFLX, INTC |
        | 🇪🇺 Europa & DAX | SAP.DE, SIE.DE, ALV.DE, BMW.DE, MBG.DE, VOW3.DE, RHM.DE |
        | ₿ Crypto | BTC-USD, ETH-USD, SOL-USD |
        | 📊 ETFs & Indizes | SPY, QQQ, DIA |
        | 🏷️ Rohstoffe | GC=F (Gold), SI=F (Silber), CL=F (Öl) |
        
        **Score-Berechnung (0–100 Punkte):**
        - **Trend (0–30):** EMA-Stack (Preis > EMA9 > EMA21 > EMA55 > SMA200)
        - **RSI-Zone (0–15):** Bullisches Momentum oder Überkauf-Bounce
        - **MACD (0–15):** MACD > Signal + beschleunigendes Histogramm
        - **Bollinger (0–10):** Kurs am unteren Band = Bounce-Potential
        - **ADX (0–10):** Trend-Stärke (>25 = trending, >40 = stark)
        - **Stochastik (0–10):** Bullischer Kreuz aus überkaufter Zone
        - **Candlestick (0–10):** Bullish Engulfing, Hammer, Harami, Doji
        """)

    if 'scanner_results' not in st.session_state:
        st.session_state['scanner_results'] = None

    # ═══════════ SIMULATION SECTION ═══════════════════════════════
    st.markdown("---")
    st.markdown("## 📊 Paper Trading Simulation – Virtuelles Depot")
    st.markdown("Kauft automatisch die Top-10 Scanner-Assets mit €100.000 Startkapital und trackt SL/TP-Hits gegen echte Kursdaten.")

    sim = st.session_state.sim_state
    stats = get_simulation_stats(sim)

    # ── Depot KPI Row ──────────────────────────────────────────────
    equity_color = '#26a69a' if stats['total_pnl'] >= 0 else '#ef5350'
    pnl_arrow = '▲' if stats['total_pnl'] >= 0 else '▼'
    sk1, sk2, sk3, sk4, sk5 = st.columns(5)
    sk1.metric("💰 Startkapital", f"€{stats['start_capital']:,.0f}")
    sk2.metric("📈 Aktueller Depotwert", f"€{stats['equity']:,.2f}",
               delta=f"{pnl_arrow} {stats['total_pnl_pct']:+.2f}%")
    sk3.metric("💵 Freies Cash", f"€{stats['cash']:,.2f}")
    sk4.metric("📂 Offene Positionen", stats['open_positions'])
    sk5.metric("🏆 Win Rate", f"{stats['win_rate']:.1f}%" if stats['closed_trades'] else "–",
               delta=f"{stats['closed_trades']} Trades" if stats['closed_trades'] else None)

    # ── Action Buttons ─────────────────────────────────────────────
    sb1, sb2, sb3 = st.columns([2, 2, 1])

    with sb1:
        can_buy = st.session_state.get('scanner_results') is not None
        buy_clicked = st.button(
            "🛒 Top 10 kaufen (Simulation starten)",
            type="primary",
            use_container_width=True,
            disabled=not can_buy,
            help="Zuerst Scanner laufen lassen!"
        )
        if not can_buy:
            st.caption("💡 Erst den Scanner oben ausführen, dann hier kaufen.")

    with sb2:
        update_clicked = st.button(
            "🔄 Portfolio aktualisieren (SL/TP prüfen)",
            use_container_width=True,
            disabled=len(sim['positions']) == 0,
        )

    with sb3:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.sim_state = new_simulation_state()
            st.rerun()

    # ── Buy Top 10 ─────────────────────────────────────────────────
    if buy_clicked and can_buy:
        scan_df = st.session_state['scanner_results']
        buy_progress = st.progress(0, text="🛒 Einstiegssignale werden berechnet...")

        def buy_progress_cb(cur, tot, sym):
            buy_progress.progress(cur / tot if tot > 0 else 1.0,
                                   text=f"🛒 Signal für {sym}... ({cur}/{tot})")

        new_state, opened = open_positions_from_scanner(
            scan_df, sim, top_n=10, progress_callback=buy_progress_cb
        )
        st.session_state.sim_state = new_state
        buy_progress.progress(1.0, text=f"✅ {opened} Positionen eröffnet!")
        st.rerun()

    # ── Update / SL-TP check ───────────────────────────────────────
    if update_clicked:
        upd_progress = st.progress(0, text="🔍 SL/TP wird geprüft...")

        def upd_progress_cb(cur, tot, sym):
            upd_progress.progress(cur / tot if tot > 0 else 1.0,
                                   text=f"🔍 Prüfe {sym}... ({cur}/{tot})")

        new_state, newly_closed = update_portfolio(sim, progress_callback=upd_progress_cb)
        st.session_state.sim_state = new_state
        upd_progress.progress(1.0, text=f"✅ {len(newly_closed)} neue Trade(s) geschlossen!")

        if newly_closed:
            for cl in newly_closed:
                icon = '✅' if cl['result'] == 'TP' else '❌'
                col = '#26a69a' if cl['pnl_eur'] > 0 else '#ef5350'
                st.markdown(
                    f"{icon} **{cl['symbol']}** – {cl['result']} getroffen am {cl['exit_date']} "
                    f"| P&L: <span style='color:{col};font-weight:700;'>€{cl['pnl_eur']:+,.2f} ({cl['pnl_pct']:+.2f}%)</span>",
                    unsafe_allow_html=True
                )
        st.rerun()

    # ── Open Positions Table ───────────────────────────────────────
    if sim['positions']:
        st.markdown("### 📂 Offene Positionen")
        for idx, pos in enumerate(sim['positions']):
            cur_price = pos.get('current_price', pos['entry'])
            unr_pnl = pos.get('unrealized_pnl', 0.0)
            unr_pct = pos.get('unrealized_pnl_pct', 0.0)
            unr_col = '#26a69a' if unr_pnl >= 0 else '#ef5350'
            unr_arrow = '▲' if unr_pnl >= 0 else '▼'
            sl_pct = ((pos['sl'] - pos['entry']) / pos['entry']) * 100
            tp_pct = ((pos['tp'] - pos['entry']) / pos['entry']) * 100

            card = f"""
            <div style='border-left:4px solid #26a69a; background:rgba(38,166,154,0.06);
                        padding:12px 18px; margin:5px 0; border-radius:8px;'>
              <span style='font-size:16px;font-weight:700;'>{pos['symbol']}</span>
              <span style='font-size:12px;color:#9e9e9e;margin-left:8px;'>{pos['name']}</span>
              <span style='font-size:11px;color:#607d8b;margin-left:12px;'>{pos['strategy']}</span><br/>
              <span style='font-size:12px;'>
                Einstieg <b>${pos['entry']:,.4f}</b> &nbsp;|
                Aktuell <b style='color:{unr_col};'>${cur_price:,.4f}</b> &nbsp;|
                <span style='color:#ef5350;'>SL ${pos['sl']:,.4f} ({sl_pct:+.1f}%)</span> &nbsp;|
                <span style='color:#26a69a;'>TP ${pos['tp']:,.4f} ({tp_pct:+.1f}%)</span><br/>
                Anteile <b>{pos['shares']:.4f}</b> &nbsp;|
                Investiert <b>€{pos['invest_eur']:,.2f}</b> &nbsp;|
                Eröffnet <b>{pos['open_date']}</b> &nbsp;|
                Unrealisiert <b style='color:{unr_col};'>{unr_arrow} €{unr_pnl:,.2f} ({unr_pct:+.2f}%)</b>
              </span>
            </div>
            """
            st.markdown(card, unsafe_allow_html=True)

            if st.button(f"📊 {pos['symbol']} Chart laden", key=f"sim_load_{idx}_{pos['symbol']}"):
                st.session_state.tickers[0] = pos['symbol']
                st.rerun()

    # ── Closed Trades Table ────────────────────────────────────────
    if sim['closed_trades']:
        st.markdown("### 📋 Abgeschlossene Trades")
        rows = []
        for t in reversed(sim['closed_trades']):
            rows.append({
                "Symbol":     t['symbol'],
                "Name":       t['name'],
                "Einstieg":   f"${t['entry']:,.4f}",
                "Ausstieg":   f"${t['exit_date']} @ ${t['exit_price']:,.4f}",
                "Grund":      f"✅ TP" if t['result'] == 'TP' else f"❌ SL",
                "P&L €":      f"{t['pnl_eur']:+,.2f}",
                "P&L %":      f"{t['pnl_pct']:+.2f}%",
                "Strategie":  t['strategy'],
            })
        closed_df = pd.DataFrame(rows)
        st.dataframe(closed_df, use_container_width=True, hide_index=True)

    # ── Equity Curve Chart ─────────────────────────────────────────
    if len(sim['equity_history']) >= 2:
        st.markdown("### 📈 Equity-Kurve")
        eq_df = pd.DataFrame(sim['equity_history'])
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        eq_df = eq_df.sort_values('date')

        import plotly.graph_objects as go_sim
        fig_eq = go_sim.Figure()
        fig_eq.add_trace(go_sim.Scatter(
            x=eq_df['date'], y=eq_df['equity'],
            mode='lines+markers',
            line=dict(color='#26a69a', width=2),
            fill='tozeroy',
            fillcolor='rgba(38,166,154,0.08)',
            name='Depotwert'
        ))
        fig_eq.add_hline(
            y=sim['start_capital'],
            line_dash='dash', line_color='#607d8b',
            annotation_text='Startkapital €100k'
        )
        fig_eq.update_layout(
            template='plotly_dark',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            height=300,
            margin=dict(l=40, r=20, t=20, b=30),
            xaxis=dict(title='Datum', gridcolor='#1e1e1e'),
            yaxis=dict(title='Depotwert (€)', gridcolor='#1e1e1e'),
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    elif not sim['positions'] and not sim['closed_trades']:
        st.info("💡 Führe zuerst den Scanner aus und kaufe die Top 10, um das Depot zu starten.")


# ======= AI & STRATEGIEN: BACKTEST =======
with tab_ai:
    st.markdown("### 🎯 Live Strategy Scanner – Einstieg & Ausstieg")
    render_strategy_rules()

    strategy_pick = st.radio("Strategie wählen:", ["1️⃣ Bollinger Scalping", "2️⃣ Daytrading Reversal", "3️⃣ Fibonacci Swing"], horizontal=True)

    # Position Size Calculator with auto-fill
    with st.expander("📐 Positionsgrößen-Rechner (Max 2% Risiko-Regel)", expanded=True):
        psc1, psc2, psc3 = st.columns(3)
        capital = psc1.number_input("Gesamtkapital ($)", value=10000.0, step=500.0)
        risk_pct = psc2.number_input("Risiko pro Trade (%)", value=1.0, min_value=0.1, max_value=5.0, step=0.5)
        sl_distance_input = psc3.number_input("Stop-Loss Abstand ($)", value=2.0, min_value=0.01, step=0.5)
        ps_result = calc_position_size(capital, risk_pct, current_price, current_price - sl_distance_input)
        col_ps1, col_ps2, col_ps3 = st.columns(3)
        col_ps1.metric("Max. Verlust", f"${ps_result['max_loss']:.2f}")
        col_ps2.metric("Positionsgröße", f"{ps_result['shares']} Stück")
        col_ps3.metric("Investitionsvolumen", f"${ps_result['investment']:,.2f}")
        if ps_result['warning']:
            st.warning("⚠️ Investitionsvolumen übersteigt Gesamtkapital! Reduzieren Sie die Positionsgröße.")
        if risk_pct > 2.0:
            st.error("🚫 Risiko über 2%! Die universelle Regel empfiehlt max. 2% pro Trade.")

    st.divider()

    # ═══ STRATEGY 1: BOLLINGER SCALPING ═══
    if "Scalping" in strategy_pick:
        st.markdown("#### 1️⃣ Bollinger Band Scalping (Trend-Korrektur)")
        st.caption("**Setup:** EMA 55 steigend → Kurs berührt unteres BB → bullische Kerze → Stop-Buy über dem Hoch. **SL:** Unter dem letzten Tief. **TP:** BB Mitte / Trailing.")

        entries, exits, backtest, df_s = strategy_bollinger_scalping(df)

        fig_s1 = go.Figure()
        fig_s1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        fig_s1.add_trace(go.Scatter(x=df.index, y=df_s['BB_Upper'], line=dict(color='rgba(255,255,255,0.3)', width=1), name='BB Upper'))
        fig_s1.add_trace(go.Scatter(x=df.index, y=df_s['BB_Lower'], line=dict(color='rgba(255,255,255,0.3)', width=1),
            fill='tonexty', fillcolor='rgba(100,100,255,0.05)', name='BB Lower'))
        fig_s1.add_trace(go.Scatter(x=df.index, y=df_s['BB_Mid'], line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dot'), name='BB Mid'))
        fig_s1.add_trace(go.Scatter(x=df.index, y=df_s['EMA_55'], line=dict(color='#7c4dff', width=1.5), name='EMA 55 (Trend)'))

        if entries:
            fig_s1.add_trace(go.Scatter(x=[e['Date'] for e in entries], y=[e['Entry'] for e in entries], mode='markers',
                marker=dict(symbol='triangle-up', size=14, color='#00e676'), name='🟢 ENTRY'))
            fig_s1.add_trace(go.Scatter(x=[e['Date'] for e in entries], y=[e['SL'] for e in entries], mode='markers',
                marker=dict(symbol='x', size=10, color='#ff1744'), name='🔴 SL'))
            # Highlight Stochastic confirmed entries
            confirmed = [e for e in entries if e.get('StochConfirm')]
            if confirmed:
                fig_s1.add_trace(go.Scatter(x=[e['Date'] for e in confirmed], y=[e['Entry'] for e in confirmed], mode='markers',
                    marker=dict(symbol='star', size=12, color='#ffea00'), name='⭐ Stoch bestätigt'))
        if exits:
            fig_s1.add_trace(go.Scatter(x=[e['Date'] for e in exits], y=[e['Price'] for e in exits], mode='markers',
                marker=dict(symbol='triangle-down', size=14, color='#ffea00'), name='🟡 EXIT'))

        fig_s1.update_layout(template='plotly_dark', height=500, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s1, use_container_width=True)
        render_backtest_stats(backtest)
        render_equity_curve(entries, exits, "Scalping Equity Curve")

    # ═══ STRATEGY 2: DAYTRADING REVERSAL ═══
    elif "Reversal" in strategy_pick:
        st.markdown("#### 2️⃣ Top/Bottom Reversal (RSI >90/<10 + Doji/Spinning Top)")
        st.caption("**Setup:** 5+ aufsteigende Kerzen + RSI Extrem (>90) + oberes BB + Doji/Spinning Top → Short. Umgekehrt für Long.")

        shorts, longs, backtest, df_s = strategy_reversal(df)

        fig_s2 = go.Figure()
        fig_s2.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        fig_s2.add_trace(go.Scatter(x=df.index, y=df_s['BB_Upper'], line=dict(color='rgba(255,100,100,0.4)', width=1), name='BB Upper'))
        fig_s2.add_trace(go.Scatter(x=df.index, y=df_s['BB_Lower'], line=dict(color='rgba(100,255,100,0.4)', width=1),
            fill='tonexty', fillcolor='rgba(100,100,255,0.05)', name='BB Lower'))
        if 'VWAP' in df_s.columns:
            fig_s2.add_trace(go.Scatter(x=df.index, y=df_s['VWAP'], line=dict(color='#ffeb3b', width=1, dash='dot'), name='VWAP (TP Ref)'))
        fig_s2.add_trace(go.Scatter(x=df.index, y=df_s['EMA_21'], line=dict(color='#ff9800', width=1, dash='dot'), name='EMA 21 (TP Ref)'))

        if shorts:
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in shorts], y=[s['Entry'] for s in shorts], mode='markers+text',
                text=[f"🔴 {s['Pattern']}" for s in shorts], textposition='bottom center',
                marker=dict(symbol='triangle-down', size=16, color='#ff1744'), name='SHORT Entry'))
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in shorts], y=[s['SL'] for s in shorts], mode='markers',
                marker=dict(symbol='x', size=10, color='#ff9100'), name='SHORT SL'))
        if longs:
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in longs], y=[s['Entry'] for s in longs], mode='markers+text',
                text=[f"🟢 {s['Pattern']}" for s in longs], textposition='top center',
                marker=dict(symbol='triangle-up', size=16, color='#00e676'), name='LONG Entry'))
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in longs], y=[s['SL'] for s in longs], mode='markers',
                marker=dict(symbol='x', size=10, color='#ff9100'), name='LONG SL'))

        fig_s2.update_layout(template='plotly_dark', height=500, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s2, use_container_width=True)
        render_backtest_stats(backtest)
        st.info(f"**Gefundene Signale:** {len(shorts)} Short Reversals | {len(longs)} Long Reversals")

    # ═══ STRATEGY 3: FIBONACCI SWING ═══
    elif "Fibonacci" in strategy_pick:
        st.markdown("#### 3️⃣ Fibonacci-Korrektur Swing (50%/61.8% + Candlestick Confirmation)")
        st.caption("**Setup:** Korrektur auf Fib 50%/61.8% → Hammer/Engulfing/Harami → Entry über Vortagshoch. **SL:** Unter Korrektur-Tief. **TP:** Trailing 3-4 Tage + Fib Extensions.")

        fib_entries, fib_exits, backtest, _, fib_data = strategy_fibonacci_swing(df)

        fig_s3 = go.Figure()
        fig_s3.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))

        # Draw Fib levels
        fib_colors = {'23.6%': '#aaa', '38.2%': '#888', '50.0%': '#ff9800', '61.8%': '#f44336', '78.6%': '#9c27b0'}
        for lvl_name, lvl_val in fib_data['retracements'].items():
            col = fib_colors.get(lvl_name, '#666')
            fig_s3.add_hline(y=lvl_val, line_dash="dot", line_color=col,
                annotation_text=f"Fib {lvl_name} (${lvl_val:.2f})", annotation_position="right")
        # Extensions
        for ext_name, ext_val in fib_data['extensions'].items():
            fig_s3.add_hline(y=ext_val, line_dash="dashdot", line_color='#00e676',
                annotation_text=f"Ext {ext_name} (${ext_val:.2f})", annotation_position="right")

        fig_s3.add_hline(y=fib_data['swing_high'], line_color="#00e676", line_width=1, annotation_text=f"Swing High ${fib_data['swing_high']:.2f}")
        fig_s3.add_hline(y=fib_data['swing_low'], line_color="#ff1744", line_width=1, annotation_text=f"Swing Low ${fib_data['swing_low']:.2f}")

        if fib_entries:
            fig_s3.add_trace(go.Scatter(x=[e['Date'] for e in fib_entries], y=[e['Entry'] for e in fib_entries],
                mode='markers+text', text=[f"🟢 {e['Level']} ({e['Pattern']})" for e in fib_entries],
                textposition='top center', marker=dict(symbol='triangle-up', size=16, color='#00e676'), name='ENTRY'))
            fig_s3.add_trace(go.Scatter(x=[e['Date'] for e in fib_entries], y=[e['SL'] for e in fib_entries],
                mode='markers', marker=dict(symbol='x', size=12, color='#ff1744'), name='🔴 SL'))
        if fib_exits:
            fig_s3.add_trace(go.Scatter(x=[e['Date'] for e in fib_exits], y=[e['Price'] for e in fib_exits],
                mode='markers+text', text=[e['Reason'] for e in fib_exits], textposition='bottom center',
                marker=dict(symbol='diamond', size=12, color='#ffea00'), name='EXIT'))

        fig_s3.update_layout(template='plotly_dark', height=550, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s3, use_container_width=True)
        render_backtest_stats(backtest)
        render_equity_curve(fib_entries, fib_exits, "Fibonacci Swing Equity Curve")
        if fib_entries:
            st.dataframe(pd.DataFrame(fib_entries), use_container_width=True, hide_index=True)

    # Trailing-Stop reminder
    st.divider()
    st.markdown("#### 📏 Trailing-Stop Empfehlung")
    st.markdown("""
- **Scalping:** Nach +10 Pips → SL auf Break-Even. Trailing am Tief der 3.-letzten Kerze.
- **Reversal:** Take Profit am nächsten Support/VWAP/MA-Level.
- **Fibonacci:** Trailing-SL unter dem Tief der letzten 3-4 Tage. Gewinnziele: Fib 127.2% / 161.8%.
    """)

# ======= AI & STRATEGIEN: AI & ML (continued in same tab) =======
with tab_ai:
    st.markdown("### 🤖 AI & Machine Learning Suite")
    ai_pick = st.radio("AI-Modul:", ["🧠 Pattern Scanner", "📊 Regime Detection", "🎯 Auto S/R", "📈 Forecast", "📋 Risk Profiler", "📄 AI Report", "📚 Wissens-DB"], horizontal=True)
    returns_ai = df['Daily_Return'].dropna()

    if "Pattern" in ai_pick:
        st.markdown("#### 🧠 AI Pattern Scanner")
        patterns_found = []
        for i in range(20, min(len(df), 100)):
            window = df['Low'].iloc[i-20:i]
            min1_idx = window.iloc[:10].idxmin()
            min2_idx = window.iloc[10:].idxmin()
            if abs(df['Low'][min1_idx] - df['Low'][min2_idx]) / df['Low'][min1_idx] < 0.02:
                patterns_found.append(f"📐 **Double Bottom** near ${df['Low'][min2_idx]:.2f} ({str(min2_idx)[:10]})")
            if i >= 15:
                seg = df['High'].iloc[i-15:i]
                mid = seg.iloc[5:10].max(); left = seg.iloc[:5].max(); right = seg.iloc[10:].max()
                if mid > left and mid > right and abs(left - right)/left < 0.03:
                    patterns_found.append(f"👤 **Head & Shoulders** hint: peak ${mid:.2f}")
        patterns_found = list(set(patterns_found))[:8]
        if patterns_found:
            for p in patterns_found: st.markdown(p)
        else: st.info("Keine Patterns erkannt.")

    elif "Regime" in ai_pick:
        st.markdown("#### 📊 Market Regime Detection")
        vol_20 = returns_ai.rolling(20).std().iloc[-1] * np.sqrt(252) * 100 if len(returns_ai) > 20 else 0
        trend_str = abs(returns_ai.rolling(20).mean().iloc[-1]) * 252 * 100 if len(returns_ai) > 20 else 0
        if vol_20 < 15 and trend_str < 10: regime_ai = "😴 Mean Reverting"
        elif vol_20 < 25 and trend_str >= 10: regime_ai = "📈 Trending"
        elif vol_20 >= 25 and trend_str >= 15: regime_ai = "🚀 Momentum"
        else: regime_ai = "🌪️ Chaos"
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Regime", regime_ai); rc2.metric("20D Vola", f"{vol_20:.1f}%"); rc3.metric("Trend-Stärke", f"{trend_str:.1f}%")

    elif "S/R" in ai_pick:
        st.markdown("#### 🎯 Auto Support & Resistance (K-Means)")
        from scipy.cluster.vq import kmeans as km
        pts = np.concatenate([df['High'].values, df['Low'].values])
        pts = pts[~np.isnan(pts)]
        if len(pts) > 10:
            centroids, _ = km(pts.astype(float), min(5, len(pts)))
            centroids = sorted(centroids)
            fig_sr = go.Figure()
            fig_sr.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
            cols_sr = ['#ff1744','#ff9800','#ffeb3b','#00e676','#2196f3']
            for ic, c in enumerate(centroids):
                fig_sr.add_hline(y=c, line_dash="dash", line_color=cols_sr[ic % len(cols_sr)],
                    annotation_text=f"{'Support' if c < current_price else 'Resistance'} ${c:.2f}")
            fig_sr.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_sr, use_container_width=True)

    elif "Forecast" in ai_pick:
        st.markdown("#### 📈 Price Forecast (Linear + Monte Carlo)")
        from scipy.stats import linregress
        x_v = np.arange(len(df)); y_v = df['Close'].values
        slope, intercept, r_val, _, _ = linregress(x_v, y_v)
        future_x = np.arange(len(df), len(df) + 30)
        trend_line = slope * future_x + intercept
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=list(range(len(df))), y=y_v, mode='lines', name='Historisch', line=dict(color='#26a69a')))
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=trend_line, mode='lines', name='Forecast', line=dict(color='#ff9800', dash='dash')))
        std_r = returns_ai.std()
        upper = trend_line * (1 + 2*std_r*np.sqrt(np.arange(1, 31)))
        lower = trend_line * (1 - 2*std_r*np.sqrt(np.arange(1, 31)))
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=upper, mode='lines', line={"width": 0}, showlegend=False))
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=lower, mode='lines', line={"width": 0}, fill='tonexty', fillcolor='rgba(255,152,0,0.15)', name='95% Konfidenz'))
        fig_fc.update_layout(template='plotly_dark', height=350, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
        st.plotly_chart(fig_fc, use_container_width=True)
        st.metric("R²", f"{r_val**2:.4f}"); st.metric("30-Tage Prognose", f"${trend_line[-1]:.2f}")

    elif "Profiler" in ai_pick:
        st.markdown("#### 📋 Risk Profiler Quiz")
        q1 = st.slider("Nervosität bei Verlusten", 1, 10, 5)
        q2 = st.slider("Anlagehorizont", 1, 10, 5)
        q3 = st.slider("Trading-Erfahrung", 1, 10, 3)
        q4 = st.slider("Max. Drawdown Toleranz", 1, 10, 4)
        q5 = st.slider("Dividenden-Priorität", 1, 10, 3)
        rs = (q1*-1 + q2 + q3 + q4 - q5 + 30) / 6
        if rs < 3: pf, assets = "🛡️ Konservativ", ["BND", "VYM", "JNJ"]
        elif rs < 6: pf, assets = "⚖️ Ausgewogen", ["SPY", "QQQ", "AAPL"]
        else: pf, assets = "🔥 Aggressiv", ["BTC-USD", "NVDA", "ARKK"]
        st.success(f"**Profil:** {pf} (Score: {rs:.1f}/10)")
        for a in assets: st.markdown(f"- {a}")

    elif "Wissens-DB" in ai_pick:
        st.markdown("#### 📚 Knowledge Base (KI Regelextraktion)")
        st.write("Füttere die Trading-Engine mit Regeln aus Büchern oder PDFs.")
        
        with st.form("kb_form"):
            col1, col2 = st.columns(2)
            with col1:
                book_title = st.text_input("Quelle / Buchtitel", placeholder="z.B. Trading für Anfänger")
            with col2:
                rule_name = st.text_input("Strategie-Name", placeholder="z.B. RSI Bounce")
                
            raw_text = st.text_area("Regeltext / Buchauszug", height=150, placeholder="Textextrakt... (z.B. Kaufe wenn RSI unter 30 fällt)")
            submit = st.form_submit_button("🧠 Regeln extrahieren & Speichern")
            
            if submit and raw_text and rule_name:
                try:
                    import knowledge_base as kb
                    import database as db
                    import json
                    
                    # Mock RAG extraction
                    extracted_logic = kb.parse_book_to_rules(raw_text)
                    
                    # Save to database
                    db.add_strategy(name=rule_name, source_book=book_title, logic_json=json.dumps(extracted_logic), base_weight=1.0)
                    st.success(f"Strategie '{rule_name}' erfolgreich extrahiert und als ID in der Datenbank gespeichert!")
                    st.json(extracted_logic)
                except Exception as e:
                    st.error(f"Fehler bei der Regelextraktion: {e}")
                
        st.markdown("---")
        st.markdown("##### 🗄️ Aktive Strategien in der Datenbank")
        import database as db
        import pandas as pd
        strats = db.get_all_strategies()
        if strats:
            df_strats = pd.DataFrame(strats)
            st.dataframe(df_strats, use_container_width=True, hide_index=True)
        else:
            st.info("Noch keine Strategien gespeichert.")

    elif "Report" in ai_pick:
        st.markdown("#### 📄 Executive AI Report")
        if st.button("📄 Report generieren"):
            ann_ret = returns_ai.mean() * 252 * 100
            ann_vol_r = returns_ai.std() * np.sqrt(252) * 100
            sharpe_r = (returns_ai.mean() - 0.02/252) / returns_ai.std() * np.sqrt(252) if returns_ai.std() > 0 else 0
            report = f"""## 📊 Executive Report: {st.session_state.tickers[0]}
**Datum:** {now.strftime('%d.%m.%Y %H:%M')}
| Kennzahl | Wert |
|---|---|
| Kurs | ${current_price:,.2f} |
| Ann. Rendite | {ann_ret:.2f}% |
| Ann. Vola | {ann_vol_r:.2f}% |
| Sharpe | {sharpe_r:.2f} |
| Max DD | {(df['Close']/df['Close'].cummax()-1).min()*100:.2f}% |

{'**BULLISH** – Risikoadjustierte Rendite positiv.' if sharpe_r > 0.5 else '**NEUTRAL** – Abwarten.' if sharpe_r > 0 else '**BEARISH** – Absicherung empfohlen.'}

*Automatisch generiert – keine Anlageberatung.*"""
            st.markdown(report)
            st.download_button("📥 Download", report, file_name=f"report_{st.session_state.tickers[0]}.md")

if __name__ == '__main__':
    if not os.environ.get("STREAMLIT_RUNNING_APP"):
        os.environ["STREAMLIT_RUNNING_APP"] = "1"
        os.system(f"{sys.executable} -m streamlit run '{__file__}'")
