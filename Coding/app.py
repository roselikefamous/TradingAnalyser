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

from trading_terminal.indicators import (
    apply_core_indicators, calc_heikin_ashi, calc_ichimoku, calc_parabolic_sar,
    calc_supertrend, calc_volume_profile, auto_fibonacci, calc_bollinger_bands,
    calc_ema, calc_sma, find_swing_points
)
from trading_terminal.ui import (
    TERMINAL_CSS,
    render_ai_ml_suite_section,
    render_ai_strategy_backtest_section,
    render_home_tab,
    render_signal_card,
    render_market_signals_section,
    render_market_scanner_simulation_section,
    INDICATOR_GUIDE,
)
from trading_terminal.data import fetch_market_data, apply_indicators
from trading_terminal.scanner import scan_all_assets_df
from trading_terminal.data_qa.feed_monitor import check_feed_quality
from streamlit_autorefresh import st_autorefresh
from simulation import (
    new_simulation_state
)
from trading_terminal.utils.logger import structured_logger, log_error, log_trade_event

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
        _cur = yf.Ticker(_sym).fast_info.get('lastPrice')
        if _cur:
            if _cur <= _pos['sl']:
                st.toast(f"🛑 SL getriggert: {_sym} fiel auf ${_cur:.2f} (SL war ${_pos['sl']:.2f})", icon="🔴")
                st.session_state.sltp_alerted.add(_alert_key)
            elif _cur >= _pos['tp']:
                st.toast(f"🎯 TP erreicht! {_sym} stieg auf ${_cur:.2f} (TP war ${_pos['tp']:.2f})", icon="🟢")
                st.session_state.sltp_alerted.add(_alert_key)
    except Exception as e:
        log_error("ALERT_FAILURE", f"Failed to check SL/TP for {_sym}: {str(e)}")

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
        scan_df = scan_all_assets_df(progress_callback=update_progress)

    progress_bar.progress(1.0, text="✅ Scan abgeschlossen!")
    st.session_state['scanner_results'] = scan_df
    st.session_state['scanner_running'] = False
    import time; time.sleep(1) # let user see 100%
    _prog_container.empty()

# ================== SIDEBAR ==================
with st.sidebar:
    st.title("⚙️ Terminal Controls")

    st.markdown("### 🧭 Navigation")
    active_page = st.radio("Gehe zu:", [
        "🏠 Dashboard & Scanner",
        "📈 Chart & Setup",
        "💼 Portfolio"
    ], label_visibility="collapsed")
    st.divider()

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
market_data = fetch_market_data(st.session_state.tickers[0], period, interval)
if market_data is None or market_data.df is None or len(market_data.df) == 0:
    st.error(f"❌ Keine Daten für das Symbol '{st.session_state.tickers[0]}' gefunden. Bitte überprüfe die Schreibweise.")
    if st.button("🔄 Zurück setzen (Reset)", type="primary"):
        st.session_state.tickers[0] = "AAPL"
        # We don't modify custom_ticker_input or other widget keys directly
        st.rerun()
    st.stop()

df = market_data.df
info = market_data.info
news = market_data.news
dividends = market_data.dividends
earnings_dates = market_data.earnings_dates

# ================== MLOps & QA CHECKS ==================
is_clean, qa_issues = check_feed_quality(df)
if not is_clean:
    st.toast("⚠️ MLOps Warnung: Mögliche Fehleinspeisung/Anomalie im Datenfeed erkannt.", icon="⚠️")
    with st.expander("🚨 Feed Quality Issues Detected", expanded=True):
        for issue in qa_issues:
            st.error(issue)

# ================== APPLY INDICATORS ==================
df = apply_indicators(df, ema1_len, ema2_len, rsi_len)
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


# ======= TAB HOME: DASHBOARD =======
if active_page == "🏠 Dashboard & Scanner":
    render_home_tab(now, market_status)

# ======= MÄRKTE: SIGNAL DASHBOARD =======
if active_page == "🏠 Dashboard & Scanner":
    render_market_signals_section(now)


# ======= CHART & ANALYSE =======
if active_page == "📈 Chart & Setup":
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
                line={"color": '#26a69a', "width": 2}, name='Close'), row=1, col=1)
        elif chart_type == "Area":
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], mode='lines',
                line={"color": '#26a69a', "width": 2}, fill='tozeroy',
                fillcolor='rgba(38,166,154,0.15)', name='Close'), row=1, col=1)

        # EMA Overlays
        if show_ema:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_1'], line={"color": ema1_col, "width": 1.5}, name=f'EMA {ema1_len}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_2'], line={"color": ema2_col, "width": 1.5}, name=f'EMA {ema2_len}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_55'], line={"color": '#7c4dff', "width": 1, "dash": 'dot'}, name='EMA 55'), row=1, col=1)

        # SMA Overlays
        if show_sma:
            if 'SMA_50' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line={"color": '#ff9800', "width": 1.5, "dash": 'dash'}, name='SMA 50'), row=1, col=1)
            if 'SMA_200' in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line={"color": '#f44336', "width": 1.5, "dash": 'dash'}, name='SMA 200'), row=1, col=1)

        # VWAP
        if show_vwap and 'VWAP' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line={"color": '#ffeb3b', "width": 1.5}, name='VWAP'), row=1, col=1)

        # Ichimoku Cloud
        if show_ichimoku:
            tenkan, kijun, senkou_a, senkou_b, chikou = calc_ichimoku(df)
            fig.add_trace(go.Scatter(x=df.index, y=tenkan, line={"color": '#2196f3', "width": 1}, name='Tenkan'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=kijun, line={"color": '#f44336', "width": 1}, name='Kijun'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=senkou_a, line=dict(color='rgba(0,230,118,0.4)', width=0.5), name='Senkou A'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=senkou_b, line=dict(color='rgba(239,83,80,0.4)', width=0.5),
                fill='tonexty', fillcolor='rgba(38,166,154,0.08)', name='Senkou B'), row=1, col=1)

        # Parabolic SAR
        if show_sar:
            sar = calc_parabolic_sar(df)
            bull_sar = sar.where(sar < df['Close'])
            bear_sar = sar.where(sar >= df['Close'])
            fig.add_trace(go.Scatter(x=df.index, y=bull_sar, mode='markers', marker={"size": 3, "color": '#00e676'}, name='SAR Bull'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bear_sar, mode='markers', marker={"size": 3, "color": '#ff1744'}, name='SAR Bear'), row=1, col=1)

        # SuperTrend
        if show_supertrend:
            st_line, st_dir = calc_supertrend(df)
            bull_st = st_line.where(st_dir == 1)
            bear_st = st_line.where(st_dir == -1)
            fig.add_trace(go.Scatter(x=df.index, y=bull_st, line={"color": '#00e676', "width": 2}, name='SuperTrend ↑'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=bear_st, line={"color": '#ff1744', "width": 2}, name='SuperTrend ↓'), row=1, col=1)

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
                if not comp_df.empty:
                    comp_scaled = comp_df['Close'] * (plot_df['Close'].iloc[0] / comp_df['Close'].iloc[0])
                    fig.add_trace(go.Scatter(x=comp_df.index, y=comp_scaled, mode='lines',
                        line={"color": 'yellow', "width": 1.5}, name=compare_asset), row=1, col=1)
            except Exception:
                pass

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
            fig.add_trace(go.Scatter(x=df.index, y=pp, line={"color": '#ab47bc', "width": 1, "dash": 'dot'}, name='Pivot'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=r1, line={"color": '#ef5350', "width": 1, "dash": 'dot'}, name='R1'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=s1, line={"color": '#26a69a', "width": 1, "dash": 'dot'}, name='S1'), row=1, col=1)

        # Candlestick Pattern Markers
        doji_idx = df.index[df['Doji']]
        hammer_idx = df.index[df['Hammer']]
        engulf_idx = df.index[df['Bullish_Engulfing']]
        if len(doji_idx) > 0:
            fig.add_trace(go.Scatter(x=doji_idx, y=df.loc[doji_idx, 'Close'], mode='markers',
                marker=dict(symbol='diamond', size=7, color='yellow'), name='Doji'), row=1, col=1)
        if len(hammer_idx) > 0:
            fig.add_trace(go.Scatter(x=hammer_idx, y=df.loc[hammer_idx, 'Low'] * 0.99, mode='text',
                text='🔨', textfont={'size': 12}, name='Hammer'), row=1, col=1)
        if len(engulf_idx) > 0:
            fig.add_trace(go.Scatter(x=engulf_idx, y=df.loc[engulf_idx, 'High'] * 1.005, mode='text',
                text='🟢', textfont={'size': 10}, name='Engulfing'), row=1, col=1)

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
            fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], line={"color": '#ffeb3b', "width": 1.5}, name='OBV'), row=current_row, col=1)
            current_row += 1
        if "RSI" in subplots:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line={"color": '#ab47bc', "width": 1.5}, name='RSI'), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.1)", row=current_row, col=1)
            current_row += 1
        if "MACD" in subplots:
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'],
                marker_color=['#26a69a' if v > 0 else '#ef5350' for v in df['MACD_Hist']], name='Histogram'), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line={"color": '#2962ff', "width": 1.5}, name='MACD'), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line={"color": '#ff6d00', "width": 1}, name='Signal'), row=current_row, col=1)
            current_row += 1
        if "Stoch" in subplots:
            fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line={"color": '#2196f3', "width": 1.5}, name='%K'), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line={"color": '#ff9800', "width": 1}, name='%D'), row=current_row, col=1)
            fig.add_hline(y=80, line_dash="dot", line_color="gray", row=current_row, col=1)
            fig.add_hline(y=20, line_dash="dot", line_color="gray", row=current_row, col=1)
            current_row += 1
        if "ATR" in subplots:
            fig.add_trace(go.Scatter(x=df.index, y=df['ATR'], line={"color": '#ff9800', "width": 1.5}, name='ATR'), row=current_row, col=1)
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
                _bc_color = '#ff1744'; _b_icon = '🔴'; _b_action = 'VERKAUFEN (SHORT)'
                _b_msg = f'SHORT bei ${_banner_entry:,.2f} • SL ${_banner_sl:,.2f} • TP ${_banner_tp:,.2f} • {_pos_shares} Anteile • Risiko €{_risk2:.0f}'
            else:
                _bc_color = '#ffea00'; _b_icon = '🟡'; _b_action = 'ABWARTEN'
                _b_msg = f'Kein klares Signal. Warte auf Score ≥3 oder ≤-3. RSI: {_banner_rsi:.0f}'
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
            from strategy_engine import calculate_smart_entry_zone
            
            ez_data = calculate_smart_entry_zone(df)
            
            score = ez_data['score']
            max_score = ez_data['max_score']
            score_label = ez_data['score_label']
            direction = ez_data['direction']
            d_emoji = ez_data['d_emoji']
            d_color = ez_data['d_color']
            regime = ez_data['regime']
            confidence = ez_data['confidence']
            entry_low = ez_data['entry_low']
            entry_high = ez_data['entry_high']
            sl_level = ez_data['sl_level']
            target_1 = ez_data['target_1']
            target_2 = ez_data['target_2']
            target_3 = ez_data['target_3']
            last_price = ez_data['last_price']
            key_levels = ez_data['key_levels']
            signals = ez_data['signals']
            last_adx = ez_data['last_adx']

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
            fig_ez.add_trace(go.Scatter(x=df_w.index, y=df_w['EMA_1'], line={"color": ema1_col, "width": 1.2}, name=f'EMA {ema1_len}'))
            fig_ez.add_trace(go.Scatter(x=df_w.index, y=df_w['EMA_2'], line={"color": ema2_col, "width": 1.2}, name=f'EMA {ema2_len}'))
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
                showlegend=True, legend=dict(orientation="h", y=-0.05, font={'size': 10}),
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
            
            ann_factor = 365 if "-USD" in st.session_state.tickers[0] else 252
            rf = 0.02 / ann_factor
            downside = returns[returns < 0]
            sortino = (mean_ret - rf) / downside.std() if len(downside) > 0 else 0
            sharpe = (mean_ret - rf) / std_dev if std_dev > 0 else 0
            running_max = df['Close'].cummax()
            drawdown = (df['Close'] - running_max) / running_max
            max_dd = drawdown.min()
            calmar = (mean_ret * ann_factor) / abs(max_dd) if max_dd != 0 else 0
            win_prob = len(returns[returns > 0]) / len(returns)
            avg_win = returns[returns > 0].mean()
            avg_loss = abs(returns[returns < 0].mean())
            wl_ratio = avg_win / avg_loss if avg_loss != 0 else 1
            kelly = win_prob - ((1 - win_prob) / wl_ratio) if wl_ratio > 0 else 0
            
            c1.metric("Value at Risk (95%)", f"{var_95*100:.2f}%")
            c2.metric("CVaR (Exp. Shortfall)", f"{cvar_95*100:.2f}%")
            c3.metric("Sortino Ratio", f"{sortino * np.sqrt(ann_factor):.2f}")
            c4.metric("Calmar Ratio", f"{calmar:.2f}")

            st.markdown("---")
            r1c, r2c = st.columns([2, 1], gap="large")
            with r1c:
                with st.container(border=True):
                    st.markdown("#### 📉 Max Drawdown (Underwater Chart)")
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown*100, fill='tozeroy', mode='none', fillcolor='rgba(239,83,80,0.5)'))
                    fig_dd.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=20), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(title="Drawdown %", gridcolor='#1e1e1e'), xaxis=dict(gridcolor='#1e1e1e'))
                    st.plotly_chart(fig_dd, use_container_width=True)
                
                with st.container(border=True):
                    st.markdown("#### 🌪️ 30-Day Rolling Volatility")
                    roll_vol = returns.rolling(30).std() * np.sqrt(ann_factor) * 100
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, line=dict(color='#ff9800', width=2)))
                    fig_vol.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=20), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(title="Volatility %", gridcolor='#1e1e1e'), xaxis=dict(gridcolor='#1e1e1e'))
                    st.plotly_chart(fig_vol, use_container_width=True)

                with st.container(border=True):
                    st.markdown("#### 🎲 Monte Carlo Price Simulation (30 Days)")
                    if st.button("🚀 Run 100 Simulations", use_container_width=True):
                        mc_fig = go.Figure()
                        lp = df['Close'].iloc[-1]
                        for _ in range(100):
                            shocks = np.random.normal(loc=(mean_ret - 0.5*std_dev**2), scale=std_dev, size=30)
                            price_path = lp * np.exp(np.cumsum(shocks))
                            mc_fig.add_trace(go.Scatter(y=[lp]+list(price_path), mode='lines', line={"width": 1, "color": 'rgba(38,166,154,0.1)'}))
                        mc_fig.update_layout(height=280, showlegend=False, margin=dict(l=0,r=0,t=10,b=0), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(gridcolor='#1e1e1e'), xaxis=dict(gridcolor='#1e1e1e'))
                        st.plotly_chart(mc_fig, use_container_width=True)
            with r2c:
                with st.container(border=True):
                    st.markdown("#### ⚡ Annualisierte Volatilität")
                    ann_vol = std_dev * np.sqrt(ann_factor) * 100
                    gc = "green" if ann_vol < 15 else "yellow" if ann_vol < 30 else "red"
                    fig_gauge = go.Figure(go.Indicator(mode="gauge+number", value=ann_vol, number={'suffix': "%"},
                        gauge={'axis': {'range': [None, max(100, ann_vol*1.2)]}, 'bar': {'color': gc},
                               'steps': [{'range': [0,15], 'color': 'rgba(0,255,0,0.1)'}, {'range': [15,30], 'color': 'rgba(255,255,0,0.1)'}, {'range': [30,max(100, ann_vol*1.2)], 'color': 'rgba(255,0,0,0.1)'}]}))
                    fig_gauge.update_layout(height=220, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with st.container(border=True):
                    st.markdown("#### 💡 Risk Insights")
                    st.info(f"**Sharpe Ratio (ann.):** {sharpe * np.sqrt(ann_factor):.2f}\n\n*(Ein Wert > 1.0 ist gut)*")
                    st.success(f"**Kelly Criterion:** Optimal {max(0, kelly)*100:.2f}% des Portfolios riskieren.")
                    st.warning(f"**Win Rate:** {win_prob*100:.1f}%\n\n**W/L Ratio:** {wl_ratio:.2f}")

# ======= PORTFOLIO =======
if active_page == "💼 Portfolio":
    st.markdown("### 💼 Portfolio Management & Analytics")
    p1, p2 = st.columns([2, 1], gap="large")
    with p1:
        st.markdown("#### 🟢 Offene Positionen")
        port_df = st.session_state.portfolio.copy()
        if not port_df.empty:
            port_df["CurrentPrice"] = port_df["EntryPrice"] * 1.05
            port_df["PnL %"] = ((port_df["CurrentPrice"] - port_df["EntryPrice"]) / port_df["EntryPrice"]) * 100
            port_df["TotalValue"] = port_df["CurrentPrice"] * port_df["Shares"]
            st.dataframe(port_df, use_container_width=True, hide_index=True)
        else:
            st.info("Keine manuellen Positionen im Depot.")

        st.markdown("---")
        st.markdown("#### 📅 10-Jahres Saisonalität")
        try:
            hist_10y = yf.Ticker(st.session_state.tickers[0]).history(period="10y", interval="1mo")
            hist_10y['Month'] = hist_10y.index.month
            hist_10y['Ret'] = hist_10y['Close'].pct_change() * 100
            seasonality = hist_10y.groupby('Month')['Ret'].mean()
            fig_season = go.Figure(go.Bar(
                x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                y=seasonality,
                marker_color=['#26a69a' if v > 0 else '#ef5350' for v in seasonality],
                opacity=0.85
            ))
            fig_season.update_layout(
                height=250, margin=dict(l=0,r=0,t=10,b=20),
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Ø Rendite (%)", xaxis_title="Monat"
            )
            st.plotly_chart(fig_season, use_container_width=True)
        except Exception:
            st.warning("⚠️ Saisonalitätsdaten für dieses Asset aktuell nicht verfügbar.")

    with p2:
        with st.container(border=True):
            st.markdown("#### 🥧 Asset Allocation")
            if "TotalValue" in port_df.columns and not port_df.empty:
                fig_pie = go.Figure(go.Pie(
                    labels=port_df["Symbol"], values=port_df["TotalValue"],
                    hole=0.6, marker=dict(colors=['#26a69a', '#29b6f6', '#ab47bc', '#ffca28', '#ef5350'])
                ))
                fig_pie.update_layout(
                    height=220, margin=dict(l=0,r=0,t=0,b=0),
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.caption("Keine Daten für Allocation-Chart.")

        with st.container(border=True):
            st.markdown("#### 📝 Manueller Trade")
            cA, cB = st.columns(2)
            def add_manual_trade():
                sym = st.session_state.trade_sym
                shares = st.session_state.trade_shares_input
                price = st.session_state.trade_price_input
                new_pos = pd.DataFrame([{"Symbol": sym, "Shares": shares, "EntryPrice": price}])
                import datetime
                db.add_position({
                    'symbol': sym, 'direction': 'BUY',
                    'entry_price': price, 'sl': price * 0.95,
                    'tp': price * 1.10, 'shares': shares,
                    'open_date': datetime.datetime.now().isoformat()
                })
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_pos], ignore_index=True)

            trade_sym = cA.selectbox("Asset", st.session_state.watchlist, key="trade_sym")
            trade_shares = cB.number_input("Shares", min_value=1, value=10, key="trade_shares_input")
            trade_price = st.number_input("Einstiegspreis ($)", value=float(current_price), key="trade_price_input")
            st.button("🛒 Position Hinzufügen", use_container_width=True, type="primary", on_click=add_manual_trade)

        with st.expander("⚙️ Erweiterte Tools", expanded=True):
            st.radio("Base Currency", ["USD", "EUR", "GBP"], horizontal=True, key="base_curr")
            st.dataframe({"Sector": ["Tech", "Energy", "Health"], "Perf": ["+1.2%", "-0.5%", "+0.3%"]}, use_container_width=True)
            st.markdown("**Beispiel-Holdings:**")
            st.progress(0.12, text="AAPL (12%)")
            st.progress(0.10, text="MSFT (10%)")
            st.progress(0.08, text="NVDA (8%)")

# ======= MÄRKTE: SCANNER (secondary section inside tab_markt) =======
if active_page == "🏠 Dashboard & Scanner":
    render_market_scanner_simulation_section(now)

# ======= AI & STRATEGIEN: BACKTEST =======
if active_page == "📈 Chart & Setup":
    render_ai_strategy_backtest_section(df, current_price)

# ======= AI & STRATEGIEN: AI & ML (continued in same tab) =======
if active_page == "📈 Chart & Setup":
    render_ai_ml_suite_section(df, current_price, now)

if __name__ == '__main__':
    if not os.environ.get("STREAMLIT_RUNNING_APP"):
        os.environ["STREAMLIT_RUNNING_APP"] = "1"
        os.system(f"{sys.executable} -m streamlit run '{__file__}'")
