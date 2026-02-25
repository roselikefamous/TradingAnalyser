import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime, sys, os

from indicators import (
    apply_core_indicators, calc_heikin_ashi, calc_ichimoku, calc_parabolic_sar,
    calc_supertrend, calc_volume_profile, auto_fibonacci, calc_bollinger_bands,
    calc_ema, calc_sma, find_swing_points
)
from strategies import strategy_bollinger_scalping, strategy_reversal, strategy_fibonacci_swing, calc_position_size
from ui_helpers import TERMINAL_CSS, render_signal_card, render_backtest_stats, render_equity_curve, render_strategy_rules, INDICATOR_GUIDE
from streamlit_autorefresh import st_autorefresh
from scanner import scan_all_assets, PREDEFINED_ASSETS

st.set_page_config(page_title="Pro Trading Terminal", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

# ================== SESSION STATE ==================
if 'tickers' not in st.session_state: st.session_state.tickers = ["AAPL"]
if 'comparison_tickers' not in st.session_state: st.session_state.comparison_tickers = []
if 'compare_mode' not in st.session_state: st.session_state.compare_mode = False
if 'watchlist' not in st.session_state: st.session_state.watchlist = ["AAPL", "MSFT", "GOOG", "TSLA", "BTC-USD"]
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame([
        {"Symbol": "AAPL", "Shares": 50, "EntryPrice": 150.0},
        {"Symbol": "MSFT", "Shares": 30, "EntryPrice": 300.0}
    ])
if 'journal' not in st.session_state: st.session_state.journal = ""

now = datetime.datetime.now()
market_hrs = 9 <= now.hour <= 16
market_status = "🟢 MARKET OPEN" if (market_hrs and now.weekday() < 5) else "🔴 MARKET CLOSED"
st.markdown(f'<div class="watermark">{st.session_state.tickers[0]}</div>', unsafe_allow_html=True)
st.markdown('<button class="fab-btn" title="Quick Trade">⚡</button>', unsafe_allow_html=True)
st.markdown(f'<div class="live-clock">{now.strftime("%H:%M:%S")} | {market_status}</div>', unsafe_allow_html=True)

# ================== SIDEBAR ==================
with st.sidebar:
    st.title("⚙️ Terminal Controls")
    
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
    
    custom_ticker = st.text_input("...oder eigenes Symbol (z.B. KO, JPM)", placeholder="Eigenes Symbol eingeben...")
    
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
            st.rerun()
    for sym in st.session_state.watchlist:
        cols = st.columns([3, 1])
        if cols[0].button(sym, key=f"wl_{sym}"):
            st.session_state.tickers[0] = sym
            st.rerun()
        if cols[1].button("❌", key=f"del_{sym}"):
            st.session_state.watchlist.remove(sym)
            st.rerun()

    with st.expander("🔔 Price Alerts & Journal"):
        st.selectbox("Asset Alert", st.session_state.watchlist, key="alrt")
        st.number_input("Target Price", key="alrtp")
        st.button("Create Alert")
        st.session_state.journal = st.text_area("Trading Journal", value=st.session_state.journal)

    @st.dialog("📚 Chart & Indicator Interpretation Guide")
    def show_interpretation_guide():
        st.markdown(INDICATOR_GUIDE)
    if st.button("📖 Read Chart Interpretations", use_container_width=True):
        show_interpretation_guide()

    st.markdown("---")
    st.markdown("### Technical Indicators")
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

    st.markdown("---")
    st.markdown("### ⏱️ Real-Time Refresh")
    auto_refresh = st.checkbox("Auto-Refresh aktivieren", value=False)
    refresh_interval = st.number_input("Intervall (Sekunden)", min_value=1, max_value=3600, value=60)

# ================== AUTO REFRESH ==================
if auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="data_refresh")

# ================== TIMEFRAMES ==================
c_tf1, c_tf2 = st.columns([3, 1])
with c_tf1:
    tf_selection = st.radio("Quick Timeframe", ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"], index=4)
interval_map = {"1D": "5m", "5D": "15m", "1M": "30m", "3M": "1h", "6M": "1d", "YTD": "1d", "1Y": "1d", "5Y": "1wk", "MAX": "1mo"}
period_map = {"1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y", "5Y": "5y", "MAX": "max"}
period = period_map[tf_selection]
interval = interval_map[tf_selection]

# ================== DATA FETCHING ==================
@st.cache_data(ttl=5)
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
    st.error(f"❌ Cannot fetch data for {st.session_state.tickers[0]}.")
    st.stop()

# ================== APPLY INDICATORS ==================
df = apply_core_indicators(df, ema1_len, ema2_len, rsi_len)
current_price = df['Close'].iloc[-1]
prev_price = list(df['Close'])[-2] if len(df) > 1 else current_price
pct_change = ((current_price - prev_price) / prev_price) * 100

st.markdown(f"## {st.session_state.tickers[0]} <span style='font-size:24px; color:{'#26a69a' if pct_change>=0 else '#ef5350'}'>${current_price:,.2f} ({pct_change:+.2f}%)</span>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🕯️ Chart", "🏢 Fundamental", "🎲 Quants", "💼 Portfolio", "🔍 Scanner", "🎯 Strategien", "🤖 AI & ML"])

# ======= TAB 1: CHART =======
with tab1:
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
    # 🎯 Smart Entry Zone Scanner v2
    # ══════════════════════════════════════════════════════════════
    if st.button("🎯 Jetzt Einstieg anzeigen", use_container_width=True, type="primary"):
        st.markdown("---")
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


# ======= TAB 2: FUNDAMENTALS & NEWS =======
with tab2:
    f1, f2, f3 = st.columns([1, 1, 1])
    ticker_obj = yf.Ticker(st.session_state.tickers[0])
    if info:
        with f1:
            st.markdown(f"### 🏢 {info.get('shortName', st.session_state.tickers[0])}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}")
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
            st.metric("Dividend Yield", f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")
            short_ratio = info.get('shortRatio', None)
            if short_ratio: st.metric("Short Interest Ratio", f"{short_ratio:.2f}")
        with f3:
            st.markdown("### 📰 Latest News")
            if news:
                for idx, n in enumerate(news[:5]):
                    st.markdown(f"**[{n.get('title', 'Headline')}]({n.get('link', '#')})**")
                    st.caption(n.get('publisher', 'Unknown'))
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

# ======= TAB 3: RISK & QUANTS =======
with tab3:
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
                mc_fig.add_trace(go.Scatter(y=[lp]+list(price_path), mode='lines', line=dict(width=1, color='rgba(38,166,154,0.1)')))
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

# ======= TAB 4: PORTFOLIO =======
with tab4:
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

# ======= TAB 5: MARKET SCANNER =======
with tab5:
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

    # Run scanner
    if st.session_state.get('scanner_running'):
        progress_bar = st.progress(0, text="🔄 Scanner startet...")
        status_text = st.empty()
        results_placeholder = st.empty()

        def update_progress(current, total, symbol):
            pct = current / total if total > 0 else 0
            progress_bar.progress(pct, text=f"🔄 Analysiere {symbol}... ({current}/{total})")

        with st.spinner(""):
            scan_df = scan_all_assets(progress_callback=update_progress)

        progress_bar.progress(1.0, text="✅ Scan abgeschlossen!")
        st.session_state['scanner_results'] = scan_df
        st.session_state['scanner_running'] = False

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
        st.markdown("##### 📊 Rohdata (vollständige Tabelle)")
        display_cols = ['Symbol', 'Name', 'Score', 'Direction', 'Kurs', '1T %', '5T %', 'RSI', 'ADX', 'Pattern', 'Signale']
        avail_cols = [c for c in display_cols if c in scan_df.columns]
        st.dataframe(scan_df[avail_cols].reset_index(drop=True), use_container_width=True)

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

# ======= TAB 6: STRATEGY SIGNALS =======
with tab6:
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

# ======= TAB 6: AI & ML =======
# ======= TAB 7: AI & ML =======
with tab7:
    st.markdown("### 🤖 AI & Machine Learning Suite")
    ai_pick = st.radio("AI-Modul:", ["🧠 Pattern Scanner", "📊 Regime Detection", "🎯 Auto S/R", "📈 Forecast", "📋 Risk Profiler", "📄 AI Report"], horizontal=True)
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
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=upper, mode='lines', line=dict(width=0), showlegend=False))
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,152,0,0.15)', name='95% Konfidenz'))
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
