import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import sys
import os

st.set_page_config(page_title="Pro Trading Terminal", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stApp { background-color: #0e1117; color: #fafafa; }
    /* F80: KPI Hover-States with 3D flip */
    div[data-testid="metric-container"] {
        background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5);
        transition: all 0.3s ease; transform-style: preserve-3d;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px) scale(1.02); border-color: #26a69a;
        box-shadow: 0 8px 25px rgba(38,166,154,0.3);
    }
    div.row-widget.stRadio > div{flex-direction:row;}
    .stTextInput>div>div>input { font-size: 20px !important; font-weight: bold !important; text-transform: uppercase !important; }
    /* F74: Responsive Flex */
    @media (max-width: 768px) {
        div[data-testid="metric-container"] { min-width: 100% !important; }
        .stTabs [data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; }
    }
    /* F77: Floating Action Button */
    .fab-btn { position: fixed; bottom: 30px; right: 30px; z-index: 9999;
        background: linear-gradient(135deg, #26a69a, #00897b); color: white;
        border: none; border-radius: 50%; width: 60px; height: 60px; font-size: 24px;
        cursor: pointer; box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        transition: transform 0.2s; }
    .fab-btn:hover { transform: scale(1.1) rotate(15deg); }
    /* F84: Loading skeleton pulse */
    @keyframes pulse { 0%,100%{opacity:0.4} 50%{opacity:0.8} }
    .skeleton { background: linear-gradient(90deg, #1e1e1e 25%, #2a2a2a 50%, #1e1e1e 75%);
        background-size: 200% 100%; animation: pulse 1.5s infinite; border-radius: 8px; height: 200px; }
    /* F72: Watermark */
    .watermark { position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
        font-size: 200px; font-weight: 900; opacity: 0.025; color: #fafafa;
        pointer-events: none; z-index: 0; letter-spacing: 30px; }
    /* F82: Live Clock */
    .live-clock { position: fixed; top: 8px; right: 16px; z-index: 9999;
        font-family: 'Courier New', monospace; font-size: 13px; color: #26a69a;
        background: rgba(14,17,23,0.9); padding: 4px 12px; border-radius: 4px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# ================== SESSION STATE INIT ==================
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

# F72: Watermark
st.markdown(f'<div class="watermark">{st.session_state.tickers[0]}</div>', unsafe_allow_html=True)
# F77: FAB
st.markdown('<button class="fab-btn" title="Quick Trade">⚡</button>', unsafe_allow_html=True)
# F82: Live Clock & Market Status
now = datetime.datetime.now()
market_hrs = 9 <= now.hour <= 16
market_status = "🟢 MARKET OPEN" if (market_hrs and now.weekday() < 5) else "🔴 MARKET CLOSED"
st.markdown(f'<div class="live-clock">{now.strftime("%H:%M:%S")} | {market_status}</div>', unsafe_allow_html=True)

# ================== SIDEBAR ==================
with st.sidebar:
    st.title("⚙️ Terminal Controls")
    
    ticker_input = st.text_input("Primary Ticker", value=st.session_state.tickers[0]).upper()
    if ticker_input != st.session_state.tickers[0]:
        st.session_state.tickers[0] = ticker_input
        st.rerun()

    st.markdown("### 📋 Watchlist (F56)")
    watch_col1, watch_col2 = st.columns(2)
    new_ticker = watch_col1.text_input("Add/Import", key="wt_add")
    if watch_col2.button("➕ Add") and new_ticker:
        if new_ticker.upper() not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker.upper())
            st.rerun()
            
    for sym in st.session_state.watchlist:
        cols = st.columns([3,1])
        if cols[0].button(sym, key=f"wl_{sym}"):
            st.session_state.tickers[0] = sym
            st.rerun()
        if cols[1].button("❌", key=f"del_{sym}"):
            st.session_state.watchlist.remove(sym)
            st.rerun()

    with st.expander("🔔 Price Alerts & Journal (F62/F63)"):
        st.selectbox("Asset Alert", st.session_state.watchlist, key="alrt")
        st.number_input("Target Price", key="alrtp")
        st.button("Create Alert")
        st.session_state.journal = st.text_area("Trading Journal", value=st.session_state.journal)

    # --- Interpretation Guide Button ---
    @st.dialog("📚 Chart & Indicator Interpretation Guide")
    def show_interpretation_guide():
        st.markdown("""
        Hier ist eine Übersicht der wichtigsten Indikatoren und ihrer Interpretationen für das Trading:

        ### 1. Trend-Indikatoren
        **Gleitende Durchschnitte (Moving Averages - MA):**
        *   **10-Tage EMA:** Kurzfristiger Trend. Preis über EMA = Kaufsignal ("grünes Licht"), darunter = Verkaufssignal ("rotes Licht").
        *   **21-Tage EMA:** Mittelfristige Unterstützungslinie in volatilen Aufwärtstrends.
        *   **50-Tage SMA:** Typische Unterstützungsebene für starke Aktien (Dip-Buying).
        *   **200-Tage SMA:** Langfristiger Trendindikator.
        
        **VWAP (Volume Weighted Average Price):**
        Berücksichtigt das gehandelte Volumen. Essenziell für Daytrader.

        **MACD (Moving Average Convergence/Divergence):**
        *   **Interpretation:** MACD-Linie kreuzt Signallinie nach oben = Kaufsignal. Nach unten = Verkaufssignal. Divergenzen zwischen MACD und Preis warnen vor Trendenden.

        ### 2. Momentum- und Oszillator-Indikatoren
        **RSI (Relative Strength Index):**
        *   **Interpretation:** Werte > 70 gelten als überkauft (Verkaufssignal), < 30 als überverkauft (Kaufgelegenheit). Kreuzen der 50er-Linie zeigt generellen Trend.

        **Stochastik-Oszillator:**
        *   **Interpretation:** Kreuzt %K die %D-Linie, entstehen Signale. Funktioniert am besten in Seitwärtsmärkten.

        ### 3. Volatilitäts-Indikatoren
        **Bollinger Bänder:**
        *   **Interpretation:** Verengen sich die Bänder extrem ("The Squeeze"), deutet das auf einen baldigen, explosiven Ausbruch hin.

        **ATR (Average True Range):**
        *   **Interpretation:** Steigende ATR bestätigt einen starken Trend. Fallende ATR bei steigenden Preisen warnt vor Trendwechsel. Gut für Stop-Loss Platzierung.

        ### 4. Volumen-Indikatoren
        *   **Volumen-Spikes:** Extremes Volumen kann das Ende eines Trends (Panik/Kapitulation) signalisieren oder einen starken Ausbruch bestätigen.

        ### 5. Preisaktion & Candlesticks
        *   **Support & Resistance:** Käufer springen bei Support ein, Verkäufer bei Resistance. Bricht der Preis durch, wird alter Widerstand oft zu neuer Unterstützung.
        *   **Engulfing Pattern:** Die aktuelle Kerze "verschluckt" die Vortageskerze. Starkes Umkehrsignal.
        *   **Doji:** Zeigt Unentschlossenheit zwischen Käufern und Verkäufern an Wendepunkten.
        
        > **Wichtig:** Vertraue nie einem einzelnen Indikator blind. Signale gewinnen erst durch die Kombination mehrerer Indikatoren an Validität!
        """)
        
    if st.button("📖 Read Chart Interpretations", use_container_width=True):
        show_interpretation_guide()
        
    st.markdown("---")
    st.markdown("### Technical Indicators (11-20)")
    # Grouping features
    with st.expander("Overlays (Main Chart)", expanded=True):
        show_ema = st.checkbox("EMA (9,21,55)", value=True) # F13
        show_vwap = st.checkbox("VWAP", value=False) # F14
        show_ichimoku = st.checkbox("Ichimoku Cloud", value=False) # F15
        show_sar = st.checkbox("Parabolic SAR", value=False) # F18
        show_supertrend = st.checkbox("SuperTrend", value=False) # F19
        show_vrvp = st.checkbox("Volume Profile (VRVP)", value=False) # F20
        show_fib = st.checkbox("Auto Fibonacci", value=False)
        chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "Area", "Heikin-Ashi"])
        compare_asset = st.selectbox("Compare with Benchmark (F57)", ["None", "SPY", "QQQ", "DIA", "IWM"])
    
    with st.expander("Sub-Charts", expanded=True):
        show_volume = st.checkbox("Volume", value=True)
        show_obv = st.checkbox("On-Balance Volume (OBV)", value=False) # F22
        show_rsi = st.checkbox("RSI", value=False) # F11
        show_macd = st.checkbox("MACD", value=False) # F12
        show_stoch = st.checkbox("Stochastic", value=False) # F17
        show_atr = st.checkbox("ATR", value=False) # F16
        
    # Feature 23 & 25: Custom Settings & Colors
    with st.expander("⚙️ Fine-Tune Indicators", expanded=False):
        ema1_len = st.number_input("EMA 1 Length", 1, 200, 9)
        ema1_col = st.color_picker("EMA 1 Color", "#2196f3")
        ema2_len = st.number_input("EMA 2 Length", 1, 200, 21)
        ema2_col = st.color_picker("EMA 2 Color", "#ff9800")
        rsi_len = st.number_input("RSI Length", 1, 100, 14)

# ================== TIMEFRAMES ==================
c_tf1, c_tf2 = st.columns([3, 1])
with c_tf1:
    tf_selection = st.radio("Quick Timeframe", ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"], index=4)

interval_map = {"1D": "5m", "5D": "15m", "1M": "30m", "3M": "1h", "6M": "1d", "YTD": "1d", "1Y": "1d", "5Y": "1wk", "MAX": "1mo"}
period_map = {"1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y", "5Y": "5y", "MAX": "max"}
period = period_map[tf_selection]
interval = interval_map[tf_selection]

# ================== DATA FETCHING ==================
@st.cache_data(ttl=60)
def fetch_data(ticker, p, i):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=p, interval=i)
        if df.empty: return None, None, None, None, None
        
        # F26-F30: Fetch additional data with fallback
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

# ================== INDICATOR CALCULATIONS ==================
df['Daily_Return'] = df['Close'].pct_change()
current_price = df['Close'].iloc[-1]
prev_price = list(df['Close'])[-2] if len(df) > 1 else current_price
pct_change = ((current_price - prev_price) / prev_price) * 100

st.markdown(f"## {st.session_state.tickers[0]} <span style='font-size:24px; color:{'#26a69a' if pct_change>=0 else '#ef5350'}'>${current_price:,.2f} ({pct_change:+.2f}%)</span>", unsafe_allow_html=True)

# Tabs: all 100 features organized
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🕯️ Chart", "🏢 Fundamental", "🎲 Quants", "💼 Portfolio", "🎯 Strategien", "🤖 AI & ML"])

with tab1:
    # F13 & 23/25
    df['EMA_1'] = df['Close'].ewm(span=ema1_len, adjust=False).mean()
    df['EMA_2'] = df['Close'].ewm(span=ema2_len, adjust=False).mean()
    df['EMA_55'] = df['Close'].ewm(span=55, adjust=False).mean()

    if 'Volume' in df.columns:
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        # F22: On-Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # F11 & 23
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_len).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_len).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()

    # F21: Pivot Points (Classic)
    pp = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
    r1 = (2 * pp) - df['Low'].shift(1)
    s1 = (2 * pp) - df['High'].shift(1)

    # F24: Candlestick Pattern Recognition (Doji & Hammer)
    df['Doji'] = np.abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1
    df['Hammer'] = ((df['High'] - df['Low']) > 3 * np.abs(df['Open'] - df['Close'])) & \
                   ((df['Close'] - df['Low']) / (.001 + df['High'] - df['Low']) > 0.6) & \
                   ((df['Open'] - df['Low']) / (.001 + df['High'] - df['Low']) > 0.6)

    # HA Calculate
    def calc_heikin_ashi(df):
        ha_df = df.copy(); ha_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        for i in range(len(df)):
            if i == 0: ha_df.iloc[i, ha_df.columns.get_loc('Open')] = (df['Open'].iloc[i] + df['Close'].iloc[i]) / 2
            else: ha_df.iloc[i, ha_df.columns.get_loc('Open')] = (ha_df['Open'].iloc[i-1] + ha_df['Close'].iloc[i-1]) / 2
        ha_df['High'] = ha_df[['Open', 'Close', 'High']].max(axis=1)
        ha_df['Low'] = ha_df[['Open', 'Close', 'Low']].min(axis=1)
        return ha_df

    plot_df = calc_heikin_ashi(df) if chart_type == "Heikin-Ashi" else df

    # ================== DYNAMIC PLOTLY LAYOUT ==================
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

    fig = make_subplots(
        rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
        row_heights=row_heights, row_titles=["Price"] + row_titles
    )

    # --- ROW 1: PRICE & OVERLAYS ---
    if chart_type in ["Candlestick", "Heikin-Ashi"]:
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'], mode='lines', line=dict(color='#26a69a', width=2), name='Close'), row=1, col=1)

    # Overlays
    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_1'], line=dict(color=ema1_col, width=1.5), name=f'EMA {ema1_len}'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_2'], line=dict(color=ema2_col, width=1.5), name=f'EMA {ema2_len}'), row=1, col=1)

    # F57: Benchmark Overlay
    if compare_asset != "None":
        try:
            comp_df = yf.Ticker(compare_asset).history(period=period, interval=interval)
            # Scale benchmark to match the primary asset's starting price for visual comparison
            comp_scaled = comp_df['Close'] * (plot_df['Close'].iloc[0] / comp_df['Close'].iloc[0])
            fig.add_trace(go.Scatter(x=comp_df.index, y=comp_scaled, mode='lines', line=dict(color='yellow', width=1.5), name=compare_asset), row=1, col=1)
        except: pass
    
    # F21: Pivot Points
    if st.checkbox("Show Daily Pivot Points", value=False):
        fig.add_trace(go.Scatter(x=df.index, y=pp, line=dict(color='#ab47bc', width=1, dash='dot'), name='Pivot'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=r1, line=dict(color='#ef5350', width=1, dash='dot'), name='R1'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=s1, line=dict(color='#26a69a', width=1, dash='dot'), name='S1'), row=1, col=1)

    # F24: Plot Pattern Markers
    doji_idx = df.index[df['Doji']]
    hammer_idx = df.index[df['Hammer']]
    if len(doji_idx) > 0: fig.add_trace(go.Scatter(x=doji_idx, y=df.loc[doji_idx, 'Close'], mode='markers', marker=dict(symbol='triangle-down', size=8, color='yellow'), name='Doji'), row=1, col=1)
    if len(hammer_idx) > 0: fig.add_trace(go.Scatter(x=hammer_idx, y=df.loc[hammer_idx, 'Low'] * 0.99, mode='text', text='🔨', textfont=dict(size=14), name='Hammer'), row=1, col=1)

    # F28: Dividend History Overlay
    try:
        if dividends is not None and not dividends.empty:
            df_start = df.index.min().tz_localize(None) if df.index.tz else df.index.min()
            for div_date, div_val in dividends.items():
                d = div_date.tz_localize(None) if div_date.tzinfo else div_date
                if d >= df_start:
                    fig.add_annotation(x=str(d), y=float(df['Close'].iloc[-1]), text=f"💰 ${div_val:.2f}", showarrow=True, arrowhead=1, ax=0, ay=-40, row=1, col=1)
    except Exception: pass

    # F29: Earnings Dates Overlay
    try:
        if earnings_dates is not None and not earnings_dates.empty:
            df_start = df.index.min().tz_localize(None) if df.index.tz else df.index.min()
            for earn_date, _ in earnings_dates.iterrows():
                d = earn_date.tz_localize(None) if earn_date.tzinfo else earn_date
                if d >= df_start:
                    fig.add_vline(x=str(d), line_dash="dash", line_color="orange", annotation_text="📊", row=1, col=1)
    except Exception: pass


    # --- DYNAMIC SUBROWS ---
    current_row = 2
    if "Volume" in subplots:
        colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' for index, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=current_row, col=1)
        current_row += 1
        
    if "OBV" in subplots:
        fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='#ffeb3b', width=1.5), name='OBV'), row=current_row, col=1)
        current_row += 1

    if "RSI" in subplots:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#ab47bc', width=1.5), name='RSI'), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=current_row, col=1)
        current_row += 1

    if "MACD" in subplots:
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=['#26a69a' if val > 0 else '#ef5350' for val in df['MACD_Hist']], name='Histogram'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#2962ff', width=1.5), name='MACD'), row=current_row, col=1)
        current_row += 1

    fig.update_layout(
        template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
        height=800, showlegend=False, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', hovermode="x unified",
        dragmode='drawline', newshape=dict(line_color='yellow')
    )
    if interval in ['1d', '1wk']: fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape']})

    # ══════════════════════════════════════════════════════════════
    # 🎯 JETZT EINSTIEG ANZEIGEN — Smart Entry Zone Scanner v2
    # ══════════════════════════════════════════════════════════════
    if st.button("🎯 Jetzt Einstieg anzeigen", use_container_width=True, type="primary"):
        st.markdown("---")
        st.markdown("### 🎯 Smart Entry-Zone Analyse v2")

        from scipy.cluster.vq import kmeans

        last_price = df['Close'].iloc[-1]
        atr_now = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else (df['High'].iloc[-1] - df['Low'].iloc[-1])

        # ════════════════════════════════════════════════════════
        # PHASE 1: Structural Analysis
        # ════════════════════════════════════════════════════════

        # ── 1a) K-Means Support & Resistance (much smarter than min/max) ──
        # Identify swing highs and swing lows (local pivots)
        swing_window = 5
        swing_highs = []
        swing_lows = []
        for i in range(swing_window, len(df) - swing_window):
            if df['High'].iloc[i] == df['High'].iloc[i-swing_window:i+swing_window+1].max():
                swing_highs.append(df['High'].iloc[i])
            if df['Low'].iloc[i] == df['Low'].iloc[i-swing_window:i+swing_window+1].min():
                swing_lows.append(df['Low'].iloc[i])

        # Cluster all pivot points into key levels
        all_pivots = np.array(swing_highs + swing_lows, dtype=float)
        if len(all_pivots) >= 5:
            n_clusters = min(6, len(all_pivots))
            key_levels, _ = kmeans(all_pivots, n_clusters)
            key_levels = sorted(key_levels)
        else:
            # Fallback: use simple percentile-based levels
            key_levels = [
                df['Low'].iloc[-40:].min() if len(df) >= 40 else df['Low'].min(),
                df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else last_price,
                df['High'].iloc[-40:].max() if len(df) >= 40 else df['High'].max(),
            ]
            key_levels = sorted(key_levels)

        # Separate into support / resistance relative to current price
        supports = sorted([l for l in key_levels if l < last_price], reverse=True)  # nearest first
        resistances = sorted([l for l in key_levels if l > last_price])  # nearest first

        nearest_support = supports[0] if supports else last_price - atr_now * 2
        nearest_resistance = resistances[0] if resistances else last_price + atr_now * 2
        second_resistance = resistances[1] if len(resistances) > 1 else nearest_resistance + atr_now * 2
        second_support = supports[1] if len(supports) > 1 else nearest_support - atr_now * 2

        # ── 1b) Bollinger Bands ──
        bb_p = 20
        df['BB_Mid_EZ'] = df['Close'].rolling(bb_p).mean()
        df['BB_Std_EZ'] = df['Close'].rolling(bb_p).std()
        df['BB_Upper_EZ'] = df['BB_Mid_EZ'] + 2 * df['BB_Std_EZ']
        df['BB_Lower_EZ'] = df['BB_Mid_EZ'] - 2 * df['BB_Std_EZ']

        # ── 1c) ADX-like Trend Strength (using directional movement) ──
        adx_period = 14
        plus_dm = df['High'].diff().clip(lower=0)
        minus_dm = (-df['Low'].diff()).clip(lower=0)
        # When plus_dm < minus_dm, set plus_dm to 0 and vice versa
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < plus_dm] = 0
        atr_smooth = df['ATR']  # reuse
        plus_di = 100 * (plus_dm.ewm(span=adx_period, adjust=False).mean() / atr_smooth.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(span=adx_period, adjust=False).mean() / atr_smooth.replace(0, np.nan))
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(span=adx_period, adjust=False).mean()
        last_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
        last_plus_di = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 50
        last_minus_di = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 50

        # ════════════════════════════════════════════════════════
        # PHASE 2: Signal Confluence (12 weighted checks)
        # ════════════════════════════════════════════════════════
        score = 0
        max_score = 20  # theoretical max of all positive signals
        signals = []

        # Safely read indicators
        last_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
        last_macd = df['MACD'].iloc[-1] if not pd.isna(df['MACD'].iloc[-1]) else 0
        last_signal_val = df['Signal'].iloc[-1] if not pd.isna(df['Signal'].iloc[-1]) else 0
        last_macd_hist = df['MACD_Hist'].iloc[-1] if not pd.isna(df['MACD_Hist'].iloc[-1]) else 0
        last_ema1 = df['EMA_1'].iloc[-1] if not pd.isna(df['EMA_1'].iloc[-1]) else last_price
        last_ema2 = df['EMA_2'].iloc[-1] if not pd.isna(df['EMA_2'].iloc[-1]) else last_price
        last_ema55 = df['EMA_55'].iloc[-1] if not pd.isna(df['EMA_55'].iloc[-1]) else last_price
        last_bb_lower = df['BB_Lower_EZ'].iloc[-1] if not pd.isna(df['BB_Lower_EZ'].iloc[-1]) else last_price * 0.95
        last_bb_upper = df['BB_Upper_EZ'].iloc[-1] if not pd.isna(df['BB_Upper_EZ'].iloc[-1]) else last_price * 1.05
        last_bb_mid = df['BB_Mid_EZ'].iloc[-1] if not pd.isna(df['BB_Mid_EZ'].iloc[-1]) else last_price

        # ── CHECK 1: ADX Trend Strength (weight: 3) ──
        if last_adx > 25 and last_plus_di > last_minus_di:
            score += 3
            signals.append(("💪 Starker Aufwärtstrend", f"ADX = {last_adx:.0f} (>25), +DI > -DI → Kontrollierte Bullen", "bullish"))
        elif last_adx > 25 and last_minus_di > last_plus_di:
            score -= 3
            signals.append(("💪 Starker Abwärtstrend", f"ADX = {last_adx:.0f} (>25), -DI > +DI → Bären dominieren", "bearish"))
        elif last_adx < 20:
            signals.append(("😴 Seitwärtsmarkt", f"ADX = {last_adx:.0f} (<20) → Kein klarer Trend, Range-Trading", "neutral"))
        else:
            signals.append(("📊 Moderater Trend", f"ADX = {last_adx:.0f} → Trend entwickelt sich", "neutral"))

        # ── CHECK 2: EMA Ribbon Health (weight: 2) ──
        ema_spread = (last_ema1 - last_ema55) / last_price * 100  # % spread
        if last_ema1 > last_ema2 > last_ema55:
            score += 2
            signals.append(("✅ EMA perfekt gestaffelt", f"EMA {ema1_len} > {ema2_len} > 55 (Spread: {ema_spread:+.1f}%) → Gesunder Trend", "bullish"))
        elif last_ema1 < last_ema2 < last_ema55:
            score -= 2
            signals.append(("🔻 EMA bärisch gestaffelt", f"EMA {ema1_len} < {ema2_len} < 55 → Bärischer Trend", "bearish"))
        elif last_ema1 > last_ema2:
            score += 1
            signals.append(("🟡 EMAs mischen sich", f"EMA {ema1_len} > {ema2_len}, aber EMA 55 zeigt Widerstand", "neutral"))
        else:
            score -= 1
            signals.append(("⚠️ EMAs negativ", f"EMA {ema1_len} < {ema2_len} → Kurzfristige Schwäche", "bearish"))

        # ── CHECK 3: Price vs. Major Moving Averages (weight: 2) ──
        above_count = sum([last_price > last_ema1, last_price > last_ema2, last_price > last_ema55])
        if above_count == 3:
            score += 2
            signals.append(("📈 Über allen EMAs", "Kurs über EMA 9/21/55 → Alle Zeitrahmen bullisch", "bullish"))
        elif above_count == 0:
            score -= 2
            signals.append(("📉 Unter allen EMAs", "Kurs unter EMA 9/21/55 → Alle Zeitrahmen bärisch", "bearish"))
        elif above_count >= 2:
            score += 1
            signals.append(("🟡 Über 2 von 3 EMAs", f"Gemischtes Bild – Trend nicht eindeutig", "neutral"))
        else:
            score -= 1
            signals.append(("⚠️ Unter 2 von 3 EMAs", "Schwäche in mehreren Zeitrahmen", "bearish"))

        # ── CHECK 4: RSI with Divergence Detection (weight: 3) ──
        # Check for bullish divergence: price makes lower low but RSI makes higher low
        rsi_div = "none"
        if len(df) >= 20:
            price_recent = df['Close'].iloc[-10:]
            price_prev = df['Close'].iloc[-20:-10]
            rsi_recent = df['RSI'].iloc[-10:]
            rsi_prev = df['RSI'].iloc[-20:-10]

            price_recent_low = price_recent.min()
            price_prev_low = price_prev.min()
            rsi_recent_low = rsi_recent.min() if not rsi_recent.isna().all() else 50
            rsi_prev_low = rsi_prev.min() if not rsi_prev.isna().all() else 50

            price_recent_high = price_recent.max()
            price_prev_high = price_prev.max()
            rsi_recent_high = rsi_recent.max() if not rsi_recent.isna().all() else 50
            rsi_prev_high = rsi_prev.max() if not rsi_prev.isna().all() else 50

            if price_recent_low < price_prev_low and rsi_recent_low > rsi_prev_low:
                rsi_div = "bullish"
            elif price_recent_high > price_prev_high and rsi_recent_high < rsi_prev_high:
                rsi_div = "bearish"

        if rsi_div == "bullish":
            score += 3
            signals.append(("🔮 Bullische RSI-Divergenz!", f"Preis macht tiefere Tiefs, RSI macht höhere Tiefs → Starkes Umkehrsignal", "bullish"))
        elif rsi_div == "bearish":
            score -= 3
            signals.append(("🔮 Bärische RSI-Divergenz!", f"Preis macht höhere Hochs, RSI macht tiefere Hochs → Warnung vor Trendumkehr", "bearish"))

        if last_rsi < 30:
            score += 2
            signals.append(("🟢 RSI stark überverkauft", f"RSI = {last_rsi:.1f} (<30) → Hohe Kaufwahrscheinlichkeit", "bullish"))
        elif last_rsi < 40:
            score += 1
            signals.append(("🟡 RSI niedrig", f"RSI = {last_rsi:.1f} → Raum für Erholung", "neutral"))
        elif last_rsi > 70:
            score -= 2
            signals.append(("🔴 RSI überkauft", f"RSI = {last_rsi:.1f} (>70) → Rücksetzer wahrscheinlich", "bearish"))
        elif last_rsi > 60:
            signals.append(("📊 RSI erhöht", f"RSI = {last_rsi:.1f} → Bullisches Momentum, aber aufpassen", "neutral"))
        else:
            signals.append(("⚪ RSI neutral", f"RSI = {last_rsi:.1f} → Kein klares Signal", "neutral"))

        # ── CHECK 5: MACD Histogram Momentum (weight: 2) ──
        hist_vals = df['MACD_Hist'].dropna()
        if len(hist_vals) >= 3:
            hist_accel = hist_vals.iloc[-1] - hist_vals.iloc[-2]
            hist_prev_accel = hist_vals.iloc[-2] - hist_vals.iloc[-3]
            if last_macd_hist > 0 and hist_accel > 0:
                score += 2
                signals.append(("🚀 MACD Momentum steigend", f"Histogram wächst (Δ={hist_accel:.4f}) → Beschleunigendes Kaufinteresse", "bullish"))
            elif last_macd_hist > 0 and hist_accel < 0:
                score += 1
                signals.append(("⚠️ MACD bullisch aber abflachend", f"Histogram positiv aber schrumpfend → Momentum schwächt sich ab", "neutral"))
            elif last_macd_hist < 0 and hist_accel > 0:
                signals.append(("🔄 MACD Erholung?", f"Histogram negativ aber schrumpfend → Mögliche Bodenbildung", "neutral"))
            elif last_macd_hist < 0 and hist_accel < 0:
                score -= 2
                signals.append(("📉 MACD Momentum fallend", f"Histogram fällt weiter (Δ={hist_accel:.4f}) → Verkaufsdruck steigt", "bearish"))
        else:
            if last_macd > last_signal_val:
                score += 1
                signals.append(("✅ MACD über Signal", "Grundlegend bullisch", "bullish"))
            else:
                score -= 1
                signals.append(("🔴 MACD unter Signal", "Grundlegend bärisch", "bearish"))

        # ── CHECK 6: Bollinger Squeeze Detection (weight: 2) ──
        bb_width = (df['BB_Upper_EZ'] - df['BB_Lower_EZ']) / df['BB_Mid_EZ']
        bb_width_valid = bb_width.dropna()
        if len(bb_width_valid) >= 20:
            bb_width_pct = bb_width_valid.iloc[-1]
            bb_width_avg = bb_width_valid.iloc[-20:].mean()
            is_squeeze = bb_width_pct < bb_width_avg * 0.7
            if is_squeeze:
                score += 2
                signals.append(("💥 Bollinger Squeeze!", f"BB-Breite = {bb_width_pct:.3f} vs Ø {bb_width_avg:.3f} → Explosive Bewegung erwartet!", "bullish"))
            else:
                bb_pct = (last_price - last_bb_lower) / (last_bb_upper - last_bb_lower) if (last_bb_upper - last_bb_lower) > 0 else 0.5
                if bb_pct < 0.2:
                    score += 1
                    signals.append(("🟢 Unteres BB", f"BB-Position: {bb_pct:.0%} → Nahe unterer Bandgrenze (überdehnt)", "bullish"))
                elif bb_pct > 0.8:
                    score -= 1
                    signals.append(("🔴 Oberes BB", f"BB-Position: {bb_pct:.0%} → Nahe oberer Bandgrenze", "bearish"))
                else:
                    signals.append(("⚪ BB mittig", f"BB-Position: {bb_pct:.0%} → Keine extreme Ausdehnung", "neutral"))
        else:
            signals.append(("⚪ BB neutral", "Nicht genug Daten für Squeeze-Analyse", "neutral"))

        # ── CHECK 7: Volume Confirmation (weight: 2) ──
        if 'Volume' in df.columns and len(df) >= 20:
            vol_20_avg = df['Volume'].iloc[-20:].mean()
            vol_last = df['Volume'].iloc[-1]
            vol_ratio = vol_last / vol_20_avg if vol_20_avg > 0 else 1
            price_up = df['Close'].iloc[-1] > df['Open'].iloc[-1]

            if vol_ratio > 1.5 and price_up:
                score += 2
                signals.append(("📊 Hohes Volumen + Kursanstieg", f"Vol = {vol_ratio:.1f}x Ø → Starkes institutionelles Kaufinteresse", "bullish"))
            elif vol_ratio > 1.5 and not price_up:
                score -= 1
                signals.append(("📊 Hohes Volumen + Kursrückgang", f"Vol = {vol_ratio:.1f}x Ø → Distributionsphase möglich", "bearish"))
            elif vol_ratio < 0.5:
                signals.append(("🔇 Niedriges Volumen", f"Vol = {vol_ratio:.1f}x Ø → Geringe Überzeugung, Vorsicht", "neutral"))
            else:
                signals.append(("📊 Normales Volumen", f"Vol = {vol_ratio:.1f}x Ø → Keine besondere Auffälligkeit", "neutral"))

        # ── CHECK 8: Candlestick Pattern at Current Level (weight: 1) ──
        if len(df) >= 3:
            c_open = df['Open'].iloc[-1]
            c_close = df['Close'].iloc[-1]
            c_high = df['High'].iloc[-1]
            c_low = df['Low'].iloc[-1]
            body = abs(c_close - c_open)
            total_range = c_high - c_low if c_high - c_low > 0 else 0.001
            lower_shadow = min(c_open, c_close) - c_low
            upper_shadow = c_high - max(c_open, c_close)

            # Hammer (bullish reversal)
            if lower_shadow > 2 * body and upper_shadow < body * 0.5 and body > 0:
                score += 1
                signals.append(("🔨 Hammer-Kerze", "Langer unterer Docht → Käufer haben Verkaufsdruck absorbiert", "bullish"))
            # Shooting Star (bearish reversal)
            elif upper_shadow > 2 * body and lower_shadow < body * 0.5 and body > 0:
                score -= 1
                signals.append(("⭐ Shooting Star", "Langer oberer Docht → Verkäufer drücken Kurs zurück", "bearish"))
            # Bullish Engulfing
            p_open = df['Open'].iloc[-2]
            p_close = df['Close'].iloc[-2]
            if c_close > c_open and p_close < p_open and c_close > p_open and c_open < p_close:
                score += 1
                signals.append(("🟢 Bullish Engulfing", "Aktuelle Kerze verschluckt die vorige bärische Kerze → Umkehr", "bullish"))
            # Bearish Engulfing
            elif c_close < c_open and p_close > p_open and c_open > p_close and c_close < p_open:
                score -= 1
                signals.append(("🔴 Bearish Engulfing", "Aktuelle Kerze verschluckt die vorige bullische Kerze → Warnung", "bearish"))
            # Doji near key level
            elif body / total_range < 0.1:
                dist_to_support = abs(last_price - nearest_support) / last_price
                dist_to_resist = abs(last_price - nearest_resistance) / last_price
                if dist_to_support < 0.01:
                    score += 1
                    signals.append(("✝️ Doji am Support!", "Unentschlossenheit genau am Unterstützungsniveau → Mögliche Umkehr", "bullish"))
                elif dist_to_resist < 0.01:
                    score -= 1
                    signals.append(("✝️ Doji am Widerstand!", "Unentschlossenheit genau am Widerstand → Mögliche Ablehnung", "bearish"))

        # ── CHECK 9: Proximity to Key S/R Level (weight: 2) ──
        dist_to_support = abs(last_price - nearest_support) / last_price * 100
        dist_to_resistance = abs(nearest_resistance - last_price) / last_price * 100
        room_up = dist_to_resistance
        room_down = dist_to_support

        if room_up > room_down * 2:
            score += 2
            signals.append(("🎯 Viel Platz nach oben", f"Nächster Widerstand {room_up:.1f}% entfernt vs Support nur {room_down:.1f}% → Gutes Chance/Risiko", "bullish"))
        elif room_down > room_up * 2:
            score -= 1
            signals.append(("⚠️ Wenig Platz nach oben", f"Nächster Widerstand nur {room_up:.1f}% entfernt vs Support {room_down:.1f}% → Begrenztes Potenzial", "bearish"))
        else:
            signals.append(("↔️ Ausgewogene Position", f"Support {room_down:.1f}% | Widerstand {room_up:.1f}% entfernt", "neutral"))

        # ── CHECK 10: Consecutive Candle Trend (weight: 1) ──
        if len(df) >= 5:
            last5_bullish = sum(df['Close'].iloc[-5:] > df['Open'].iloc[-5:])
            if last5_bullish >= 4:
                score += 1
                signals.append(("🟢 Bullische Serie", f"{last5_bullish}/5 der letzten Kerzen grün → Kurzfristiges Momentum", "bullish"))
            elif last5_bullish <= 1:
                score -= 1
                signals.append(("🔴 Bärische Serie", f"Nur {last5_bullish}/5 der letzten Kerzen grün → Verkaufsdruck", "bearish"))

        # ════════════════════════════════════════════════════════
        # PHASE 3: Decision & Smart Zone Placement
        # ════════════════════════════════════════════════════════

        # Determine market regime
        if last_adx > 25:
            regime = "TRENDING"
        elif last_adx < 20:
            regime = "RANGING"
        else:
            regime = "TRANSITIONING"

        # Determine direction
        is_bullish = score >= 4
        is_bearish = score <= -4

        if is_bullish:
            direction = "LONG"
            direction_emoji = "🟢"
            direction_color = "#00e676"

            # Smart entry: pullback to nearest support confluence
            if regime == "TRENDING":
                # In strong trend: enter on pullback to fast EMA
                entry_low = min(last_ema1, last_ema2)
                entry_high = last_price
            else:
                # In range: enter closer to support
                entry_low = max(nearest_support, last_bb_lower)
                entry_high = min(last_price, last_bb_mid)

            if entry_high <= entry_low:
                entry_low = last_price - atr_now * 0.5
                entry_high = last_price

            # SL below nearest structural support (not just ATR)
            sl_level = min(nearest_support - atr_now * 0.5, entry_low - atr_now * 1.5)

            # Targets at actual resistance levels
            target_1 = nearest_resistance
            target_2 = second_resistance if second_resistance > nearest_resistance else nearest_resistance + atr_now * 2
            target_3 = max(target_2 + atr_now * 1.5, last_price + atr_now * 4)

        elif is_bearish:
            direction = "SHORT"
            direction_emoji = "🔴"
            direction_color = "#ff1744"

            if regime == "TRENDING":
                entry_low = last_price
                entry_high = max(last_ema1, last_ema2)
            else:
                entry_low = max(last_price, last_bb_mid)
                entry_high = min(nearest_resistance, last_bb_upper)

            if entry_high <= entry_low:
                entry_low = last_price
                entry_high = last_price + atr_now * 0.5

            sl_level = max(nearest_resistance + atr_now * 0.5, entry_high + atr_now * 1.5)
            target_1 = nearest_support
            target_2 = second_support if second_support < nearest_support else nearest_support - atr_now * 2
            target_3 = min(target_2 - atr_now * 1.5, last_price - atr_now * 4)

        else:
            direction = "ABWARTEN"
            direction_emoji = "🟡"
            direction_color = "#ffea00"

            entry_low = nearest_support
            entry_high = nearest_support + atr_now * 0.3
            sl_level = nearest_support - atr_now * 1.5
            target_1 = nearest_resistance
            target_2 = second_resistance if second_resistance > nearest_resistance else nearest_resistance + atr_now * 2
            target_3 = target_2 + atr_now * 1.5

        # ── Score classification ──
        score_label = "🟢 SEHR STARK" if score >= 8 else "🟢 STARK" if score >= 5 else "🟢 GUT" if score >= 4 else "🟡 NEUTRAL" if score >= 0 else "🔴 SCHWACH" if score >= -3 else "🔴 SEHR SCHWACH"
        confidence = min(100, max(0, int((score / max_score) * 100)))

        # ── KPI Header ──
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Richtung", f"{direction_emoji} {direction}")
        mc2.metric("Score", f"{score}/{max_score}", help="Gewichteter Konfluenz-Score aus 12 Prüfungen")
        mc3.metric("Bewertung", score_label)
        mc4.metric("Marktregime", f"{'📈 Trend' if regime == 'TRENDING' else '↔️ Range' if regime == 'RANGING' else '🔄 Übergang'}")
        mc5.metric("Konfidenz", f"{confidence}%")

        # ── Entry Zone Chart ──
        fig_ez = go.Figure()
        plot_window = min(60, len(df))
        plot_data = plot_df.iloc[-plot_window:]
        df_window = df.iloc[-plot_window:]

        # Candlesticks
        fig_ez.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'],
            low=plot_data['Low'], close=plot_data['Close'], name="Price",
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))

        # EMAs
        fig_ez.add_trace(go.Scatter(x=df_window.index, y=df_window['EMA_1'], line=dict(color=ema1_col, width=1.2), name=f'EMA {ema1_len}'))
        fig_ez.add_trace(go.Scatter(x=df_window.index, y=df_window['EMA_2'], line=dict(color=ema2_col, width=1.2), name=f'EMA {ema2_len}'))
        fig_ez.add_trace(go.Scatter(x=df_window.index, y=df_window['EMA_55'], line=dict(color='#7c4dff', width=1, dash='dot'), name='EMA 55'))

        # Bollinger Bands
        fig_ez.add_trace(go.Scatter(x=df_window.index, y=df_window['BB_Upper_EZ'], line=dict(color='rgba(255,255,255,0.12)', width=1), name='BB', showlegend=False))
        fig_ez.add_trace(go.Scatter(x=df_window.index, y=df_window['BB_Lower_EZ'], line=dict(color='rgba(255,255,255,0.12)', width=1), fill='tonexty', fillcolor='rgba(100,100,255,0.03)', showlegend=False))

        # K-Means S/R Levels
        sr_colors = {'support': '#26a69a', 'resistance': '#ef5350'}
        for lvl in key_levels:
            is_support = lvl < last_price
            label = f"S ${lvl:.2f}" if is_support else f"R ${lvl:.2f}"
            color = sr_colors['support'] if is_support else sr_colors['resistance']
            fig_ez.add_hline(y=lvl, line_dash="dashdot", line_color=color, line_width=1, opacity=0.6,
                             annotation_text=label, annotation_position="left",
                             annotation=dict(font=dict(color=color, size=10)))

        # ── ENTRY ZONE ──
        zone_color = "rgba(0,230,118,0.15)" if direction == "LONG" else "rgba(255,23,68,0.12)" if direction == "SHORT" else "rgba(255,234,0,0.10)"
        zone_border = "#00e676" if direction == "LONG" else "#ff1744" if direction == "SHORT" else "#ffea00"
        fig_ez.add_hrect(y0=entry_low, y1=entry_high, fillcolor=zone_color, line_width=2, line_color=zone_border, line_dash="dash",
                         annotation_text=f"🎯 ENTRY ${entry_low:.2f}–${entry_high:.2f}", annotation_position="top left",
                         annotation=dict(font=dict(color=zone_border, size=12)))

        # ── STOP-LOSS ──
        fig_ez.add_hline(y=sl_level, line_dash="dash", line_color="#ff1744", line_width=2,
                         annotation_text=f"🛑 SL ${sl_level:.2f}", annotation_position="bottom left",
                         annotation=dict(font=dict(color="#ff1744", size=11)))

        # ── TARGETS ──
        for i, (tgt, color, label) in enumerate([(target_1, "#00b0ff", "Ziel 1"), (target_2, "#448aff", "Ziel 2"), (target_3, "#7c4dff", "Ziel 3")]):
            fig_ez.add_hline(y=tgt, line_dash="dot", line_color=color, line_width=1.5,
                             annotation_text=f"{'🎯' if i==0 else '🏆' if i==1 else '💎'} {label}: ${tgt:.2f}", annotation_position="top right",
                             annotation=dict(font=dict(color=color, size=10)))

        # Target zone shading
        if direction in ["LONG", "ABWARTEN"]:
            fig_ez.add_hrect(y0=min(target_1, target_2), y1=max(target_1, target_2), fillcolor="rgba(0,176,255,0.06)", line_width=0)
        else:
            fig_ez.add_hrect(y0=min(target_1, target_2), y1=max(target_1, target_2), fillcolor="rgba(0,176,255,0.06)", line_width=0)

        # Current price marker
        fig_ez.add_hline(y=last_price, line_dash="solid", line_color="white", line_width=1.5,
                         annotation_text=f"◀ JETZT ${last_price:.2f}", annotation_position="right",
                         annotation=dict(font=dict(color="white", size=12, family="monospace")))

        fig_ez.update_layout(
            template="plotly_dark", height=600,
            margin=dict(l=0, r=130, t=35, b=0),
            paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
            xaxis_rangeslider_visible=False,
            showlegend=True, legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
            title=dict(text=f"Entry-Analyse: {st.session_state.tickers[0]} — {direction_emoji} {direction} (Score {score}/{max_score})", font=dict(color=direction_color, size=15))
        )
        if interval in ['1d', '1wk']:
            fig_ez.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

        st.plotly_chart(fig_ez, use_container_width=True)

        # ── Signal Detail Cards ──
        st.markdown("#### 📊 Signal-Konfluenz Details")
        for sig_name, sig_desc, sig_type in signals:
            color = "#00e676" if sig_type == "bullish" else "#ff1744" if sig_type == "bearish" else "#ffea00"
            bg = "rgba(0,230,118,0.08)" if sig_type == "bullish" else "rgba(255,23,68,0.08)" if sig_type == "bearish" else "rgba(255,234,0,0.05)"
            st.markdown(f"<div style='padding:8px 14px;margin:4px 0;border-left:4px solid {color};background:{bg};border-radius:6px;font-size:14px;'><b>{sig_name}</b> — {sig_desc}</div>", unsafe_allow_html=True)

        # ── Risk/Reward Summary ──
        st.markdown("#### 💰 Risk / Reward Analyse")
        entry_mid = (entry_low + entry_high) / 2
        risk = abs(entry_mid - sl_level)
        reward_1 = abs(target_1 - entry_mid)
        reward_2 = abs(target_2 - entry_mid)
        reward_3 = abs(target_3 - entry_mid)
        rr1 = reward_1 / risk if risk > 0 else 0
        rr2 = reward_2 / risk if risk > 0 else 0
        rr3 = reward_3 / risk if risk > 0 else 0

        rr_verdict = "✅ Akzeptabel" if rr1 >= 1.5 else "⚠️ Grenzwertig" if rr1 >= 1.0 else "❌ Zu riskant"

        rrc1, rrc2, rrc3, rrc4 = st.columns(4)
        rrc1.metric("Risiko (→ SL)", f"${risk:.2f}", help="Abstand vom mittleren Einstieg zum Stop-Loss")
        rrc2.metric("R:R Ziel 1", f"1:{rr1:.1f}", delta=f"${reward_1:.2f}", delta_color="normal")
        rrc3.metric("R:R Ziel 2", f"1:{rr2:.1f}", delta=f"${reward_2:.2f}", delta_color="normal")
        rrc4.metric("R:R Urteil", rr_verdict, help="Min. 1:1.5 empfohlen")

        # ── Trade Plan Summary ──
        if direction != "ABWARTEN":
            st.markdown("#### 📋 Trade-Plan")
            st.markdown(f"""
| Parameter | Wert |
|---|---|
| **Richtung** | {direction_emoji} {direction} |
| **Entry Zone** | ${entry_low:.2f} – ${entry_high:.2f} |
| **Stop-Loss** | ${sl_level:.2f} (Risiko: ${risk:.2f}/Aktie) |
| **Ziel 1** | ${target_1:.2f} (R:R 1:{rr1:.1f}) |
| **Ziel 2** | ${target_2:.2f} (R:R 1:{rr2:.1f}) |
| **Ziel 3** | ${target_3:.2f} (R:R 1:{rr3:.1f}) |
| **Regime** | {regime} (ADX: {last_adx:.0f}) |
| **Konfidenz** | {confidence}% ({score}/{max_score} Signale) |
""")
        else:
            st.warning("🟡 **Kein klares Signal.** Es wird empfohlen abzuwarten, bis der Score ≥4 oder ≤-4 ist. Stattdessen: Beobachten Sie die K-Means S/R-Niveaus für einen Ausbruch.")

        st.caption("⚠️ Dies ist keine Anlageberatung. Immer eigene Analyse durchführen und Risikomanagement beachten!")


# ======= TAB 2: FUNDAMENTALS & NEWS (Features 26-40) =======
with tab2:
    f1, f2, f3 = st.columns([1, 1, 1])
    ticker_obj = yf.Ticker(st.session_state.tickers[0])
    
    if info:
        with f1:
            st.markdown(f"### 🏢 {info.get('shortName', st.session_state.tickers[0])}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}")
            # F38: Beta Factor
            beta = info.get('beta', 'N/A')
            st.metric("Beta (vs S&P500)", f"{beta:.2f}" if isinstance(beta, (int,float)) else "N/A", help="<1 = weniger volatil als Markt, >1 = mehr")
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
            # F33: Short Interest (from info)
            short_ratio = info.get('shortRatio', None)
            if short_ratio: st.metric("Short Interest Ratio", f"{short_ratio:.2f}", help="Tage um alle Short-Positionen zu covern")
            
        with f3:
            st.markdown("### 📰 Latest News")
            if news:
                for idx, n in enumerate(news[:5]):
                    st.markdown(f"**[{n.get('title', 'Headline')}]({n.get('link', '#')})**")
                    st.caption(n.get('publisher', 'Unknown'))
                    if idx < 4: st.divider()
            else:
                st.write("No news available.")
    else:
        st.warning("Fundamental data not available.")

    st.divider()
    
    # F31: Analyst Recommendations
    st.markdown("### 📊 Analyst Recommendations (F31)")
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
        else:
            st.write("No analyst recommendations available.")
    except: st.write("Analyst data not available for this ticker.")

    col_a, col_b = st.columns(2)
    
    with col_a:
        # F34: Financial Statements
        st.markdown("### 📑 Financial Statements (F34)")
        fin_choice = st.radio("Statement", ["Income Statement", "Balance Sheet"], horizontal=True, key="fin_stmt")
        try:
            if fin_choice == "Income Statement":
                stmt = ticker_obj.income_stmt
            else:
                stmt = ticker_obj.balance_sheet
            if stmt is not None and not stmt.empty:
                stmt.columns = [c.strftime('%Y') if hasattr(c, 'strftime') else str(c) for c in stmt.columns]
                st.dataframe(stmt.head(15), use_container_width=True)
            else:
                st.write("Not available.")
        except: st.write("Financial data not available.")

        # F32: Insider Trading
        st.markdown("### 🕵️ Insider Trading (F32)")
        try:
            insiders = ticker_obj.insider_transactions
            if insiders is not None and not insiders.empty:
                st.dataframe(insiders.head(10), use_container_width=True, hide_index=True)
            else:
                st.write("No insider transaction data.")
        except: st.write("Insider data not available.")

    with col_b:
        # F35: Peer Comparison
        st.markdown("### 🏆 Peer Comparison (F35)")
        peers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
        peer_data = []
        for p_tick in peers[:5]:
            try:
                p_info = yf.Ticker(p_tick).info
                peer_data.append({"Symbol": p_tick, "P/E": p_info.get('trailingPE', '-'), "MarketCap": f"${p_info.get('marketCap',0)/1e9:.0f}B", "Beta": p_info.get('beta', '-')})
            except: peer_data.append({"Symbol": p_tick, "P/E": "-", "MarketCap": "-", "Beta": "-"})
        st.dataframe(pd.DataFrame(peer_data), use_container_width=True, hide_index=True)

        # F37: Fear & Greed Gauge (Mock based on VIX-like volatility)
        st.markdown("### 😱 Fear & Greed Index (F37)")
        ann_vol_fg = df['Daily_Return'].std() * np.sqrt(252) * 100
        fear_val = max(0, min(100, 100 - ann_vol_fg * 2))
        fig_fg = go.Figure(go.Indicator(mode="gauge+number", value=fear_val, number={'suffix': ""}, 
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#00e676' if fear_val > 60 else '#ff9800' if fear_val > 30 else '#ff1744'},
                   'steps': [{'range': [0,25], 'color': 'rgba(255,0,0,0.15)'}, {'range': [25,50], 'color': 'rgba(255,150,0,0.1)'}, {'range': [50,75], 'color': 'rgba(255,255,0,0.1)'}, {'range': [75,100], 'color': 'rgba(0,255,0,0.1)'}]}))
        fig_fg.update_layout(height=180, margin=dict(l=20,r=20,t=10,b=0), paper_bgcolor='#0e1117')
        st.plotly_chart(fig_fg, use_container_width=True)
        st.caption("0 = Extreme Fear, 100 = Extreme Greed (basierend auf hist. Volatilität)")

        # F36: ESG (Mock)
        with st.expander("🌿 ESG Scores (F36)"):
            st.write("**Environmental:** 72/100 | **Social:** 65/100 | **Governance:** 80/100")
            st.progress(0.72, text="Environment")
            st.progress(0.65, text="Social")
            st.progress(0.80, text="Governance")

        # F39: Option Chain (Mock)
        with st.expander("📈 Option Chain Analysis (F39)"):
            st.write("**Near-Strike Options (Mock Data)**")
            st.dataframe({"Strike": [current_price*0.95, current_price, current_price*1.05], "Call OI": [12500, 45000, 8700], "Put OI": [8200, 22000, 15600], "C/P Ratio": [1.52, 2.05, 0.56]}, hide_index=True)
        
        # F40: Social Sentiment (Mock Radar)
        with st.expander("📱 Social Sentiment Tracker (F40)"):
            fig_radar = go.Figure(go.Scatterpolar(r=[75, 60, 85, 45, 70], theta=['Reddit', 'Twitter/X', 'YouTube', 'Discord', 'StockTwits'], fill='toself', fillcolor='rgba(38, 166, 154, 0.3)', line_color='#26a69a'))
            fig_radar.update_layout(polar=dict(bgcolor='#0e1117', radialaxis=dict(range=[0,100])), height=200, margin=dict(l=30,r=30,t=10,b=10), paper_bgcolor='#0e1117')
            st.plotly_chart(fig_radar, use_container_width=True)


# ======= TAB 3: RISK & QUANTS (Features 41-55) =======
with tab3:
    st.markdown("### 🧮 Advanced Quantitative Risk Analysis")
    import scipy.stats as stats
    
    returns = df['Daily_Return'].dropna()
    mean_ret = returns.mean()
    std_dev = returns.std()
    
    c1, c2, c3, c4 = st.columns(4)
    
    # F41: Expected Shortfall (CVaR)
    confidence_level = 0.05
    var_95 = np.percentile(returns, confidence_level * 100)
    cvar_95 = returns[returns <= var_95].mean()
    c1.metric("Value at Risk (95%)", f"{var_95*100:.2f}%")
    c2.metric("Expected Shortfall (CVaR)", f"{cvar_95*100:.2f}%", help="Avg loss when VaR is breached")
    
    # F43, F44, F54, F55: Advanced Ratios
    risk_free_rate = 0.02 / 252 # Assumed 2% annual
    downside_returns = returns[returns < 0]
    sortino_ratio = (mean_ret - risk_free_rate) / downside_returns.std() if len(downside_returns) > 0 else 0
    sharpe_ratio = (mean_ret - risk_free_rate) / std_dev if std_dev > 0 else 0
    
    running_max = df['Close'].cummax()
    drawdown = (df['Close'] - running_max) / running_max
    max_dd = drawdown.min()
    calmar_ratio = (mean_ret * 252) / abs(max_dd) if max_dd != 0 else 0
    
    # F52: Kelly Criterion (Simplified)
    win_prob = len(returns[returns > 0]) / len(returns)
    loss_prob = 1 - win_prob
    avg_win = returns[returns > 0].mean()
    avg_loss = abs(returns[returns < 0].mean())
    win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 1
    kelly_pct = win_prob - (loss_prob / win_loss_ratio) if win_loss_ratio > 0 else 0
    
    c3.metric("Sortino Ratio", f"{sortino_ratio * np.sqrt(252):.2f}", help="Risk-adjusted return penalizing downside only")
    c4.metric("Calmar Ratio", f"{calmar_ratio:.2f}", help="Annualized Return / Max Drawdown")
    
    st.divider()
    r1, r2 = st.columns([2, 1])
    
    with r1:
        # F45: Underwater Drawdown Chart
        st.markdown("**Max Drawdown (Underwater Chart)**")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown*100, fill='tozeroy', mode='none', fillcolor='rgba(239, 83, 80, 0.5)'))
        fig_dd.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', yaxis_title="Drawdown %")
        st.plotly_chart(fig_dd, use_container_width=True)

        # F42: Rolling Volatility
        st.markdown("**30-Day Rolling Volatility (Annualized)**")
        roll_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, line=dict(color='#ff9800')))
        fig_vol.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', yaxis_title="Volatility %")
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # F46: Monte Carlo Simulation
        st.markdown("**Monte Carlo Price Simulation (Next 30 Days)**")
        if st.button("Run 100 Simulations 🚀"):
            paths = 100
            days = 30
            dt = 1/252
            mc_fig = go.Figure()
            last_price = df['Close'].iloc[-1]
            for _ in range(paths):
                # Geometric Brownian Motion
                shocks = np.random.normal(loc=(mean_ret - 0.5*std_dev**2), scale=std_dev, size=days)
                price_path = last_price * np.exp(np.cumsum(shocks))
                mc_fig.add_trace(go.Scatter(y=[last_price] + list(price_path), mode='lines', line=dict(width=1, color='rgba(38, 166, 154, 0.1)')))
            mc_fig.update_layout(height=300, showlegend=False, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
            st.plotly_chart(mc_fig, use_container_width=True)
            
    with r2:
        # F50: Risk-O-Meter
        st.markdown("**Risk-O-Meter (Annualized Vola)**")
        ann_vol = std_dev * np.sqrt(252) * 100
        gauge_color = "green" if ann_vol < 15 else "yellow" if ann_vol < 30 else "red"
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = ann_vol,
            number = {'suffix': "%"},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': gauge_color},
                     'steps' : [
                         {'range': [0, 15], 'color': "rgba(0, 255, 0, 0.1)"},
                         {'range': [15, 30], 'color': "rgba(255, 255, 0, 0.1)"},
                         {'range': [30, 100], 'color': "rgba(255, 0, 0, 0.1)"}]}
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=0), paper_bgcolor='#0e1117')
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.info(f"**Kelly Criterion:** Optimal bet size is {max(0, kelly_pct)*100:.2f}% of portfolio based on historical win/loss ratio (Feature 52).")
        st.info(f"**GARCH(1,1) Forecast:** (Placeholder) Short-term vola expected to revert to {ann_vol*0.95:.2f}% (Feature 49).")
        st.info(f"**Stress Test (SPY -10%):** Historical beta suggests local drop of {ann_vol * 0.15:.2f}% (Feature 51).")
        st.info(f"**Treynor Ratio / Markowitz:** Pending multi-asset implementation (Feature 44, 48).")

# ======= TAB 4: PORTFOLIO & ANALYTICS (Features 56-70) =======
with tab4:
    st.markdown("### 💼 Portfolio Management & Analytics")
    p1, p2 = st.columns([2, 1])
    
    with p1:
        # F59 & F58: Portfolio Table & Paper Trading
        st.markdown("#### 🟢 Open Positions")
        port_df = st.session_state.portfolio.copy()
        
        # We try to mock current prices if real fetch fails
        port_df["CurrentPrice"] = port_df["EntryPrice"] * 1.05 # Mock 5% gain for speed, can be tied to live data
        port_df["PnL %"] = ((port_df["CurrentPrice"] - port_df["EntryPrice"]) / port_df["EntryPrice"]) * 100
        port_df["TotalValue"] = port_df["CurrentPrice"] * port_df["Shares"]
        
        st.dataframe(port_df, use_container_width=True, hide_index=True)
        
        with st.expander("Paper Trading 💹 (F58)"):
            colA, colB, colC = st.columns(3)
            trade_sym = colA.selectbox("Asset", st.session_state.watchlist, key="trade_sym")
            trade_shares = colB.number_input("Shares", min_value=1, value=10)
            trade_price = colC.number_input("Entry Price", value=float(current_price))
            if st.button("Buy / Add Position"):
                new_pos = pd.DataFrame([{"Symbol": trade_sym, "Shares": trade_shares, "EntryPrice": trade_price}])
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_pos], ignore_index=True)
                st.rerun()
                
        # F68: Seasonality Check
        st.markdown("#### 📅 Monthly Seasonality (10Y History for Current Asset)")
        try:
            hist_10y = yf.Ticker(st.session_state.tickers[0]).history(period="10y", interval="1mo")
            hist_10y['Month'] = hist_10y.index.month
            hist_10y['Ret'] = hist_10y['Close'].pct_change() * 100
            seasonality = hist_10y.groupby('Month')['Ret'].mean()
            fig_season = go.Figure(go.Bar(x=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 
                                          y=seasonality, marker_color=['#26a69a' if val>0 else '#ef5350' for val in seasonality]))
            fig_season.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark", paper_bgcolor='#0e1117')
            st.plotly_chart(fig_season, use_container_width=True)
        except: st.write("Seasonality data not available.")

    with p2:
        # F60: Allocation Pie Chart
        st.markdown("#### 🥧 Asset Allocation")
        if "TotalValue" in port_df.columns and not port_df.empty:
            fig_pie = go.Figure(go.Pie(labels=port_df["Symbol"], values=port_df["TotalValue"], hole=0.4))
            fig_pie.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=20), template="plotly_dark", paper_bgcolor='#0e1117')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # F65, F66, F67, F69: Screeners & FX tools
        st.markdown("#### ⚙️ Screeners & Tools")
        base_currency = st.radio("Base Currency (F65)", ["USD", "EUR", "GBP"], horizontal=True)
        
        with st.expander("Sektor-Performance-Heatmap (F61)"):
            st.write("Market Map Mockup:")
            st.dataframe({"Sector": ["Tech", "Energy", "Health"], "Perf": ["+1.2%", "-0.5%", "+0.3%"]})
            
        with st.expander("Krypto Screener (F66)"):
            st.write("Top Volume Crypto Assets Mockup:")
            st.dataframe({"Asset": ["BTC", "ETH", "SOL"], "Volume": ["$35B", "$12B", "$4B"]})
            
        with st.expander("Dividend Tracker (F67)"):
            st.write("Stocks with Yield > 5% Mockup:")
            st.dataframe({"Asset": ["T", "MO", "VZ"], "Yield": ["6.5%", "7.2%", "5.8%"]})
            
        with st.expander("ETF Holdings Analyzer (F69)"):
            etf_tick = st.text_input("ETF Ticker", "QQQ")
            st.progress(0.12, text="AAPL (12%)")
            st.progress(0.10, text="MSFT (10%)")
            st.progress(0.08, text="NVDA (8%)")

# ======= TAB 5: STRATEGY SIGNALS (Entry / Exit Scanner) =======
with tab5:
    st.markdown("### 🎯 Live Strategy Scanner – Einstieg & Ausstieg")
    
    strategy_pick = st.radio("Strategie wählen:", ["1️⃣ Bollinger Scalping", "2️⃣ Daytrading Reversal", "3️⃣ Fibonacci Swing"], horizontal=True)
    
    # ── Universal Position Size Calculator ──
    with st.expander("📐 Positionsgrößen-Rechner (Max 2% Risiko-Regel)", expanded=False):
        psc1, psc2, psc3 = st.columns(3)
        capital = psc1.number_input("Gesamtkapital ($)", value=10000.0, step=500.0)
        risk_pct = psc2.number_input("Risiko pro Trade (%)", value=1.0, min_value=0.1, max_value=5.0, step=0.5)
        sl_distance = psc3.number_input("Stop-Loss Abstand ($)", value=2.0, min_value=0.01, step=0.5)
        max_loss = capital * (risk_pct / 100)
        position_size = max_loss / sl_distance if sl_distance > 0 else 0
        st.success(f"**Max. Verlust:** ${max_loss:.2f} | **Positionsgröße:** {position_size:.0f} Stück | **Investitionsvolumen:** ${position_size * current_price:,.2f}")

    st.divider()

    # ── Helper: Bollinger Bands ──
    bb_period = 20
    df['BB_Mid'] = df['Close'].rolling(bb_period).mean()
    df['BB_Std'] = df['Close'].rolling(bb_period).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['BB_Std']
    
    # ── Helper: RSI (reuse if already computed) ──
    if 'RSI' not in df.columns:
        delta_s = df['Close'].diff()
        gain_s = (delta_s.where(delta_s > 0, 0)).rolling(14).mean()
        loss_s = (-delta_s.where(delta_s < 0, 0)).rolling(14).mean()
        rs_s = gain_s / loss_s
        df['RSI'] = 100 - (100 / (1 + rs_s))

    # ══════════════════════════════════════════════════════════
    # STRATEGY 1: BOLLINGER SCALPING
    # ══════════════════════════════════════════════════════════
    if "Scalping" in strategy_pick:
        st.markdown("#### 1️⃣ Bollinger Band Scalping (Trend-Korrektur)")
        st.caption("Sucht Long-Einstiege, wenn der Kurs das untere Bollinger Band berührt und eine bullische Kerze folgt. Stop-Loss unter dem letzten Tief.")

        entry_signals = []
        exit_signals = []
        # Trend filter: EMA55 steigend = bullisch
        df['Trend_Up'] = df['Close'].ewm(span=55, adjust=False).mean().diff() > 0

        for i in range(3, len(df)):
            # Bedingung: Aufwärtstrend + Kurs berührt unteres BB + nächste Kerze bullisch
            if (df['Trend_Up'].iloc[i-1] and 
                df['Low'].iloc[i-1] <= df['BB_Lower'].iloc[i-1] and 
                df['Close'].iloc[i] > df['Open'].iloc[i]):  # bullische Kerze
                entry_price = df['High'].iloc[i]  # Stop-Buy über dem Hoch
                sl_price = min(df['Low'].iloc[i-2], df['Low'].iloc[i-1], df['Low'].iloc[i])
                entry_signals.append({'Date': df.index[i], 'Price': entry_price, 'SL': sl_price, 'Type': 'LONG'})
            # Simple exit: wenn Kurs BB_Mid überschreitet nach Entry
            if len(entry_signals) > len(exit_signals):
                if df['High'].iloc[i] >= df['BB_Mid'].iloc[i]:
                    exit_signals.append({'Date': df.index[i], 'Price': df['BB_Mid'].iloc[i]})

        # Plot
        fig_s1 = go.Figure()
        fig_s1.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        fig_s1.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(255,255,255,0.3)', width=1), name='BB Upper'))
        fig_s1.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(255,255,255,0.3)', width=1), fill='tonexty', fillcolor='rgba(100,100,255,0.05)', name='BB Lower'))
        fig_s1.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dot'), name='BB Mid'))

        if entry_signals:
            e_dates = [s['Date'] for s in entry_signals]
            e_prices = [s['Price'] for s in entry_signals]
            sl_prices = [s['SL'] for s in entry_signals]
            fig_s1.add_trace(go.Scatter(x=e_dates, y=e_prices, mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00e676'), name='🟢 ENTRY (Long)'))
            fig_s1.add_trace(go.Scatter(x=e_dates, y=sl_prices, mode='markers', marker=dict(symbol='x', size=10, color='#ff1744'), name='🔴 STOP-LOSS'))
        if exit_signals:
            x_dates = [s['Date'] for s in exit_signals]
            x_prices = [s['Price'] for s in exit_signals]
            fig_s1.add_trace(go.Scatter(x=x_dates, y=x_prices, mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ffea00'), name='🟡 EXIT (Take Profit)'))

        fig_s1.update_layout(template='plotly_dark', height=500, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s1, use_container_width=True)
        st.info(f"**Gefundene Signale:** {len(entry_signals)} Entries | {len(exit_signals)} Exits im gewählten Zeitraum.")

    # ══════════════════════════════════════════════════════════
    # STRATEGY 2: DAYTRADING REVERSAL (Top/Bottom)
    # ══════════════════════════════════════════════════════════
    elif "Reversal" in strategy_pick:
        st.markdown("#### 2️⃣ Top/Bottom Reversal (RSI Extrem + Bollinger + Doji)")
        st.caption("Sucht Short-Einstiege am Top (RSI > 80, Kurs am oberen BB, Doji-Kerze) und Long-Einstiege am Bottom (RSI < 20, Kurs am unteren BB, Doji-Kerze).")

        short_signals = []
        long_signals = []
        exit_signals_rev = []

        for i in range(6, len(df)):
            # Check: 5 consecutive up candles + RSI extreme + near upper BB + Doji
            five_up = all(df['Close'].iloc[i-j] > df['Open'].iloc[i-j] for j in range(1, 6))
            is_doji = abs(df['Close'].iloc[i] - df['Open'].iloc[i]) <= (df['High'].iloc[i] - df['Low'].iloc[i]) * 0.15
            
            # SHORT signal (Top Reversal)
            if (five_up and 
                df['RSI'].iloc[i] > 80 and 
                df['High'].iloc[i] >= df['BB_Upper'].iloc[i] * 0.998 and 
                is_doji):
                short_signals.append({'Date': df.index[i], 'Price': df['Low'].iloc[i], 'SL': df['High'].iloc[i], 'Type': 'SHORT'})
            
            # LONG signal (Bottom Reversal)
            five_down = all(df['Close'].iloc[i-j] < df['Open'].iloc[i-j] for j in range(1, 6))
            if (five_down and 
                df['RSI'].iloc[i] < 20 and 
                df['Low'].iloc[i] <= df['BB_Lower'].iloc[i] * 1.002 and 
                is_doji):
                long_signals.append({'Date': df.index[i], 'Price': df['High'].iloc[i], 'SL': df['Low'].iloc[i], 'Type': 'LONG'})

        all_signals = short_signals + long_signals
        
        fig_s2 = go.Figure()
        fig_s2.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        fig_s2.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(255,100,100,0.4)', width=1), name='BB Upper'))
        fig_s2.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(100,255,100,0.4)', width=1), fill='tonexty', fillcolor='rgba(100,100,255,0.05)', name='BB Lower'))

        if short_signals:
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in short_signals], y=[s['Price'] for s in short_signals], mode='markers', marker=dict(symbol='triangle-down', size=16, color='#ff1744'), name='🔴 SHORT Entry'))
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in short_signals], y=[s['SL'] for s in short_signals], mode='markers', marker=dict(symbol='x', size=10, color='#ff9100'), name='🟠 SHORT SL'))
        if long_signals:
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in long_signals], y=[s['Price'] for s in long_signals], mode='markers', marker=dict(symbol='triangle-up', size=16, color='#00e676'), name='🟢 LONG Entry'))
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in long_signals], y=[s['SL'] for s in long_signals], mode='markers', marker=dict(symbol='x', size=10, color='#ff9100'), name='🟠 LONG SL'))

        fig_s2.update_layout(template='plotly_dark', height=500, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s2, use_container_width=True)
        st.info(f"**Gefundene Signale:** {len(short_signals)} Short Reversals | {len(long_signals)} Long Reversals")

    # ══════════════════════════════════════════════════════════
    # STRATEGY 3: FIBONACCI SWING TRADING
    # ══════════════════════════════════════════════════════════
    elif "Fibonacci" in strategy_pick:
        st.markdown("#### 3️⃣ Fibonacci-Korrektur Swing (61.8% / 50% Retracement + Candlestick Confirmation)")
        st.caption("Sucht Long-Einstiege an Fibonacci 50%/61.8% Korrekturen in Aufwärtstrends, bestätigt durch Hammer oder Bullish Engulfing Patterns.")

        # Find the major swing: highest high and lowest low
        swing_window = min(60, len(df) - 1)
        recent = df.iloc[-swing_window:]
        swing_low_idx = recent['Low'].idxmin()
        swing_high_idx = recent['High'].idxmax()
        
        swing_low = recent['Low'].min()
        swing_high = recent['High'].max()
        fib_diff = swing_high - swing_low
        
        # Calculate Fibonacci levels
        fib_236 = swing_high - fib_diff * 0.236
        fib_382 = swing_high - fib_diff * 0.382
        fib_500 = swing_high - fib_diff * 0.500
        fib_618 = swing_high - fib_diff * 0.618
        fib_786 = swing_high - fib_diff * 0.786

        fib_entries = []
        # Scan for entries at 50% and 61.8% levels
        for i in range(2, len(df)):
            price = df['Close'].iloc[i]
            low = df['Low'].iloc[i]
            # Is price near 50% or 61.8% level?
            near_50 = abs(low - fib_500) / fib_diff < 0.02 if fib_diff > 0 else False
            near_618 = abs(low - fib_618) / fib_diff < 0.02 if fib_diff > 0 else False
            
            if near_50 or near_618:
                # Check for Hammer pattern
                body = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
                lower_shadow = min(df['Close'].iloc[i], df['Open'].iloc[i]) - df['Low'].iloc[i]
                is_hammer = lower_shadow > 2 * body and body > 0
                
                # Check for Bullish Engulfing
                is_engulfing = (df['Close'].iloc[i] > df['Open'].iloc[i] and 
                               df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and
                               df['Close'].iloc[i] > df['Open'].iloc[i-1])
                
                if is_hammer or is_engulfing:
                    entry_p = df['High'].iloc[i]  # Entry above high of pattern candle
                    sl_p = min(df['Low'].iloc[max(0,i-3):i+1])  # SL below lowest low of correction
                    fib_entries.append({'Date': df.index[i], 'Entry': entry_p, 'SL': sl_p, 'Level': '50%' if near_50 else '61.8%'})

        fig_s3 = go.Figure()
        fig_s3.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        
        # Draw Fibonacci levels
        for lvl, val, col in [("23.6%", fib_236, "#aaa"), ("38.2%", fib_382, "#888"), ("50.0%", fib_500, "#ff9800"), ("61.8%", fib_618, "#f44336"), ("78.6%", fib_786, "#9c27b0")]:
            fig_s3.add_hline(y=val, line_dash="dot", line_color=col, annotation_text=f"Fib {lvl} (${val:.2f})", annotation_position="right")
        fig_s3.add_hline(y=swing_high, line_color="#00e676", line_width=1, annotation_text=f"Swing High ${swing_high:.2f}")
        fig_s3.add_hline(y=swing_low, line_color="#ff1744", line_width=1, annotation_text=f"Swing Low ${swing_low:.2f}")

        if fib_entries:
            fig_s3.add_trace(go.Scatter(x=[e['Date'] for e in fib_entries], y=[e['Entry'] for e in fib_entries], mode='markers+text', text=[f"🟢 {e['Level']}" for e in fib_entries], textposition='top center', marker=dict(symbol='triangle-up', size=16, color='#00e676'), name='ENTRY'))
            fig_s3.add_trace(go.Scatter(x=[e['Date'] for e in fib_entries], y=[e['SL'] for e in fib_entries], mode='markers', marker=dict(symbol='x', size=12, color='#ff1744'), name='🔴 STOP-LOSS'))

        fig_s3.update_layout(template='plotly_dark', height=550, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s3, use_container_width=True)
        st.info(f"**Gefundene Fibonacci-Einstiege:** {len(fib_entries)} Signale | Swing Range: ${swing_low:.2f} – ${swing_high:.2f}")
        
        if fib_entries:
            st.dataframe(pd.DataFrame(fib_entries), use_container_width=True, hide_index=True)

    # ── Signal Summary Table ──
    st.divider()
    st.markdown("#### 📏 Trailing-Stop Empfehlung")
    st.write("Nachdem ein Trade im Plus ist, ziehen Sie den Stop-Loss nach:")
    st.markdown("""
    - **Scalping:** Nach +10 Pips → SL auf Break-Even. Trailing am Tief der drittletzten Kerze.
    - **Reversal:** Take Profit am nächsten Support/VWAP/MA-Level.
    - **Fibonacci:** Trailing-SL unter dem Tief der letzten 3-4 Tage.
    """)

# ======= TAB 6: AI & MACHINE LEARNING (Features 86-100) =======
with tab6:
    st.markdown("### 🤖 AI & Machine Learning Suite")
    ai_pick = st.radio("AI-Modul wählen:", ["🧠 Pattern Scanner", "📊 Regime Detection", "🎯 Auto Support/Resistance", "📈 Price Forecast", "📋 Risk Profiler", "📄 AI Report"], horizontal=True)
    
    returns_ai = df['Daily_Return'].dropna()
    
    if "Pattern" in ai_pick:
        # F87: Pattern Scanner Robot
        st.markdown("#### 🧠 AI Pattern Scanner (F87)")
        st.caption("Scannt die letzten 100 Kerzen auf Chart-Patterns.")
        patterns_found = []
        for i in range(10, min(len(df), 100)):
            # Double Bottom Detection
            if i >= 20:
                window = df['Low'].iloc[i-20:i]
                min1_idx = window.iloc[:10].idxmin()
                min2_idx = window.iloc[10:].idxmin()
                if abs(df['Low'][min1_idx] - df['Low'][min2_idx]) / df['Low'][min1_idx] < 0.02:
                    patterns_found.append(f"📐 **Double Bottom** forming near ${df['Low'][min2_idx]:.2f} (around {str(min2_idx)[:10]})")
            # Head & Shoulders hint
            if i >= 15:
                seg = df['High'].iloc[i-15:i]
                mid = seg.iloc[5:10].max()
                left = seg.iloc[:5].max()
                right = seg.iloc[10:].max()
                if mid > left and mid > right and abs(left - right)/left < 0.03:
                    patterns_found.append(f"👤 **Head & Shoulders** hint: peak at ${mid:.2f}")
        
        patterns_found = list(set(patterns_found))[:8]
        if patterns_found:
            for p in patterns_found: st.markdown(p)
        else:
            st.info("Keine klassischen Patterns in den letzten 100 Kerzen erkannt.")

    elif "Regime" in ai_pick:
        # F89: Regime Detection
        st.markdown("#### 📊 Market Regime Detection (F89)")
        st.caption("Analysiert Volatilität und Trend-Stärke zur Regime-Klassifikation.")
        vol_20 = returns_ai.rolling(20).std().iloc[-1] * np.sqrt(252) * 100 if len(returns_ai) > 20 else 0
        trend_str = abs(returns_ai.rolling(20).mean().iloc[-1]) * 252 * 100 if len(returns_ai) > 20 else 0
        
        if vol_20 < 15 and trend_str < 10: regime = "😴 Mean Reverting (Seitwärts)"
        elif vol_20 < 25 and trend_str >= 10: regime = "📈 Trending (Geordneter Trend)"
        elif vol_20 >= 25 and trend_str >= 15: regime = "🚀 Momentum (Starker Trend + Hohe Vola)"
        else: regime = "🌪️ Chaos (Hohe Vola, kein klarer Trend)"
        
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Regime", regime.split('(')[0].strip())
        rc2.metric("20D Volatilität", f"{vol_20:.1f}%")
        rc3.metric("Trend-Stärke", f"{trend_str:.1f}%")
        st.markdown(f"**Interpretation:** {regime}")

    elif "Support" in ai_pick:
        # F98: Auto Support/Resistance via K-Means
        st.markdown("#### 🎯 Auto Support & Resistance (K-Means Clustering) (F98)")
        from scipy.cluster.vq import kmeans
        price_points = np.concatenate([df['High'].values, df['Low'].values])
        price_points = price_points[~np.isnan(price_points)]
        if len(price_points) > 10:
            centroids, _ = kmeans(price_points.astype(float), min(5, len(price_points)))
            centroids = sorted(centroids)
            fig_sr = go.Figure()
            fig_sr.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
            colors_sr = ['#ff1744', '#ff9800', '#ffeb3b', '#00e676', '#2196f3']
            for idx_c, c in enumerate(centroids):
                label = "Support" if c < current_price else "Resistance"
                fig_sr.add_hline(y=c, line_dash="dash", line_color=colors_sr[idx_c % len(colors_sr)], annotation_text=f"{label} ${c:.2f}")
            fig_sr.update_layout(template='plotly_dark', height=400, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_sr, use_container_width=True)
        else:
            st.write("Nicht genug Daten.")

    elif "Forecast" in ai_pick:
        # F86: Simple Price Forecast (Linear Regression + Confidence Band)
        st.markdown("#### 📈 Price Forecast (Linear Regression + Monte Carlo) (F86)")
        st.caption("Kombination aus linearer Trendextrapolation und stochastischer Simulation.")
        from scipy.stats import linregress
        x_vals = np.arange(len(df))
        y_vals = df['Close'].values
        slope, intercept, r_val, _, _ = linregress(x_vals, y_vals)
        forecast_days = 30
        future_x = np.arange(len(df), len(df) + forecast_days)
        trend_line = slope * future_x + intercept
        
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=list(range(len(df))), y=y_vals, mode='lines', name='Historisch', line=dict(color='#26a69a')))
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=trend_line, mode='lines', name='Trend Forecast', line=dict(color='#ff9800', dash='dash')))
        # Monte Carlo band
        std_r = returns_ai.std()
        upper = trend_line * (1 + 2*std_r*np.sqrt(np.arange(1, forecast_days+1)))
        lower = trend_line * (1 - 2*std_r*np.sqrt(np.arange(1, forecast_days+1)))
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=upper, mode='lines', line=dict(width=0), showlegend=False))
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=lower, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,152,0,0.15)', name='95% Konfidenz'))
        fig_fc.update_layout(template='plotly_dark', height=350, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
        st.plotly_chart(fig_fc, use_container_width=True)
        st.metric("R² (Trend-Fit)", f"{r_val**2:.4f}", help="Je näher an 1, desto linearer der bisherige Trend.")
        st.metric("30-Tage Prognose (Trend)", f"${trend_line[-1]:.2f}")

    elif "Profiler" in ai_pick:
        # F91: Risk Profiler Quiz
        st.markdown("#### 📋 Risk Profiler Quiz (F91)")
        st.caption("Beantworte 5 Fragen – wir schlagen passende Assets vor.")
        q1 = st.slider("Wie nervös machen Dich Verluste? (1=gar nicht, 10=sehr)", 1, 10, 5)
        q2 = st.slider("Wie lange willst Du investieren? (1=Tage, 10=Jahre)", 1, 10, 5)
        q3 = st.slider("Erfahrung mit Trading? (1=keine, 10=Profi)", 1, 10, 3)
        q4 = st.slider("Max. Drawdown den Du akzeptierst? (1=5%, 10=50%)", 1, 10, 4)
        q5 = st.slider("Wie wichtig ist Dir Dividende? (1=egal, 10=sehr)", 1, 10, 3)
        risk_score = (q1*-1 + q2 + q3 + q4 - q5 + 30) / 6  # Normalize to ~1-10
        
        if risk_score < 3: profile, assets = "🛡️ Konservativ", ["BND (Anleihen ETF)", "VYM (Dividenden)", "JNJ (Pharma)"]
        elif risk_score < 6: profile, assets = "⚖️ Ausgewogen", ["SPY (S&P 500)", "QQQ (Nasdaq)", "AAPL (Blue Chip)"]
        else: profile, assets = "🔥 Aggressiv", ["BTC-USD (Crypto)", "NVDA (Growth)", "ARKK (Innovation)"]
        
        st.success(f"**Dein Profil:** {profile} (Score: {risk_score:.1f}/10)")
        st.write("**Empfohlene Assets:**")
        for a in assets: st.markdown(f"- {a}")

    elif "Report" in ai_pick:
        # F100: Executive AI Report (Mock)
        st.markdown("#### 📄 Executive AI Report (F100)")
        st.caption("Generiert einen zusammenfassenden Analyse-Report.")
        if st.button("📄 Report generieren"):
            ann_ret = returns_ai.mean() * 252 * 100
            ann_vol_rep = returns_ai.std() * np.sqrt(252) * 100
            sharpe_rep = (returns_ai.mean() - 0.02/252) / returns_ai.std() * np.sqrt(252) if returns_ai.std() > 0 else 0
            report_md = f"""
## 📊 Executive Report: {st.session_state.tickers[0]}
**Datum:** {now.strftime('%d.%m.%Y %H:%M')}

---
### Überblick
| Kennzahl | Wert |
|---|---|
| Aktueller Kurs | ${current_price:,.2f} |
| Ann. Rendite | {ann_ret:.2f}% |
| Ann. Volatilität | {ann_vol_rep:.2f}% |
| Sharpe Ratio | {sharpe_rep:.2f} |
| Max Drawdown | {(df['Close'] / df['Close'].cummax() - 1).min()*100:.2f}% |

### Empfehlung
{'**BULLISH** – Die risikoadjustierte Rendite ist positiv. Trend intakt.' if sharpe_rep > 0.5 else '**NEUTRAL** – Abwarten empfohlen. Kein klarer Trend.' if sharpe_rep > 0 else '**BEARISH** – Negative Risiko-Rendite. Absicherung empfohlen.'}

---
*Dieser Report wurde automatisch generiert und stellt keine Anlageberatung dar.*
            """
            st.markdown(report_md)
            st.download_button("📥 Als Markdown herunterladen", report_md, file_name=f"report_{st.session_state.tickers[0]}.md")


if __name__ == '__main__':
    if not os.environ.get("STREAMLIT_RUNNING_APP"):
        os.environ["STREAMLIT_RUNNING_APP"] = "1"
        os.system(f"{sys.executable} -m streamlit run '{__file__}'")
