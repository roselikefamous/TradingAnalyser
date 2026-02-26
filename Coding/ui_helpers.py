"""
ui_helpers.py – UI Utilities for the Trading Terminal
CSS styles, signal card renderers, strategy docs, backtest visualizations.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


# ═══════════════════════════════════════════════════════════
# CSS STYLES
# ═══════════════════════════════════════════════════════════

TERMINAL_CSS = """
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stApp { background-color: #0e1117; color: #fafafa; }
    /* KPI cards with premium 3D hover effect */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #161922, #101217);
        border: 1px solid #2d3139; padding: 18px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease, border-color 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="metric-container"]::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, transparent, #26a69a, transparent); opacity: 0;
        transition: opacity 0.3s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: #26a69a;
        box-shadow: 0 12px 25px rgba(0,0,0,0.5), 0 0 15px rgba(38,166,154,0.15);
    }
    div[data-testid="metric-container"]:hover::before { opacity: 1; }
    
    div.row-widget.stRadio > div { flex-direction: row; }
    .stTextInput>div>div>input {
        font-size: 16px !important; font-weight: 600 !important;
        background-color: #1a1d24 !important; border-color: #2d3139 !important;
        border-radius: 8px !important;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .stTextInput>div>div>input:focus {
        border-color: #26a69a !important; box-shadow: 0 0 0 2px rgba(38,166,154,0.2) !important;
    }
    /* Responsive */
    @media (max-width: 768px) {
        div[data-testid="metric-container"] { min-width: 100% !important; }
        .stTabs [data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; padding-bottom: 5px; }
    }
    /* Tab styling (Premium) */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important; font-size: 15px !important; color: #9e9e9e;
        padding: 10px 16px !important; transition: color 0.2s;
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;
        border-bottom: 3px solid #26a69a !important;
    }
    /* Floating Action Button */
    .fab-btn {
        position: fixed; bottom: 30px; right: 30px; z-index: 9999;
        background: linear-gradient(135deg, #26a69a, #00897b); color: white;
        border: none; border-radius: 50%; width: 56px; height: 56px; font-size: 22px;
        cursor: pointer; box-shadow: 0 6px 20px rgba(0,0,0,0.5);
        transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s;
    }
    .fab-btn:hover { transform: scale(1.15) rotate(15deg); box-shadow: 0 8px 25px rgba(38,166,154,0.4); }
    /* Skeleton loading animation */
    @keyframes pulse { 0% {opacity:0.6} 50% {opacity:1} 100% {opacity:0.6} }
    .skeleton {
        background: linear-gradient(90deg, #1e1e1e 25%, #2a2a2a 50%, #1e1e1e 75%);
        background-size: 200% 100%; animation: pulse 1.5s ease-in-out infinite; border-radius: 8px; height: 200px;
    }
    /* Watermark */
    .watermark {
        position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
        font-size: 180px; font-weight: 900; opacity: 0.02; color: #fafafa;
        pointer-events: none; z-index: 0; letter-spacing: 25px;
    }
    /* Live Clock */
    .live-clock {
        position: fixed; top: 8px; right: 16px; z-index: 9999;
        font-family: 'JetBrains Mono', 'Courier New', monospace; font-size: 12px; color: #26a69a;
        background: rgba(14,17,23,0.95); padding: 4px 14px; border-radius: 6px;
        border: 1px solid #2d3139; backdrop-filter: blur(10px);
    }
    /* Strategy card styling */
    .strategy-card {
        padding: 16px 20px; margin: 8px 0; border-radius: 10px;
        background: linear-gradient(135deg, rgba(30,30,30,0.8), rgba(20,20,20,0.9));
        border: 1px solid #2d3139;
    }
    /* Signal cards */
    .signal-card {
        padding: 10px 16px; margin: 4px 0; border-radius: 8px; font-size: 14px;
        transition: all 0.2s;
    }
    .signal-card:hover {
        transform: translateX(4px);
    }
    /* Backtest stats card */
    .backtest-card {
        background: linear-gradient(135deg, #1a1d24, #12151a);
        border: 1px solid #2d3139; border-radius: 12px;
        padding: 20px; margin: 10px 0;
    }
    /* ── Sidebar Fixes ─────────────────────────────────────────
       Prevent text going vertical when sidebar is narrow.     */
    section[data-testid="stSidebar"] {
        min-width: 260px !important;
        writing-mode: horizontal-tb !important;
    }
    section[data-testid="stSidebar"] * {
        writing-mode: horizontal-tb !important;
        text-orientation: mixed !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stSlider,
    section[data-testid="stSidebar"] .stCheckbox {
        white-space: normal !important;
    }
    section[data-testid="stSidebar"] h1 { font-size: 1.1rem !important; }
    /* Custom scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #2d3139; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #26a69a; }
</style>
"""


# ═══════════════════════════════════════════════════════════
# SIGNAL CARD RENDERERS
# ═══════════════════════════════════════════════════════════

def render_signal_card(name: str, description: str, signal_type: str):
    """Render a styled signal card (bullish/bearish/neutral)."""
    colors = {
        'bullish': ('#00e676', 'rgba(0,230,118,0.08)'),
        'bearish': ('#ff1744', 'rgba(255,23,68,0.08)'),
        'neutral': ('#ffea00', 'rgba(255,234,0,0.05)'),
    }
    color, bg = colors.get(signal_type, colors['neutral'])
    st.markdown(
        f"<div class='signal-card' style='border-left:4px solid {color};background:{bg};'>"
        f"<b>{name}</b> — {description}</div>",
        unsafe_allow_html=True
    )


def render_backtest_stats(stats: dict, title="📊 Backtest-Ergebnisse"):
    """Render backtest statistics in a styled card."""
    if stats['total_signals'] == 0:
        st.info("Keine Signale im gewählten Zeitraum gefunden.")
        return

    st.markdown(f"#### {title}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Signale", stats['total_signals'])
    c2.metric("Exits", stats.get('total_exits', stats.get('short_count', 0) + stats.get('long_count', 0)))

    win_rate = stats.get('win_rate')
    if win_rate is not None:
        c3.metric("Win Rate", f"{win_rate:.0f}%")
    else:
        c3.metric("Win Rate", "N/A")

    c4.metric("Ø R:R", f"1:{stats.get('avg_rr', 0):.1f}")

    if 'net_pnl_pct' in stats:
        c5.metric("Net P&L", f"{stats['net_pnl_pct']:.2f}%",
                  delta=f"{'✅' if stats['net_pnl_pct'] > 0 else '❌'}")
    elif 'short_count' in stats:
        c5.metric("Short/Long", f"{stats.get('short_count', 0)}/{stats.get('long_count', 0)}")


def render_equity_curve(entries, exits, title="Equity Curve"):
    """Render a simple equity curve from trade results."""
    if not entries or not exits:
        return

    paired = min(len(entries), len(exits))
    if paired == 0:
        return

    equity = [0]
    for i in range(paired):
        entry = entries[i]['Entry']
        exit_price = exits[i]['Price']
        pnl_pct = (exit_price - entry) / entry * 100 if entries[i]['Type'] == 'LONG' else (entry - exit_price) / entry * 100
        equity.append(equity[-1] + pnl_pct)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity, mode='lines+markers',
        line=dict(color='#26a69a' if equity[-1] >= 0 else '#ef5350', width=2),
        marker=dict(size=5),
        fill='tozeroy',
        fillcolor='rgba(38,166,154,0.1)' if equity[-1] >= 0 else 'rgba(239,83,80,0.1)',
    ))
    fig.update_layout(
        title=title, template='plotly_dark', height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        yaxis_title="Kumulative P&L (%)", showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# STRATEGY DOCUMENTATION
# ═══════════════════════════════════════════════════════════

STRATEGY_DOC = """
## 📖 Strategie-Regelwerk

---

### 1️⃣ Bollinger Band Scalping (Schnelle Gewinne im Trend)

| Parameter | Regel |
|---|---|
| **Zeitrahmen** | 5-Min Chart (Entry), Tageschart (Trendrichtung) |
| **Indikatoren** | Bollinger Bänder (20, 2σ), EMA 55, Stochastik |
| **Trend-Filter** | EMA 55 muss steigen → Nur LONG Trades |
| **Entry** | Kurs berührt unteres BB → bullische Kerze → Stop-Buy über dem Hoch |
| **Stop-Loss** | Unter dem signifikanten Tief der letzten 3 Kerzen |
| **Take Profit** | Nach +10 Pips → SL auf Break-Even. Trailing am Tief der 3.-letzten Kerze |
| **Stochastik** | %K < 30 (überverkauft) = zusätzliche Bestätigung |

---

### 2️⃣ Daytrading Top/Bottom Reversal (Umkehr-Strategie)

| Parameter | Regel |
|---|---|
| **Zeitrahmen** | 1-Min oder 5-Min Chart |
| **Indikatoren** | Bollinger Bänder, RSI, VWAP, EMA 21, Candlestick-Muster |
| **Short-Setup** | 5+ aufsteigende Kerzen + RSI > **90** + Kurs am oberen BB + Doji/Spinning Top |
| **Long-Setup** | 5+ fallende Kerzen + RSI < **10** + Kurs am unteren BB + Doji/Spinning Top |
| **Entry** | Short: Unter dem Tief der Doji-Kerze. Long: Über dem Hoch der Doji-Kerze |
| **Stop-Loss** | Am absoluten Hoch/Tief der Umkehrkerze |
| **Take Profit** | Nächstes Support/Resistance, VWAP, oder EMA 21 |

> ⚠️ **Wichtig:** Steige NIE blind in einen steigenden Kurs ein! Warte immer auf die Bestätigungskerze (Doji/Spinning Top).

---

### 3️⃣ Fibonacci-Korrektur Swing Trading

| Parameter | Regel |
|---|---|
| **Zeitrahmen** | Tageschart |
| **Indikatoren** | Fibonacci Retracement (50%/61.8%), Candlestick-Muster |
| **Setup** | Aufwärtstrend → Korrektur auf 50% oder 61.8% Fibonacci-Level |
| **Confirmation** | Hammer, Bullish Engulfing, oder Harami am Fibonacci-Level |
| **Entry** | Kaufe erst, wenn der Kurs das Hoch der Vortags-Kerze übersteigt! |
| **Stop-Loss** | Unter dem tiefsten Tief der Korrektur |
| **Take Profit** | Trailing-SL unter dem Tief der letzten 3-4 Tage |
| **Extensions** | Fibonacci 127.2% und 161.8% als Gewinnziele |

---

### 💰 Universelle Risiko-Regel (Der "Heilige Gral")

> **Max. 2% des Gesamtkapitals pro Trade riskieren!** (besser 1%)

**Berechnung:**
1. **Max. Verlust** = Gesamtkapital × Risiko-% (z.B. 10.000$ × 1% = 100$)
2. **SL-Abstand** = Entry-Preis − Stop-Loss (z.B. 150$ − 148$ = 2$)
3. **Positionsgröße** = Max. Verlust ÷ SL-Abstand (z.B. 100$ ÷ 2$ = 50 Aktien)

> ⚠️ Kein einzelner Indikator ist perfekt! Signale gewinnen erst durch die **Kombination mehrerer Indikatoren** an Validität.
"""


def render_strategy_rules():
    """Render the complete strategy rules document."""
    with st.expander("📖 Strategie-Regelwerk (alle Regeln im Detail)", expanded=False):
        st.markdown(STRATEGY_DOC)


# ═══════════════════════════════════════════════════════════
# INTERPRETATION GUIDE
# ═══════════════════════════════════════════════════════════

INDICATOR_GUIDE = """
### 1. Trend-Indikatoren
**Gleitende Durchschnitte (Moving Averages):**
- **10-Tage EMA:** Kurzfristiger Trend. Preis über EMA = Kaufsignal, darunter = Verkaufssignal.
- **21-Tage EMA:** Mittelfristige Unterstützung in volatilen Aufwärtstrends.
- **50-Tage SMA:** Typische Unterstützung für starke Aktien (Dip-Buying).
- **200-Tage SMA:** Langfristiger Trendindikator. Kurs darüber = Bull Market.

**VWAP:** Berücksichtigt Volumen. Über VWAP = bullisch, unter VWAP = bärisch.

**MACD:** MACD kreuzt Signal nach oben = Kauf. Nach unten = Verkauf. Divergenzen warnen vor Trendenden.

### 2. Momentum & Oszillatoren
**RSI:** >70 = überkauft, <30 = überverkauft. Über 50 = bullischer Trend.

**Stochastik:** %K kreuzt %D nach oben = Kauf. Funktioniert am besten in Seitwärtsmärkten.

### 3. Volatilität
**Bollinger Bänder:** Squeeze (enge Bänder) = explosive Bewegung erwartet.

**ATR:** Steigende ATR bestätigt Trend. Fallende ATR bei steigenden Preisen warnt vor Trendwechsel.

### 4. Volumen
**OBV:** Steigendes OBV + steigende Preise = gesunder Trend. Divergenz = Warnung.

**Volumen-Spikes:** Extremes Volumen = Ende eines Trends oder starker Ausbruch.

### 5. Candlestick-Muster
- **Hammer:** Bullische Umkehr am Tief. Langer unterer Docht.
- **Engulfing:** Starkes Umkehrsignal. Körper verschluckt Vortageskerze.
- **Doji:** Unentschlossenheit an Wendepunkten.
- **Harami:** Kleiner Körper innerhalb des Vortags-Körpers.

> **Wichtig:** Vertraue nie einem einzelnen Indikator blind!
"""
