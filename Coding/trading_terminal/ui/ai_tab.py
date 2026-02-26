import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trading_terminal.strategies import (
    compute_position_size,
    run_bollinger_scalping,
    run_fibonacci_swing,
    run_reversal,
    run_price_action,
    run_candlestick_reversal,
    run_walk_forward_backtest,
    to_legacy_exit_dicts,
    to_legacy_signal_dicts,
)
from ui_helpers import render_backtest_stats, render_equity_curve, render_strategy_rules

def render_latest_setup_cards(entries):
    if not entries:
        st.info("ℹ️ Für die aktuelle Marktphase liegt kein aktives Setup-Signal dieser Strategie vor.")
        return
        
    latest_entry = entries[-1]
    
    entry_price = latest_entry.get('Entry', 0)
    sl_price = latest_entry.get('SL', 0)
    tp_price = latest_entry.get('TP')
    
    if entry_price and sl_price:
        risk = abs(entry_price - sl_price)
        # Determine direction
        direction = latest_entry.get('Type')
        if not direction:
            direction = "LONG" if entry_price > sl_price else "SHORT"
            
        if not tp_price:
            tp_price = entry_price + (risk * 2) if direction == "LONG" else entry_price - (risk * 2)
        
        try:
            date_str = latest_entry['Date'].strftime('%d.%m.%Y')
        except AttributeError:
            date_str = str(latest_entry['Date'])
            
        st.success(f"🔔 **Aktuelles {direction} Setup (Vom {date_str})**")
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("🛒 ENTRY PREIS", f"${entry_price:.2f}")
        ec2.metric("🛑 STOP LOSS", f"${sl_price:.2f}")
        ec3.metric("🎯 TAKE PROFIT (Mindestens)", f"${tp_price:.2f}")
        st.divider()


def render_ai_strategy_backtest_section(df: pd.DataFrame, current_price: float) -> None:
    st.markdown("### 🎯 Live Strategy Scanner – Einstieg & Ausstieg")
    render_strategy_rules()

    strategy_pick = st.radio(
        "Strategie wählen:",
        ["1️⃣ Bollinger Scalping", "2️⃣ Daytrading Reversal", "3️⃣ Fibonacci Swing", "4️⃣ Price Action Breakout", "5️⃣ Candlestick Reversal"],
        horizontal=True,
    )

    st.divider()

    if "Scalping" in strategy_pick:
        st.markdown("#### 1️⃣ Bollinger Band Scalping (Trend-Korrektur)")
        st.caption("**Setup:** EMA 55 steigend → Kurs berührt unteres BB → bullische Kerze → Stop-Buy über dem Hoch. **SL:** Unter dem letzten Tief. **TP:** BB Mitte / Trailing.")

        entry_contracts, exit_contracts, backtest, df_s = run_bollinger_scalping(df)
        entries = to_legacy_signal_dicts(entry_contracts)
        exits = to_legacy_exit_dicts(exit_contracts)

        render_latest_setup_cards(entries)

        fig_s1 = go.Figure()
        fig_s1.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
            )
        )
        fig_s1.add_trace(go.Scatter(x=df.index, y=df_s['BB_Upper'], line=dict(color='rgba(255,255,255,0.3)', width=1), name='BB Upper'))
        fig_s1.add_trace(
            go.Scatter(
                x=df.index,
                y=df_s['BB_Lower'],
                line=dict(color='rgba(255,255,255,0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(100,100,255,0.05)',
                name='BB Lower',
            )
        )
        fig_s1.add_trace(go.Scatter(x=df.index, y=df_s['BB_Mid'], line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dot'), name='BB Mid'))
        fig_s1.add_trace(go.Scatter(x=df.index, y=df_s['EMA_55'], line={"color": '#7c4dff', "width": 1.5}, name='EMA 55 (Trend)'))

        if entries:
            fig_s1.add_trace(go.Scatter(x=[e['Date'] for e in entries], y=[e['Entry'] for e in entries], mode='markers', marker=dict(symbol='triangle-up', size=14, color='#00e676'), name='🟢 ENTRY'))
            fig_s1.add_trace(go.Scatter(x=[e['Date'] for e in entries], y=[e['SL'] for e in entries], mode='markers', marker=dict(symbol='x', size=10, color='#ff1744'), name='🔴 SL'))
            confirmed = [e for e in entries if e.get('StochConfirm')]
            if confirmed:
                fig_s1.add_trace(go.Scatter(x=[e['Date'] for e in confirmed], y=[e['Entry'] for e in confirmed], mode='markers', marker=dict(symbol='star', size=12, color='#ffea00'), name='⭐ Stoch bestätigt'))
        if exits:
            fig_s1.add_trace(go.Scatter(x=[e['Date'] for e in exits], y=[e['Price'] for e in exits], mode='markers', marker=dict(symbol='triangle-down', size=14, color='#ffea00'), name='🟡 EXIT'))

        fig_s1.update_layout(template='plotly_dark', height=500, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s1, use_container_width=True)
        render_backtest_stats(backtest)
        render_equity_curve(entries, exits, "Scalping Equity Curve")

    elif "Reversal" in strategy_pick:
        st.markdown("#### 2️⃣ Top/Bottom Reversal (RSI >90/<10 + Doji/Spinning Top)")
        st.caption("**Setup:** 5+ aufsteigende Kerzen + RSI Extrem (>90) + oberes BB + Doji/Spinning Top → Short. Umgekehrt für Long.")

        short_contracts, long_contracts, backtest, df_s = run_reversal(df)
        shorts = to_legacy_signal_dicts(short_contracts)
        longs = to_legacy_signal_dicts(long_contracts)

        combined = sorted(shorts + longs, key=lambda x: x['Date'])
        render_latest_setup_cards(combined)

        fig_s2 = go.Figure()
        fig_s2.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        fig_s2.add_trace(go.Scatter(x=df.index, y=df_s['BB_Upper'], line=dict(color='rgba(255,100,100,0.4)', width=1), name='BB Upper'))
        fig_s2.add_trace(go.Scatter(x=df.index, y=df_s['BB_Lower'], line=dict(color='rgba(100,255,100,0.4)', width=1), fill='tonexty', fillcolor='rgba(100,100,255,0.05)', name='BB Lower'))
        if 'VWAP' in df_s.columns:
            fig_s2.add_trace(go.Scatter(x=df.index, y=df_s['VWAP'], line={"color": '#ffeb3b', "width": 1, "dash": 'dot'}, name='VWAP (TP Ref)'))
        fig_s2.add_trace(go.Scatter(x=df.index, y=df_s['EMA_21'], line={"color": '#ff9800', "width": 1, "dash": 'dot'}, name='EMA 21 (TP Ref)'))

        if shorts:
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in shorts], y=[s['Entry'] for s in shorts], mode='markers+text', text=[f"🔴 {s['Pattern']}" for s in shorts], textposition='bottom center', marker=dict(symbol='triangle-down', size=16, color='#ff1744'), name='SHORT Entry'))
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in shorts], y=[s['SL'] for s in shorts], mode='markers', marker=dict(symbol='x', size=10, color='#ff9100'), name='SHORT SL'))
        if longs:
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in longs], y=[s['Entry'] for s in longs], mode='markers+text', text=[f"🟢 {s['Pattern']}" for s in longs], textposition='top center', marker=dict(symbol='triangle-up', size=16, color='#00e676'), name='LONG Entry'))
            fig_s2.add_trace(go.Scatter(x=[s['Date'] for s in longs], y=[s['SL'] for s in longs], mode='markers', marker=dict(symbol='x', size=10, color='#ff9100'), name='LONG SL'))

        fig_s2.update_layout(template='plotly_dark', height=500, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s2, use_container_width=True)
        render_backtest_stats(backtest)
        st.info(f"**Gefundene Signale:** {len(shorts)} Short Reversals | {len(longs)} Long Reversals")

    elif "Fibonacci" in strategy_pick:
        st.markdown("#### 3️⃣ Fibonacci-Korrektur Swing (50%/61.8% + Candlestick Confirmation)")
        st.caption("**Setup:** Korrektur auf Fib 50%/61.8% → Hammer/Engulfing/Harami → Entry über Vortagshoch. **SL:** Unter Korrektur-Tief. **TP:** Trailing 3-4 Tage + Fib Extensions.")

        fib_entry_contracts, fib_exit_contracts, backtest, _, fib_data = run_fibonacci_swing(df)
        fib_entries = to_legacy_signal_dicts(fib_entry_contracts)
        fib_exits = to_legacy_exit_dicts(fib_exit_contracts)

        render_latest_setup_cards(fib_entries)

        fig_s3 = go.Figure()
        fig_s3.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))

        fib_colors = {'23.6%': '#aaa', '38.2%': '#888', '50.0%': '#ff9800', '61.8%': '#f44336', '78.6%': '#9c27b0'}
        for lvl_name, lvl_val in fib_data['retracements'].items():
            col = fib_colors.get(lvl_name, '#666')
            fig_s3.add_hline(y=lvl_val, line_dash="dot", line_color=col, annotation_text=f"Fib {lvl_name} (${lvl_val:.2f})", annotation_position="right")
        for ext_name, ext_val in fib_data['extensions'].items():
            fig_s3.add_hline(y=ext_val, line_dash="dashdot", line_color='#00e676', annotation_text=f"Ext {ext_name} (${ext_val:.2f})", annotation_position="right")

        fig_s3.add_hline(y=fib_data['swing_high'], line_color="#00e676", line_width=1, annotation_text=f"Swing High ${fib_data['swing_high']:.2f}")
        fig_s3.add_hline(y=fib_data['swing_low'], line_color="#ff1744", line_width=1, annotation_text=f"Swing Low ${fib_data['swing_low']:.2f}")

        if fib_entries:
            fig_s3.add_trace(go.Scatter(x=[e['Date'] for e in fib_entries], y=[e['Entry'] for e in fib_entries], mode='markers+text', text=[f"🟢 {e['Level']} ({e['Pattern']})" for e in fib_entries], textposition='top center', marker=dict(symbol='triangle-up', size=16, color='#00e676'), name='ENTRY'))
            fig_s3.add_trace(go.Scatter(x=[e['Date'] for e in fib_entries], y=[e['SL'] for e in fib_entries], mode='markers', marker=dict(symbol='x', size=12, color='#ff1744'), name='🔴 SL'))
        if fib_exits:
            fig_s3.add_trace(go.Scatter(x=[e['Date'] for e in fib_exits], y=[e['Price'] for e in fib_exits], mode='markers+text', text=[e['Reason'] for e in fib_exits], textposition='bottom center', marker=dict(symbol='diamond', size=12, color='#ffea00'), name='EXIT'))

        fig_s3.update_layout(template='plotly_dark', height=550, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s3, use_container_width=True)
        render_backtest_stats(backtest)
        render_equity_curve(fib_entries, fib_exits, "Fibonacci Swing Equity Curve")
        if fib_entries:
            st.dataframe(pd.DataFrame(fib_entries), use_container_width=True, hide_index=True)

    elif "Price Action" in strategy_pick:
        st.markdown("#### 4️⃣ Price Action Breakout (Inside Bar)")
        st.caption("**Setup:** Inside Bar (High < Prev High, Low > Prev Low) in Trendrichtung. **Entry:** Ausbruch. **TP/SL:** Ratio 1:1.5")

        short_contracts, long_contracts, backtest, df_s = run_price_action(df)
        shorts = to_legacy_signal_dicts(short_contracts)
        longs = to_legacy_signal_dicts(long_contracts)

        combined = sorted(shorts + longs, key=lambda x: x['Date'])
        render_latest_setup_cards(combined)

        fig_s4 = go.Figure()
        fig_s4.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        fig_s4.add_trace(go.Scatter(x=df.index, y=df_s['EMA_21'], line={"color": '#ff9800', "width": 1.5}, name='EMA 21'))
        fig_s4.add_trace(go.Scatter(x=df.index, y=df_s['EMA_55'], line={"color": '#7c4dff', "width": 1.5, "dash": 'dot'}, name='EMA 55'))

        if shorts:
            fig_s4.add_trace(go.Scatter(x=[s['Date'] for s in shorts], y=[s['Entry'] for s in shorts], mode='markers+text', text=[f"🔴 {s['Pattern']}" for s in shorts], textposition='bottom center', marker=dict(symbol='triangle-down', size=16, color='#ff1744'), name='SHORT Entry'))
            fig_s4.add_trace(go.Scatter(x=[s['Date'] for s in shorts], y=[s['SL'] for s in shorts], mode='markers', marker=dict(symbol='x', size=10, color='#ff9100'), name='SHORT SL'))
        if longs:
            fig_s4.add_trace(go.Scatter(x=[s['Date'] for s in longs], y=[s['Entry'] for s in longs], mode='markers+text', text=[f"🟢 {s['Pattern']}" for s in longs], textposition='top center', marker=dict(symbol='triangle-up', size=16, color='#00e676'), name='LONG Entry'))
            fig_s4.add_trace(go.Scatter(x=[s['Date'] for s in longs], y=[s['SL'] for s in longs], mode='markers', marker=dict(symbol='x', size=10, color='#ff9100'), name='LONG SL'))

        fig_s4.update_layout(template='plotly_dark', height=500, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s4, use_container_width=True)
        render_backtest_stats(backtest)
        st.info(f"**Gefundene Signale:** {len(shorts)} Shorts | {len(longs)} Longs")

    elif "Candlestick" in strategy_pick:
        st.markdown("#### 5️⃣ Candlestick Reversal an Support/Resistance")
        st.caption("**Setup:** Starke Umkehrmuster (Engulfing, Pinbar) bestätigt durch RSI >65/<35. **Entry:** Nächster Tag Open.")

        short_contracts, long_contracts, backtest, df_s = run_candlestick_reversal(df)
        shorts = to_legacy_signal_dicts(short_contracts)
        longs = to_legacy_signal_dicts(long_contracts)

        combined = sorted(shorts + longs, key=lambda x: x['Date'])
        render_latest_setup_cards(combined)

        fig_s5 = go.Figure()
        fig_s5.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
        if 'SMA_50' in df_s.columns:
            fig_s5.add_trace(go.Scatter(x=df.index, y=df_s['SMA_50'], line={"color": '#ffeb3b', "width": 1.5, "dash": 'dot'}, name='SMA 50 (S/R)'))

        if shorts:
            fig_s5.add_trace(go.Scatter(x=[s['Date'] for s in shorts], y=[s['Entry'] for s in shorts], mode='markers+text', text=[f"🔴 {s['Pattern']}" for s in shorts], textposition='bottom center', marker={"symbol": 'triangle-down', "size": 16, "color": '#ff1744'}, name='SHORT Entry'))
            fig_s5.add_trace(go.Scatter(x=[s['Date'] for s in shorts], y=[s['SL'] for s in shorts], mode='markers', marker={"symbol": 'x', "size": 10, "color": '#ff9100'}, name='SHORT SL'))
        if longs:
            fig_s5.add_trace(go.Scatter(x=[s['Date'] for s in longs], y=[s['Entry'] for s in longs], mode='markers+text', text=[f"🟢 {s['Pattern']}" for s in longs], textposition='top center', marker={"symbol": 'triangle-up', "size": 16, "color": '#00e676'}, name='LONG Entry'))
            fig_s5.add_trace(go.Scatter(x=[s['Date'] for s in longs], y=[s['SL'] for s in longs], mode='markers', marker={"symbol": 'x', "size": 10, "color": '#ff9100'}, name='LONG SL'))

        fig_s5.update_layout(template='plotly_dark', height=500, margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False, showlegend=True)
        st.plotly_chart(fig_s5, use_container_width=True)
        render_backtest_stats(backtest)
        st.info(f"**Gefundene Signale:** {len(shorts)} Shorts | {len(longs)} Longs")

    st.divider()
    with st.expander("🧪 Walk-Forward Validation (Out-of-Sample)"):
        wf_col1, wf_col2, wf_col3 = st.columns(3)
        wf_strategy_ui = wf_col1.selectbox(
            "Strategie",
            options=[
                "bollinger_scalping",
                "fibonacci_swing",
                "reversal",
                "price_action",
                "candlestick_reversal",
            ],
            index=0,
        )
        wf_train = wf_col2.number_input("Train-Bars", min_value=100, max_value=2000, value=252, step=21)
        wf_test = wf_col3.number_input("Test-Bars", min_value=20, max_value=500, value=63, step=7)
        
        wf_col4, wf_col5 = st.columns(2)
        wf_step = wf_col4.number_input("Step-Bars", min_value=20, max_value=500, value=63, step=7)
        wf_use_sim = wf_col5.checkbox("Nutze PortfolioSimEngine (Realistic)", value=True, help="Nutzt die event-driven Engine mit Kosten & Slippage anstelle der Vector-Logik.")

        if st.button("▶ Walk-Forward starten", use_container_width=True):
            try:
                folds, summary = run_walk_forward_backtest(
                    df=df,
                    strategy_name=wf_strategy_ui,
                    train_size=int(wf_train),
                    test_size=int(wf_test),
                    step_size=int(wf_step),
                    use_portfolio_sim=wf_use_sim,
                )
                st.success(f"{summary.folds} Folds berechnet ({summary.strategy}).")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Profitable Folds", f"{summary.profitable_folds}/{summary.folds}")
                s2.metric("Pass-Rate", f"{summary.pass_rate_pct:.1f}%")
                s3.metric("Ø OOS P&L", f"{summary.avg_oos_net_pnl_pct:.2f}%")
                s4.metric("Ø OOS R:R", f"1:{summary.avg_oos_rr:.2f}")

                checks = {
                    "mind. 3 Folds": summary.folds >= 3,
                    "Pass-Rate >= 55%": summary.pass_rate_pct >= 55.0,
                    "Ø OOS P&L > 0": summary.avg_oos_net_pnl_pct > 0.0,
                    "Ø OOS R:R >= 0.8": summary.avg_oos_rr >= 0.8,
                }
                passed = sum(1 for ok in checks.values() if ok)
                if passed == len(checks):
                    st.success("✅ Go/No-Go: GO (alle Validierungsregeln erfüllt)")
                elif passed >= 2:
                    st.warning("🟡 Go/No-Go: CONDITIONAL (teilweise erfüllt, Risiko erhöht)")
                else:
                    st.error("🔴 Go/No-Go: NO-GO (zu schwache OOS-Qualität)")
                st.caption("Validierungs-Checks: " + " | ".join([f"{'✅' if ok else '❌'} {name}" for name, ok in checks.items()]))

                wf_df = pd.DataFrame(
                    [
                        {
                            "Fold": f.fold,
                            "Train Start": f.train_start.date(),
                            "Train End": f.train_end.date(),
                            "Test Start": f.test_start.date(),
                            "Test End": f.test_end.date(),
                            "Test Signals": f.test_signals,
                            "Test Exits": f.test_exits,
                            "Win Rate %": round(f.test_win_rate, 2),
                            "Avg RR": round(f.test_avg_rr, 3),
                            "Net PnL %": round(f.test_net_pnl_pct, 3),
                        }
                        for f in folds
                    ]
                )
                st.dataframe(wf_df, use_container_width=True, hide_index=True)
                if wf_strategy_ui in {"reversal", "price_action", "candlestick_reversal"}:
                    st.caption("Hinweis: Exits für diese Strategie werden im Walk-Forward per TP/SL-Bar-Simulation approximiert.")
            except Exception as e:
                st.error(f"Walk-Forward fehlgeschlagen: {e}")

    st.divider()
    st.markdown("#### 📏 Trailing-Stop Empfehlung")
    st.markdown("""
- **Scalping:** Nach +10 Pips → SL auf Break-Even. Trailing am Tief der 3.-letzten Kerze.
- **Reversal:** Take Profit am nächsten Support/VWAP/MA-Level.
- **Fibonacci:** Trailing-SL unter dem Tief der letzten 3-4 Tage. Gewinnziele: Fib 127.2% / 161.8%.
- **Price Action (Inside Bar):** Take Profit statisch oder Trailing am gleitenden Durchschnitt.
- **Candlestick Reversal:** Fester CRV Ziel (z.B. 2-fach Risiko) oder an großer EMA/SMA Widerstand.
    """)


def render_ai_ml_suite_section(df: pd.DataFrame, current_price: float, now: datetime.datetime) -> None:
    st.markdown("### 🤖 AI & Machine Learning Suite")
    ai_pick = st.radio("AI-Modul:", ["🧠 Pattern Scanner", "📊 Regime Detection", "🎯 Auto S/R", "📈 Forecast", "📋 Risk Profiler", "📄 AI Report", "📚 Wissens-DB"], horizontal=True)
    returns_ai = df['Daily_Return'].dropna()

    if "Pattern" in ai_pick:
        st.markdown("#### 🧠 AI Pattern Scanner")
        patterns_found = []
        for i in range(20, min(len(df), 100)):
            window = df['Low'].iloc[i - 20:i]
            min1_idx = window.iloc[:10].idxmin()
            min2_idx = window.iloc[10:].idxmin()
            if abs(df['Low'][min1_idx] - df['Low'][min2_idx]) / df['Low'][min1_idx] < 0.02:
                patterns_found.append(f"📐 **Double Bottom** near ${df['Low'][min2_idx]:.2f} ({str(min2_idx)[0:10]})")
            if i >= 15:
                seg = df['High'].iloc[i - 15:i]
                mid = seg.iloc[5:10].max()
                left = seg.iloc[:5].max()
                right = seg.iloc[10:].max()
                if mid > left and mid > right and abs(left - right) / left < 0.03:
                    patterns_found.append(f"👤 **Head & Shoulders** hint: peak ${mid:.2f}")
        patterns_found = list(set(patterns_found))[:8]
        if patterns_found:
            for p in patterns_found:
                st.markdown(p)
        else:
            st.info("Keine Patterns erkannt.")

    elif "Regime" in ai_pick:
        st.markdown("#### 📊 Market Regime Detection")
        if len(returns_ai) < 30:
            st.warning("Wenig Daten für robustes Regime-Label (<30 Returns). Interpretation mit Vorsicht.")
        vol_20 = returns_ai.rolling(20).std().iloc[-1] * np.sqrt(252) * 100 if len(returns_ai) > 20 else 0
        trend_str = abs(returns_ai.rolling(20).mean().iloc[-1]) * 252 * 100 if len(returns_ai) > 20 else 0
        if vol_20 < 15 and trend_str < 10:
            regime_ai = "😴 Mean Reverting"
        elif vol_20 < 25 and trend_str >= 10:
            regime_ai = "📈 Trending"
        elif vol_20 >= 25 and trend_str >= 15:
            regime_ai = "🚀 Momentum"
        else:
            regime_ai = "🌪️ Chaos"
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Regime", regime_ai)
        rc2.metric("20D Vola", f"{vol_20:.1f}%")
        rc3.metric("Trend-Stärke", f"{trend_str:.1f}%")

    elif "S/R" in ai_pick:
        st.markdown("#### 🎯 Auto Support & Resistance (K-Means)")
        from scipy.cluster.vq import kmeans as km

        pts = np.concatenate([df['High'].values, df['Low'].values])
        pts = pts[~np.isnan(pts)]
        if len(pts) > 10:
            centroids, _ = km(pts.astype(float), min(5, len(pts)))
            centroids = sorted(centroids)
            fig_sr = go.Figure()
            fig_sr.add_trace(go.Candlestick(x=list(df.index), open=list(df['Open']), high=list(df['High']), low=list(df['Low']), close=list(df['Close']), increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))
            cols_sr = ['#ff1744', '#ff9800', '#ffeb3b', '#00e676', '#2196f3']
            for ic, c in enumerate(centroids):
                fig_sr.add_hline(y=c, line_dash="dash", line_color=cols_sr[ic % len(cols_sr)], annotation_text=f"{'Support' if c < current_price else 'Resistance'} ${c:.2f}")
            fig_sr.update_layout(template='plotly_dark', height=400, margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_sr, use_container_width=True)

    elif "Forecast" in ai_pick:
        st.markdown("#### 📈 Price Forecast (Linear + Monte Carlo)")
        from scipy.stats import linregress

        if len(df) < 60:
            st.warning("Forecast basiert auf kurzer Historie (<60 Kerzen) und ist statistisch unsicherer.")

        x_v = np.arange(len(df))
        y_v = df['Close'].values
        slope, intercept, r_val, _, _ = linregress(x_v, y_v)
        future_x = np.arange(len(df), len(df) + 30)
        trend_line = slope * future_x + intercept
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=list(range(len(df))), y=y_v, mode='lines', name='Historisch', line=dict(color='#26a69a')))
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=trend_line, mode='lines', name='Forecast', line=dict(color='#ff9800', dash='dash')))
        std_r = returns_ai.std()
        upper = trend_line * (1 + 2 * std_r * np.sqrt(np.arange(1, 31)))
        lower = trend_line * (1 - 2 * std_r * np.sqrt(np.arange(1, 31)))
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=upper, mode='lines', line={"width": 0}, showlegend=False))
        fig_fc.add_trace(go.Scatter(x=list(future_x), y=lower, mode='lines', line={"width": 0}, fill='tonexty', fillcolor='rgba(255,152,0,0.15)', name='95% Konfidenz'))
        fig_fc.update_layout(template='plotly_dark', height=350, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117')
        st.plotly_chart(fig_fc, use_container_width=True)
        st.metric("R²", f"{r_val**2:.4f}")
        st.metric("30-Tage Prognose", f"${trend_line[-1]:.2f}")
        st.caption("Hinweis: lineares Modell + konstante Volatilität; kein Regimewechsel-Modell.")

    elif "Profiler" in ai_pick:
        st.markdown("#### 📋 Risk Profiler Quiz")
        q1 = st.slider("Nervosität bei Verlusten", 1, 10, 5)
        q2 = st.slider("Anlagehorizont", 1, 10, 5)
        q3 = st.slider("Trading-Erfahrung", 1, 10, 3)
        q4 = st.slider("Max. Drawdown Toleranz", 1, 10, 4)
        q5 = st.slider("Dividenden-Priorität", 1, 10, 3)
        rs = (q1 * -1 + q2 + q3 + q4 - q5 + 30) / 6
        if rs < 3:
            pf, assets = "🛡️ Konservativ", ["BND", "VYM", "JNJ"]
        elif rs < 6:
            pf, assets = "⚖️ Ausgewogen", ["SPY", "QQQ", "AAPL"]
        else:
            pf, assets = "🔥 Aggressiv", ["BTC-USD", "NVDA", "ARKK"]
        st.success(f"**Profil:** {pf} (Score: {rs:.1f}/10)")
        for a in assets:
            st.markdown(f"- {a}")

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
                    import database as db
                    import json
                    import knowledge_base as kb

                    extracted_logic = kb.parse_book_to_rules(raw_text)
                    db.add_strategy(name=rule_name, source_book=book_title, logic_json=json.dumps(extracted_logic), base_weight=1.0)
                    st.success(f"Strategie '{rule_name}' erfolgreich extrahiert und als ID in der Datenbank gespeichert!")
                    st.json(extracted_logic)
                except Exception as e:
                    st.error(f"Fehler bei der Regelextraktion: {e}")

        st.markdown("---")
        st.markdown("##### 🗄️ Aktive Strategien in der Datenbank")
        import database as db

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
            sharpe_r = (returns_ai.mean() - 0.02 / 252) / returns_ai.std() * np.sqrt(252) if returns_ai.std() > 0 else 0
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

    st.divider()
    st.markdown("### 🔌 Live Paper Trading & Execution Layer")
    
    col_exec1, col_exec2 = st.columns([2, 1])
    with col_exec1:
        st.markdown("Sende manuell ermittelte Signale direkt an den **PaperBroker** (Simulation mit Slippage & Latenz).")
        with st.form("paper_trade_form"):
            s_col1, s_col2, s_col3 = st.columns(3)
            exec_side = s_col1.selectbox("Richtung", ["BUY", "SELL"])
            exec_type = s_col2.selectbox("Order Typ", ["MARKET", "LIMIT", "STOP"])
            exec_qty = s_col3.number_input("Menge", min_value=0.01, value=1.0)
            
            exec_target = st.number_input("Limit/Stop Preis (Optional)", value=current_price, format="%.2f")
            
            submit_exec = st.form_submit_button("🚀 Order an PaperBroker senden")
            
            if submit_exec:
                from trading_terminal.execution.broker import PaperBroker
                from trading_terminal.execution.signals import ExecutionOrder, OrderSide, OrderType
                
                # Init or grab broker from session state to persist it across renders
                if 'live_broker' not in st.session_state:
                    st.session_state.live_broker = PaperBroker()
                
                broker: PaperBroker = st.session_state.live_broker
                
                side_enum = OrderSide.BUY if exec_side == "BUY" else OrderSide.SELL
                type_enum = OrderType.MARKET if exec_type == "MARKET" else (OrderType.LIMIT if exec_type == "LIMIT" else OrderType.STOP)
                target = exec_target if type_enum != OrderType.MARKET else None
                
                new_order = ExecutionOrder(
                    symbol=st.session_state.tickers[0],
                    side=side_enum,
                    order_type=type_enum,
                    quantity=exec_qty,
                    price_target=target
                )
                
                o_id = broker.place_order(new_order)
                # Tick the broker immediately to simulate time passing for MARKET orders
                broker.process_market_tick(now, st.session_state.tickers[0], current_price - 0.05, current_price + 0.05)
                
                st.success(f"Order #{o_id} platziert! Status im Broker-Log überprüfen.")
                
    with col_exec2:
        st.markdown("#### Broker Status")
        if 'live_broker' in st.session_state:
            active_count = len(st.session_state.live_broker.active_orders)
            fills_count = len(st.session_state.live_broker.fills)
            st.metric("Pending Orders", active_count)
            st.metric("Total Fills (Session)", fills_count)
            
            if st.button("Logs & Fills anzeigen"):
                st.write([f"Fill: {f.side} {f.quantity} @ {f.price:.2f}" for f in st.session_state.live_broker.fills])
        else:
            st.info("Noch keine Orders in dieser Session gesendet.")

    st.divider()
    st.markdown("### 🔍 MLOps: Volatility Drift Detection")
    st.markdown("Vergleicht die Volatilität der letzten 10 Tage mit den vorherigen 90 Tagen (Trainings-Referenz).")
    if st.button("Drift-Analyse starten"):
        from trading_terminal.mlops.drift import calculate_volatility_regime_drift
        
        if len(df) > 100:
            df_train = df.iloc[-100:-10]
            df_live = df.iloc[-10:]
            
            report = calculate_volatility_regime_drift(df_train, df_live)
            
            col_d1, col_d2 = st.columns(2)
            col_d1.metric("Ref. Volatilität (Train)", f"{report.reference_mean*100:.1f}%")
            col_d2.metric("Aktuelle Volatilität (Live)", f"{report.current_mean*100:.1f}%")
            
            if report.is_drifting:
                st.error(f"🚨 **Drift erkannt!** Das Volatilitäts-Regime hat sich signifikant verändert (Ratio: {report.z_score:.2f}). Walk-Forward-Modelle sollten neu trainiert werden.")
            else:
                st.success(f"✅ **Kein Drift.** Das Marktregime ist stabil (Ratio: {report.z_score:.2f}). Modelle können weiterlaufen.")
        else:
            st.warning("Nicht genug Daten für eine sinnvolle Drift-Detection (<100 Kerzen).")
