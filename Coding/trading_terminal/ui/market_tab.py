import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import database as db
from simulation import (
    get_simulation_stats,
    new_simulation_state,
    open_positions_from_scanner,
    update_portfolio,
)
from trading_terminal.scanner import (
    SCANNER_DISPLAY_COLUMNS,
    ScannerDirection,
    score_asset_contract,
)


def render_market_signals_section(now: datetime.datetime) -> None:
    st.markdown("## 🚀 Heutige Handelssignale")
    st.caption("Live-Signale deiner Watchlist – berechnet aus RSI, EMA, MACD, ADX und Support/Resistance.")

    _watchlist_signals = []
    _n_syms = len(st.session_state.watchlist)
    if _n_syms > 0:
        _prog = st.progress(0, text="📡 Analysiere Markt-Dynamik (AI Engine)...")
        active_strategies = db.get_all_strategies()
        for _wi, _wsym in enumerate(st.session_state.watchlist):
            _prog.progress((_wi + 1) / _n_syms, text=f"📡 {_wsym} (MTF AI Scan)...")
            try:
                _res = score_asset_contract(_wsym, "", active_strategies)
            except Exception as e:
                print(f"Error scanning {_wsym}: {e}")
                _res = None
            if _res:
                sc = _res.score
                d = _res.direction_key
                p = _res.price
                en = _res.entry
                sl = _res.sl
                tp = _res.tp
                pat = _res.signal_notes
                _watchlist_signals.append((_wsym, sc, d, p, en, sl, tp, pat))
        _prog.empty()

    _buy_sigs = [s for s in _watchlist_signals if s[2] == ScannerDirection.BUY]
    _sell_sigs = [s for s in _watchlist_signals if s[2] == ScannerDirection.SELL]
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

    _sig_cols = st.columns(2)
    with _sig_cols[0]:
        st.markdown("### 🟢 Top Kaufsignale")
        _buy_sorted = sorted(_buy_sigs, key=lambda x: x[1], reverse=True)[:3]
        if not _buy_sorted:
            st.info("Keine klaren Kaufsignale in der Watchlist.")
        for _bs in _buy_sorted:
            _bsym, _bsc, _, _bprice, _bentry, _bsl, _btp, _bpat = _bs
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
                st.session_state.tickers[0] = _bsym
                st.rerun()
            _order_str = f"BUY {_shares}x {_bsym} @ ${_bentry:.2f} | SL ${_bsl:.2f} | TP ${_btp:.2f}"
            _bc2.code(_order_str, language=None)

    with _sig_cols[1]:
        st.markdown("### 🔴 Top Vorsicht-Signale")
        _sell_sorted = sorted(_sell_sigs, key=lambda x: x[1])[:3]
        if not _sell_sorted:
            st.info("Keine klaren Verkaufssignale in der Watchlist.")
        for _ss in _sell_sorted:
            _ssym, _ssc, _, _sprice, _sentry, _ssl, _stp, _spat = _ss
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
                st.session_state.tickers[0] = _ssym
                st.rerun()

    if st.session_state.morning_scan_date == now.strftime("%Y-%m-%d"):
        st.success(f"✅ Morgenscan abgeschlossen ({now.strftime('%H:%M')} Uhr)")


def render_market_scanner_simulation_section(now: datetime.datetime) -> None:
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

    if st.session_state.get('scanner_results') is not None:
        scan_df = st.session_state['scanner_results'].copy()

        if direction_filter == "🟢 NUR KAUFEN":
            scan_df = scan_df[scan_df['DirectionKey'] == ScannerDirection.BUY]
        elif direction_filter == "🟡 NUR NEUTRAL":
            scan_df = scan_df[scan_df['DirectionKey'] == ScannerDirection.NEUTRAL]
        elif direction_filter == "🔴 NUR VORSICHT":
            scan_df = scan_df[scan_df['DirectionKey'] == ScannerDirection.SELL]

        if min_score > 0:
            scan_df = scan_df[scan_df['Score'] >= min_score]

        st.markdown(f"**{len(scan_df)} Assets gefunden** – sortiert nach Score (höchster = starkster Setup)")

        for rank_idx, (_, row) in enumerate(scan_df.iterrows(), start=1):
            dk = row.get('DirectionKey', ScannerDirection.NEUTRAL)
            if dk == ScannerDirection.BUY:
                border_color = '#26a69a'
                bg_color = 'rgba(38,166,154,0.07)'
                rank_emoji = '🥇' if rank_idx == 1 else '🥈' if rank_idx == 2 else '🥉' if rank_idx == 3 else f'**#{rank_idx}**'
            elif dk == ScannerDirection.SELL:
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

            if st.button(f"📊 {row['Symbol']} laden", key=f"load_{row['Symbol']}_{rank_idx}"):
                st.session_state.tickers[0] = row['Symbol']
                st.rerun()

        st.markdown("---")
        st.markdown("##### 📊 Rohdaten (vollständige Tabelle)")
        avail_cols = [c for c in SCANNER_DISPLAY_COLUMNS if c in scan_df.columns]
        st.dataframe(scan_df[avail_cols].reset_index(drop=True), use_container_width=True)

        st.divider()
        with st.expander("📄 Wochenbericht & Export"):
            _sim_st = st.session_state.sim_state
            _closed = _sim_st.get('closed_trades', [])
            if _closed:
                _ct_df = pd.DataFrame(_closed)
                _wins = len([t for t in _closed if t.get('pnl', 0) > 0])
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

        st.divider()
        st.markdown("### 🏆 Dein Trading-Fortschritt")
        _sim_g = st.session_state.sim_state
        _g_equity = _sim_g.get('equity_curve', [])
        _g_closed = _sim_g.get('closed_trades', [])
        _g_start = _sim_g.get('initial_cash', 100_000)
        _g_current = _g_equity[-1] if _g_equity else _g_start
        _g_target = _g_start * 2
        _g_progress = min(1.0, (_g_current - _g_start) / (_g_target - _g_start))

        _gc1, _gc2, _gc3 = st.columns(3)
        with _gc1:
            st.markdown("**💰 Kapital**")
            st.progress(max(0.0, _g_progress), text=f"€{_g_current:,.0f} von €{_g_target:,.0f} Ziel")
        with _gc2:
            _streak = 0
            for _t in reversed(_g_closed):
                if _t.get('pnl', 0) > 0:
                    _streak += 1
                else:
                    break
            _streak_icon = "🔥" * min(_streak, 5) if _streak > 0 else "❄️"
            st.metric(f"{_streak_icon} Gewinn-Serie", f"{_streak} Trades")
        with _gc3:
            _total_trades = len(_g_closed)
            _win_rate_g = len([t for t in _g_closed if t.get('pnl', 0) > 0]) / max(1, _total_trades)
            if _total_trades >= 20 and _win_rate_g >= 0.6 and _g_current > _g_start * 1.3:
                _level = "👑 Expert"
            elif _total_trades >= 10 and _win_rate_g >= 0.5:
                _level = "🧙 Pro"
            elif _total_trades >= 5:
                _level = "💼 Trader"
            else:
                _level = "🌱 Rookie"
            st.markdown(f"<div style='text-align:center; font-size:28px;'>{_level}</div>"
                        f"<div style='text-align:center; font-size:11px; color:#9e9e9e;'>{_total_trades} Trades abgeschlossen</div>",
                        unsafe_allow_html=True)

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

    st.markdown("---")
    st.markdown("## 📊 Paper Trading Simulation – Virtuelles Depot")
    st.markdown("Kauft automatisch die Top-10 Scanner-Assets mit €100.000 Startkapital und trackt SL/TP-Hits gegen echte Kursdaten.")

    sim = st.session_state.sim_state
    stats = get_simulation_stats(sim)

    pnl_arrow = '▲' if stats['total_pnl'] >= 0 else '▼'
    sk1, sk2, sk3, sk4, sk5 = st.columns(5)
    sk1.metric("💰 Startkapital", f"€{stats['start_capital']:,.0f}")
    sk2.metric("📈 Aktueller Depotwert", f"€{stats['equity']:,.2f}",
               delta=f"{pnl_arrow} {stats['total_pnl_pct']:+.2f}%")
    sk3.metric("💵 Freies Cash", f"€{stats['cash']:,.2f}")
    sk4.metric("📂 Offene Positionen", stats['open_positions'])
    sk5.metric("🏆 Win Rate", f"{stats['win_rate']:.1f}%" if stats['closed_trades'] else "–",
               delta=f"{stats['closed_trades']} Trades" if stats['closed_trades'] else None)

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
            db.reset_portfolio()
            st.session_state.sim_state = new_simulation_state()
            st.rerun()

    if buy_clicked and can_buy:
        scan_df = st.session_state['scanner_results']
        buy_progress = st.progress(0, text="🛒 Einstiegssignale werden berechnet...")

        def buy_progress_cb(cur, tot, sym):
            buy_progress.progress(cur / tot if tot > 0 else 1.0, text=f"🛒 Signal für {sym}... ({cur}/{tot})")

        new_state, opened = open_positions_from_scanner(scan_df, sim, top_n=10, progress_callback=buy_progress_cb)
        st.session_state.sim_state = new_state
        buy_progress.progress(1.0, text=f"✅ {opened} Positionen eröffnet!")
        st.rerun()

    if update_clicked:
        upd_progress = st.progress(0, text="🔍 SL/TP wird geprüft...")

        def upd_progress_cb(cur, tot, sym):
            upd_progress.progress(cur / tot if tot > 0 else 1.0, text=f"🔍 Prüfe {sym}... ({cur}/{tot})")

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

    if sim['closed_trades']:
        st.markdown("### 📋 Abgeschlossene Trades")
        rows = []
        for t in reversed(sim['closed_trades']):
            rows.append({
                "Symbol": t['symbol'],
                "Name": t['name'],
                "Einstieg": f"${t['entry']:,.4f}",
                "Ausstieg": f"${t['exit_date']} @ ${t['exit_price']:,.4f}",
                "Grund": f"✅ TP" if t['result'] == 'TP' else "❌ SL",
                "P&L €": f"{t['pnl_eur']:+,.2f}",
                "P&L %": f"{t['pnl_pct']:+.2f}%",
                "Strategie": t['strategy'],
            })
        closed_df = pd.DataFrame(rows)
        st.dataframe(closed_df, use_container_width=True, hide_index=True)

    if len(sim['equity_history']) >= 2:
        st.markdown("### 📈 Equity-Kurve")
        eq_df = pd.DataFrame(sim['equity_history'])
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        eq_df = eq_df.sort_values('date')

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_df['date'],
            y=eq_df['equity'],
            mode='lines+markers',
            line=dict(color='#26a69a', width=2),
            fill='tozeroy',
            fillcolor='rgba(38,166,154,0.08)',
            name='Depotwert'
        ))
        fig_eq.add_hline(
            y=sim['start_capital'],
            line_dash='dash',
            line_color='#607d8b',
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

