import datetime

import streamlit as st
import yfinance as yf

import database as db
from simulation import get_current_equity, get_simulation_stats
from trading_terminal.scanner import ScannerDirection, score_asset_contract


def render_home_tab(now: datetime.datetime, market_status: str) -> None:
    st.markdown("## 🏠 Trading Dashboard")
    st.caption("Dein persönliches Cockpit – auf einen Blick alles Wichtige für den heutigen Handelstag.")

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

    st.markdown("### 📡 Markt-Ampel (Watchlist)")
    _wl_h = st.session_state.watchlist
    _buy_h, _sell_h, _neutral_h = 0, 0, 0
    _top_buys_h = []
    _top_sells_h = []
    try:
        _tf = yf.Ticker(st.session_state.tickers[0]).history(period="2d", interval="1d")
        if len(_tf) >= 2:
            _ = (_tf['Close'].iloc[-1] - _tf['Close'].iloc[-2]) / _tf['Close'].iloc[-2] * 100
    except Exception:
        pass

    with st.spinner("📡 Analysiere Markt-Dynamik (AI Strategy Engine)..."):
        active_strategies = db.get_all_strategies()
        for _hs in _wl_h:
            try:
                _r = score_asset_contract(_hs, "", active_strategies)
                if _r:
                    sc = _r.score
                    d = _r.direction_key
                    p = _r.price
                    en = _r.entry
                    sl = _r.sl
                    tp = _r.tp
                    pat = _r.signal_notes

                    if d == ScannerDirection.BUY:
                        _buy_h += 1
                        _top_buys_h.append((_hs, sc, d, p, en, sl, tp, pat))
                    elif d == ScannerDirection.SELL:
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

    _top3_buys_h = sorted(_top_buys_h, key=lambda x: x[1], reverse=True)[:3]
    _top3_sells_h = sorted(_top_sells_h, key=lambda x: x[1])[:3]

    _home_sig_cols = st.columns(3)
    _sig_items = _top3_buys_h + _top3_sells_h
    for _ci, _hs_item in enumerate((_sig_items)[:3]):
        _hs_sym, _hs_sc, _hs_dir, _hs_p, _hs_en, _hs_sl, _hs_tp, _hs_pat = _hs_item
        _is_buy = _hs_dir == ScannerDirection.BUY
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
    except Exception:
        st.info("News konnten nicht geladen werden.")

