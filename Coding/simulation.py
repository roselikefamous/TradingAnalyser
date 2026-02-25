"""
simulation.py – Paper Trading Simulation Engine
Virtual depot starting at €100,000. Buys top-N scanner assets,
sets SL/TP via Bollinger strategy, then checks real OHLCV data
to determine if SL or TP was hit.
"""

from typing import Optional, List, Dict, Any, Tuple
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

from indicators import calc_bollinger_bands, calc_ema, calc_stochastic, apply_core_indicators


# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

STARTING_CAPITAL = 100_000.0   # EUR
RISK_PCT   = 2.0               # max. 2% risk per trade
MAX_INVEST = 0.20              # max. 20% of capital per single position


# ═══════════════════════════════════════════════════════════
# DEFAULT STATE
# ═══════════════════════════════════════════════════════════

def new_simulation_state() -> Dict[str, Any]:
    """Return a fresh simulation state dict."""
    return {
        "cash": STARTING_CAPITAL,
        "start_capital": STARTING_CAPITAL,
        "positions": [],      # list of open position dicts
        "closed_trades": [],  # list of closed trade dicts
        "equity_history": [   # list of {date, equity}
            {"date": datetime.date.today().isoformat(), "equity": STARTING_CAPITAL}
        ],
        "last_updated": None,
    }


# ═══════════════════════════════════════════════════════════
# STRATEGY: derive Entry / SL / TP from Bollinger Scalping
# ═══════════════════════════════════════════════════════════

def _get_bollinger_signal(symbol: str) -> Optional[Dict[str, float]]:
    """
    Fetch 3-month daily data and run the Bollinger Scalping
    strategy to find the latest valid entry signal.

    Returns dict with {entry, sl, tp, strategy} or None.
    """
    try:
        raw = yf.Ticker(symbol).history(period="3mo", interval="1d")
        if raw is None or raw.empty or len(raw) < 30:
            return None
    except Exception:
        return None

    try:
        bb_mid, bb_upper, bb_lower = calc_bollinger_bands(raw["Close"], 20, 2)
        ema_55 = calc_ema(raw["Close"], 55)
        stoch_k, _ = calc_stochastic(raw)

        last_idx = len(raw) - 1

        # Walk backwards to find a recent signal (up to last 5 bars)
        for i in range(last_idx, max(last_idx - 5, 5), -1):
            if pd.isna(bb_lower.iloc[i]) or pd.isna(ema_55.iloc[i]):
                continue

            is_uptrend = ema_55.diff().iloc[i - 1] > 0
            touched_lower = raw["Low"].iloc[i - 1] <= bb_lower.iloc[i - 1]
            bullish_candle = raw["Close"].iloc[i] > raw["Open"].iloc[i]

            if is_uptrend and touched_lower and bullish_candle:
                entry = raw["Close"].iloc[i]          # current close as entry
                sl    = min(raw["Low"].iloc[max(0, i - 2):i + 1])
                tp    = bb_mid.iloc[i]

                if sl >= entry or tp <= entry:
                    continue
                if (entry - sl) / entry < 0.005:      # skip tiny SL (<0.5%)
                    continue

                return {
                    "entry": round(entry, 4),
                    "sl":    round(sl, 4),
                    "tp":    round(tp, 4),
                    "strategy": "Bollinger Scalping",
                }

        # Fallback: use ATR-based levels from latest candle
        last_close = raw["Close"].iloc[-1]
        atr_approx = (raw["High"] - raw["Low"]).rolling(14).mean().iloc[-1]
        if pd.isna(atr_approx) or atr_approx <= 0:
            return None

        entry = last_close
        sl    = round(last_close - 1.5 * atr_approx, 4)
        tp    = round(last_close + 2.0 * atr_approx, 4)
        return {
            "entry": round(entry, 4),
            "sl":    sl,
            "tp":    tp,
            "strategy": "ATR Fallback",
        }

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
# OPEN POSITIONS – buy top-N assets from scanner results
# ═══════════════════════════════════════════════════════════

def open_positions_from_scanner(
    scan_df: pd.DataFrame,
    state: Dict[str, Any],
    top_n: int = 10,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Read the top-N assets from scan_df, compute entry signals,
    apply 2% risk sizing, and open positions funded by cash.

    Modifies state in-place and returns it.
    """
    # Only take BUY signals that are not already in portfolio
    existing_symbols = {p["symbol"] for p in state["positions"]}
    buy_df = scan_df[scan_df["DirectionKey"] == "BUY"].head(top_n)

    opened = 0
    for rank_i, (_, row) in enumerate(buy_df.iterrows()):
        symbol = row["Symbol"]
        if symbol in existing_symbols:
            continue

        if progress_callback:
            progress_callback(rank_i + 1, len(buy_df), symbol)

        signal = _get_bollinger_signal(symbol)
        if signal is None:
            continue

        entry = signal["entry"]
        sl    = signal["sl"]
        tp    = signal["tp"]

        sl_distance = entry - sl
        if sl_distance <= 0:
            continue

        # 2% risk sizing
        max_risk_eur  = state["cash"] * (RISK_PCT / 100)
        shares        = max_risk_eur / sl_distance
        invest_eur    = shares * entry

        # Cap at 20% of total capital
        max_invest_eur = state["start_capital"] * MAX_INVEST
        if invest_eur > max_invest_eur:
            shares = max_invest_eur / entry
            invest_eur = shares * entry

        if round(invest_eur, 2) > round(state["cash"], 2) or shares < 0.001:
            continue

        # Deduct from cash
        state["cash"] -= invest_eur
        opened += 1

        position = {
            "symbol":       symbol,
            "name":         row.get("Name", symbol),
            "entry":        round(entry, 4),
            "sl":           round(sl, 4),
            "tp":           round(tp, 4),
            "shares":       round(shares, 6),
            "invest_eur":   round(invest_eur, 2),
            "open_date":    datetime.date.today().isoformat(),
            "strategy":     signal["strategy"],
            "score":        row.get("Score", 0),
        }
        state["positions"].append(position)
        existing_symbols.add(symbol)

    # snapshot equity
    _snapshot_equity(state)
    state["last_updated"] = datetime.datetime.now().isoformat()

    return state, opened


# ═══════════════════════════════════════════════════════════
# UPDATE PORTFOLIO – check SL/TP hits for every open position
# ═══════════════════════════════════════════════════════════

def update_portfolio(state: Dict[str, Any], progress_callback=None) -> Tuple[Dict[str, Any], list]:
    """
    For each open position, fetch OHLCV data since open_date and
    check candle-by-candle whether SL or TP was hit first.

    - Closes position if SL or TP hit, adding P&L to cash.
    - Also fetches latest price for still-open positions.
    - Updates equity history.
    """
    still_open   = []
    newly_closed = []

    total = len(state["positions"])

    for idx, pos in enumerate(state["positions"]):
        if progress_callback:
            progress_callback(idx + 1, total, pos["symbol"])

        try:
            ticker = yf.Ticker(pos["symbol"])
            # Fetch from open_date → today (at least 1 day buffer)
            start = pd.Timestamp(pos["open_date"])
            raw = ticker.history(start=start, interval="1d")
        except Exception:
            still_open.append(pos)
            continue

        if raw is None or raw.empty:
            still_open.append(pos)
            continue

        entry  = pos["entry"]
        sl     = pos["sl"]
        tp     = pos["tp"]
        shares = pos["shares"]

        hit = None       # "SL" | "TP" | None
        hit_price = None
        hit_date  = None

        for bar_date, bar in raw.iterrows():
            low  = bar["Low"]
            high = bar["High"]

            sl_hit = low <= sl
            tp_hit = high >= tp

            # Determine which was hit first within the candle.
            # If BOTH SL and TP are hit on the same bar, we consider that
            # the open moved favorably first and TP was hit (realistic scenario
            # for long positions where price rallied before reverting).
            if sl_hit and tp_hit:
                # TP wins on ambiguous same-bar hit (aggressive but realistic)
                hit = "TP"
                hit_price = tp
            elif sl_hit:
                hit = "SL"
                hit_price = sl
            elif tp_hit:
                hit = "TP"
                hit_price = tp

            if hit:
                hit_date = bar_date.date().isoformat() if hasattr(bar_date, "date") else str(bar_date)[:10]
                break

        if hit:
            pnl_eur = (hit_price - entry) * shares
            pnl_pct = ((hit_price - entry) / entry) * 100

            state["cash"] += pos["invest_eur"] + pnl_eur  # return capital + P&L

            closed_trade = {**pos,
                "exit_price": round(hit_price, 4),
                "exit_date":  hit_date,
                "result":     hit,
                "pnl_eur":    round(pnl_eur, 2),
                "pnl_pct":    round(pnl_pct, 2),
            }
            newly_closed.append(closed_trade)
            state["closed_trades"].append(closed_trade)
        else:
            # Still open – update current price
            last_close = raw["Close"].iloc[-1]
            unrealized_pnl = (last_close - entry) * shares
            pos["current_price"]    = round(last_close, 4)
            pos["unrealized_pnl"]   = round(unrealized_pnl, 2)
            pos["unrealized_pnl_pct"] = round((last_close - entry) / entry * 100, 2)
            still_open.append(pos)

    state["positions"] = still_open
    _snapshot_equity(state)
    state["last_updated"] = datetime.datetime.now().isoformat()

    return state, newly_closed


# ═══════════════════════════════════════════════════════════
# EQUITY HELPER
# ═══════════════════════════════════════════════════════════

def _snapshot_equity(state: Dict[str, Any]):
    """Add current equity snapshot to history (deduped by date)."""
    today = datetime.date.today().isoformat()

    # Compute current equity: cash + market value of open positions
    open_value = sum(
        p.get("current_price", p["entry"]) * p["shares"]
        for p in state["positions"]
    )
    equity = state["cash"] + open_value

    # Replace today's entry if already present
    history = [h for h in state["equity_history"] if h["date"] != today]
    history.append({"date": today, "equity": round(equity, 2)})
    history.sort(key=lambda x: x["date"])
    state["equity_history"] = history


def get_current_equity(state: Dict[str, Any]) -> float:
    """Return current total equity (cash + open market value)."""
    open_value = sum(
        p.get("current_price", p["entry"]) * p["shares"]
        for p in state["positions"]
    )
    return round(state["cash"] + open_value, 2)


def get_simulation_stats(state: Dict[str, Any]) -> Dict[str, Any]:
    """Return summary statistics for the simulation."""
    equity = get_current_equity(state)
    start  = state["start_capital"]
    closed = state["closed_trades"]

    total_pnl     = equity - start
    total_pnl_pct = (total_pnl / start) * 100

    wins   = [t for t in closed if t["pnl_eur"] > 0]
    losses = [t for t in closed if t["pnl_eur"] <= 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0

    avg_win  = sum(t["pnl_eur"] for t in wins)  / max(len(wins), 1)
    avg_loss = sum(t["pnl_eur"] for t in losses) / max(len(losses), 1)

    return {
        "equity":         equity,
        "cash":           round(state["cash"], 2),
        "start_capital":  start,
        "total_pnl":      round(total_pnl, 2),
        "total_pnl_pct":  round(total_pnl_pct, 2),
        "open_positions": len(state["positions"]),
        "closed_trades":  len(closed),
        "win_rate":       round(win_rate, 1),
        "avg_win_eur":    round(avg_win, 2),
        "avg_loss_eur":   round(avg_loss, 2),
    }
