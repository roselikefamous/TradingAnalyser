from dataclasses import dataclass
from typing import Optional

import pandas as pd

from trading_terminal.contracts import (
    BacktestRunResult,
    BacktestSignal,
    Trade,
)
from trading_terminal.risk_engine import RiskConfig, RiskEngine


@dataclass
class BacktestConfig:
    initial_cash: float = 100_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 3.0
    default_notional_pct: float = 0.10
    max_holding_bars: Optional[int] = None
    intrabar_mode: str = "conservative"  # conservative | optimistic | mid


def run_backtest(
    df: pd.DataFrame,
    signals: list[BacktestSignal],
    config: Optional[BacktestConfig] = None,
    risk_config: Optional[RiskConfig] = None,
) -> BacktestRunResult:
    cfg = config or BacktestConfig()
    if cfg.intrabar_mode not in {"conservative", "optimistic", "mid"}:
        raise ValueError("intrabar_mode must be one of: conservative, optimistic, mid")
    risk = RiskEngine(risk_config or RiskConfig())
    if df.empty:
        return BacktestRunResult(
            trades=[],
            equity_curve=pd.DataFrame(columns=["Date", "Equity", "Cash"]),
            risk_snapshots=[],
            final_cash=cfg.initial_cash,
            final_equity=cfg.initial_cash,
        )

    signals_by_date: dict[pd.Timestamp, list[BacktestSignal]] = {}
    for s in signals:
        signals_by_date.setdefault(pd.Timestamp(s.date), []).append(s)

    cash = float(cfg.initial_cash)
    trades: list[Trade] = []
    risk_snaps = []
    eq_rows = []
    active_positions = []

    for i, (ts, row) in enumerate(df.iterrows()):
        ts = pd.Timestamp(ts)
        close = float(row["Close"])
        open_px = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])

        gross_exposure = sum(abs(p["qty"] * close) for p in active_positions)
        unrealized_value = 0.0
        for p in active_positions:
            if p["side"] == "LONG":
                unrealized_value += p["qty"] * close
            else:
                unrealized_value += p["qty"] * p["entry_price"] + (p["entry_price"] - close) * p["qty"]
        equity = cash + unrealized_value

        closed_positions = []
        for p in list(active_positions):
            side = p["side"]
            tp = p["tp"]
            sl = p["sl"]
            exit_price = None
            reason = None

            hit_tp = high >= tp if side == "LONG" else low <= tp
            hit_sl = low <= sl if side == "LONG" else high >= sl

            if hit_tp and hit_sl:
                if cfg.intrabar_mode == "conservative":
                    exit_price = sl
                    reason = "SL/TP same bar (conservative SL)"
                elif cfg.intrabar_mode == "optimistic":
                    exit_price = tp
                    reason = "SL/TP same bar (optimistic TP)"
                else:
                    exit_price = (tp + sl) / 2
                    reason = "SL/TP same bar (midpoint)"
            elif hit_sl:
                exit_price = sl
                reason = "SL"
            elif hit_tp:
                exit_price = tp
                reason = "TP"
            elif cfg.max_holding_bars is not None and (i - p["entry_i"]) >= cfg.max_holding_bars:
                exit_price = close
                reason = "TIME"

            if exit_price is not None:
                slippage = cfg.slippage_bps / 10000
                exec_price = exit_price * (1 - slippage if side == "LONG" else 1 + slippage)
                fee = abs(exec_price * p["qty"]) * (cfg.fee_bps / 10000)
                
                if side == "LONG":
                    cash += p["qty"] * exec_price - fee
                    gross_pnl = (exec_price - p["entry_price"]) * p["qty"]
                else:
                    gross_pnl = (p["entry_price"] - exec_price) * p["qty"]
                    cash += p["qty"] * p["entry_price"] + gross_pnl - fee
                    
                net_pnl = gross_pnl - p["entry_fee"] - fee
                trade = Trade(
                    symbol=p["symbol"],
                    side=side,
                    entry_date=p["entry_date"],
                    exit_date=ts,
                    quantity=float(p["qty"]),
                    entry_price=float(p["entry_price"]),
                    exit_price=float(exec_price),
                    gross_pnl=float(gross_pnl),
                    net_pnl=float(net_pnl),
                    fee_total=float(p["entry_fee"] + fee),
                    exit_reason=str(reason),
                )
                trades.append(trade)
                equity = cash # update intermediate equity
                risk.on_realized_pnl(ts, net_pnl, equity)
                closed_positions.append(p)

        for p in closed_positions:
            active_positions.remove(p)
            
        gross_exposure = sum(abs(p["qty"] * close) for p in active_positions)

        for sig in signals_by_date.get(ts, []):
            side = sig.side.upper()
            notional_target = cash * cfg.default_notional_pct
            entry_slippage = cfg.slippage_bps / 10000
            entry_price = open_px * (1 + entry_slippage if side == "LONG" else 1 - entry_slippage)
            qty = sig.quantity if sig.quantity is not None else (notional_target / entry_price if entry_price > 0 else 0.0)
            notional = abs(qty * entry_price)

            ok, _ = risk.can_open_trade(
                date=ts,
                equity=equity,
                proposed_notional=notional,
                gross_exposure=gross_exposure,
                open_positions=len(active_positions),
            )
            if not ok or qty <= 0:
                continue

            fee = notional * (cfg.fee_bps / 10000)
            cash -= (qty * entry_price + fee)
            new_pos = {
                "symbol": sig.symbol,
                "side": side,
                "qty": float(qty),
                "entry_price": float(entry_price),
                "entry_fee": float(fee),
                "entry_date": ts,
                "entry_i": i,
                "tp": float(sig.tp),
                "sl": float(sig.sl),
            }
            active_positions.append(new_pos)
            gross_exposure += abs(new_pos["qty"] * close)

        unrealized_value = 0.0
        for p in active_positions:
            if p["side"] == "LONG":
                unrealized_value += p["qty"] * close
            else:
                unrealized_value += p["qty"] * p["entry_price"] + (p["entry_price"] - close) * p["qty"]
        equity = cash + unrealized_value
        
        eq_rows.append({"Date": ts, "Equity": equity, "Cash": cash})
        risk_snaps.append(
            risk.snapshot(
                date=ts,
                equity=equity,
                cash=cash,
                gross_exposure=gross_exposure,
                open_positions=len(active_positions),
            )
        )

    ts = pd.Timestamp(df.index[-1])
    close = float(df["Close"].iloc[-1])
    for p in active_positions:
        side = p["side"]
        slippage = cfg.slippage_bps / 10000
        exec_price = close * (1 - slippage if side == "LONG" else 1 + slippage)
        fee = abs(exec_price * p["qty"]) * (cfg.fee_bps / 10000)
        
        if side == "LONG":
            cash += p["qty"] * exec_price - fee
            gross_pnl = (exec_price - p["entry_price"]) * p["qty"]
        else:
            gross_pnl = (p["entry_price"] - exec_price) * p["qty"]
            cash += p["qty"] * p["entry_price"] + gross_pnl - fee
            
        net_pnl = gross_pnl - p["entry_fee"] - fee
        trades.append(
            Trade(
                symbol=p["symbol"],
                side=side,
                entry_date=p["entry_date"],
                exit_date=ts,
                quantity=float(p["qty"]),
                entry_price=float(p["entry_price"]),
                exit_price=float(exec_price),
                gross_pnl=float(gross_pnl),
                net_pnl=float(net_pnl),
                fee_total=float(p["entry_fee"] + fee),
                exit_reason="EOD",
            )
        )

    equity_curve = pd.DataFrame(eq_rows)
    final_equity = cash
    return BacktestRunResult(
        trades=trades,
        equity_curve=equity_curve,
        risk_snapshots=risk_snaps,
        final_cash=float(cash),
        final_equity=float(final_equity),
    )
