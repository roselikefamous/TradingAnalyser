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


def run_backtest(
    df: pd.DataFrame,
    signals: list[BacktestSignal],
    config: Optional[BacktestConfig] = None,
    risk_config: Optional[RiskConfig] = None,
) -> BacktestRunResult:
    cfg = config or BacktestConfig()
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
    open_pos = None

    for i, (ts, row) in enumerate(df.iterrows()):
        ts = pd.Timestamp(ts)
        close = float(row["Close"])
        open_px = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])

        gross_exposure = 0.0
        if open_pos is not None:
            gross_exposure = abs(open_pos["qty"] * close)
        equity = cash + (open_pos["qty"] * close if open_pos is not None else 0.0)

        if open_pos is not None:
            side = open_pos["side"]
            tp = open_pos["tp"]
            sl = open_pos["sl"]
            exit_price = None
            reason = None

            hit_tp = high >= tp if side == "LONG" else low <= tp
            hit_sl = low <= sl if side == "LONG" else high >= sl

            if hit_tp and hit_sl:
                exit_price = sl
                reason = "SL/TP same bar (conservative SL)"
            elif hit_sl:
                exit_price = sl
                reason = "SL"
            elif hit_tp:
                exit_price = tp
                reason = "TP"
            elif cfg.max_holding_bars is not None and (i - open_pos["entry_i"]) >= cfg.max_holding_bars:
                exit_price = close
                reason = "TIME"

            if exit_price is not None:
                slippage = cfg.slippage_bps / 10000
                exec_price = exit_price * (1 - slippage if side == "LONG" else 1 + slippage)
                fee = abs(exec_price * open_pos["qty"]) * (cfg.fee_bps / 10000)
                cash += open_pos["qty"] * exec_price - fee
                gross_pnl = (exec_price - open_pos["entry_price"]) * open_pos["qty"]
                net_pnl = gross_pnl - open_pos["entry_fee"] - fee
                trade = Trade(
                    symbol=open_pos["symbol"],
                    side=side,
                    entry_date=open_pos["entry_date"],
                    exit_date=ts,
                    quantity=float(open_pos["qty"]),
                    entry_price=float(open_pos["entry_price"]),
                    exit_price=float(exec_price),
                    gross_pnl=float(gross_pnl),
                    net_pnl=float(net_pnl),
                    fee_total=float(open_pos["entry_fee"] + fee),
                    exit_reason=str(reason),
                )
                trades.append(trade)
                equity = cash
                risk.on_realized_pnl(ts, net_pnl, equity)
                open_pos = None
                gross_exposure = 0.0

        if open_pos is None:
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
                    open_positions=1 if open_pos is not None else 0,
                )
                if not ok or qty <= 0:
                    continue

                fee = notional * (cfg.fee_bps / 10000)
                cash -= (qty * entry_price + fee)
                open_pos = {
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
                gross_exposure = abs(open_pos["qty"] * close)
                break

        equity = cash + (open_pos["qty"] * close if open_pos is not None else 0.0)
        eq_rows.append({"Date": ts, "Equity": equity, "Cash": cash})
        risk_snaps.append(
            risk.snapshot(
                date=ts,
                equity=equity,
                cash=cash,
                gross_exposure=gross_exposure,
                open_positions=1 if open_pos is not None else 0,
            )
        )

    if open_pos is not None:
        ts = pd.Timestamp(df.index[-1])
        close = float(df["Close"].iloc[-1])
        side = open_pos["side"]
        slippage = cfg.slippage_bps / 10000
        exec_price = close * (1 - slippage if side == "LONG" else 1 + slippage)
        fee = abs(exec_price * open_pos["qty"]) * (cfg.fee_bps / 10000)
        cash += open_pos["qty"] * exec_price - fee
        gross_pnl = (exec_price - open_pos["entry_price"]) * open_pos["qty"]
        net_pnl = gross_pnl - open_pos["entry_fee"] - fee
        trades.append(
            Trade(
                symbol=open_pos["symbol"],
                side=side,
                entry_date=open_pos["entry_date"],
                exit_date=ts,
                quantity=float(open_pos["qty"]),
                entry_price=float(open_pos["entry_price"]),
                exit_price=float(exec_price),
                gross_pnl=float(gross_pnl),
                net_pnl=float(net_pnl),
                fee_total=float(open_pos["entry_fee"] + fee),
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

