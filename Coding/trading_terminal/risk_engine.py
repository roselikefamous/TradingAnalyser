from dataclasses import dataclass
from typing import Optional

import pandas as pd

from trading_terminal.contracts import RiskSnapshot


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.20
    max_open_positions: int = 1
    max_gross_exposure_pct: float = 1.0


class RiskEngine:
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.peak_equity: Optional[float] = None
        self.day_start_equity: Optional[float] = None
        self.current_day: Optional[pd.Timestamp] = None
        self.daily_pnl: float = 0.0
        self.halted: bool = False

    def _roll_day(self, date: pd.Timestamp, equity: float) -> None:
        d = pd.Timestamp(date).normalize()
        if self.current_day is None or d != self.current_day:
            self.current_day = d
            self.day_start_equity = equity
            self.daily_pnl = 0.0

    def on_mark(self, date: pd.Timestamp, equity: float) -> None:
        self._roll_day(date, equity)
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        dd_pct = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if dd_pct >= self.config.max_drawdown_pct:
            self.halted = True

    def on_realized_pnl(self, date: pd.Timestamp, pnl: float, equity_after: float) -> None:
        self._roll_day(date, equity_after)
        self.daily_pnl += pnl
        day_base = self.day_start_equity if self.day_start_equity else max(1.0, equity_after)
        daily_loss_pct = max(0.0, -self.daily_pnl / day_base)
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            self.halted = True
        self.on_mark(date, equity_after)

    def can_open_trade(
        self,
        date: pd.Timestamp,
        equity: float,
        proposed_notional: float,
        gross_exposure: float,
        open_positions: int,
    ) -> tuple[bool, str]:
        self.on_mark(date, equity)
        if self.halted:
            return False, "Risk halt active"
        if open_positions >= self.config.max_open_positions:
            return False, "Max open positions reached"

        max_gross = equity * self.config.max_gross_exposure_pct
        if gross_exposure + proposed_notional > max_gross:
            return False, "Gross exposure limit breached"
        return True, "OK"

    def snapshot(
        self,
        date: pd.Timestamp,
        equity: float,
        cash: float,
        gross_exposure: float,
        open_positions: int,
    ) -> RiskSnapshot:
        self.on_mark(date, equity)
        dd_pct = 0.0
        if self.peak_equity and self.peak_equity > 0:
            dd_pct = (self.peak_equity - equity) / self.peak_equity
        return RiskSnapshot(
            date=pd.Timestamp(date),
            equity=float(equity),
            cash=float(cash),
            gross_exposure=float(gross_exposure),
            open_positions=int(open_positions),
            drawdown_pct=float(dd_pct),
            daily_pnl=float(self.daily_pnl),
            halted=bool(self.halted),
        )

