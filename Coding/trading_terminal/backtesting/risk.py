import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskLimits:
    max_daily_loss_pct: float = 0.05      # 5% max daily loss
    max_drawdown_pct: float = 0.15        # 15% max drawdown from peak
    max_open_exposure_pct: float = 1.0    # 100% max gross exposure (no margin scaling by default)
    kill_switch: bool = False             # Global manual halt


class RiskManager:
    """Evaluates real-time portfolio state against hard risk gates."""
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        
        # Internal state tracking
        self.peak_equity: float = 0.0
        self.start_of_day_equity: float = 0.0
        self.current_equity: float = 0.0
        self.current_exposure: float = 0.0
        self.is_halted: bool = False
        self.halt_reason: str = ""
        self.current_date: Optional[pd.Timestamp] = None

    def update_state(self, date: pd.Timestamp, equity: float, gross_exposure: float):
        """Update the risk manager with the latest MTM portfolio state."""
        # Detect new trading day
        if self.current_date is None or self.current_date.date() != date.date():
            self.start_of_day_equity = equity
            self.current_date = date
        
        self.current_equity = equity
        self.current_exposure = gross_exposure
        
        # Update high-water mark
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        self._check_limits()

    def _check_limits(self):
        """Internal check to halt trading if global risk gates are breached."""
        if self.is_halted:
            return

        if self.limits.kill_switch:
            self.halt("Global kill switch activated by user/admin.")
            return

        if self.peak_equity > 0:
            dd = (self.peak_equity - self.current_equity) / self.peak_equity
            if dd >= self.limits.max_drawdown_pct:
                self.halt(f"Max drawdown breached: {dd:.2%} >= {self.limits.max_drawdown_pct:.2%}")

        if self.start_of_day_equity > 0:
            daily_loss = (self.start_of_day_equity - self.current_equity) / self.start_of_day_equity
            if daily_loss >= self.limits.max_daily_loss_pct:
                self.halt(f"Daily loss limit breached: {daily_loss:.2%} >= {self.limits.max_daily_loss_pct:.2%}")

    def halt(self, reason: str):
        """Halt trading persistently."""
        if not self.is_halted:
            self.is_halted = True
            self.halt_reason = reason

    def check_pre_trade(self, estimated_order_value: float) -> tuple[bool, str]:
        """
        Pre-trade check before emitting an order.
        Returns: (is_approved, rejection_reason)
        """
        if self.is_halted:
            return False, f"Trading halted: {self.halt_reason}"
        
        new_exposure = self.current_exposure + estimated_order_value
        exposure_pct = new_exposure / self.current_equity if self.current_equity > 0 else 0
        
        if exposure_pct > self.limits.max_open_exposure_pct:
            return False, f"Order exceeds max exposure ({exposure_pct:.2%} > {self.limits.max_open_exposure_pct:.2%})"
        
        return True, "OK"
