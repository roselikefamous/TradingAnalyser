import pandas as pd
from typing import Optional, Dict

from trading_terminal.contracts import Trade, RiskSnapshot, BacktestRunResult, StrategySignal
from trading_terminal.backtesting.risk import RiskManager

class Position:
    def __init__(self, symbol: str, quantity: float, entry_price: float, side: str, entry_date: pd.Timestamp, sl: float, tp: Optional[float]):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.side = side.upper()
        self.entry_date = entry_date
        self.sl = sl
        self.tp = tp

    def pnl(self, current_price: float) -> float:
        if self.side == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def gross_exposure(self, current_price: float) -> float:
        return self.quantity * current_price


class PortfolioSimEngine:
    """Event-driven simulation engine modeling portfolio cash, equity, and risk."""
    
    def __init__(self, initial_capital: float = 10000.0, fee_bps: float = 5.0, slippage_bps: float = 3.0, risk_manager: Optional[RiskManager] = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.risk = risk_manager or RiskManager()
        
        # We track positions by symbol, assuming at most 1 position per symbol for now
        self.positions: Dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.risk_snapshots: list[RiskSnapshot] = []
        
        self.equity_curve_dates = []
        self.equity_curve_values = []

    def _calc_fees(self, notional: float) -> tuple[float, float]:
        """Returns (fee_amount, slippage_amount)"""
        fee = notional * (self.fee_bps / 10000)
        slip = notional * (self.slippage_bps / 10000)
        return fee, slip

    def get_equity(self, current_price: float) -> float:
        open_pnl = sum(p.pnl(current_price) for p in self.positions.values())
        return self.cash + open_pnl + sum(p.entry_price * p.quantity for p in self.positions.values() if p.side == "LONG") - sum(p.entry_price * p.quantity for p in self.positions.values() if p.side == "SHORT")

    def run(self, df: pd.DataFrame, signals: list[StrategySignal], symbol: str = "ASSET") -> BacktestRunResult:
        if df.empty:
            return BacktestRunResult([], pd.DataFrame(), [], self.cash, self.cash)

        # Pre-group signals by date for quick access
        signal_map = {}
        for s in signals:
            d = s.date.normalize() if hasattr(s.date, 'normalize') else s.date
            if d not in signal_map:
                signal_map[d] = []
            signal_map[d].append(s)

        for i, (idx, row) in enumerate(df.iterrows()):
            curr_date = pd.Timestamp(idx).normalize() if hasattr(idx, 'normalize') else pd.Timestamp(idx)
            current_price = float(row['Close'])
            high = float(row['High'])
            low = float(row['Low'])

            # 1. Update Mark-to-Market and Risk State
            # (In a real intraday tick engine, this happens continuously. Here we approximate end-of-bar)
            equity = self.cash
            gross_exp = 0.0
            for p in self.positions.values():
                gross_exp += p.gross_exposure(current_price)
                if p.side == "LONG":
                    equity += p.quantity * current_price
                else:
                    # short proceeds are in cash during entry, but we need to subtract the liability
                    equity -= p.quantity * current_price

            self.risk.update_state(curr_date, equity, gross_exp)

            # Record snapshot
            prev_equity = self.risk_snapshots[-1].equity if self.risk_snapshots else self.initial_capital
            daily_pnl = equity - prev_equity
            dd_pct = (self.risk.peak_equity - equity) / self.risk.peak_equity if self.risk.peak_equity > 0 else 0.0
            
            self.risk_snapshots.append(RiskSnapshot(
                date=pd.Timestamp(idx),
                equity=equity,
                cash=self.cash,
                gross_exposure=gross_exp,
                open_positions=len(self.positions),
                drawdown_pct=float(dd_pct),
                daily_pnl=float(daily_pnl),
                halted=self.risk.is_halted
            ))
            
            self.equity_curve_dates.append(idx)
            self.equity_curve_values.append(equity)

            # 2. Halt logic: close all on halt
            to_close = []
            if self.risk.is_halted:
                for sym, p in self.positions.items():
                    to_close.append((sym, current_price, f"Risk Halt: {self.risk.halt_reason}"))
            else:
                # 3. Check for SL / TP hits first inline
                for sym, p in self.positions.items():
                    if p.side == "LONG":
                        if low <= p.sl:
                            to_close.append((sym, p.sl, "Stop Loss"))
                        elif p.tp and high >= p.tp:
                            to_close.append((sym, p.tp, "Take Profit"))
                    else: # SHORT
                        if high >= p.sl:
                            to_close.append((sym, p.sl, "Stop Loss"))
                        elif p.tp and low <= p.tp:
                            to_close.append((sym, p.tp, "Take Profit"))

            # Execute Closures
            for sym, exit_price, reason in to_close:
                if sym not in self.positions:
                    continue # already closed
                p = self.positions.pop(sym)
                
                notional = p.quantity * exit_price
                fee, slip = self._calc_fees(notional)
                total_cost = fee + slip
                
                if p.side == "LONG":
                    self.cash += (notional - total_cost)
                    gross_pnl = (exit_price - p.entry_price) * p.quantity
                else:
                    # Cash was increased by short proceeds at entry, now we repurchase
                    self.cash -= (notional + total_cost)
                    gross_pnl = (p.entry_price - exit_price) * p.quantity
                    
                net_pnl = gross_pnl - total_cost - p._entry_cost # we need to attach entry cost
                
                self.trades.append(Trade(
                    symbol=sym,
                    side=p.side,
                    entry_date=p.entry_date,
                    exit_date=pd.Timestamp(idx),
                    quantity=p.quantity,
                    entry_price=p.entry_price,
                    exit_price=exit_price,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    fee_total=total_cost + p._entry_fee, # store full cycle
                    exit_reason=reason
                ))

            # 4. Process new signals (only if not halted)
            if not self.risk.is_halted and curr_date in signal_map:
                daily_sigs = signal_map[curr_date]
                for sig in daily_sigs:
                    if symbol in self.positions:
                        continue # simple: 1 position per asset max
                    
                    # Estimate value
                    # Risk rule: 1% risk per trade logic could be applied here instead of passing qty.
                    # For simplicity, if not provided, we risk 1% of current equity.
                    risk_allowance = equity * 0.01
                    risk_per_share = abs(sig.entry - sig.sl)
                    if risk_per_share <= 0:
                        continue
                        
                    qty = risk_allowance / risk_per_share
                    # Ensure we don't buy more than we can afford (no margin initially)
                    if sig.side.upper() == "LONG":
                        max_qty = self.cash / sig.entry
                        qty = min(qty, max_qty)
                    
                    if qty <= 0.0001:
                        continue
                        
                    est_val = qty * sig.entry
                    approved, reason = self.risk.check_pre_trade(est_val)
                    
                    if approved:
                        notional = qty * sig.entry
                        fee, slip = self._calc_fees(notional)
                        cost = fee + slip
                        
                        side = sig.side.upper()
                        if side == "LONG":
                            self.cash -= (notional + cost)
                        else:
                            self.cash += (notional - cost)
                            
                        pos = Position(
                            symbol=symbol,
                            quantity=qty,
                            entry_price=sig.entry,
                            side=side,
                            entry_date=pd.Timestamp(idx),
                            sl=sig.sl,
                            tp=sig.tp
                        )
                        pos._entry_cost = cost
                        pos._entry_fee = fee
                        self.positions[symbol] = pos

        # End of Backtest: Force Close EOP
        if self.positions:
            last_date = pd.Timestamp(df.index[-1])
            last_price = float(df['Close'].iloc[-1])
            for sym, p in list(self.positions.items()):
                p = self.positions.pop(sym)
                notional = p.quantity * last_price
                fee, slip = self._calc_fees(notional)
                total_cost = fee + slip
                
                if p.side == "LONG":
                    self.cash += (notional - total_cost)
                    gross_pnl = (last_price - p.entry_price) * p.quantity
                else:
                    self.cash -= (notional + total_cost)
                    gross_pnl = (p.entry_price - last_price) * p.quantity
                    
                net_pnl = gross_pnl - total_cost - p._entry_cost
                
                self.trades.append(Trade(
                    symbol=sym,
                    side=p.side,
                    entry_date=p.entry_date,
                    exit_date=last_date,
                    quantity=p.quantity,
                    entry_price=p.entry_price,
                    exit_price=last_price,
                    gross_pnl=gross_pnl,
                    net_pnl=net_pnl,
                    fee_total=total_cost + p._entry_fee,
                    exit_reason="End of Portfolio Sim Mark-to-Market"
                ))

        eq_df = pd.DataFrame({"Date": self.equity_curve_dates, "Equity": self.equity_curve_values})
        if not eq_df.empty:
            eq_df.set_index("Date", inplace=True)

        return BacktestRunResult(
            trades=self.trades,
            equity_curve=eq_df,
            risk_snapshots=self.risk_snapshots,
            final_cash=self.cash,
            final_equity=self.equity_curve_values[-1] if self.equity_curve_values else self.initial_capital
        )
