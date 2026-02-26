import pandas as pd
import numpy as np
import pytest

from trading_terminal.backtesting.portfolio import PortfolioSimEngine
from trading_terminal.backtesting.risk import RiskManager, RiskLimits
from trading_terminal.contracts import StrategySignal

@pytest.fixture
def sample_market_data():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Open": [100, 102, 101, 105, 108, 107, 110, 112, 115, 114],
        "High": [105, 104, 106, 109, 110, 112, 115, 116, 118, 116],
        "Low": [98, 100, 99, 102, 105, 106, 108, 110, 112, 110],
        "Close": [102, 101, 105, 108, 107, 110, 112, 115, 114, 113],
        "Volume": [1000] * 10
    }, index=dates)
    return df

def test_risk_manager_drawdown_halt():
    limits = RiskLimits(max_drawdown_pct=0.10)
    rm = RiskManager(limits=limits)
    
    rm.update_state(pd.Timestamp("2023-01-01"), 10000, 0)
    assert not rm.is_halted
    
    # Peak equity reaches 10,000. Now drops to 8,900 (-11%)
    rm.update_state(pd.Timestamp("2023-01-02"), 8900, 0)
    
    assert rm.is_halted
    assert "drawdown" in rm.halt_reason.lower()

def test_risk_manager_daily_loss_halt():
    limits = RiskLimits(max_daily_loss_pct=0.05)
    rm = RiskManager(limits=limits)
    
    # Start of day 1: 10000
    rm.update_state(pd.Timestamp("2023-01-01 09:30"), 10000, 0)
    assert rm.start_of_day_equity == 10000
    
    # Intraday drop to 9400 (-6%)
    rm.update_state(pd.Timestamp("2023-01-01 14:00"), 9400, 0)
    
    assert rm.is_halted
    assert "daily loss" in rm.halt_reason.lower()

def test_portfolio_sim_engine_basic_long(sample_market_data):
    sim = PortfolioSimEngine(initial_capital=10000, fee_bps=0, slippage_bps=0)
    
    # Entry at 100 on Jan 1, SL at 90, TP at 110
    sig1 = StrategySignal(
        date=pd.Timestamp("2023-01-01"),
        entry=100.0,
        sl=90.0,
        side="LONG",
        tp=110.0
    )
    
    res = sim.run(sample_market_data, [sig1], symbol="AAPL")
    
    assert len(res.trades) == 1
    trade = res.trades[0]
    
    assert trade.side == "LONG"
    # Jan 7 High is 115, which hits the TP of 110
    assert trade.exit_price == 110.0
    assert trade.exit_reason == "Take Profit"
    assert trade.net_pnl > 0

def test_portfolio_sim_engine_risk_halt(sample_market_data):
    limits = RiskLimits(max_drawdown_pct=0.0001) # Very tight DD
    rm = RiskManager(limits=limits)
    sim = PortfolioSimEngine(initial_capital=10000, fee_bps=0, slippage_bps=0, risk_manager=rm)
    
    sig1 = StrategySignal(
        date=pd.Timestamp("2023-01-01"),
        entry=100.0,
        sl=50.0,
        side="LONG",
        tp=150.0
    )
    
    res = sim.run(sample_market_data, [sig1], symbol="AAPL")
    
    # The trade should be stopped out by Risk Halt, not SL/TP
    assert len(res.trades) == 1
    trade = res.trades[0]
    assert "Risk Halt" in trade.exit_reason
