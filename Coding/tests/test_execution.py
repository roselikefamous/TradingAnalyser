import pytest
import pandas as pd

from trading_terminal.execution.signals import ExecutionOrder, OrderStatus, OrderType, OrderSide
from trading_terminal.execution.broker import PaperBroker
from trading_terminal.execution.reconciliation import reconcile_fills
from trading_terminal.contracts import Fill

def test_paper_broker_market_fill():
    broker = PaperBroker(latency_ms=0, slippage_bps=5.0, fee_bps=5.0)
    
    order = ExecutionOrder(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0
    )
    
    order_id = broker.place_order(order)
    assert order_id in broker.active_orders
    
    # Tick simulation
    date = pd.Timestamp("2023-01-01 10:00:00")
    broker.process_market_tick(date, "AAPL", bid=100.0, ask=101.0)
    
    assert order_id not in broker.active_orders
    assert order_id in broker.completed_orders
    
    completed = broker.completed_orders[order_id]
    assert completed.status == OrderStatus.FILLED
    
    # Buy at ask 101.0 + 5bps slippage (0.0505) = 101.0505
    assert completed.avg_fill_price == pytest.approx(101.0505)
    
    assert len(broker.fills) == 1
    fill = broker.fills[0]
    assert fill.quantity == 10.0
    assert fill.price == pytest.approx(101.0505)


def test_paper_broker_limit_order():
    broker = PaperBroker()
    
    order = ExecutionOrder(
        symbol="TSLA",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=5.0,
        price_target=200.0
    )
    
    broker.place_order(order)
    
    # Tick 1: Ask is 201 (Too high, won't fill)
    broker.process_market_tick(pd.Timestamp("2023-01-01 10:00:00"), "TSLA", bid=200.0, ask=201.0)
    assert len(broker.fills) == 0
    
    # Tick 2: Ask drops to 199 (Will fill!)
    broker.process_market_tick(pd.Timestamp("2023-01-01 10:01:00"), "TSLA", bid=198.0, ask=199.0)
    assert len(broker.fills) == 1


def test_paper_broker_cancel():
    broker = PaperBroker()
    
    order = ExecutionOrder(
        symbol="MSFT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=1.0
    )
    order_id = broker.place_order(order)
    
    success = broker.cancel_order(order_id)
    assert success
    
    assert order_id not in broker.active_orders
    assert broker.completed_orders[order_id].status == OrderStatus.CANCELED


def test_reconciliation():
    expected_fills = [
        Fill(pd.Timestamp("2023-01-01 10:00:00"), "AAPL", "BUY", 10.0, 100.0, 0, 0),
        Fill(pd.Timestamp("2023-01-01 11:00:00"), "TSLA", "SELL", 5.0, 200.0, 0, 0)
    ]
    
    actual_fills = [
        # Match for AAPL but slippage is high!
        Fill(pd.Timestamp("2023-01-01 10:00:05"), "AAPL", "BUY", 10.0, 105.0, 1.0, 3.0),
        # Unmatched actual fill
        Fill(pd.Timestamp("2023-01-01 12:00:00"), "MSFT", "BUY", 1.0, 300.0, 1.0, 3.0)
    ]
    
    res = reconcile_fills(expected_fills, actual_fills)
    
    assert res.matched_fills == 1
    assert res.unmatched_expected == 1 # TSLA missing
    assert res.unmatched_actual == 1   # MSFT orphan
    
    # Slippage error AAPL: (105-100)/100 = 5%
    assert res.avg_slippage_error_pct == pytest.approx(5.0)
    assert "Severe Price Mismatch" in res.issues[0]
