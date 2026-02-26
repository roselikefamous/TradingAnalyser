from typing import List, Dict, Optional
import pandas as pd
import uuid

from trading_terminal.execution.signals import ExecutionOrder, OrderStatus, OrderType, OrderSide
from trading_terminal.contracts import Fill
from trading_terminal.utils.logger import log_trade_event

class PaperBroker:
    """Mock broker simulating live asynchronous execution with latency and slippage."""
    
    def __init__(self, latency_ms: int = 50, slippage_bps: float = 3.0, fee_bps: float = 5.0):
        self.latency_ms = latency_ms
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        
        # Order Book
        self.active_orders: Dict[str, ExecutionOrder] = {}
        self.completed_orders: Dict[str, ExecutionOrder] = {}
        self.fills: List[Fill] = []

    def place_order(self, order: ExecutionOrder) -> str:
        """Submit a new order to the broker."""
        order.status = OrderStatus.PENDING
        self.active_orders[order.order_id] = order
        
        log_trade_event(
            event_type="ORDER_PLACED",
            message=f"Received {order.side.name} {order.order_type.name} order for {order.symbol}",
            order_id=order.order_id,
            symbol=order.symbol,
            quantity=order.quantity
        )
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            order.status = OrderStatus.CANCELED
            self.completed_orders[order_id] = order
            
            log_trade_event(
                event_type="ORDER_CANCELED",
                message=f"Canceled order {order_id}",
                order_id=order_id
            )
            return True
        return False

    def process_market_tick(self, date: pd.Timestamp, symbol: str, bid: float, ask: float):
        """Simulate real-time price tick processing to fill orders."""
        # Process a copy to allow modification during iteration
        for order_id, order in list(self.active_orders.items()):
            if order.symbol != symbol:
                continue

            if order.status == OrderStatus.PENDING:
                # Simulate simple fill logic
                if order.order_type == OrderType.MARKET:
                    self._execute_fill(order, bid, ask, date)
                
                elif order.order_type == OrderType.LIMIT and order.price_target:
                    if order.side == OrderSide.BUY and ask <= order.price_target:
                        self._execute_fill(order, bid, ask, date)
                    elif order.side == OrderSide.SELL and bid >= order.price_target:
                        self._execute_fill(order, bid, ask, date)
                        
                elif order.order_type == OrderType.STOP and order.price_target:
                    if order.side == OrderSide.BUY and ask >= order.price_target:
                        # Stop triggered, turns into market order
                        self._execute_fill(order, bid, ask, date)
                    elif order.side == OrderSide.SELL and bid <= order.price_target:
                        self._execute_fill(order, bid, ask, date)

    def _execute_fill(self, order: ExecutionOrder, bid: float, ask: float, date: pd.Timestamp):
        """Internal logic to compute slippage, fee, and generate a Fill."""
        # Simulate slippage by slightly worsening the fill price from the BBO
        mid = (bid + ask) / 2
        
        if order.side == OrderSide.BUY:
            raw_price = ask
            fill_price = raw_price * (1 + (self.slippage_bps / 10000))
        else:
            raw_price = bid
            fill_price = raw_price * (1 - (self.slippage_bps / 10000))
            
        fee_amount = (fill_price * order.quantity) * (self.fee_bps / 10000)

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        
        self.active_orders.pop(order.order_id)
        self.completed_orders[order.order_id] = order

        fill = Fill(
            date=date,
            symbol=order.symbol,
            side=order.side.name,
            quantity=order.filled_quantity,
            price=fill_price,
            fee=fee_amount,
            slippage_bps=self.slippage_bps
        )
        self.fills.append(fill)
        
        log_trade_event(
            event_type="ORDER_FILLED",
            message=f"Filled {order.side.name} {order.quantity} {order.symbol} @ {fill_price:.2f}",
            order_id=order.order_id,
            symbol=order.symbol,
            fill_price=fill_price,
            fee=fee_amount
        )
