from dataclasses import dataclass, field
from enum import Enum
import uuid
import pandas as pd
from typing import Optional

class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class ExecutionOrder:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price_target: Optional[float] = None  # Relevant for limit/stop orders
    
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    submit_date: Optional[pd.Timestamp] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    
    # Context
    strategy_name: str = "Manual"
    signal_id: Optional[str] = None
