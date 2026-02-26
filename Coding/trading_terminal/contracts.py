from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class MarketDataBundle:
    df: pd.DataFrame
    info: dict[str, Any]
    news: list[dict[str, Any]]
    dividends: Optional[pd.Series]
    earnings_dates: Optional[pd.DataFrame]


@dataclass(frozen=True)
class StrategySignal:
    date: pd.Timestamp
    entry: float
    sl: float
    side: str
    tp: Optional[float] = None
    pattern: Optional[str] = None
    rsi: Optional[float] = None
    level: Optional[str] = None
    stoch_confirm: Optional[bool] = None

    @staticmethod
    def from_legacy(payload: dict[str, Any]) -> "StrategySignal":
        return StrategySignal(
            date=pd.Timestamp(payload["Date"]),
            entry=float(payload["Entry"]),
            sl=float(payload["SL"]),
            tp=float(payload["TP"]) if payload.get("TP") is not None else None,
            side=str(payload.get("Type", "LONG")),
            pattern=payload.get("Pattern"),
            rsi=float(payload["RSI"]) if payload.get("RSI") is not None else None,
            level=payload.get("Level"),
            stoch_confirm=(
                bool(payload["StochConfirm"])
                if payload.get("StochConfirm") is not None
                else None
            ),
        )

    def to_legacy(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "Date": self.date,
            "Entry": self.entry,
            "SL": self.sl,
            "Type": self.side,
        }
        if self.tp is not None:
            payload["TP"] = self.tp
        if self.pattern is not None:
            payload["Pattern"] = self.pattern
        if self.rsi is not None:
            payload["RSI"] = self.rsi
        if self.level is not None:
            payload["Level"] = self.level
        if self.stoch_confirm is not None:
            payload["StochConfirm"] = self.stoch_confirm
        return payload


@dataclass(frozen=True)
class StrategyExit:
    date: pd.Timestamp
    price: float
    reason: str

    @staticmethod
    def from_legacy(payload: dict[str, Any]) -> "StrategyExit":
        return StrategyExit(
            date=pd.Timestamp(payload["Date"]),
            price=float(payload["Price"]),
            reason=str(payload["Reason"]),
        )

    def to_legacy(self) -> dict[str, Any]:
        return {"Date": self.date, "Price": self.price, "Reason": self.reason}


@dataclass(frozen=True)
class PositionSizeResult:
    max_loss: float
    sl_distance: float
    shares: int
    investment: float
    risk_pct_actual: float
    warning: bool

    @staticmethod
    def from_legacy(payload: dict[str, Any]) -> "PositionSizeResult":
        return PositionSizeResult(
            max_loss=float(payload["max_loss"]),
            sl_distance=float(payload["sl_distance"]),
            shares=int(payload["shares"]),
            investment=float(payload["investment"]),
            risk_pct_actual=float(payload["risk_pct_actual"]),
            warning=bool(payload["warning"]),
        )
