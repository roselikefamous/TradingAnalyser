from dataclasses import dataclass
from typing import Any, Optional


class ScannerDirection:
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


SCANNER_SCHEMA_COLUMNS = [
    "Symbol",
    "Name",
    "Score",
    "Direction",
    "DirectionKey",
    "Kurs",
    "Entry",
    "SL",
    "TP",
    "1T %",
    "5T %",
    "RSI",
    "ADX",
    "Pattern",
    "Signale",
    "EMA_Layers",
]

SCANNER_DISPLAY_COLUMNS = [
    "Symbol",
    "Name",
    "Score",
    "Direction",
    "Kurs",
    "1T %",
    "5T %",
    "RSI",
    "ADX",
    "Pattern",
    "Signale",
]


@dataclass(frozen=True)
class ScannerSignal:
    symbol: str
    name: str
    score: float
    direction: str
    direction_key: str
    price: float
    entry: float
    sl: float
    tp: float
    perf_1d_pct: float
    perf_5d_pct: float
    rsi: Optional[float]
    adx: Optional[float]
    pattern: str
    signal_notes: str
    ema_layers: int

    @staticmethod
    def from_legacy(payload: dict[str, Any]) -> "ScannerSignal":
        return ScannerSignal(
            symbol=str(payload.get("Symbol", "")),
            name=str(payload.get("Name", "")),
            score=float(payload.get("Score", 0.0)),
            direction=str(payload.get("Direction", "🟡 NEUTRAL")),
            direction_key=str(payload.get("DirectionKey", ScannerDirection.NEUTRAL)),
            price=float(payload.get("Kurs", 0.0)),
            entry=float(payload.get("Entry", payload.get("Kurs", 0.0))),
            sl=float(payload.get("SL", 0.0)),
            tp=float(payload.get("TP", 0.0)),
            perf_1d_pct=float(payload.get("1T %", 0.0)),
            perf_5d_pct=float(payload.get("5T %", 0.0)),
            rsi=float(payload["RSI"]) if payload.get("RSI") is not None else None,
            adx=float(payload["ADX"]) if payload.get("ADX") is not None else None,
            pattern=str(payload.get("Pattern", "")),
            signal_notes=str(payload.get("Signale", "")),
            ema_layers=int(payload.get("EMA_Layers", 0)),
        )

    def to_legacy(self) -> dict[str, Any]:
        return {
            "Symbol": self.symbol,
            "Name": self.name,
            "Score": self.score,
            "Direction": self.direction,
            "DirectionKey": self.direction_key,
            "Kurs": self.price,
            "Entry": self.entry,
            "SL": self.sl,
            "TP": self.tp,
            "1T %": self.perf_1d_pct,
            "5T %": self.perf_5d_pct,
            "RSI": self.rsi,
            "ADX": self.adx,
            "Pattern": self.pattern,
            "Signale": self.signal_notes,
            "EMA_Layers": self.ema_layers,
        }
