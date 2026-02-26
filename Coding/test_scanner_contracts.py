import pandas as pd

from trading_terminal.scanner.contracts import (
    SCANNER_SCHEMA_COLUMNS,
    ScannerDirection,
    ScannerSignal,
)
from trading_terminal.scanner.service import scan_all_assets_df, score_asset_contract


def _legacy_signal_payload():
    return {
        "Symbol": "AAPL",
        "Name": "Apple Inc.",
        "Score": 77.5,
        "Direction": "🟢 KAUFEN",
        "DirectionKey": "BUY",
        "Kurs": 201.25,
        "Entry": 201.25,
        "SL": 196.0,
        "TP": 210.5,
        "1T %": 0.8,
        "5T %": 2.6,
        "RSI": 58.2,
        "ADX": 24.1,
        "Pattern": "Trend: BULL",
        "Signale": "✅ MTF Confluence",
        "EMA_Layers": 0,
    }


def test_scanner_signal_roundtrip():
    payload = _legacy_signal_payload()
    signal = ScannerSignal.from_legacy(payload)
    assert signal.direction_key == ScannerDirection.BUY
    assert signal.symbol == "AAPL"
    assert signal.signal_notes == "✅ MTF Confluence"
    assert signal.to_legacy() == payload


def test_score_asset_contract_maps_from_legacy(monkeypatch):
    payload = _legacy_signal_payload()

    def fake_score_asset(symbol, name, strategies):
        return payload

    monkeypatch.setattr("trading_terminal.scanner.service.score_asset", fake_score_asset)
    out = score_asset_contract("AAPL", "Apple Inc.", [])
    assert out is not None
    assert out.symbol == "AAPL"
    assert out.score == 77.5


def test_scan_all_assets_df_normalizes_schema(monkeypatch):
    raw_df = pd.DataFrame(
        [
            {"Symbol": "MSFT", "Score": 60.0, "DirectionKey": "BUY", "Kurs": 100.0},
            {"Symbol": "AAPL", "Score": 80.0, "DirectionKey": "SELL", "Kurs": 90.0},
        ]
    )

    def fake_scan_all_assets(assets=None, progress_callback=None):
        return raw_df

    monkeypatch.setattr(
        "trading_terminal.scanner.service.scan_all_assets", fake_scan_all_assets
    )
    out = scan_all_assets_df()
    assert list(out["Symbol"]) == ["AAPL", "MSFT"]
    assert list(out.columns) == SCANNER_SCHEMA_COLUMNS
