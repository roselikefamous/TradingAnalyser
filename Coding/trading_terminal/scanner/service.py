from typing import Optional

import pandas as pd

from scanner import scan_all_assets, score_asset
from trading_terminal.scanner.contracts import SCANNER_SCHEMA_COLUMNS, ScannerSignal


def score_asset_contract(
    symbol: str, name: str, strategies: list
) -> Optional[ScannerSignal]:
    result = score_asset(symbol, name, strategies)
    if not result:
        return None
    return ScannerSignal.from_legacy(result)


def scan_all_assets_df(assets: dict = None, progress_callback=None) -> pd.DataFrame:
    df = scan_all_assets(assets=assets, progress_callback=progress_callback)
    if df is None or df.empty:
        return pd.DataFrame()
    # Standardize and normalize schema through contract conversion.
    normalized = [ScannerSignal.from_legacy(row.to_dict()).to_legacy() for _, row in df.iterrows()]
    out = pd.DataFrame(normalized)
    if out.empty:
        return out
    for col in SCANNER_SCHEMA_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[SCANNER_SCHEMA_COLUMNS]
    out = out.sort_values("Score", ascending=False).reset_index(drop=True)
    out.index = out.index + 1
    return out
