import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

def check_feed_quality(df: pd.DataFrame, max_gap_pct: float = 0.15) -> Tuple[bool, List[str]]:
    """
    Analyzes a price dataframe for structural anomalies.
    Returns (is_clean, list_of_issues)
    """
    if df.empty:
        return False, ["DataFrame is empty"]
        
    issues = []
    
    # 1. Missing Columns
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        issues.append(f"Missing required columns: {missing}")
        return False, issues

    # 2. NaN Checks
    nan_counts = df[list(required_cols)].isna().sum()
    if nan_counts.any():
        for col, count in nan_counts.items():
            if count > 0:
                issues.append(f"Column '{col}' has {count} NaN values.")

    # 3. Structural Price Errors
    # High must be >= Low, High >= Open, High >= Close
    # Low must be <= Open, Low <= Close
    invalid_high_low = df[(df['High'] < df['Low']) | 
                          (df['High'] < df['Open']) | 
                          (df['High'] < df['Close']) |
                          (df['Low'] > df['Open']) | 
                          (df['Low'] > df['Close'])]
    
    if not invalid_high_low.empty:
        count = len(invalid_high_low)
        issues.append(f"Found {count} rows with structural price errors (e.g. Low > High).")

    # 4. Extreme Gap Detection (Potential Split / Data error)
    # Compare Open of D(i) with Close of D(i-1)
    if len(df) > 1:
        prev_close = df['Close'].shift(1)
        current_open = df['Open']
        
        # Calculate gap percentage
        gap_pct = abs(current_open - prev_close) / prev_close
        extreme_gaps = df[gap_pct > max_gap_pct]
        
        if not extreme_gaps.empty:
            count = len(extreme_gaps)
            issues.append(f"Found {count} extreme gaps exceeding {max_gap_pct*100:.1f}% overnight.")

    # 5. Zero Volume anomalies
    zero_vol = df[df['Volume'] == 0]
    if len(zero_vol) > (len(df) * 0.1): # If more than 10% of data has 0 volume
        issues.append(f"Suspiciously high amount of zero-volume days ({len(zero_vol)} rows).")

    is_clean = len(issues) == 0
    return is_clean, issues
