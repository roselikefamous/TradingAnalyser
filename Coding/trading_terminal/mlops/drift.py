import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DriftReport:
    feature_name: str
    reference_mean: float
    reference_std: float
    current_mean: float
    current_std: float
    z_score: float
    is_drifting: bool


from trading_terminal.config import settings

class FeatureDriftDetector:
    """Detects distribution drift between historical backtest data and live data."""
    
    def __init__(self, z_score_threshold: float = None):
        self.zs_threshold = z_score_threshold if z_score_threshold is not None else settings.drift_z_threshold
        # Stores reference distributions extracted from Walk-Forward
        self.reference_stats: dict[str, dict[str, float]] = {}

    def fit_reference(self, df: pd.DataFrame, feature_cols: list[str] = None):
        """Calculates mean/std of features from historical "golden" dataset."""
        if feature_cols is None:
            feature_cols = ['Close', 'Volume'] # default naive features
            
        for col in feature_cols:
            if col in df.columns:
                series = df[col].dropna()
                self.reference_stats[col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std())
                }

    def detect_drift(self, recent_df: pd.DataFrame, feature_cols: list[str] = None) -> list[DriftReport]:
        """Evaluates recent dataframe against reference statistics."""
        reports = []
        if feature_cols is None:
            feature_cols = list(self.reference_stats.keys())

        for col in feature_cols:
            if col not in self.reference_stats or col not in recent_df.columns:
                continue
                
            ref = self.reference_stats[col]
            recent_series = recent_df[col].dropna()
            
            if recent_series.empty:
                continue
                
            curr_mean = float(recent_series.mean())
            curr_std = float(recent_series.std())
            
            # Simple Z-score of the new mean against the reference distribution
            # Assuming central limit theorem for sample mean
            n = len(recent_series)
            if ref["std"] == 0 or n == 0:
                z_score = 0.0
            else:
                standard_error = ref["std"] / np.sqrt(n)
                z_score = (curr_mean - ref["mean"]) / standard_error

            is_drifting = abs(z_score) > self.zs_threshold
            
            reports.append(DriftReport(
                feature_name=col,
                reference_mean=ref["mean"],
                reference_std=ref["std"],
                current_mean=curr_mean,
                current_std=curr_std,
                z_score=float(z_score),
                is_drifting=bool(is_drifting)
            ))
            
        return reports


def calculate_volatility_regime_drift(df_train: pd.DataFrame, df_live: pd.DataFrame, return_col: str = "Close") -> DriftReport:
    """Specialized drift detector targeting solely market volatility regimes (Daily Returns StdDev)."""
    
    # Calculate daily returns
    train_rets = df_train[return_col].pct_change().dropna()
    live_rets = df_live[return_col].pct_change().dropna()
    
    train_vol = float(train_rets.std() * np.sqrt(252))
    live_vol = float(live_rets.std() * np.sqrt(252))
    
    # If the live volatility exceeds training volatility by more than 50% relative
    drift_ratio = live_vol / train_vol if train_vol > 0 else 1.0
    is_drifting = drift_ratio > 1.5 or drift_ratio < 0.5
    
    return DriftReport(
        feature_name="Annualized Volatility",
        reference_mean=train_vol,
        reference_std=0.0, # N/A here
        current_mean=live_vol,
        current_std=0.0,
        z_score=float(drift_ratio), # Repurpose Z-score field as ratio for simplicity
        is_drifting=is_drifting
    )
