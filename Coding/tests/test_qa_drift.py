import pytest
import pandas as pd
import numpy as np

from trading_terminal.data_qa.feed_monitor import check_feed_quality
from trading_terminal.mlops.drift import FeatureDriftDetector, calculate_volatility_regime_drift

def test_feed_quality_clean():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "Open": [100, 102, 101, 105, 108],
        "High": [105, 104, 106, 109, 110],
        "Low": [98, 100, 99, 102, 105],
        "Close": [102, 101, 105, 108, 107],
        "Volume": [1000] * 5
    }, index=dates)
    
    is_clean, issues = check_feed_quality(df)
    assert is_clean
    assert len(issues) == 0

def test_feed_quality_anomaly():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    df = pd.DataFrame({
        "Open": [100, 100],
        "High": [105, 90], # 90 High is lower than 100 Open -> Anomaly!
        "Low": [98, 98],
        "Close": [102, 102],
        "Volume": [1000, 0] # 50% 0 volume
    }, index=dates)
    
    is_clean, issues = check_feed_quality(df)
    assert not is_clean
    assert len(issues) >= 2 # Should find Structural Price Error + Zero Vol Anomaly
    
def test_feed_quality_gap():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    df = pd.DataFrame({
        "Open": [100, 200], # 100% GAP!
        "High": [105, 205], 
        "Low": [98, 195],
        "Close": [102, 202],
        "Volume": [1000, 1000] 
    }, index=dates)
    
    is_clean, issues = check_feed_quality(df, max_gap_pct=0.2)
    assert not is_clean
    assert any("extreme gaps" in i.lower() for i in issues)

def test_drift_detector_volatility():
    # Train is low volatility 
    dates_train = pd.date_range("2022-01-01", periods=100, freq="D")
    # Returns tightly clustered around 0
    df_train = pd.DataFrame({
        "Close": np.linspace(100, 105, 100)
    }, index=dates_train)
    
    # Live is extremely high volatility
    dates_live = pd.date_range("2023-01-01", periods=10, freq="D")
    df_live = pd.DataFrame({
        "Close": [100, 110, 90, 120, 80, 130, 70, 140, 60, 150]
    }, index=dates_live)
    
    report = calculate_volatility_regime_drift(df_train, df_live)
    assert report.is_drifting
    assert report.current_mean > report.reference_mean * 2

def test_feature_drift_detector():
    dates_train = pd.date_range("2022-01-01", periods=100, freq="D")
    df_train = pd.DataFrame({
        "Volume": np.random.normal(1000, 100, 100)
    }, index=dates_train)
    
    detector = FeatureDriftDetector(z_score_threshold=3.0)
    detector.fit_reference(df_train, feature_cols=["Volume"])
    
    # Very anamolous new data (mean 5000, instead of 1000)
    dates_live = pd.date_range("2023-01-01", periods=10, freq="D")
    df_live = pd.DataFrame({
        "Volume": np.random.normal(5000, 10, 10)
    }, index=dates_live)
    
    reports = detector.detect_drift(df_live)
    assert len(reports) == 1
    assert reports[0].is_drifting
