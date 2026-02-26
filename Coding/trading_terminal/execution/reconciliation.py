from typing import List, Dict
import pandas as pd
from dataclasses import dataclass

from trading_terminal.contracts import Fill

@dataclass
class ReconcileResult:
    matched_fills: int
    unmatched_expected: int
    unmatched_actual: int
    avg_slippage_error_pct: float
    issues: List[str]


def reconcile_fills(expected_fills: List[Fill], actual_fills: List[Fill]) -> ReconcileResult:
    """Vergleicht vom Backtester erwartete Fills mit tatsächlichen Broker-Fills."""
    issues = []
    
    # Very naïve matching logic (by combination of side, quantity and approx date).
    # In a real environment, you'd match by ClientOrderId mapped via the broker API.
    matched = 0
    total_slip_error = 0.0
    
    expected_pool = expected_fills.copy()
    actual_pool = actual_fills.copy()
    
    for exp in expected_pool[:]:
        best_match = None
        best_time_diff = float("inf")
        
        for act in actual_pool:
            if exp.symbol == act.symbol and exp.side == act.side and exp.quantity == act.quantity:
                # Time diff in seconds
                t_diff = abs((exp.date - act.date).total_seconds())
                if t_diff < best_time_diff:
                    best_match = act
                    best_time_diff = t_diff
                    
        if best_match and best_time_diff < 3600: # Found within 1 hour
            matched += 1
            actual_pool.remove(best_match)
            expected_pool.remove(exp)
            
            # Slippage error 
            diff_pct = abs(exp.price - best_match.price) / exp.price
            total_slip_error += diff_pct
            
            if diff_pct > 0.02: # 2% mismatch
                issues.append(f"Severe Price Mismatch: Exp {exp.price:.2f} vs Actual {best_match.price:.2f} ({(diff_pct*100):.2f}%)")
                
    unmatched_exp_count = len(expected_pool)
    unmatched_act_count = len(actual_pool)
    
    if unmatched_exp_count > 0:
        issues.append(f"Missing {unmatched_exp_count} executions that were expected by strategy.")
    if unmatched_act_count > 0:
        issues.append(f"Found {unmatched_act_count} orphan executions in broker not matched to strategy.")
        
    avg_error = (total_slip_error / matched) * 100 if matched > 0 else 0.0
    
    return ReconcileResult(
        matched_fills=matched,
        unmatched_expected=unmatched_exp_count,
        unmatched_actual=unmatched_act_count,
        avg_slippage_error_pct=avg_error,
        issues=issues
    )
