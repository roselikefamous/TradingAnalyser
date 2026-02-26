import json
import ast
import pandas as pd
import numpy as np

def validate_condition(condition: dict, df: pd.DataFrame, current_idx: int = -1) -> bool:
    \"\"\"
    Evaluates a single JSON condition against the DataFrame at a specific index.
    Condition format: {"indicator": "RSI_14", "operator": "<", "value": 30}
    \"\"\"
    try:
        ind = condition.get("indicator")
        op = condition.get("operator")
        val = condition.get("value")
        
        # Ensure the indicator exists in the DataFrame
        if ind not in df.columns and ind != "Close":
            return False
            
        current_value = df[ind].iloc[current_idx]
        
        # If the target value is another indicator (e.g. "EMA_21")
        if isinstance(val, str) and val in df.columns:
            target_value = df[val].iloc[current_idx]
        else:
            try:
                target_value = float(val)
            except (ValueError, TypeError):
                return False # Cannot parse target value

        # Evaluate operator
        if op == ">": return float(current_value) > target_value
        elif op == "<": return float(current_value) < target_value
        elif op == ">=": return float(current_value) >= target_value
        elif op == "<=": return float(current_value) <= target_value
        elif op == "==": return float(current_value) == target_value
        elif op == "crosses_above":
            # Rough cross over logic
            if current_idx - 1 < -len(df): return False
            prev_value = df[ind].iloc[current_idx - 1]
            if isinstance(val, str) and val in df.columns:
                prev_target = df[val].iloc[current_idx - 1]
            else:
                prev_target = target_value
            return prev_value <= prev_target and float(current_value) > target_value
            
        elif op == "crosses_below":
            # Rough cross under logic
            if current_idx - 1 < -len(df): return False
            prev_value = df[ind].iloc[current_idx - 1]
            if isinstance(val, str) and val in df.columns:
                prev_target = df[val].iloc[current_idx - 1]
            else:
                prev_target = target_value
            return prev_value >= prev_target and float(current_value) < target_value
            
        return False
        
    except Exception as e:
        print(f"Error validating condition {condition}: {e}")
        return False

def run_strategies(df: pd.DataFrame, ai_strategies: list) -> dict:
    \"\"\"
    Applies the active database rules across the input technical DataFrame.
    Returns a summarized Confluence Result.
    \"\"\"
    if df is None or len(df) < 2:
        return {"action": "NEUTRAL", "score": 0, "matching_books": []}
        
    total_buy_weight = 0.0
    total_sell_weight = 0.0
    matching_buy_strats = []
    matching_sell_strats = []
    
    for strat in ai_strategies:
        try:
            # Parse logic JSON
            logic_str = strat["logic_json"]
            if isinstance(logic_str, str):
                logic = json.loads(logic_str)
            else:
                logic = logic_str # already parsed dict
                
            conditions = logic.get("conditions", [])
            action = logic.get("action", "NEUTRAL")
            weight = float(strat.get("base_weight", 1.0))
            
            # Check ALL conditions (AND logic)
            all_passed = True
            for cond in conditions:
                if not validate_condition(cond, df):
                    all_passed = False
                    break
                    
            if all_passed:
                info_tuple = (strat.get("name", "Unknown"), strat.get("source_book", "Unknown"))
                if action == "BUY":
                    total_buy_weight += weight
                    matching_buy_strats.append(info_tuple)
                elif action == "SELL":
                    total_sell_weight += weight
                    matching_sell_strats.append(info_tuple)
                    
        except Exception as e:
            print(f"Error executing strategy {strat.get('name')}: {e}")

    # Calculate final Confluence Score (0-100%)
    # Very crude score: sum of matching weights against a typical max limit (e.g. 5)
    max_expected_weight = 5.0
    
    if total_buy_weight > total_sell_weight:
        raw_score = total_buy_weight / max_expected_weight * 100
        score = min(max(raw_score, 0), 100)
        return {
            "action": "BUY",
            "score": score,
            "matching_books": matching_buy_strats
        }
    elif total_sell_weight > total_buy_weight:
        raw_score = total_sell_weight / max_expected_weight * 100
        score = min(max(raw_score, 0), 100)
        return {
            "action": "SELL",
            "score": score,
            "matching_books": matching_sell_strats
        }
        
    return {"action": "NEUTRAL", "score": 0, "matching_books": []}
