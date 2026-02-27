import json
import ast
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans
from trading_terminal.indicators import find_swing_points

def validate_condition(condition: dict, df: pd.DataFrame, current_idx: int = -1) -> bool:
    """
    Evaluates a single JSON condition against the DataFrame at a specific index.
    Condition format: {"indicator": "RSI_14", "operator": "<", "value": 30}
    """
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
    """
    Applies the active database rules across the input technical DataFrame.
    Returns a summarized Confluence Result.
    """
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


def calculate_smart_entry_zone(df: pd.DataFrame) -> dict:
    """Calculates all metrics and levels for the Smart Entry Zone v2."""
    last_price = df['Close'].iloc[-1]
    atr_now = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else (df['High'].iloc[-1] - df['Low'].iloc[-1])

    # K-Means S/R
    swing_highs, swing_lows, _, _ = find_swing_points(df)
    all_pivots = np.array(swing_highs + swing_lows, dtype=float)
    if len(all_pivots) >= 5:
        n_clusters = min(6, len(all_pivots))
        key_levels, _ = kmeans(all_pivots, n_clusters)
        key_levels = sorted(key_levels)
    else:
        key_levels = [df['Low'].iloc[-40:].min() if len(df) >= 40 else df['Low'].min(),
                      df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else last_price,
                      df['High'].iloc[-40:].max() if len(df) >= 40 else df['High'].max()]
        key_levels = sorted(key_levels)

    supports = sorted([l for l in key_levels if l < last_price], reverse=True)
    resistances = sorted([l for l in key_levels if l > last_price])
    nearest_support = supports[0] if supports else last_price - atr_now * 2
    nearest_resistance = resistances[0] if resistances else last_price + atr_now * 2
    second_resistance = resistances[1] if len(resistances) > 1 else nearest_resistance + atr_now * 2
    second_support = supports[1] if len(supports) > 1 else nearest_support - atr_now * 2

    # Confluence scoring
    score = 0
    max_score = 20
    signals = []

    last_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
    last_macd_hist = df['MACD_Hist'].iloc[-1] if not pd.isna(df['MACD_Hist'].iloc[-1]) else 0
    last_ema1 = df['EMA_1'].iloc[-1] if not pd.isna(df['EMA_1'].iloc[-1]) else last_price
    last_ema2 = df['EMA_2'].iloc[-1] if not pd.isna(df['EMA_2'].iloc[-1]) else last_price
    last_ema55 = df['EMA_55'].iloc[-1] if not pd.isna(df['EMA_55'].iloc[-1]) else last_price
    last_adx = df['ADX'].iloc[-1] if not pd.isna(df['ADX'].iloc[-1]) else 25
    last_plus_di = df['Plus_DI'].iloc[-1] if not pd.isna(df['Plus_DI'].iloc[-1]) else 50
    last_minus_di = df['Minus_DI'].iloc[-1] if not pd.isna(df['Minus_DI'].iloc[-1]) else 50
    last_bb_lower = df['BB_Lower'].iloc[-1] if not pd.isna(df['BB_Lower'].iloc[-1]) else last_price * 0.95
    last_bb_upper = df['BB_Upper'].iloc[-1] if not pd.isna(df['BB_Upper'].iloc[-1]) else last_price * 1.05
    last_bb_mid = df['BB_Mid'].iloc[-1] if not pd.isna(df['BB_Mid'].iloc[-1]) else last_price

    # CHECK 1: ADX
    if last_adx > 25 and last_plus_di > last_minus_di:
        score += 3; signals.append(("💪 Starker Aufwärtstrend", f"ADX={last_adx:.0f}, +DI > -DI", "bullish"))
    elif last_adx > 25 and last_minus_di > last_plus_di:
        score -= 3; signals.append(("💪 Starker Abwärtstrend", f"ADX={last_adx:.0f}, -DI > +DI", "bearish"))
    elif last_adx < 20:
        signals.append(("😴 Seitwärtsmarkt", f"ADX={last_adx:.0f} → Range", "neutral"))
    else:
        signals.append(("📊 Moderater Trend", f"ADX={last_adx:.0f}", "neutral"))

    # CHECK 2: EMA Ribbon
    if last_ema1 > last_ema2 > last_ema55:
        score += 2; signals.append(("✅ EMA perfekt gestaffelt", "Gesunder Aufwärtstrend", "bullish"))
    elif last_ema1 < last_ema2 < last_ema55:
        score -= 2; signals.append(("🔻 EMA bärisch", "Abwärtstrend intakt", "bearish"))
    elif last_ema1 > last_ema2:
        score += 1; signals.append(("🟡 EMAs gemischt", "Kurzfristig bullisch", "neutral"))
    else:
        score -= 1; signals.append(("⚠️ EMAs negativ", "Kurzfristige Schwäche", "bearish"))

    # CHECK 3: Price vs EMAs
    above_count = sum([last_price > last_ema1, last_price > last_ema2, last_price > last_ema55])
    if above_count == 3: score += 2; signals.append(("📈 Über allen EMAs", "Bullisch", "bullish"))
    elif above_count == 0: score -= 2; signals.append(("📉 Unter allen EMAs", "Bärisch", "bearish"))

    # CHECK 4: RSI + Divergence
    if len(df) >= 20:
        p_rec = df['Close'].iloc[-10:]; p_prev = df['Close'].iloc[-20:-10]
        r_rec = df['RSI'].iloc[-10:]; r_prev = df['RSI'].iloc[-20:-10]
        if p_rec.min() < p_prev.min() and r_rec.min() > r_prev.min():
            score += 3; signals.append(("🔮 Bullische RSI-Divergenz!", "Starkes Umkehrsignal", "bullish"))
        elif p_rec.max() > p_prev.max() and r_rec.max() < r_prev.max():
            score -= 3; signals.append(("🔮 Bärische RSI-Divergenz!", "Warnung vor Umkehr", "bearish"))
    if last_rsi < 30: score += 2; signals.append(("🟢 RSI überverkauft", f"RSI={last_rsi:.1f}", "bullish"))
    elif last_rsi > 70: score -= 2; signals.append(("🔴 RSI überkauft", f"RSI={last_rsi:.1f}", "bearish"))

    # CHECK 5: MACD Momentum
    hist_vals = df['MACD_Hist'].dropna()
    if len(hist_vals) >= 3:
        hist_accel = hist_vals.iloc[-1] - hist_vals.iloc[-2]
        if last_macd_hist > 0 and hist_accel > 0: score += 2; signals.append(("🚀 MACD Momentum steigend", "Kaufinteresse", "bullish"))
        elif last_macd_hist < 0 and hist_accel < 0: score -= 2; signals.append(("📉 MACD Momentum fallend", "Verkaufsdruck", "bearish"))

    # CHECK 6: Bollinger
    bb_width = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
    bb_valid = bb_width.dropna()
    if len(bb_valid) >= 20:
        if bb_valid.iloc[-1] < bb_valid.iloc[-20:].mean() * 0.7:
            score += 2; signals.append(("💥 Bollinger Squeeze!", "Explosion erwartet!", "bullish"))
        else:
            bb_pct = (last_price - last_bb_lower) / (last_bb_upper - last_bb_lower) if (last_bb_upper - last_bb_lower) > 0 else 0.5
            if bb_pct < 0.2: score += 1; signals.append(("🟢 Unteres BB", f"BB%={bb_pct:.0%}", "bullish"))
            elif bb_pct > 0.8: score -= 1; signals.append(("🔴 Oberes BB", f"BB%={bb_pct:.0%}", "bearish"))

    # CHECK 7: Volume
    if 'Volume' in df.columns and len(df) >= 20:
        vol_ratio = df['Volume'].iloc[-1] / df['Volume'].iloc[-20:].mean()
        price_up = df['Close'].iloc[-1] > df['Open'].iloc[-1]
        if vol_ratio > 1.5 and price_up: score += 2; signals.append(("📊 Hohes Vol + Anstieg", f"{vol_ratio:.1f}x", "bullish"))
        elif vol_ratio > 1.5 and not price_up: score -= 1; signals.append(("📊 Hohes Vol + Rückgang", f"{vol_ratio:.1f}x", "bearish"))

    # CHECK 8: Candlestick patterns
    if df['Hammer'].iloc[-1]: score += 1; signals.append(("🔨 Hammer", "Käufer absorbieren", "bullish"))
    elif df['Shooting_Star'].iloc[-1]: score -= 1; signals.append(("⭐ Shooting Star", "Verkäufer drücken", "bearish"))
    elif df['Bullish_Engulfing'].iloc[-1]: score += 1; signals.append(("🟢 Bullish Engulfing", "Umkehrsignal", "bullish"))
    elif df['Bearish_Engulfing'].iloc[-1]: score -= 1; signals.append(("🔴 Bearish Engulfing", "Warnung", "bearish"))

    # CHECK 9: S/R proximity
    dist_sup = abs(last_price - nearest_support) / last_price * 100
    dist_res = abs(nearest_resistance - last_price) / last_price * 100
    if dist_res > dist_sup * 2: score += 2; signals.append(("🎯 Viel Platz nach oben", f"R:{dist_res:.1f}% S:{dist_sup:.1f}%", "bullish"))
    elif dist_sup > dist_res * 2: score -= 1; signals.append(("⚠️ Wenig Platz oben", f"R:{dist_res:.1f}%", "bearish"))

    # CHECK 10: Stochastic
    stk = df['Stoch_K'].iloc[-1] if not pd.isna(df['Stoch_K'].iloc[-1]) else 50
    if stk < 20: score += 1; signals.append(("📉 Stochastik überverkauft", f"%K={stk:.0f}", "bullish"))
    elif stk > 80: score -= 1; signals.append(("📈 Stochastik überkauft", f"%K={stk:.0f}", "bearish"))

    # Decision
    regime = "TRENDING" if last_adx > 25 else "RANGING" if last_adx < 20 else "TRANSITIONING"
    is_bullish = score >= 4
    is_bearish = score <= -4

    if is_bullish:
        direction, d_emoji, d_color = "LONG", "🟢", "#00e676"
        entry_low = min(last_ema1, last_ema2) if regime == "TRENDING" else max(nearest_support, last_bb_lower)
        entry_high = last_price if regime == "TRENDING" else min(last_price, last_bb_mid)
        if entry_high <= entry_low: entry_low, entry_high = last_price - atr_now*0.5, last_price
        sl_level = min(nearest_support - atr_now*0.5, entry_low - atr_now*1.5)
        target_1, target_2 = nearest_resistance, second_resistance
        target_3 = max(target_2 + atr_now*1.5, last_price + atr_now*4)
    elif is_bearish:
        direction, d_emoji, d_color = "SHORT", "🔴", "#ff1744"
        # Sell zone: between price and nearest resistance/EMA
        entry_low = last_price if regime == "TRENDING" else max(last_price, last_bb_mid)
        entry_high = max(last_ema1, last_ema2) if regime == "TRENDING" else min(nearest_resistance, last_bb_upper)
        if entry_high <= entry_low: entry_low, entry_high = last_price, last_price + atr_now*0.5
        # Stop-Loss: above nearest resistance
        sl_level = max(nearest_resistance + atr_now*0.75, entry_high + atr_now*1.5)
        # Targets: below current price
        target_1, target_2 = nearest_support, second_support
        target_3 = min(target_2 - atr_now*1.5, last_price - atr_now*4)
    else:
        direction, d_emoji, d_color = "NEUTRAL", "🟡", "#ffea00"
        entry_low, entry_high = last_price - atr_now*0.5, last_price + atr_now*0.5
        sl_level = nearest_support - atr_now*1.5
        target_1, target_2 = nearest_resistance, second_resistance if second_resistance > nearest_resistance else nearest_resistance + atr_now*2
        target_3 = target_2 + atr_now*1.5

    score_label = "🟢 SEHR STARK" if score >= 8 else "🟢 STARK" if score >= 5 else "🟢 GUT" if score >= 4 else "🟡 NEUTRAL" if score >= 0 else "🔴 SCHWACH" if score >= -3 else "🔴 SEHR SCHWACH"
    confidence = min(100, max(0, int((score / max_score) * 100)))

    return {
        "last_price": last_price,
        "key_levels": key_levels,
        "direction": direction,
        "d_emoji": d_emoji,
        "d_color": d_color,
        "score": score,
        "max_score": max_score,
        "score_label": score_label,
        "regime": regime,
        "confidence": confidence,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "sl_level": sl_level,
        "target_1": target_1,
        "target_2": target_2,
        "target_3": target_3,
        "signals": signals,
        "last_adx": last_adx
    }

