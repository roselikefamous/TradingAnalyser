import json

def parse_book_to_rules(text_input: str) -> dict:
    """
    Simulates a RAG (Retrieval-Augmented Generation) pipeline extracting
    trading rules from natural language text using an LLM.
    
    In a full production environment, this would call OpenAI/Anthropic API 
    with a strict JSON schema prompt to extract the exact indicators,
    operators, and values.
    """
    
    text_lower = text_input.lower()
    
    # Mocking a few standard pattern extractions based on keywords
    if "bollinger" in text_lower and "ausbruch" in text_lower:
        return {
            "description": "Bollinger Band Breakout",
            "conditions": [
                {"indicator": "Close", "operator": ">", "value": "BB_Upper"}
            ],
            "action": "BUY"
        }
        
    if "rsi" in text_lower and "unter 30" in text_lower:
        return {
            "description": "RSI Oversold Reversal",
            "conditions": [
                {"indicator": "RSI_14", "operator": "<", "value": 30}
            ],
            "action": "BUY"
        }
        
    if "macd" in text_lower and "kreuzt" in text_lower:
        return {
            "description": "MACD Crossover",
            "conditions": [
                {"indicator": "MACD", "operator": ">", "value": "MACD_Signal"}
            ],
            "action": "BUY"
        }
        
    # Default fallback if LLM cannot parse
    return {
        "description": "Unknown Strategy (LLM Parse Failed)",
        "conditions": [],
        "action": "NEUTRAL"
    }

def generate_mock_prompt(book_title: str, text_excerpt: str) -> str:
    """ Generates the prompt that would be sent to an LLM. """
    prompt = f"""
    You are an expert quantitative trading engineer. 
    Analyze the following excerpt from the trading book '{book_title}'.
    Extract the precise technical indicator trading strategy described.
    
    Excerpt:
    "{text_excerpt}"
    
    Output strictly in the following JSON format:
    {{
        "description": "Short description of the strategy",
        "conditions": [
            {{"indicator": "IndicatorName", "operator": "<|>|==", "value": "NumberOrOtherIndicator"}}
        ],
        "action": "BUY|SELL"
    }}
    """
    return prompt
