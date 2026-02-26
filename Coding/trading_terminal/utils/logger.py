import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure logs directory exists
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Attach any extra data passed via the `extra` kwarg
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


def setup_structured_logger(name: str = "trading_system") -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    
    formatter = JSONFormatter()
    
    # Console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File output (rotating log pattern could be implemented here)
    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

structured_logger = setup_structured_logger()

def log_trade_event(event_type: str, message: str, **kwargs: Any):
    structured_logger.info(message, extra={"extra_data": {"event_type": event_type, **kwargs}})

def log_error(event_type: str, message: str, **kwargs: Any):
    structured_logger.error(message, extra={"extra_data": {"event_type": event_type, **kwargs}})
