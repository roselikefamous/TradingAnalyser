import os
import streamlit as st
from typing import Any, Optional

class Settings:
    """Centralized configuration for the Trading Terminal."""
    
    # Data Provider Settings
    @property
    def data_provider(self) -> str:
        return self._get_secret_or_env("TRADING_DATA_PROVIDER", "yfinance").lower()

    # Alpaca API Credentials
    @property
    def alpaca_api_key(self) -> Optional[str]:
        return self._get_secret_or_env("ALPACA_API_KEY")

    @property
    def alpaca_secret_key(self) -> Optional[str]:
        return self._get_secret_or_env("ALPACA_SECRET_KEY")

    @property
    def alpaca_base_url(self) -> str:
        return self._get_secret_or_env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # Drift Detection Settings
    @property
    def drift_z_threshold(self) -> float:
        return float(self._get_secret_or_env("DRIFT_Z_THRESHOLD", "3.0"))

    # UI / App Settings
    @property
    def app_name(self) -> str:
        return "Pro Trading Terminal"

    def _get_secret_or_env(self, key: str, default: Any = None) -> Any:
        # Check Streamlit Secrets first
        if key in st.secrets:
            return st.secrets[key]
        # Fallback to Environment Variables
        return os.getenv(key, default)

# Global settings instance
settings = Settings()
