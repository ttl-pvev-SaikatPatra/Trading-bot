# File: app/core/config.py

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Load environment variables from .env file
# Pydantic's SettingsConfigDict handles this, so load_dotenv() is often redundant
# but can be kept for local dev simplicity.
# from dotenv import load_dotenv
# load_dotenv()

class Settings(BaseSettings):
    """
    Application settings for the trading bot.
    Environment variables are loaded from the .env file.
    """

    # Pydantic-settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True
    )

    # Zerodha API credentials
    ZERODHA_API_KEY: str
    ZERODHA_API_SECRET: str
    ZERODHA_REDIRECT_URL: str

    # Frontend URL for CORS
    FRONTEND_URL: str = "http://localhost:3000"

    # Application Security
    APP_SECRET_KEY: str
    CRON_SECRET_KEY: str

    # Database settings
    SQLITE_DB_PATH: Path = Path("database.db")

    # Risk Management
    RISK_PCT_PER_TRADE: float = 0.005
    REWARD_RISK_RATIO: float = 2.0
    ATR_MULTIPLIER_SL: float = 1.5
    MAX_CONCURRENT_POSITIONS: int = 5
    DAILY_LOSS_LIMIT_PCT: float = 0.02

    # Trading Strategy
    TRADING_INTERVAL: str = "5m" # e.g., "5m", "15m"

settings = Settings()

# Ensure the database directory exists
# This is crucial for a smooth deployment on Render
settings.SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
