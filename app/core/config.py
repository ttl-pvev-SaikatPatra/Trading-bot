# File: app/core/config.py

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Zerodha Credentials
    ZERODHA_API_KEY: str
    ZERODHA_API_SECRET: str
    ZERODHA_REDIRECT_URL: str

    # Application Security
    APP_SECRET_KEY: str
    CRON_SECRET_KEY: str

    # Database
    DATABASE_URL: str = "sqlite:///./trading_bot.db"
    SQLITE_DB_PATH: str = "database.db" # Add this line

    # Risk Management
    RISK_PCT_PER_TRADE: float = 0.005
    REWARD_RISK_RATIO: float = 2.0
    ATR_MULTIPLIER_SL: float = 1.5
    MAX_CONCURRENT_POSITIONS: int = 5
    DAILY_LOSS_LIMIT_PCT: float = 0.02

    # Trading Strategy
    TRADING_INTERVAL: str = "5m" # e.g., "5m", "15m"

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
