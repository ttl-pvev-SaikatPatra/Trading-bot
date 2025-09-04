# File: app/services/market_data_provider.py

import logging
import pandas as pd
import yfinance as yf
from functools import lru_cache
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

class MarketDataProvider:
    @staticmethod
    @lru_cache(maxsize=128)
    def get_candles(symbol: str, interval: str, lookback_days: int = 5):
        """
        Fetches historical candle data from Yahoo Finance.
        Includes caching to avoid repeated calls for the same data.
        """
        try:
            # yfinance uses '.NS' for NSE stocks
            yf_symbol = f"{symbol.upper()}.NS"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            log.info(f"Fetching {interval} data for {yf_symbol} from {start_date} to {end_date}")

            data = yf.download(
                tickers=yf_symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                log.warning(f"No data received from yfinance for {yf_symbol}")
                return pd.DataFrame()

            # Ensure columns are lowercase for consistency
            data.columns = [col.lower() for col in data.columns]
            return data
        except Exception as e:
            log.error(f"Error fetching data for {symbol} from yfinance: {e}")
            return pd.DataFrame()
