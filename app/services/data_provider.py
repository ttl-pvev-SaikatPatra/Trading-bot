import logging
import yfinance as yf
import pandas as pd
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure the yfinance logger to be less verbose
logging.getLogger('yfinance').setLevel(logging.WARNING)
log = logging.getLogger(__name__)

class MarketDataProvider:
    """
    Handles fetching market data from external sources concurrently.
    """
    def __init__(self, max_workers: int = 10):
        """
        Initializes the data provider.
        Args:
            max_workers: The number of parallel threads to use for fetching data.
        """
        self.max_workers = max_workers

    def _fetch_single_symbol_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetches 5-minute historical data for a single NSE symbol.
        """
        # yfinance requires '.NS' suffix for NSE stocks
        ticker_symbol = f"{symbol}.NS"
        try:
            # Fetch last 7 days of data to ensure we have enough for indicators
            ticker = yf.Ticker(ticker_symbol)
            # '5m' interval data is available for the last 60 days
            data = ticker.history(period="7d", interval="5m", auto_adjust=True)
            
            if data.empty:
                log.debug(f"[{symbol}] No data returned from yfinance.")
                return pd.DataFrame()
                
            # Clean up columns for consistency
            data.rename(columns={
                "Open": "open", "High": "high", "Low": "low", 
                "Close": "close", "Volume": "volume"
            }, inplace=True)
            
            return data[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            log.error(f"[{symbol}] Failed to fetch data from yfinance: {e}")
            return pd.DataFrame()

    def fetch_for_universe(self, universe: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetches market data for a list of symbols concurrently using a thread pool.
        """
        market_data = {}
        # Using ThreadPoolExecutor to fetch data in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a future for each symbol fetch
            future_to_symbol = {
                executor.submit(self._fetch_single_symbol_data, symbol): symbol 
                for symbol in universe
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty:
                        market_data[symbol] = data
                except Exception as e:
                    log.error(f"[{symbol}] An exception occurred in future result: {e}")
        
        return market_data
