# File: app/services/strategy_engine.py

import logging
import pandas as pd
import pandas_ta as ta
from app.services.market_data_provider import MarketDataProvider

log = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol
        self.interval = interval
        self.data = pd.DataFrame()

    def fetch_data(self):
        self.data = MarketDataProvider.get_candles(self.symbol, self.interval)
        if self.data.empty:
            return False
        return True

    def calculate_indicators(self):
        if self.data.empty:
            return
        
        # EMA
        self.data.ta.ema(length=10, append=True)
        self.data.ta.ema(length=20, append=True)
        # RSI
        self.data.ta.rsi(length=14, append=True)
        # ATR
        self.data.ta.atr(length=14, append=True)
        # VWAP (requires volume)
        if 'volume' in self.data.columns:
            self.data.ta.vwap(append=True)
        else:
            self.data['VWAP'] = pd.NA # Handle cases where volume is not available
        
        self.data.dropna(inplace=True)

    def check_signal(self):
        """
        Checks for a trading signal on the latest candle.
        Returns: 'BUY', 'SELL', or None
        """
        if len(self.data) < 2:
            return None, None # Not enough data

        last_candle = self.data.iloc[-1]
        prev_candle = self.data.iloc[-2]

        signal = None
        stop_loss_price = None

        # Long Entry: 10 EMA crosses above 20 EMA, RSI > 50, Price > VWAP
        if (prev_candle['EMA_10'] < prev_candle['EMA_20'] and
            last_candle['EMA_10'] > last_candle['EMA_20'] and
            last_candle['RSI_14'] > 55 and
            last_candle['close'] > last_candle['VWAP_D']):
            signal = 'BUY'

        # Short Entry: 10 EMA crosses below 20 EMA, RSI < 50, Price < VWAP
        elif (prev_candle['EMA_10'] > prev_candle['EMA_20'] and
              last_candle['EMA_10'] < last_candle['EMA_20'] and
              last_candle['RSI_14'] < 45 and
              last_candle['close'] < last_candle['VWAP_D']):
            signal = 'SELL'
        
        if signal:
            log.info(f"Signal found for {self.symbol}: {signal} at price {last_candle['close']:.2f}")

        return signal, last_candle
