# File: app/api/market_data.py (IN YOUR BACKEND PROJECT)

import logging
from fastapi import APIRouter, Query
from typing import List
from app.services.market_data_provider import MarketDataProvider

router = APIRouter()
log = logging.getLogger(__name__)

@router.get("/market-data", tags=["Market Data"])
def get_batch_market_data(symbols: List[str] = Query(...)):
    """
    Fetches the last known price for a list of symbols.
    NOTE: This is NOT real-time streaming data.
    """
    results = {}
    for symbol in symbols:
        try:
            # Fetch 1 day of 1-minute data to get the latest price
            df = MarketDataProvider.get_candles(symbol, interval="1m", lookback_days=1)
            if not df.empty:
                last_price = df.iloc[-1]['close']
                prev_close = df.iloc[-2]['close'] if len(df) > 1 else last_price
                change = last_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                results[symbol] = {
                    "last_price": last_price,
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2)
                }
            else:
                results[symbol] = None
        except Exception as e:
            log.error(f"Could not fetch market data for {symbol}: {e}")
            results[symbol] = None
    
    return results
