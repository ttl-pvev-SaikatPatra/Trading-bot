# File: app/services/order_manager.py

import logging
import time
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.config import settings
from app.services.kite_client import KiteClient
from app.services.strategy_engine import StrategyEngine
# We'll assume a new service for concurrent data fetching
from app.services.data_provider import MarketDataProvider
from app.db.models import SystemState, TradeLog

log = logging.getLogger(__name__)

class OrderManager:
    def __init__(self, db: Session):
        self.db = db
        self.kite = KiteClient(db)
        # The data provider is now a separate component
        self.data_provider = MarketDataProvider()
        self.state = self._load_state()

    # ... (the rest of your __init__, _load_state, _save_state methods remain the same) ...

    def run_strategy_cycle(self):
        """
        Runs the main trading strategy cycle with performance optimizations and detailed logging.
        """
        log.info("--- Starting new strategy cycle ---")

        # --- 1. Pre-flight Checks ---
        if not self.state.get('trading_enabled'):
            log.warning("Trading is disabled by user. Skipping cycle.")
            return {"status": "Cycle skipped: trading disabled."}

        if not self.kite.is_connected():
            log.error("Kite is not connected. Skipping cycle.")
            return {"status": "Cycle skipped: Kite disconnected."}

        # Check daily loss limit (assuming get_total_capital is efficient)
        capital = self.get_total_capital()
        if self.state['daily_pnl'] <= -(settings.DAILY_LOSS_LIMIT_PCT * capital):
            log.critical(f"Daily loss limit reached. Disabling trading for the day.")
            self.toggle_trading(False)
            return {"status": "Cycle stopped: daily loss limit hit."}

        # --- 2. Load Universe and Positions ---
        universe = self.get_universe() # Renamed for clarity
        if not universe:
            log.warning("Universe is empty. No stocks to scan. Skipping cycle.")
            return {"status": "Cycle skipped: empty universe."}
        log.info(f"Loaded {len(universe)} symbols for strategy scan.")

        positions = self.kite.get_positions().get('net', [])
        open_positions = {p['tradingsymbol'] for p in positions if p['product'] == 'MIS' and p['quantity'] != 0}
        
        # --- 3. Fetch Market Data Concurrently ---
        log.info(f"Fetching market data for {len(universe)} symbols concurrently...")
        start_time = time.perf_counter()
        
        # This function should use asyncio or threading to fetch all data in parallel
        market_data = self.data_provider.fetch_for_universe(universe)
        
        duration = time.perf_counter() - start_time
        log.info(f"Data fetch complete in {duration:.2f}s.")
        
        # --- 4. Process Symbols and Manage Trades ---
        signals_found = 0
        positions_managed = 0
        
        # Create a combined set of symbols to process (universe + open positions)
        symbols_to_process = set(universe) | open_positions

        for symbol in symbols_to_process:
            if symbol not in market_data:
                log.warning(f"[{symbol}] No market data received. Skipping processing.")
                continue

            candles = market_data[symbol]
            engine = StrategyEngine(symbol, settings.TRADING_INTERVAL)
            engine.calculate_indicators(candles) # Pass data to the engine

            # A. Manage existing open positions
            if symbol in open_positions:
                positions_managed += 1
                # action_taken = engine.manage_trailing_stop(candles)
                # if action_taken:
                #     log.info(f"[{symbol}] Managed open position: {action_taken['summary']}")
                log.debug(f"[{symbol}] Already in position, skipping new entry check.")
                continue

            # B. Check for new trade entries
            if len(open_positions) + signals_found >= settings.MAX_CONCURRENT_POSITIONS:
                continue # Skip new signals if we are already at max capacity

            signal, last_candle = engine.check_signal()
            if signal:
                signals_found += 1
                self.execute_trade(symbol, signal, last_candle)

        # --- 5. Final Summary ---
        log.info(
            f"--- Strategy cycle finished. Found {signals_found} new signals, "
            f"managed {positions_managed} open positions. ---"
        )
        return {"status": "Strategy cycle executed."}
    
    def get_universe(self):
        # This can be a simple wrapper around your UniverseBuilder logic
        from app.services.universe_builder import UniverseBuilder
        return UniverseBuilder(self.db).get_universe()
        
    # ... (execute_trade, get_total_capital, and other methods remain the same) ...
    # ... Make sure they also have good logging as you've already done ...
