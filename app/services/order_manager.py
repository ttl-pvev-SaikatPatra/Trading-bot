# File: app/services/order_manager.py

import logging
from datetime import datetime
from sqlalchemy.orm import Session
from app.core.config import settings
from app.services.kite_client import KiteClient
from app.services.strategy_engine import StrategyEngine
from app.services.universe_builder import UniverseBuilder
from app.db.models import SystemState, TradeLog

log = logging.getLogger(__name__)

class OrderManager:
    def __init__(self, db: Session):
        self.db = db
        self.kite = KiteClient(db)
        self.universe_builder = UniverseBuilder(db)
        self.state = self._load_state()
        log.info(
            "OrderManager initialized.",
            extra={"initial_state": self.state}
        )

    def _load_state(self):
        defaults = {
            "trading_enabled": False,
            "dry_run_mode": True,
            "daily_pnl": 0.0,
            "last_daily_cron": None,
            "last_open_cron": None,
            "last_close_cron": None,
        }
        state_db = self.db.query(SystemState).filter(SystemState.key == "main_state").first()
        if state_db:
            # NEW: Log the loaded state from DB
            log.info("Loading state from database.", extra={"db_state": state_db.value})
            defaults.update(state_db.value)
        else:
            # NEW: Log that no state was found and defaults are being used
            log.info("No state found in database, using default values.")
        return defaults

    def _save_state(self):
        state_db = self.db.query(SystemState).filter(SystemState.key == "main_state").first()
        if not state_db:
            state_db = SystemState(key="main_state", value=self.state)
            self.db.add(state_db)
        else:
            state_db.value = self.state
        
        # NEW: Log the state being saved
        log.info("Saving new state to database.", extra={"new_state": self.state})
        self.db.commit()

    def run_strategy_cycle(self):
        log.info(
            "Starting strategy cycle.", 
            extra={"trading_enabled": self.state.get('trading_enabled'), "dry_run": self.state.get('dry_run_mode')}
        )

        if not self.state.get('trading_enabled'):
            log.warning("Trading is disabled. Skipping cycle.")
            return

        if not self.kite.is_connected():
            log.error("Kite is not connected. Skipping cycle.")
            return

        # Check daily loss limit
        total_capital = self.get_total_capital()
        daily_loss_limit_amount = -(settings.DAILY_LOSS_LIMIT_PCT * total_capital)
        current_pnl = self.state.get('daily_pnl', 0.0)

        if current_pnl <= daily_loss_limit_amount:
            log.critical(
                f"Daily loss limit reached. P&L: {current_pnl:.2f}, Limit: {daily_loss_limit_amount:.2f}. Stopping trading.",
                extra={"pnl": current_pnl, "limit": daily_loss_limit_amount}
            )
            self.toggle_trading(False)
            return

        universe = self.universe_builder.get_universe()
        positions_response = self.kite.get_positions()
        if not positions_response:
             log.error("Failed to get positions from broker. Skipping cycle.")
             return

        positions = positions_response.get('net', [])
        open_positions = {p['tradingsymbol'] for p in positions if p.get('product') == 'MIS' and p.get('quantity', 0) != 0}
        
        # NEW: Added more context to logs
        log.info(f"Scanning universe of {len(universe)} stocks.", extra={"universe_size": len(universe)})
        log.info(f"Found {len(open_positions)} open MIS positions.", extra={"open_positions": list(open_positions)})

        if len(open_positions) >= settings.MAX_CONCURRENT_POSITIONS:
            log.warning(
                f"Max concurrent positions ({settings.MAX_CONCURRENT_POSITIONS}) reached. No new trades will be placed.",
                extra={"open_positions_count": len(open_positions), "max_positions": settings.MAX_CONCURRENT_POSITIONS}
            )
            # NEW: We can still check for trailing stop loss on open positions here
            # self.manage_trailing_stops(open_positions) # Example for future implementation
            return

        for symbol in universe:
            if symbol in open_positions:
                # NEW: Explicitly log skipping symbols with existing positions
                log.debug(f"Skipping {symbol} as a position is already open.", extra={"symbol": symbol})
                continue

            engine = StrategyEngine(symbol, settings.TRADING_INTERVAL)
            if not engine.fetch_data():
                log.warning(f"Could not fetch market data for {symbol}.", extra={"symbol": symbol})
                continue
            
            engine.calculate_indicators()
            signal, last_candle = engine.check_signal()

            if signal:
                self.execute_trade(symbol, signal, last_candle)
                # Refresh open positions count before checking the limit again
                open_positions_count = len(self.kite.get_positions().get('net', []))
                if open_positions_count >= settings.MAX_CONCURRENT_POSITIONS:
                    log.info("Max positions reached after placing trade. Ending cycle.", extra={"open_positions_count": open_positions_count})
                    break
    
    def get_total_capital(self):
        margins = self.kite.get_margins()
        if margins:
            capital = margins.get('equity', {}).get('available', {}).get('opening_balance', 100000)
            log.info(f"Total capital fetched from broker: {capital:.2f}", extra={"capital": capital})
            return capital
        
        log.error("Failed to fetch margins from broker. Using default capital.", extra={"default_capital": 100000})
        return 100000 # Default if API fails

    def execute_trade(self, symbol: str, signal: str, candle):
        ## ENHANCED LOGGING - Step-by-step audit trail for position sizing ##
        log.info(f"Signal '{signal}' found for {symbol}. Evaluating for trade execution.", extra={"symbol": symbol, "signal": signal})
        
        total_capital = self.get_total_capital()
        risk_per_trade_amount = total_capital * settings.RISK_PCT_PER_TRADE
        
        stop_loss_points = candle['ATRr_14'] * settings.ATR_MULTIPLIER_SL
        
        if signal == 'BUY':
            stop_loss_price = candle['close'] - stop_loss_points
        else: # SELL
            stop_loss_price = candle['close'] + stop_loss_points
        
        risk_per_share = abs(candle['close'] - stop_loss_price)

        log.info(
            "Calculating position size.",
            extra={
                "symbol": symbol,
                "total_capital": total_capital,
                "risk_pct": settings.RISK_PCT_PER_TRADE,
                "risk_per_trade_amount": risk_per_trade_amount,
                "entry_price": candle['close'],
                "atr": candle['ATRr_14'],
                "atr_multiplier": settings.ATR_MULTIPLIER_SL,
                "stop_loss_price": stop_loss_price,
                "risk_per_share": risk_per_share
            }
        )

        if risk_per_share == 0:
            log.warning(f"Risk per share is zero for {symbol}, cannot calculate quantity. Skipping trade.", extra={"symbol": symbol})
            return

        quantity = int(risk_per_trade_amount / risk_per_share)
        if quantity == 0:
            log.warning(f"Calculated quantity is 0 for {symbol}. Trade value might be too high for risk settings. Skipping trade.", extra={"symbol": symbol})
            return

        log.info(
            f"Executing {signal} trade for {quantity} shares of {symbol}.",
