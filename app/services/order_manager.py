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
            # Merge defaults with saved state
            defaults.update(state_db.value)
        return defaults

    def _save_state(self):
        state_db = self.db.query(SystemState).filter(SystemState.key == "main_state").first()
        if not state_db:
            state_db = SystemState(key="main_state", value=self.state)
            self.db.add(state_db)
        else:
            state_db.value = self.state
        self.db.commit()

    def run_strategy_cycle(self):
        log.info("Starting strategy cycle...")
        if not self.state.get('trading_enabled'):
            log.warning("Trading is disabled. Skipping cycle.")
            return

        if not self.kite.is_connected():
            log.error("Kite is not connected. Skipping cycle.")
            return

        # Check daily loss limit
        if self.state['daily_pnl'] <= -(settings.DAILY_LOSS_LIMIT_PCT * self.get_total_capital()):
            log.critical(f"Daily loss limit of {settings.DAILY_LOSS_LIMIT_PCT:.2%} reached. Stopping trading.")
            self.toggle_trading(False)
            return

        universe = self.universe_builder.get_universe()
        positions = self.kite.get_positions().get('net', [])
        open_positions = {p['tradingsymbol'] for p in positions if p['product'] == 'MIS' and p['quantity'] != 0}
        
        if len(open_positions) >= settings.MAX_CONCURRENT_POSITIONS:
            log.info(f"Max positions ({settings.MAX_CONCURRENT_POSITIONS}) reached. No new trades will be placed.")
            return

        for symbol in universe:
            if symbol in open_positions:
                # Logic for trailing stop loss could go here
                continue

            engine = StrategyEngine(symbol, settings.TRADING_INTERVAL)
            if not engine.fetch_data():
                continue
            
            engine.calculate_indicators()
            signal, last_candle = engine.check_signal()

            if signal:
                self.execute_trade(symbol, signal, last_candle)
                # Avoid taking too many trades in one cycle
                if len(open_positions) + 1 >= settings.MAX_CONCURRENT_POSITIONS:
                    break
    
    def get_total_capital(self):
        margins = self.kite.get_margins()
        if margins:
            return margins.get('equity', {}).get('available', {}).get('opening_balance', 100000) # Default capital
        return 100000 # Default if API fails

    def execute_trade(self, symbol: str, signal: str, candle):
        total_capital = self.get_total_capital()
        risk_per_trade = total_capital * settings.RISK_PCT_PER_TRADE
        
        stop_loss_points = candle['ATRr_14'] * settings.ATR_MULTIPLIER_SL
        
        if signal == 'BUY':
            stop_loss_price = candle['close'] - stop_loss_points
        else: # SELL
            stop_loss_price = candle['close'] + stop_loss_points
        
        risk_per_share = abs(candle['close'] - stop_loss_price)
        if risk_per_share == 0:
            log.warning(f"Risk per share is zero for {symbol}, cannot calculate quantity.")
            return

        quantity = int(risk_per_trade / risk_per_share)
        if quantity == 0:
            log.warning(f"Calculated quantity is 0 for {symbol}. Skipping trade.")
            return

        log.info(f"Executing trade for {symbol}: Signal={signal}, Qty={quantity}, Entry={candle['close']:.2f}, SL={stop_loss_price:.2f}")

        if self.state.get('dry_run_mode'):
            log.info(f"[DRY RUN] Would place {signal} order for {quantity} {symbol}.")
            return

        # Place Market Order
        transaction_type = 'BUY' if signal == 'BUY' else 'SELL'
        tag = f"entry_{symbol}_{int(datetime.now().timestamp())}"
        entry_order_id = self.kite.place_mis_order(symbol, transaction_type, quantity, tag)
        
        if entry_order_id:
            # Place SL-M Order
            sl_transaction_type = 'SELL' if signal == 'BUY' else 'BUY'
            sl_tag = f"sl_{symbol}_{int(datetime.now().timestamp())}"
            # Round SL to nearest valid tick
            sl_price_rounded = self.round_to_tick(stop_loss_price)
            self.kite.place_sl_order(symbol, sl_transaction_type, quantity, sl_price_rounded, sl_tag)

    @staticmethod
    def round_to_tick(price: float, tick_size: float = 0.05) -> float:
        return round(price / tick_size) * tick_size

    def toggle_trading(self, enable: bool):
        self.state['trading_enabled'] = enable
        self._save_state()
        log.info(f"Trading has been {'ENABLED' if enable else 'DISABLED'}.")

    def toggle_dry_run(self, enable: bool):
        self.state['dry_run_mode'] = enable
        self._save_state()
        log.info(f"Dry run mode is now {'ON' if enable else 'OFF'}.")

    def update_cron_timestamp(self, cron_type: str):
        self.state[f'last_{cron_type}_cron'] = datetime.now().isoformat()
        self._save_state()

    def square_off_all_positions(self):
        log.info("Initiating end-of-day square off process.")
        if self.state.get('dry_run_mode'):
            log.info("[DRY RUN] Would square off all MIS positions and cancel pending orders.")
            return
        self.kite.square_off_all()
