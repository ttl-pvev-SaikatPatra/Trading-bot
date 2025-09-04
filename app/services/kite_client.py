# File: app/services/kite_client.py

import logging
from kiteconnect import KiteConnect
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.models import UserSession
from app.core.security import encrypt_token, decrypt_token
from datetime import datetime

log = logging.getLogger(__name__)

class KiteClient:
    def __init__(self, db: Session):
        self.db = db
        self.kite = KiteConnect(api_key=settings.ZERODHA_API_KEY)
        self._load_session()

    def _load_session(self):
        # In a multi-user system, you'd key this by a user ID. For a personal bot, we use a fixed ID.
        session_data = self.db.query(UserSession).filter(UserSession.user_id == "main_user").first()
        if session_data and session_data.encrypted_access_token:
            try:
                self.kite.access_token = decrypt_token(session_data.encrypted_access_token)
                log.info("Loaded existing session from database.")
            except Exception as e:
                log.error(f"Failed to decrypt and load session: {e}")
                self.kite.access_token = None
        else:
            log.warning("No active session found in database.")

    def get_login_url(self):
        return self.kite.login_url()

    def generate_session(self, request_token: str):
        try:
            data = self.kite.generate_session(request_token, api_secret=settings.ZERODHA_API_SECRET)
            self.kite.set_access_token(data["access_token"])
            
            # Persist the session
            session_data = self.db.query(UserSession).filter(UserSession.user_id == "main_user").first()
            if not session_data:
                session_data = UserSession(user_id="main_user")
            
            session_data.encrypted_access_token = encrypt_token(data["access_token"])
            session_data.encrypted_refresh_token = encrypt_token(data.get("refresh_token", ""))
            session_data.login_time = datetime.now()
            
            self.db.add(session_data)
            self.db.commit()
            log.info("Successfully generated and saved new Kite session.")
            return True
        except Exception as e:
            log.error(f"Error generating session: {e}", exc_info=True)
            return False

    def is_connected(self):
        return self.kite.access_token is not None

    def get_margins(self):
        if not self.is_connected(): return None
        try:
            return self.kite.margins()
        except Exception as e:
            log.error(f"Error fetching margins: {e}")
            return None

    def get_positions(self):
        if not self.is_connected(): return None
        try:
            return self.kite.positions()
        except Exception as e:
            log.error(f"Error fetching positions: {e}")
            return None

    def place_mis_order(self, symbol: str, transaction_type: str, quantity: int, tag: str):
        if not self.is_connected(): return None
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=self.kite.EXCHANGE_NSE,
                transaction_type=transaction_type, # 'BUY' or 'SELL'
                quantity=quantity,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_MARKET,
                variety=self.kite.VARIETY_REGULAR,
                tag=tag
            )
            log.info(f"Placed MIS {transaction_type} order for {quantity} {symbol}. Order ID: {order_id}, Tag: {tag}")
            return order_id
        except Exception as e:
            log.error(f"Error placing order for {symbol}: {e}")
            return None

    def place_sl_order(self, symbol: str, transaction_type: str, quantity: int, trigger_price: float, tag: str):
        if not self.is_connected(): return None
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=self.kite.EXCHANGE_NSE,
                transaction_type=transaction_type, # Opposite of entry
                quantity=quantity,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_SLM,
                variety=self.kite.VARIETY_REGULAR,
                trigger_price=trigger_price,
                tag=tag
            )
            log.info(f"Placed SL-M {transaction_type} order for {quantity} {symbol} @ {trigger_price}. Order ID: {order_id}, Tag: {tag}")
            return order_id
        except Exception as e:
            log.error(f"Error placing SL order for {symbol}: {e}")
            return None
            
    def modify_order(self, order_id: str, new_trigger_price: float):
        if not self.is_connected(): return None
        try:
            # Note: Modifying SL orders is tricky. You might need to cancel and replace.
            # This is a simplified example.
            modified_order_id = self.kite.modify_order(
                variety=self.kite.VARIETY_REGULAR,
                order_id=order_id,
                trigger_price=new_trigger_price
            )
            log.info(f"Modified order {order_id} with new trigger price {new_trigger_price}. New ID: {modified_order_id}")
            return modified_order_id
        except Exception as e:
            log.error(f"Error modifying order {order_id}: {e}")
            return None

    def square_off_all(self):
        if not self.is_connected(): return
        try:
            positions = self.get_positions().get('net', [])
            for pos in positions:
                if pos['product'] == 'MIS' and pos['quantity'] != 0:
                    symbol = pos['tradingsymbol']
                    quantity = abs(pos['quantity'])
                    transaction_type = self.kite.TRANSACTION_TYPE_BUY if pos['quantity'] < 0 else self.kite.TRANSACTION_TYPE_SELL
                    self.place_mis_order(symbol, transaction_type, quantity, tag="SQUAREOFF_EOD")
                    log.info(f"Squaring off {quantity} of {symbol}")
            
            # Cancel all pending MIS orders
            pending_orders = self.kite.orders()
            for order in pending_orders:
                if order['product'] == 'MIS' and order['status'] in ['TRIGGER PENDING', 'OPEN']:
                    self.kite.cancel_order(variety=order['variety'], order_id=order['order_id'])
                    log.info(f"Cancelled pending MIS order {order['order_id']} for {order['tradingsymbol']}")

        except Exception as e:
            log.error(f"Error during square off: {e}")
