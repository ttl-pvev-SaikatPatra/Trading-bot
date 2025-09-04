# File: app/services/universe_builder.py

import logging
from sqlalchemy.orm import Session
from app.db.models import Universe

log = logging.getLogger(__name__)

# For a free tier, a static list is the most reliable approach.
# This can be expanded to fetch from a file or a free API endpoint.
NIFTY_100_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "BHARTIARTL", 
    "ITC", "SBIN", "LICI", "BAJFINANCE", "HCLTECH", "KOTAKBANK", "MARUTI", "LT",
    "AXISBANK", "ASIANPAINT", "SUNPHARMA", "TITAN", "WIPRO", "ULTRACEMCO",
    # ... Add more symbols from NIFTY 100 or your preferred list
]


class UniverseBuilder:
    def __init__(self, db: Session):
        self.db = db

    def build_and_store_universe(self):
        """
        Builds the daily universe and stores it in the database.
        Clears the old universe first.
        """
        try:
            log.info("Starting daily universe build process.")
            
            # Clear existing universe
            self.db.query(Universe).delete()

            # Add new stocks
            for symbol in NIFTY_100_STOCKS:
                stock = Universe(symbol=symbol, is_active=True)
                self.db.add(stock)
            
            self.db.commit()
            log.info(f"Successfully built and stored universe with {len(NIFTY_100_STOCKS)} stocks.")
            return True
        except Exception as e:
            self.db.rollback()
            log.error(f"Error building universe: {e}", exc_info=True)
            return False

    def get_universe(self) -> list[str]:
        """Retrieves the active universe from the database."""
        stocks = self.db.query(Universe).filter(Universe.is_active == True).all()
        return [stock.symbol for stock in stocks]
