# File: app/api/cron.py

import logging
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.core.security import verify_hmac_signature
from app.services.universe_builder import UniverseBuilder
from app.services.order_manager import OrderManager

router = APIRouter(dependencies=[Depends(verify_hmac_signature)])
log = logging.getLogger(__name__)

@router.post("/daily", tags=["Cron"])
async def run_daily_tasks(request: Request, db: Session = Depends(get_db)):
    """
    Cron job for daily pre-market tasks.
    - Builds the trading universe.
    - Expected to run at 08:45 AM IST.
    """
    log.info("Received request for /cron/daily")
    builder = UniverseBuilder(db)
    builder.build_and_store_universe()
    
    manager = OrderManager(db)
    manager.update_cron_timestamp('daily')
    
    return {"status": "Daily tasks executed."}

@router.post("/open", tags=["Cron"])
async def run_strategy_on_open(request: Request, db: Session = Depends(get_db)):
    """
    Cron job to run the strategy cycle.
    - Expected to run every 5/15 minutes during market hours.
    - First run at 09:20 AM IST after the first candle closes.
    """
    log.info("Received request for /cron/open")
    manager = OrderManager(db)
    manager.run_strategy_cycle()
    manager.update_cron_timestamp('open')
    return {"status": "Strategy cycle executed."}


@router.post("/close", tags=["Cron"])
async def run_closing_tasks(request: Request, db: Session = Depends(get_db)):
    """
    Cron job for end-of-day closing tasks.
    - Squares off all open MIS positions.
    - Cancels all pending MIS orders.
    - Expected to run at 03:20 PM IST.
    """
    log.info("Received request for /cron/close")
    manager = OrderManager(db)
    manager.square_off_all_positions()
    manager.update_cron_timestamp('close')
    return {"status": "Closing tasks executed."}
