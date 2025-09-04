# File: app/api/status.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.kite_client import KiteClient
from app.services.order_manager import OrderManager
from app.models.pydantic_models import StatusResponse, FundsResponse, Position

router = APIRouter()

@router.get("/status", response_model=StatusResponse, tags=["Status"])
def get_system_status(db: Session = Depends(get_db)):
    """Get the current status of the trading bot."""
    kite = KiteClient(db)
    manager = OrderManager(db)
    return StatusResponse(
        broker_connected=kite.is_connected(),
        trading_enabled=manager.state.get('trading_enabled', False),
        dry_run_mode=manager.state.get('dry_run_mode', True),
        last_daily_cron=manager.state.get('last_daily_cron'),
        last_open_cron=manager.state.get('last_open_cron'),
        last_close_cron=manager.state.get('last_close_cron'),
        daily_pnl=manager.state.get('daily_pnl', 0.0)
    )

@router.get("/funds", response_model=FundsResponse, tags=["Status"])
def get_funds(db: Session = Depends(get_db)):
    """Get account funds and margin information."""
    kite = KiteClient(db)
    margins = kite.get_margins()
    if margins and 'equity' in margins:
        equity = margins['equity']
        return FundsResponse(
            available_margin=equity['available']['live_balance'],
            used_margin=equity['utilised']['debits'],
            total_balance=equity['net']
        )
    return FundsResponse(available_margin=0, used_margin=0, total_balance=0)


@router.get("/positions", response_model=list[Position], tags=["Status"])
def get_positions(db: Session = Depends(get_db)):
    """Get current open positions."""
    kite = KiteClient(db)
    positions_data = kite.get_positions()
    if positions_data and 'net' in positions_data:
        # Filter for open MIS positions
        return [
            Position(**p) for p in positions_data['net']
            if p['product'] == 'MIS' and p['quantity'] != 0
        ]
    return []
