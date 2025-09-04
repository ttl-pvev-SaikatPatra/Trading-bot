# File: app/api/controls.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.order_manager import OrderManager
from app.models.pydantic_models import ControlRequest, ControlResponse

router = APIRouter()

@router.post("/controls", response_model=ControlResponse, tags=["Controls"])
def update_controls(req: ControlRequest, db: Session = Depends(get_db)):
    """Start/stop the trading engine or toggle dry-run mode."""
    manager = OrderManager(db)
    action = req.action.lower()

    if action == 'start':
        manager.toggle_trading(True)
    elif action == 'stop':
        manager.toggle_trading(False)
    elif action == 'toggle_dry_run':
        current_mode = manager.state.get('dry_run_mode', True)
        manager.toggle_dry_run(not current_mode)
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid action")

    return ControlResponse(
        status=f"Action '{action}' executed successfully.",
        trading_enabled=manager.state.get('trading_enabled'),
        dry_run_mode=manager.state.get('dry_run_mode')
    )
