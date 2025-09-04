# File: app/models/pydantic_models.py

from pydantic import BaseModel
from typing import Optional, List, Dict

class StatusResponse(BaseModel):
    broker_connected: bool
    trading_enabled: bool
    dry_run_mode: bool
    last_daily_cron: Optional[str]
    last_open_cron: Optional[str]
    last_close_cron: Optional[str]
    daily_pnl: float

class FundsResponse(BaseModel):
    available_margin: float
    used_margin: float
    total_balance: float

class Position(BaseModel):
    symbol: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    product: str

class Trade(BaseModel):
    timestamp: str
    symbol: str
    order_type: str
    quantity: int
    price: float
    status: str
    order_id: str

class ControlRequest(BaseModel):
    action: str  # 'start', 'stop', 'toggle_dry_run'

class ControlResponse(BaseModel):
    status: str
    trading_enabled: bool
    dry_run_mode: bool

class UniverseStock(BaseModel):
    symbol: str
    is_active: bool
