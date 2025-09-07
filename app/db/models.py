# File: app/db/models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
from .session import Base

# Define the declarative base here
Base = declarative_base()

class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    encrypted_access_token = Column(String)
    encrypted_refresh_token = Column(String)
    login_time = Column(DateTime)
    
class TradeLog(Base):
    __tablename__ = "trade_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    symbol = Column(String)
    order_type = Column(String) # e.g., 'BUY', 'SELL', 'SL', 'TARGET'
    quantity = Column(Integer)
    price = Column(Float)
    order_id = Column(String, unique=True)
    status = Column(String) # e.g., 'PLACED', 'EXECUTED', 'CANCELLED'
    details = Column(String)

class SystemState(Base):
    __tablename__ = "system_state"
    key = Column(String, primary_key=True, index=True)
    value = Column(JSON)

class Universe(Base):
    __tablename__ = "universe"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
