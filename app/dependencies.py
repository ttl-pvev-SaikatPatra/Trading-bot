# File: src/app/dependencies.py

import os
import sqlite3
from typing import Generator
from kiteconnect import KiteConnect
from app.core.config import settings

def get_db() -> Generator[sqlite3.Connection, None, None]:
    """
    Dependency to get a database connection.
    Yields a connection and ensures it's closed after the request.
    """
    db_path = settings.SQLITE_DB_PATH
    db = sqlite3.connect(db_path)
    try:
        yield db
    finally:
        db.close()

def get_kite_client() -> KiteConnect:
    """
    Dependency to get a KiteConnect client instance.
    Initializes the client with API key and secret from settings.
    """
    return KiteConnect(api_key=settings.ZERODHA_API_KEY)
