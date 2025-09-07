# File: src/app/db/database.py

import sqlite3
from typing import Generator
from app.core.config import settings

def get_db() -> Generator[sqlite3.Connection, None, None]:
    """
    FastAPI dependency to get a database connection.
    Yields a connection and ensures it's closed after the request.
    """
    db_path = settings.SQLITE_DB_PATH
    db = sqlite3.connect(db_path)
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initializes the SQLite database tables if they do not exist.
    """
    db_path = settings.SQLITE_DB_PATH
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    
    # Create the sessions table for storing access tokens
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            user_id TEXT PRIMARY KEY,
            access_token TEXT NOT NULL,
            public_token TEXT,
            last_login TEXT
        )
    """)

    # Create the logs table for audit trails
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            context TEXT
        )
    """)

    # Create the universe table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS universe (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            token TEXT,
            is_active INTEGER DEFAULT 1
        )
    """)
    
    db.commit()
    db.close()
