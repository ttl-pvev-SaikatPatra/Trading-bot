# File: app/main.py

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.logging_config import setup_logging, CorrelationIdMiddleware
from app.db import models
from app.db.session import engine
from app.api import auth, status, controls, cron, universe, market_data # Add market_data

# Setup logging before anything else
setup_logging()

# Create database tables
# This now works because engine is imported from session, and session imports models
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Intraday Trading Bot",
    description="An automated trading bot for Indian equities using Zerodha APIs.",
    version="1.0.0",
)

# Middleware
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this to your Vercel frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(auth.router)
app.include_router(status.router, prefix="/api")
app.include_router(controls.router, prefix="/api")
app.include_router(universe.router, prefix="/api/universe")
app.include_router(cron.router, prefix="/cron")
app.include_router(market_data.router, prefix="/api") # Add this line


@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Trading Bot API is running."}

@app.get("/health", tags=["Root"])
def health_check():
    """Health check endpoint for Render/cron jobs to ping."""
    return {"status": "ok"}
