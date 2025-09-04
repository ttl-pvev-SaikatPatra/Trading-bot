# File: app/api/auth.py

import logging
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.core.config import settings
from app.db.session import get_db
from app.services.kite_client import KiteClient

router = APIRouter()
log = logging.getLogger(__name__)

@router.get("/login", tags=["Authentication"])
def login(db: Session = Depends(get_db)):
    """Redirects the user to the Kite login page."""
    kite_client = KiteClient(db)
    login_url = kite_client.get_login_url()
    log.info("Redirecting user to Kite login page.")
    return RedirectResponse(url=login_url)

@router.get("/callback", tags=["Authentication"])
def callback(request: Request, request_token: str, db: Session = Depends(get_db)):
    """Handles the callback from Kite after successful login."""
    log.info(f"Received callback with request_token: {request_token}")
    kite_client = KiteClient(db)
    if kite_client.generate_session(request_token):
        log.info("Session generated successfully. Redirecting to frontend dashboard.")
        # In a real app, you would redirect to your frontend URL
        return {"status": "success", "message": "Authentication successful. You can close this window."}
    else:
        log.error("Failed to generate session.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate session. Check logs.",
        )
