import os
import sqlite3
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.dependencies import get_kite_client
from app.db.session import get_db

router = APIRouter(tags=["Auth"], prefix="/auth")

class TokenData(BaseModel):
    user_id: str
    access_token: str
    public_token: str
    expiry: datetime

@router.get("/login")
def login_route(kite_client=Depends(get_kite_client)):
    """
    Redirects the user to the Zerodha Kite login page.
    The Zerodha API Key and Secret are loaded from environment variables.
    """
    try:
        login_url = kite_client.login_url()
        return RedirectResponse(login_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate login URL: {str(e)}")

@router.get("/callback")
def auth_callback(
    request_token: str,
    db: sqlite3.Connection = Depends(get_db),
    kite_client=Depends(get_kite_client)
):
    """
    Handles the redirect from Zerodha after successful login.
    Exchanges the request_token for an access_token and persists it.
    """
    if not request_token:
        raise HTTPException(status_code=400, detail="Missing request_token in callback.")

    try:
        # 1. Exchange the request_token for the access_token
        data = kite_client.generate_session(request_token, api_secret=settings.ZERODHA_API_SECRET)
        access_token = data.get("access_token")
        public_token = data.get("public_token")
        user_id = data.get("user_id")

        if not access_token or not user_id:
            raise HTTPException(status_code=500, detail="Failed to retrieve access_token or user_id.")
        
        # 2. Persist the token securely in SQLite
        # Using a simple table for this purpose. In a production app, you'd want to handle
        # this with proper encryption and user session management.
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO sessions (user_id, access_token, public_token, last_login) 
            VALUES (?, ?, ?, ?)
            """,
            (user_id, access_token, public_token, datetime.now())
        )
        db.commit()
        
        # 3. Redirect to the frontend's dashboard URL
        # THIS IS THE CRITICAL FIX. The URL must be your Vercel frontend URL.
        # It's best practice to pass the user_id or a success status as a query parameter.
        # This allows the frontend to know the login was successful and trigger a fetch
        # for user-specific data, like funds.
        frontend_redirect_url = f"{settings.FRONTEND_URL}/dashboard?login_status=success&user_id={user_id}"
        print(f"Redirecting to frontend URL: {frontend_redirect_url}") # Debug log

        return RedirectResponse(url=frontend_redirect_url, status_code=status.HTTP_302_FOUND)

    except Exception as e:
        # Log the error for debugging
        print(f"Error during auth callback: {str(e)}")
        # Redirect back to the frontend's login page with an error status
        error_redirect_url = f"{settings.FRONTEND_URL}/login?login_status=failed&error={e}"
        return RedirectResponse(url=error_redirect_url, status_code=status.HTTP_302_FOUND)
