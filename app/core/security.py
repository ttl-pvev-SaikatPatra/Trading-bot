# File: app/core/security.py

import hmac
import hashlib
from fastapi import Request, HTTPException, status
from cryptography.fernet import Fernet
from app.core.config import settings

# Initialize Fernet for encryption/decryption
# NEW, CLEANER LINE
cipher_suite = Fernet(settings.APP_SECRET_KEY.encode())

async def verify_hmac_signature(request: Request):
    """Dependency to verify HMAC signature for cron jobs."""
    try:
        signature = request.headers.get("X-HMAC-Signature")
        if not signature:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="X-HMAC-Signature header missing")

        body = await request.body()
        expected_signature = hmac.new(settings.CRON_SECRET_KEY.encode(), body, hashlib.sha256).hexdigest()

        if not hmac.compare_digest(expected_signature, signature):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid HMAC signature")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Invalid signature: {e}")

def encrypt_token(token: str) -> str:
    """Encrypts a token."""
    return cipher_suite.encrypt(token.encode()).decode()

def decrypt_token(encrypted_token: str) -> str:
    """Decrypts a token."""
    return cipher_suite.decrypt(encrypted_token.encode()).decode()
