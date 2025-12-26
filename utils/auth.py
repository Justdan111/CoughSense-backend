from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from jose import jwt
import requests
import os

security = HTTPBearer()

SUPABASE_PROJECT_ID = os.getenv("SUPABASE_PROJECT_ID")
SUPABASE_JWT_URL = f"https://{SUPABASE_PROJECT_ID}.supabase.co/auth/v1/certs"

jwks = requests.get(SUPABASE_JWT_URL).json()

def verify_user(token=Depends(security)):
    try:
        payload = jwt.decode(
            token.credentials,
            jwks,
            algorithms=["RS256"],
            audience="authenticated"
        )
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")
