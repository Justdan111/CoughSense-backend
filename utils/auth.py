
import os
import requests
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from jose import jwt

security = HTTPBearer()

SUPABASE_PROJECT_ID = os.getenv("")
JWKS_URL = f"https://{SUPABASE_PROJECT_ID}.supabase.co/auth/v1/certs"

jwks = requests.get(JWKS_URL).json()

def verify_user(token=Depends(security)):
    try:
        payload = jwt.decode(
            token.credentials,
            jwks,
            algorithms=["RS256"],
            audience="authenticated",
            options={"verify_exp": True}
        )
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
