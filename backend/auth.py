"""Authentication utilities for JWT-based authentication"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from config.settings import settings

security = HTTPBearer(auto_error=False)


def create_access_token(username: str) -> str:
    """Create JWT access token"""
    expire = datetime.utcnow() + timedelta(hours=settings.jwt_expiration_hours)
    to_encode = {
        "sub": username,
        "exp": expire,
    }
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


def verify_token(token: str) -> Optional[str]:
    """Verify JWT token and return username if valid"""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None


def verify_credentials(username: str, password: str) -> bool:
    """Verify username and password against configured credentials"""
    if not settings.auth_enabled:
        return True

    if not settings.auth_username or not settings.auth_password:
        # If auth is enabled but no credentials configured, reject
        return False

    return (
        username == settings.auth_username and
        password == settings.auth_password
    )


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> str:
    """Dependency to require authentication on routes"""
    # If auth is not enabled, allow all requests
    if not settings.auth_enabled:
        return "anonymous"

    # If auth is enabled, require valid token
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = verify_token(credentials.credentials)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return username
