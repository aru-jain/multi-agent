"""Google OAuth 2.0 authentication and signed session cookie routes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Cookie, Header, HTTPException, Response
from fastapi.responses import RedirectResponse

from src.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

SESSION_COOKIE_NAME = "ma_session"
STATE_COOKIE_NAME = "ma_oauth_state"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _create_jwt(user: dict[str, Any], secret: str) -> str:
    """Create a standard HS256 JWT (RFC 7519)."""
    now = int(time.time())
    header = _b64url_encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8")
    )
    payload = _b64url_encode(
        json.dumps(
            {
                "sub": user.get("id", ""),
                "email": user.get("email", ""),
                "name": user.get("name", ""),
                "picture": user.get("picture", ""),
                "provider": user.get("provider", ""),
                "iat": now,
                "exp": now + SESSION_TTL_SECONDS,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    sig = _b64url_encode(
        hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    )
    return f"{header}.{payload}.{sig}"


def _verify_jwt(token: str, secret: str) -> dict[str, Any] | None:
    """Verify an HS256 JWT signature and expiration, returning claims."""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".", 2)
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = _b64url_encode(
            hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        )
        if not hmac.compare_digest(sig_b64, expected_sig):
            return None
        claims = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        if claims.get("exp", 0) < time.time():
            return None
        return claims
    except Exception:
        return None


def _set_session_cookie(response: Response, user: dict[str, Any], secret: str) -> str:
    """Attach HS256 JWT session cookie to response and return the JWT."""
    token = _create_jwt(user, secret)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL_SECONDS,
        path="/",
    )
    return token


@router.get("/config")
async def auth_config() -> dict[str, Any]:
    """Return whether Google OAuth is configured."""
    settings = get_settings()
    return {
        "google_enabled": bool(
            settings.google_client_id and settings.google_client_secret
        )
    }


@router.get("/google/login")
async def google_login() -> RedirectResponse:
    """Redirect the user to Google's OAuth 2.0 consent screen."""
    settings = get_settings()
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=503,
            detail="Google OAuth is not configured (set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET).",
        )

    state = secrets.token_urlsafe(24)
    redirect_uri = f"{settings.backend_url.rstrip('/')}/auth/google/callback"
    params = urlencode(
        {
            "client_id": settings.google_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "online",
            "prompt": "select_account",
        }
    )
    resp = RedirectResponse(url=f"{GOOGLE_AUTH_URL}?{params}", status_code=302)
    resp.set_cookie(
        key=STATE_COOKIE_NAME,
        value=state,
        httponly=True,
        samesite="lax",
        max_age=600,
        path="/",
    )
    return resp


@router.get("/google/callback")
async def google_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    ma_oauth_state: str | None = Cookie(default=None),
) -> RedirectResponse:
    """Handle Google OAuth callback, exchange code for userinfo, and set session cookie."""
    settings = get_settings()
    frontend = settings.frontend_url.rstrip("/")

    if error or not code:
        return RedirectResponse(url=f"{frontend}/?auth_error={error or 'missing_code'}")

    if not state or not ma_oauth_state or not hmac.compare_digest(state, ma_oauth_state):
        return RedirectResponse(url=f"{frontend}/?auth_error=invalid_state")

    redirect_uri = f"{settings.backend_url.rstrip('/')}/auth/google/callback"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            token_res = await client.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )
            token_res.raise_for_status()
            access_token = token_res.json().get("access_token")

            user_res = await client.get(
                GOOGLE_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            user_res.raise_for_status()
            info = user_res.json()
    except Exception as exc:
        logger.error("Google OAuth exchange failed: %s", exc)
        return RedirectResponse(url=f"{frontend}/?auth_error=oauth_exchange_failed")

    user = {
        "id": info.get("sub", ""),
        "email": info.get("email", ""),
        "name": info.get("name", info.get("email", "User")),
        "picture": info.get("picture", ""),
        "provider": "google",
    }

    resp = RedirectResponse(url=frontend, status_code=302)
    resp.delete_cookie(STATE_COOKIE_NAME, path="/")
    _set_session_cookie(resp, user, settings.session_secret)
    return resp


@router.post("/demo")
async def demo_login(response: Response) -> dict[str, Any]:
    """Local development fallback login when Google OAuth is not configured."""
    settings = get_settings()
    user = {
        "id": "demo-user",
        "email": "developer@localhost",
        "name": "Local Developer",
        "picture": "",
        "provider": "demo",
    }
    token = _set_session_cookie(response, user, settings.session_secret)
    return {"authenticated": True, "access_token": token, "token_type": "bearer", "user": user}


@router.get("/me")
async def get_current_user(
    ma_session: str | None = Cookie(default=None),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Return the currently authenticated user from the JWT cookie or Bearer header."""
    token = ma_session
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    settings = get_settings()
    claims = _verify_jwt(token, settings.session_secret)
    if not claims:
        raise HTTPException(status_code=401, detail="Invalid or expired JWT")

    user = {
        "id": claims.get("sub", ""),
        "email": claims.get("email", ""),
        "name": claims.get("name", ""),
        "picture": claims.get("picture", ""),
        "provider": claims.get("provider", ""),
    }
    return {"authenticated": True, "user": user}


@router.post("/logout")
async def logout(response: Response) -> dict[str, Any]:
    """Clear the session cookie."""
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"authenticated": False, "message": "Logged out"}
