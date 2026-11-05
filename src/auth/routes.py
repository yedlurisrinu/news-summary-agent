"""``/auth/login`` and ``/auth/callback``: the authorization-code + PKCE login with Keycloak.

Error mapping: only these codes ever reach the URL; ``static/login.html``
turns them into fixed text, so nothing from Keycloak is echoed to the browser.
"""

from __future__ import annotations

import logging

from authlib.integrations.base_client import MismatchingStateError, OAuthError
from fastapi import APIRouter, Request
from starlette.responses import RedirectResponse

from auth.config import BffConfig
from auth.oauth import KeycloakOAuth
from auth.sessions import LoginRequired, SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Auth"])


def _to_login(code: str) -> RedirectResponse:
    return RedirectResponse(f"/login?error={code}", status_code=302)


def _auth(request: Request) -> tuple[BffConfig, KeycloakOAuth, SessionStore]:
    state = request.app.state
    return state.bff_config, state.keycloak, state.sessions


@router.get("/auth/login", include_in_schema=False)
async def auth_login(request: Request) -> RedirectResponse:
    """Start the authorization-code + PKCE flow: 302 to Keycloak's authorize endpoint."""
    config, keycloak, _ = _auth(request)
    try:
        client = await keycloak.client()  # discovery happens here on the first login
        response: RedirectResponse = await client.authorize_redirect(
            request, config.redirect_uri, resource=config.mcp_url
        )
        return response
    except Exception as exc:  # noqa: BLE001 - login boundary: discovery/metadata failure -> fixed error code
        logger.warning("login could not start: %s: %s", type(exc).__name__, exc)
        return _to_login("login_failed")


@router.get("/auth/callback", include_in_schema=False)
async def auth_callback(request: Request) -> RedirectResponse:
    """Finish the flow: check iss, exchange the code, create the session, go to the chat."""
    config, keycloak, sessions = _auth(request)

    if request.query_params.get("error") == "access_denied":
        return _to_login("access_denied")

    # RFC 9207 mix-up defence; Authlib does not check this parameter itself.
    if keycloak.issuer is None or request.query_params.get("iss") != keycloak.issuer:
        logger.warning("callback iss mismatch: %r", request.query_params.get("iss"))
        return _to_login("login_failed")

    try:
        client = await keycloak.client()
        token = await client.authorize_access_token(request, resource=config.mcp_url)
    except MismatchingStateError:
        return _to_login("state_mismatch")
    except OAuthError as exc:
        logger.warning("callback refused: %s", exc.error)
        return _to_login(
            "access_denied" if exc.error == "access_denied" else "login_failed"
        )
    except Exception as exc:  # noqa: BLE001 - login boundary: any failure -> fixed error code
        logger.warning("callback failed: %s", type(exc).__name__)
        return _to_login("login_failed")

    try:
        sid = await sessions.create(token)
    except LoginRequired:
        logger.warning("callback refused: token response has no refresh token")
        return _to_login("login_failed")
    except Exception as exc:  # noqa: BLE001 - login boundary: session store (Vault) unavailable
        logger.warning("callback could not store the session: %s", type(exc).__name__)
        return _to_login("auth_unavailable")
    request.session.clear()  # drop Authlib's temporary state; new id after login (no fixation)
    request.session["sid"] = sid
    return RedirectResponse("/chat", status_code=302)
