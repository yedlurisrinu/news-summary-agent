"""``install_auth`` - the single call ``api/app.py`` makes to add the BFF login.

Order matters (verified in Starlette 1.10): the last ``add_middleware`` is the outermost,
so the guard is added first and ``SessionMiddleware`` second, making the session available
to the guard.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware

from auth import routes
from auth.auth_server_metadata import AuthServerMetadata
from auth.config import BffConfig
from auth.guard import AuthGuard
from auth.oauth import KeycloakOAuth
from auth.ports import RefreshTokenStore
from auth.sessions import SessionStore


def install_auth(
    app: FastAPI,
    config: BffConfig,
    store: RefreshTokenStore,
    as_meta: AuthServerMetadata,
    http_transport: Any | None = None,
) -> None:
    keycloak = KeycloakOAuth(config, as_meta=as_meta, http_transport=http_transport)
    sessions = SessionStore(store, keycloak.refresh, config.refresh_skew_seconds)

    app.state.bff_config = config
    app.state.keycloak = keycloak
    app.state.sessions = sessions

    app.include_router(routes.router)

    app.add_middleware(AuthGuard, sessions=sessions)  # inner
    app.add_middleware(  # outer: runs first, fills scope["session"]
        SessionMiddleware,
        secret_key=config.session_secret,
        session_cookie=config.session_cookie,
        max_age=config.session_max_age_sec,
        same_site="lax",  # must be lax: the return from Keycloak is a cross-site navigation
        https_only=False,  # True once the app is served over HTTPS
    )
