"""Default-deny guard: every request needs a live session, except a short allowlist.

Pure ASGI middleware (``scope``/``receive``/``send``), see guide steps 4-6:
- It must run INSIDE ``SessionMiddleware`` (so ``scope["session"]`` exists). In Starlette
  the LAST ``add_middleware`` call is the OUTERMOST, so add this guard first and
  ``SessionMiddleware`` after it (``wiring.install_auth`` does exactly that).
- Non-HTTP scopes (``lifespan``, ``websocket``) pass through untouched, otherwise the
  app's startup would never run.
- If Keycloak or the session store gives no answer, the guard returns 503 (HTML page or
 ``{"error": "auth_unavailable"}``) with ``Retry-After`` and keeps the session (F-E2E6-3).

- Every response on a protected path (allowed or refused) carries
  ``Cache-Control: no-store`` (RFC 9111 5.2.2.5), so the browser never shows a stored
  copy of an authenticated page without asking the app (F-E2E6-2). Public paths are
  left alone: ``/login`` is fine to cache.

"""

from __future__ import annotations

from collections.abc import Iterable

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse, RedirectResponse
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from auth import sessions
from auth.sessions import AuthUnavailable, LoginRequired, SessionStore

from auth.sessions import SessionStore

DEFAULT_PUBLIC_PATHS = frozenset({"/login", "/auth/login", "/auth/callback", "/health"})
RETRY_AFTER_SECONDS = "30"
UNAVAILABLE_PAGE = (
    "<!doctype html><title>Temporarily unavailable</title>"
    "<p>The sign-in service is temporarily unavailable. You are still signed in; "
    "please try again in a moment.</p>"
)

def unavailable(scope: Scope) -> Response:
    """
    503 for an auth back end that gave no answer; the session and cookie are kept.
    """
    headers = {"Retry-After": RETRY_AFTER_SECONDS}
    if is_page_request(scope=scope):
        return HTMLResponse(UNAVAILABLE_PAGE, status_code=503, headers=headers)
    return JSONResponse({"error": "auth_unavailable"}, status_code=503, headers=headers)
    
def is_page_request(scope: Scope) -> bool:
    """True for a browser navigation that expects an HTML page."""
    if scope.get("method") != "GET":
        return False
    headers = Headers(scope=scope)
    return headers.get("sec-fetch-mode") == "navigate" or "text/html" in headers.get("accept", "")

def no_store(send: Send) -> Send:
    async def send_no_store(message: Message) -> None:
        if message["type"] == "http.response.start":
            MutableHeaders(scope=message)["cache-control"] = "no-store"
        await send(message)
    return send_no_store

class AuthGuard:
    def __init__(
        self,
        app: ASGIApp,
        sessions: SessionStore,
        public_paths: Iterable[str] = DEFAULT_PUBLIC_PATHS,
    ) -> None:
        self.app = app
        self.sessions = sessions
        self.public_paths = frozenset(public_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self.public_paths:
            await self.app(scope, receive, send)
            return
        
        send = no_store(send)
        session = scope.get("session")
        if session is None:
            raise RuntimeError("AuthGuard must be installed inside SessionMiddleware")
        sid = session.get("sid")

        if sid:
            try:
                await self.sessions.access_token(sid)
            except LoginRequired:
                session.pop("sid", None) # stale cookie: forget it so the next login starts clean
            except AuthUnavailable: # Keycloak/Vault down: say so, keep the session (F-E2E6-3)
                await unavailable(scope=scope)(scope=scope, receive=receive, send=send)
                return
            else:
                await self.app(scope, receive, send)
                return
 
        if sid:  # stale cookie: forget it so the next login starts clean
            session.pop("sid", None)
        if is_page_request(scope):
            response: RedirectResponse | JSONResponse = RedirectResponse("/login", status_code=302)
        else:
            response = JSONResponse({"error": "login_required"}, status_code=401)
        await response(scope, receive, send)
