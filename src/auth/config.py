"""BFF login settings.

Non-secret values come from ``resources/application-config.yaml`` (via ``get_config``);
the cookie-signing secret comes from Vault, which ``load_secrets()`` has already copied
into the environment.

YAML keys (Srini adds them):

    BFF_BASE_URL: "http://news-summary-app:8007"     # every redirect is built from this
    MCP_URL: "http://es-mcp-lb:9599/mcp"            # resource indicator == token audience
    CIMD_CLIENT_ID: "https://news-cimd-host/clients/news-summary-app-cimd.json"
    ALLOWED_ISSUER: "http://keycloak:8180/realms/es-mcp"
    SESSION_MAX_AGE: 36000                           # optional; <= Keycloak SSO Session Max

Vault (secret/CREDENTIALS):

    NEWS_APP_SESSION_SECRET   # >= 32 random bytes, e.g. `openssl rand -base64 48`
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from config.application_config import get_config

SESSION_SECRET_ENV = "NEWS_APP_SESSION_SECRET"

@dataclass(frozen=True)
class BffConfig:
    base_url: str
    mcp_url: str
    client_id: str
    allowed_issuer: str
    session_secret: str
    session_max_age_sec: int = 36000
    refresh_skew_seconds: int = 30
    session_cookie: str = "nsa_session"

    @property
    def redirect_uri(self) -> str:
        return f"{self.base_url}/auth/callback"

    @classmethod
    def load(cls) -> BffConfig:
        """Build from application config + environment. Fails fast on a missing value."""
        secret = os.environ.get(SESSION_SECRET_ENV, "")
        if len(secret) < 32:
            raise ValueError(
                f"{SESSION_SECRET_ENV} must be set (Vault secret/CREDENTIALS) and be at least 32 characters"
            )
        try:
            max_age = int(get_config("SESSION_MAX_AGE_SEC"))
        except KeyError:
            max_age = 36000
        return cls(
            base_url=str(get_config("BFF_BASE_URL")).rstrip("/"),
            mcp_url=str(get_config("MCP_URL")),
            client_id=str(get_config("CIMD_CLIENT_ID")),
            allowed_issuer=str(get_config("ALLOWED_ISSUER_URL")),
            session_secret=secret,
            session_max_age_sec=max_age,
        )
