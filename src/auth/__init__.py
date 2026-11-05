"""BFF login for the news app: Keycloak (CIMD client) login, server-side sessions, and the
default-deny guard that keeps every endpoint behind that login.

Module map
----------
- ``ports``      : the two seams auth depends on (refresh-token store, issuer discovery).
- ``config``     : ``BffConfig``, read from application-config.yaml + the Vault-loaded env.
- ``oauth``      : lazy Authlib registration (after discovery) + the refresh-token call.
- ``sessions``   : ``SessionStore`` - memory for access tokens, the store for refresh tokens.
- ``guard``      : ``AuthGuard`` ASGI middleware (default deny).
- ``routes``     : ``/auth/login`` and ``/auth/callback``.
- ``wiring``     : ``install_auth(app, ...)`` - the one call ``api/app.py`` makes.
"""
