from typing import TypedDict


class OIDCMetadata(
    TypedDict
):  # the fields we rely on; Keycloak's extra keys stay in the dict
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str


REQUIRED = ("issuer", "authorization_endpoint", "token_endpoint", "jwks_uri")
