from typing import Any
from urllib.parse import urlsplit, urlunsplit

from httpx2 import AsyncClient

from config.application_config import get_config


async def protect_resource_metadata(mcp_url: str) -> Any:
    mcp_well_known_uri = get_config("MCP_WELL_KNOWN_URI")
    # No Exception handling as when the auth flow actually
    # calls this coroutine it the failure should be visible to caller
    parts = urlsplit(mcp_url)
    async with AsyncClient() as client:
        result = await client.get(
            urlunsplit(
                (
                    parts.scheme,
                    parts.netloc,
                    mcp_well_known_uri + parts.path.rstrip("/"),
                    parts.query,
                    "",
                )
            )
        )
        return result.raise_for_status().json()


async def discover_issuer() -> str:
    """Validating server intended mcp resource URL, issuer URL against client configured URLs"""
    mcp_url = get_config("MCP_URL")
    meta = await protect_resource_metadata(mcp_url)

    if meta["resource"] != mcp_url:
        raise ValueError(
            f"configured mcp_url: {mcp_url} does not match with mcp's protected resource metadata {meta['resource']}"
        )

    allowed_issuer = get_config("ALLOWED_ISSUER_URL")
    issuer: list[str] = meta.get("authorization_servers", [])

    if len(issuer) == 0 or issuer[0] != allowed_issuer:
        raise ValueError(
            f"configured allowed issuer: {allowed_issuer} does not match with mcp's protected resource metadata's issuers: {issuer}"
        )

    return issuer[0]
