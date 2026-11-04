from typing import Any

from auth.ports import DiscoverIssuer
from config.application_config import get_config
import asyncio
from httpx2 import AsyncClient
from data.oidc_metadata import OIDCMetadata, REQUIRED


class AuthServerMetadata:
    def __init__(
        self, discover_issuer: DiscoverIssuer, http_transport: Any | None = None
    ) -> None:
        self._discover_issuer = discover_issuer
        self._http_transport = http_transport
        self._meta: OIDCMetadata | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> OIDCMetadata:
        if self._meta is not None:
            return self._meta

        async with self._lock:
            issuer = await self._discover_issuer()
            async with AsyncClient(transport=self._http_transport) as http:
                response = await http.get(
                    url=f"{issuer}{get_config('AS_WELL_KNOWN_URL')}"
                )
                meta: OIDCMetadata = (
                    response.raise_for_status().json()
                )  # annotate: Any → OIDCMetadata

                for key in REQUIRED:
                    if not isinstance(meta.get(key), str):
                        raise ValueError(
                            f"AS metadata is missing {key}"
                        )  # check at the boundary
            if meta["issuer"] != issuer:
                raise ValueError(
                    f" MCP server issuer {issuer} is not same as issuer from AS {meta['issuer']}"
                )
            self._meta = meta
        return meta

    async def issuer(self) -> str:
        return (await self.get())["issuer"]

    async def token_endpoint(self) -> str:
        return (await self.get())["token_endpoint"]
