class ServiceTokenUnavailable(Exception): ...


import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from authlib.integrations.httpx_client import AsyncOAuth2Client

from config.application_config import get_config

log = logging.getLogger(__name__)


class ServiceAccountTokenProvider:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_endpoint: Callable[[], Awaitable[str]],
        http_transport: Any | None = None,
    ) -> None:
        self._client_id = client_id  # comes from app config
        self._client_secret = client_secret  # comes from vault
        self._token_endpoint = token_endpoint
        self._http_transport = http_transport
        self._token: dict[str, Any] | None = None
        self._lock = asyncio.Lock()

    async def token(self, force: bool = False) -> Any:
        async with self._lock:
            if (
                not force
                and self._token
                and self._token["expires_at"] - time.time()
                > get_config("SERVICE_TOKEN_LEEWAY_SEC")
            ):
                return self._token["access_token"]
            try:
                url = await self._token_endpoint()
                async with AsyncOAuth2Client(
                    client_id=self._client_id,
                    client_secret=self._client_secret,
                    token_endpoint_auth_method="client_secret_basic",
                    transport=self._http_transport,
                ) as http:
                    self._token = await http.fetch_token(
                        url, grant_type="client_credentials"
                    )

            except Exception as exc:
                log.exception("service token fetch failed")
                raise ServiceTokenUnavailable from exc

            log.info(f"service token fetched, expires_in: {self._token['expires_in']}")
            return self._token["access_token"]
