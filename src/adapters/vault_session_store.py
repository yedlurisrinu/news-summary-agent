import asyncio
from dataclasses import asdict
from typing import Any

from hvac import Client
from auth.ports import StoredSession
from hvac.exceptions import InvalidPath
import logging

logger = logging.getLogger(__file__)


class VaultRefreshTokenStore:
    # Path will be coming from application config path and token will be from environment path
    def __init__(self, addr: str, token: str, path: str) -> None:
        self._client: Client = Client(url=addr, token=token)
        if not self._client.is_authenticated():
            raise ValueError(
                "Vault authentication failed - check token information for VAULT_TOKEN variable"
            )
        if not path:
            raise KeyError(f" config secret path is missing {path}")

        self._path = path

    async def save(self, sid: str, session: StoredSession) -> None:
        await asyncio.to_thread(
            self._client.secrets.kv.v2.create_or_update_secret,
            path=f"{self._path}/{sid}",
            secret=asdict(session),
            mount_point="secret",
        )

    async def get(self, sid: str) -> StoredSession | None:
        try:
            response: dict[str, Any] = await asyncio.to_thread(
                self._client.secrets.kv.v2.read_secret_version,
                path=f"{self._path}/{sid}",
                mount_point="secret",
                raise_on_deleted_version=True,
            )
            return StoredSession(**response["data"]["data"])
        except InvalidPath as e:
            logger.error(f"Provided path : {self._path}/{sid[:8]}, not found : {e}")
        return None

    async def delete(self, sid: str) -> None:
        await asyncio.to_thread(
            self._client.secrets.kv.v2.delete_metadata_and_all_versions,
            path=f"{self._path}/{sid}",
            mount_point="secret",
        )
