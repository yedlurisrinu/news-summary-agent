import asyncio
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from hvac import Client
from hvac.exceptions import InvalidPath

from auth.ports import StoredSession

logger = logging.getLogger(__name__)


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

    def _sweep_sync(self, max_age_sec: int) -> int:
        try:
            listed = self._client.secrets.kv.v2.list_secrets(
                path=self._path, mount_point="secret"
            )
        except InvalidPath:
            return 0  # no sessions stored
        now = datetime.now(UTC)
        removed = 0
        for sid in listed["data"]["keys"]:
            if sid.endswith("/"):
                continue  # not a session entry
            meta = self._client.secrets.kv.v2.read_secret_metadata(
                path=f"{self._path}/{sid}", mount_point="secret"
            )["data"]
            # Vault returns e.g. 2026-11-04T19:28:26.123456789Z (nanoseconds): keep seconds precision
            updated = datetime.strptime(
                meta["updated_time"][:19], "%Y-%m-%dT%H:%M:%S"
            ).replace(tzinfo=UTC)
            if (now - updated).total_seconds() > max_age_sec:
                self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=f"{self._path}/{sid}", mount_point="secret"
                )
                removed += 1
        return removed

    async def sweep(self, max_age_sec: int) -> int:
        """Delete stored sessions not updated for longer than ``max_age_sec``; return how many.

        Entries pile up when a cookie simply expires (the app is never told). Use a cut-off
        above Keycloak's SSO session maximum: such a refresh token can never be used again,
        so a session that could still be refreshed is never removed.
        """
        return await asyncio.to_thread(self._sweep_sync, max_age_sec)
