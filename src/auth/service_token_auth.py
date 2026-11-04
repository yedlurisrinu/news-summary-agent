from collections.abc import AsyncGenerator

from auth.service_account_token import ServiceAccountTokenProvider
from httpx2 import Auth, Request, Response

BEARER = "Bearer"


class ServiceTokenAuth(Auth):
    def __init__(self, provider: ServiceAccountTokenProvider):
        self._provider = provider

    async def async_auth_flow(
        self, request: Request
    ) -> AsyncGenerator[Request, Response]:  # an async generator
        request.headers["Authorization"] = f"{BEARER} {await self._provider.token()}"
        response = yield request  # httpx sends it, hands back the response

        if response.status_code == 401:
            request.headers["Authorization"] = (
                f"{BEARER} {await self._provider.token(force=True)}"
            )
            yield request  # one retry only, then the caller sees the result
