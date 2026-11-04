"""PostgreSQL adapter — raw connection used to back the LangGraph checkpointer."""

import os
from typing import Any, Coroutine
from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from config.application_config import get_config


async def connect_postgres() -> AsyncConnection[dict[str, Any]]:
    """Open an autocommit Postgres connection for the ``PostgresSaver`` checkpointer.

    Connection params come from env vars (``PG_HOST``, ``PG_PORT``, ``POSTGRES_DB``,
    ``POSTGRES_USER``, ``POSTGRES_PASSWORD``) loaded from Vault at startup. The
    returned object is used as a context manager in the FastAPI lifespan.
    """
    return await AsyncConnection.connect(
        host=get_config("PG_HOST"),
        port=get_config("PG_PORT"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
