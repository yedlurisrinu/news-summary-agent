"""PostgreSQL adapter — the async connection pool that backs the LangGraph checkpointer."""

import os
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from config.application_config import get_config


def postgres_pool() -> AsyncConnectionPool[AsyncConnection[dict[str, Any]]]:
    """Build the (unopened) connection pool for ``AsyncPostgresSaver``.

    Host and port come from application config (``PG_HOST``, ``PG_PORT``); database,
    user and password from env vars (``POSTGRES_DB``, ``POSTGRES_USER``,
    ``POSTGRES_PASSWORD``) loaded from Vault at startup. Connections are autocommit
    with ``dict_row`` rows and no prepared statements, as the checkpointer requires.
    The caller opens and closes it (``async with pool:``) in the FastAPI lifespan.
    """
    return AsyncConnectionPool(
        kwargs={
            "host": get_config("PG_HOST"),
            "port": get_config("PG_PORT"),
            "dbname": os.environ["POSTGRES_DB"],
            "user": os.environ["POSTGRES_USER"],
            "password": os.environ["POSTGRES_PASSWORD"],
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
        min_size=int(get_config("PG_POOL_MIN_SIZE")),
        max_size=int(get_config("PG_POOL_MAX_SIZE")),
        open=False,
    )
