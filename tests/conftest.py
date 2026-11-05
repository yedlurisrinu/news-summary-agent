"""
conftest.py — pytest fixtures shared across all test modules.

Provides:
  - A session-wide real application config load (nodes and auth read keys from it).
  - ``anyio_backend``: async tests run under anyio's pytest plugin
    (``@pytest.mark.anyio``) on asyncio.
  - A factory for a fake chat model (FakeListChatModel) and a fake es-mcp-server
    (see ``tests/fake_mcp.py``), so nodes and the graph run with no external services.
"""

from __future__ import annotations

import pytest

from tests.fake_mcp import FakeNewsMcp


@pytest.fixture(scope="session", autouse=True)
def _load_app_config():
    """Load the real resources/application-config.yaml once for the whole test session."""
    from config.application_config import load_config

    load_config()


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def make_llm():
    """Return a factory building a FakeListChatModel from canned responses.

    The fake model is a real Runnable, so ``prompt | llm | parser`` works; it
    cycles through ``responses`` in order (JSON for classify, prose for summary).
    """
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    def _make(responses: list[str]):
        return FakeListChatModel(responses=responses)

    return _make


@pytest.fixture()
def fake_mcp() -> FakeNewsMcp:
    """An in-memory es-mcp-server stand-in; use ``fake_mcp.client()`` for the nodes."""
    return FakeNewsMcp()
