"""
tests/adapters/llm_test.py

Tests for adapters/llm.py get_chat_model():
  - model, temperature and timeout are forwarded to init_chat_model
  - lru_cache yields a singleton
"""

from __future__ import annotations

from unittest.mock import patch


def _clear_cache():
    from adapters.llm import get_chat_model

    get_chat_model.cache_clear()


class TestGetChatModel:
    def test_args_forwarded_and_cached(self):
        _clear_cache()
        sentinel = object()
        with patch("adapters.llm.init_chat_model", return_value=sentinel) as init:
            from adapters.llm import get_chat_model

            first = get_chat_model("openai:gpt-4o-mini", 0.0, 10)
            second = get_chat_model("openai:gpt-4o-mini", 0.0, 10)

        assert first is sentinel
        assert first is second
        init.assert_called_once_with(
            model="openai:gpt-4o-mini", temperature=0.0, timeout=10
        )

    def teardown_method(self):
        _clear_cache()
