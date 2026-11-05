"""Chat-model adapter — builds the LangChain chat model used by every graph node."""

from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


@lru_cache(maxsize=1)
def get_chat_model(model: str, temp: float, timeout: int) -> BaseChatModel:
    """Return a process-wide singleton chat model.

    Cached via ``lru_cache`` so every node shares one client. The values come from
    application config (``LLM_MODEL``, ``LLM_TEMPERATURE``, ``LLM_TIMEOUT_SEC``).

    Args:
        model: LangChain ``provider:model`` identifier, e.g. ``openai:gpt-4o-mini``.
        temp: Sampling temperature (0.0 = deterministic).
        timeout: Maximum time (in seconds) to wait for a response.
    """
    return init_chat_model(model=model, temperature=temp, timeout=timeout)
