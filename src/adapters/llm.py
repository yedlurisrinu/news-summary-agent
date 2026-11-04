"""Chat-model adapter — builds the LangChain chat model used by every graph node."""

from functools import lru_cache
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


@lru_cache(maxsize=1)
async def get_chat_model(
    model: str = "openai:gpt-4o-mini", temp: float = 0.0, timeout: int = 10
) -> BaseChatModel:
    """Return a process-wide singleton chat model.

    Cached via ``lru_cache`` so every node shares one client. Defaults to
    ``openai:gpt-4o-mini`` at temperature 0.0 for deterministic output.

    Args:
        model: LangChain ``provider:model`` identifier.
        temp: Sampling temperature (0.0 = deterministic).
        timeout: Maximum time (in seconds) to wait for a response.
    """
    return init_chat_model(model=model, temperature=temp, timeout=timeout)
