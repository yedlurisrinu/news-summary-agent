"""Small helpers for the judge runs: the judge chat model and interval formatting."""

import math

from langchain_openai import ChatOpenAI
from openai.types.shared import (
    ReasoningEffort,
)  # Literal['none','minimal','low','medium','high','xhigh','max'] | None


def fmt_interval(lo: float, hi: float, places: int = 4) -> str:
    """Format an interval clamped to [0, 1], rounding outwards to `places` decimals."""
    k = 10**places
    lo, hi = max(0.0, lo), min(1.0, hi)
    return f"[{math.floor(lo * k) / k:.{places}f}, {math.ceil(hi * k) / k:.{places}f}]"


async def get_model(
    model: str,
    timeout: int,
    max_retries: int,
    temperature: float,
    reasoning_effort: ReasoningEffort,
) -> ChatOpenAI:
    """Build the ChatOpenAI judge model. temperature is accepted but not sent: the
    gpt-6.1 judge models only support the default temperature.
    """
    return ChatOpenAI(
        model=model,
        timeout=timeout,
        max_retries=max_retries,
        # temperature=temperature,  'temperature' does not support 0.0 with this model (gpt-6.1-terra). Only the default (1) value is supported.
        reasoning_effort=reasoning_effort,
    )
