from typing import NotRequired, TypedDict


class Latency(TypedDict):
    """Seconds spent in each external call of one graph run (time.perf_counter deltas).

    A key is present only when its call ran: e.g. an off-topic query has only
    ``classify_llm_call``; a failed mget has no ``summary_llm_call``.
    """

    classify_llm_call: NotRequired[float]
    mcp_category_search_call: NotRequired[float]
    mcp_article_search_call: NotRequired[float]
    mcp_mget_call: NotRequired[float]
    summary_llm_call: NotRequired[float]
