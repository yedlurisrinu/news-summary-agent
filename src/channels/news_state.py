import copy
from typing import TypedDict, NotRequired, Annotated
from channels.article_reference import ArticleReference
from channels.article_hit import ArticleHit


def merge_article_hits(
    current: list[ArticleHit], update: list[ArticleHit]
) -> list[ArticleHit]:
    """Reducer: append new article hits to existing (no dedupe — refs are pre-deduped)."""
    # TODO see if we need to do dedupe as of now it is not required
    merged = copy.deepcopy(current)
    for articleHit in update:
        merged.append(articleHit)
    return merged


def merge_article_ref(
    current: list[ArticleReference], update: list[ArticleReference]
) -> list[ArticleReference]:
    """Reducer: dedupe refs by article_id keeping the higher score; return a stably-sorted,
    order-independent list (pure — inputs are not mutated)."""
    # dedupe by article_id;
    # when both sides have the same ID, keep the higher score;
    # return a list sorted so the result is order-independent, with a deterministic tie-break;
    # pure: don't mutate either input. ArticleRef is a dataclass, so it's unhashable by default; key by ID, not by object.
    # get dictionaries
    # current = [{"article_id": "id_1", "score": 0.7}, {"article_id": "id_2", "score": 0.9}]
    # update  = [{"article_id": "id_1", "score": 0.8}, {"article_id": "id_3", "score": 0.5}, {"article_id": "id_0", "score": 0.8}]
    merged_by_id: dict[str, ArticleReference] = {}

    for record in current:
        article_id = record["article_id"]
        merged_by_id[article_id] = {"article_id": article_id, "score": record["score"]}

    for update_record in update:
        article_id = update_record["article_id"]
        if (
            article_id not in merged_by_id
            or update_record["score"] > merged_by_id[article_id]["score"]
        ):
            merged_by_id[article_id] = {
                "article_id": article_id,
                "score": update_record["score"],
            }

    return sorted(
        merged_by_id.values(), key=lambda r: (-float(r["score"]), r["article_id"])
    )


def append(current: dict[str, int], additional: dict[str, int]) -> dict[str, int]:
    """Reducer: merge two timing dicts (right-hand value wins on key collision)."""
    appended = current | additional
    return appended


class NewsState(TypedDict):
    """Shared LangGraph state threaded through classify → search → summarize.

    Fields are written incrementally by nodes; ``Annotated`` fields use reducers
    to merge concurrent/sequential updates:
      query            — original user query (always present)
      is_news_query    — guard-rail #1 flag from ClassifyIntent
      fetch_latest     — recency intent
      categories       — extracted news categories
      article_refs     — (article_id, score) refs; merged + deduped by merge_article_ref
      summary          — final LLM summary text
      article_hits     — displayable article metadata; appended by merge_article_hits
      duration_seconds — per-stage timing; merged by append
    """

    query: str
    is_news_query: NotRequired[bool]
    fetch_latest: NotRequired[bool]
    categories: NotRequired[list[str]]
    article_refs: Annotated[list[ArticleReference], merge_article_ref]
    summary: NotRequired[str]
    article_hits: Annotated[list[ArticleHit], merge_article_hits]
    duration_seconds: Annotated[NotRequired[dict[str, float]], append]
