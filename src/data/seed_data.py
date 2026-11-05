from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SeedData:
    """Canned guard-rail replies (off-topic query / no results found)."""

    _OFF_TOPIC_REPLY = (
        "I'm a news search assistant and can only help with news-related queries. "
        "Please ask me about topics such as technology, politics, sports, business, "
        "science, health, climate, or world events."
    )

    _NO_NEWS_FOUND_REPLY = (
        "I searched for news articles matching your query but couldn't find any relevant results. "
        "Try broadening your search terms or asking about a different topic."
    )
