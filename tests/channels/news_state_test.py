"""
tests/channels/news_state_test.py

Unit tests for the state reducers in channels/news_state.py:
  - merge_article_ref   — dedupe by id, keep higher score, stable sort, purity
  - merge_article_hits  — append semantics, input immutability
  - append              — timing-dict merge
"""

from __future__ import annotations


class TestMergeArticleRef:
    def test_dedupe_keeps_higher_score(self):
        from channels.news_state import merge_article_ref

        current = [{"article_id": "id_1", "score": 0.7}]
        update = [{"article_id": "id_1", "score": 0.8}]
        merged = merge_article_ref(current, update)
        assert merged == [{"article_id": "id_1", "score": 0.8}]

    def test_lower_update_score_ignored(self):
        from channels.news_state import merge_article_ref

        current = [{"article_id": "id_1", "score": 0.9}]
        update = [{"article_id": "id_1", "score": 0.3}]
        merged = merge_article_ref(current, update)
        assert merged == [{"article_id": "id_1", "score": 0.9}]

    def test_sorted_by_score_desc_then_id(self):
        from channels.news_state import merge_article_ref

        current = [{"article_id": "b", "score": 0.5}]
        update = [
            {"article_id": "a", "score": 0.5},
            {"article_id": "c", "score": 0.9},
        ]
        merged = merge_article_ref(current, update)
        assert [r["article_id"] for r in merged] == ["c", "a", "b"]

    def test_pure_does_not_mutate_inputs(self):
        from channels.news_state import merge_article_ref

        current = [{"article_id": "id_1", "score": 0.7}]
        update = [{"article_id": "id_2", "score": 0.9}]
        merge_article_ref(current, update)
        assert current == [{"article_id": "id_1", "score": 0.7}]
        assert update == [{"article_id": "id_2", "score": 0.9}]


class TestMergeArticleHits:
    def test_appends_update(self):
        from channels.news_state import merge_article_hits

        current = [{"title": "A", "source": "S", "published_at": "d", "link": "l"}]
        update = [{"title": "B", "source": "S", "published_at": "d", "link": "l"}]
        merged = merge_article_hits(current, update)
        assert [h["title"] for h in merged] == ["A", "B"]

    def test_does_not_mutate_current(self):
        from channels.news_state import merge_article_hits

        current = [{"title": "A", "source": "S", "published_at": "d", "link": "l"}]
        merge_article_hits(
            current, [{"title": "B", "source": "S", "published_at": "d", "link": "l"}]
        )
        assert len(current) == 1


class TestAppend:
    def test_merges_dicts(self):
        from channels.news_state import append

        assert append({"A": 1}, {"B": 2}) == {"A": 1, "B": 2}

    def test_right_wins_on_collision(self):
        from channels.news_state import append

        assert append({"A": 1}, {"A": 9}) == {"A": 9}
