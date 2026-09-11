"""
Tests for Corrective-RAG (CRAG) Agent

Reference: Yan et al., arXiv 2401.15884
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.crag_agent import (
    GradedDocument,
    GradingResult,
    RetrievalGrade,
    _calculate_consensus,
    _classify_score,
    crag_agent,
    execute_web_fallback,
    expand_query,
    grade_retrieval,
    grade_single_document,
)


class TestRetrievalGrade:
    """Test RetrievalGrade enum and helper functions."""

    def test_classify_score_relevant(self):
        """Score >= 0.7 should be RELEVANT."""
        assert _classify_score(0.7) == RetrievalGrade.RELEVANT
        assert _classify_score(0.85) == RetrievalGrade.RELEVANT
        assert _classify_score(1.0) == RetrievalGrade.RELEVANT

    def test_classify_score_partial(self):
        """Score 0.4-0.69 should be PARTIAL."""
        assert _classify_score(0.4) == RetrievalGrade.PARTIAL
        assert _classify_score(0.5) == RetrievalGrade.PARTIAL
        assert _classify_score(0.69) == RetrievalGrade.PARTIAL

    def test_classify_score_irrelevant(self):
        """Score < 0.4 should be IRRELEVANT."""
        assert _classify_score(0.0) == RetrievalGrade.IRRELEVANT
        assert _classify_score(0.2) == RetrievalGrade.IRRELEVANT
        assert _classify_score(0.39) == RetrievalGrade.IRRELEVANT

    def test_calculate_consensus(self):
        """Test consensus calculation."""
        # High consensus (similar scores)
        high = _calculate_consensus([0.8, 0.85, 0.82])
        assert high > 0.5  # Should be relatively high

        # Low consensus (different scores)
        low = _calculate_consensus([0.1, 0.9, 0.5])
        assert low < high  # Lower than high consensus

        # Single item = perfect consensus
        assert _calculate_consensus([0.5]) == 1.0

        # Two identical items = perfect consensus
        assert _calculate_consensus([0.5, 0.5]) == 1.0


class TestGradeRetrieval:
    """Test the main grade_retrieval function."""

    @pytest.fixture
    def sample_docs(self):
        return [
            {"id": "doc1", "content": "Transformers use self-attention mechanisms."},
            {"id": "doc2", "content": "The weather today is sunny and warm."},
            {"id": "doc3", "content": "BERT is a transformer-based language model."},
        ]

    @pytest.mark.asyncio
    async def test_grade_retrieval_empty_docs(self):
        """Empty document list should return empty result."""
        result = await grade_retrieval([], "What are transformers?")

        assert result.graded_documents == []
        assert result.needs_web_fallback is False
        assert result.relevant_count == 0

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.grade_single_document")
    async def test_grade_retrieval_all_relevant(self, mock_grade_single, sample_docs):
        """When all docs are relevant, should not trigger fallback."""
        # Mock grade_single_document to return RELEVANT for all docs
        mock_grade_single.side_effect = [
            GradedDocument("doc1", 0.85, RetrievalGrade.RELEVANT, "good", "local"),
            GradedDocument("doc2", 0.85, RetrievalGrade.RELEVANT, "good", "local"),
            GradedDocument("doc3", 0.85, RetrievalGrade.RELEVANT, "good", "local"),
        ]

        result = await grade_retrieval(sample_docs, "What are transformers?")

        assert len(result.graded_documents) == 3
        assert result.needs_web_fallback is False
        assert result.relevant_count == 3

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.grade_single_document")
    async def test_grade_retrieval_mixed_relevance(self, mock_grade_single, sample_docs):
        """When some docs are irrelevant, may trigger fallback."""
        # Mock: 2 RELEVANT, 1 IRRELEVANT = 66% usable (above 50% threshold)
        mock_grade_single.side_effect = [
            GradedDocument("doc1", 0.85, RetrievalGrade.RELEVANT, "good", "local"),
            GradedDocument("doc2", 0.2, RetrievalGrade.IRRELEVANT, "bad", "local"),
            GradedDocument("doc3", 0.8, RetrievalGrade.RELEVANT, "good", "local"),
        ]

        result = await grade_retrieval(sample_docs, "What are transformers?")

        assert len(result.graded_documents) == 3
        # 2/3 docs are RELEVANT = 66% usable (above 50% threshold)
        assert result.relevant_count == 2
        assert result.irrelevant_count == 1
        assert result.needs_web_fallback is False

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent._get_client")
    async def test_grade_retrieval_low_relevance_triggers_fallback(self, mock_get_client, sample_docs):
        """When most docs are irrelevant, should trigger fallback."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # All docs irrelevant = 0% usable < 50% threshold
        mock_response.choices = [
            MagicMock(message=MagicMock(
                content='{"score": 0.2, "classification": "irrelevant", "reasoning": "Unrelated"}'
            ))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await grade_retrieval(sample_docs, "What is quantum computing?")

        assert result.needs_web_fallback is True
        assert result.irrelevant_count == 3
        assert result.fallback_reason is not None


class TestGradeSingleDocument:
    """Test grading of individual documents."""

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent._get_client")
    async def test_grade_single_document_success(self, mock_get_client):
        """Successful grading returns graded document."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(
                content='{"score": 0.85, "classification": "relevant", "reasoning": "Contains relevant info", "key_information": null}'
            ))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await grade_single_document(
            doc_id="test_doc",
            content="Transformers are important.",
            query="What are transformers?",
        )

        assert result.doc_id == "test_doc"
        assert result.score == 0.85
        assert result.grade == RetrievalGrade.RELEVANT
        assert result.source == "local"

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent._get_client")
    async def test_grade_single_document_error_fallback(self, mock_get_client):
        """Grading error should return IRRELEVANT as fail-safe."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        mock_get_client.return_value = mock_client

        result = await grade_single_document(
            doc_id="test_doc",
            content="Some content",
            query="Some query",
        )

        assert result.grade == RetrievalGrade.IRRELEVANT
        assert result.score == 0.0
        assert "failed" in result.reasoning.lower()


class TestExpandQuery:
    """Test query expansion for web search."""

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent._get_client")
    async def test_expand_query_success(self, mock_get_client):
        """Successful query expansion returns expanded query."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="transformer self-attention architecture"))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await expand_query(
            "What are transformers?",
            "Previous context about neural networks"
        )

        assert result == "transformer self-attention architecture"
        assert len(result) < 100  # Should be concise

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent._get_client")
    async def test_expand_query_fallback_on_error(self, mock_get_client):
        """On error, should return original query."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
        mock_get_client.return_value = mock_client

        original = "What are transformers?"
        result = await expand_query(original, "context")

        assert result == original

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent._get_client")
    async def test_expand_query_invalid_response(self, mock_get_client):
        """Invalid response should fall back to original query."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Empty response
        mock_response.choices = [MagicMock(message=MagicMock(content=""))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        original = "What are transformers?"
        result = await expand_query(original, "context")

        assert result == original


class TestCRAGAgent:
    """Test CRAG agent as LangGraph node."""

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.settings")
    async def test_crag_disabled(self, mock_settings):
        """When CRAG is disabled, should skip processing."""
        mock_settings.CRAG_ENABLED = False

        state = {"query": "test query", "reranked_chunks": [{"id": "doc1", "content": "test"}]}

        result = await crag_agent(state)

        assert result["crag_trace"].get("enabled") is False

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.settings")
    async def test_crag_no_documents(self, mock_settings):
        """When no documents, should skip processing."""
        mock_settings.CRAG_ENABLED = True

        state = {"query": "test query", "reranked_chunks": []}

        result = await crag_agent(state)

        assert result["crag_trace"].get("no_documents") is True

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.grade_retrieval")
    @patch("app.agents.crag_agent.settings")
    async def test_crag_passes_local_retrieval(self, mock_settings, mock_grade):
        """When local retrieval passes, no web fallback needed."""
        mock_settings.CRAG_ENABLED = True
        mock_settings.CRAG_FALLBACK_THRESHOLD = 0.5
        mock_settings.CRAG_MAX_WEB_RESULTS = 10

        # Mock: 2/3 docs relevant
        mock_grade.return_value = GradingResult(
            graded_documents=[
                GradedDocument("doc1", 0.85, RetrievalGrade.RELEVANT, "good", "local"),
                GradedDocument("doc2", 0.8, RetrievalGrade.RELEVANT, "good", "local"),
                GradedDocument("doc3", 0.3, RetrievalGrade.IRRELEVANT, "bad", "local"),
            ],
            needs_web_fallback=False,
            relevant_count=2,
            partial_count=0,
            irrelevant_count=1,
        )

        state = {
            "query": "test",
            "rewritten_query": "test",
            "reranked_chunks": [
                {"id": "doc1", "content": "content1"},
                {"id": "doc2", "content": "content2"},
                {"id": "doc3", "content": "content3"},
            ],
        }

        result = await crag_agent(state)

        assert result["crag_trace"]["enabled"] is True
        assert result["agent_trace"]["crag"]["mode"] == "local_pass"

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.execute_web_fallback")
    @patch("app.agents.crag_agent.grade_retrieval")
    @patch("app.agents.crag_agent.settings")
    async def test_crag_triggers_web_fallback(self, mock_settings, mock_grade, mock_fallback):
        """When local retrieval fails, should trigger web fallback."""
        mock_settings.CRAG_ENABLED = True
        mock_settings.CRAG_FALLBACK_THRESHOLD = 0.5
        mock_settings.CRAG_MAX_WEB_RESULTS = 10

        # First call: low relevance (triggers fallback)
        mock_grade.return_value = GradingResult(
            graded_documents=[
                GradedDocument("doc1", 0.2, RetrievalGrade.IRRELEVANT, "bad", "local"),
                GradedDocument("doc2", 0.15, RetrievalGrade.IRRELEVANT, "bad", "local"),
            ],
            needs_web_fallback=True,
            fallback_reason="Low relevance",
            relevant_count=0,
            partial_count=0,
            irrelevant_count=2,
        )

        # Second call: re-grading after web fallback
        mock_grade.side_effect = [
            GradingResult(
                graded_documents=[
                    GradedDocument("doc1", 0.2, RetrievalGrade.IRRELEVANT, "bad", "local"),
                    GradedDocument("doc2", 0.15, RetrievalGrade.IRRELEVANT, "bad", "local"),
                ],
                needs_web_fallback=True,
                relevant_count=0,
                partial_count=0,
                irrelevant_count=2,
            ),
            GradingResult(
                graded_documents=[
                    GradedDocument("doc1", 0.24, RetrievalGrade.IRRELEVANT, "bad", "local"),
                    GradedDocument("doc2", 0.18, RetrievalGrade.IRRELEVANT, "bad", "local"),
                    GradedDocument("web_0", 0.7, RetrievalGrade.RELEVANT, "good", "web"),
                ],
                needs_web_fallback=False,
                relevant_count=1,
                partial_count=0,
                irrelevant_count=2,
            ),
        ]

        mock_fallback.return_value = MagicMock(
            merged_documents=[
                {"id": "web_0", "content": "web content", "metadata": {"source": "web"}},
            ],
            search_query_used="expanded query",
            domains_included=["example.com"],
        )

        state = {
            "query": "test",
            "rewritten_query": "test",
            "reranked_chunks": [
                {"id": "doc1", "content": "content1"},
                {"id": "doc2", "content": "content2"},
            ],
        }

        result = await crag_agent(state)

        assert result["crag_trace"]["enabled"] is True
        assert result["crag_trace"]["web_fallback"]["triggered"] is True
        assert result["agent_trace"]["crag"]["mode"] == "web_fallback"
        mock_fallback.assert_called_once()


class TestWebFallback:
    """Test web fallback functionality."""

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.tavily_search")
    @patch("app.agents.crag_agent.expand_query")
    @patch("app.agents.crag_agent.settings")
    async def test_web_fallback_with_tavily(self, mock_settings, mock_expand, mock_tavily):
        """Test web fallback with Tavily API."""
        mock_settings.TAVILY_API_KEY = "test-key"

        mock_expand.return_value = "expanded query"
        mock_tavily.return_value = []

        result = await execute_web_fallback("test query")

        assert result.search_query_used == "expanded query"
        mock_tavily.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.duckduckgo_search")
    @patch("app.agents.crag_agent.expand_query")
    @patch("app.agents.crag_agent.settings")
    async def test_web_fallback_duckduckgo_fallback(self, mock_settings, mock_expand, mock_ddg):
        """Test fallback to DuckDuckGo when Tavily is not configured."""
        mock_settings.TAVILY_API_KEY = ""

        mock_expand.return_value = "expanded query"
        mock_ddg.return_value = []

        result = await execute_web_fallback("test query")

        assert result.search_query_used == "expanded query"
        mock_ddg.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.tavily_search")
    @patch("app.agents.crag_agent.expand_query")
    @patch("app.agents.crag_agent.settings")
    async def test_web_fallback_filters_blocked_domains(self, mock_settings, mock_expand, mock_tavily):
        """Test that blocked domains are filtered from results."""
        from app.agents.crag_agent import WebSearchResult

        mock_settings.TAVILY_API_KEY = "test-key"
        mock_expand.return_value = "test query"

        # Tavily returns a blocked domain
        mock_tavily.return_value = [
            WebSearchResult(
                url="https://spam-site.org/page",
                title="Spam",
                content="Spam content",
                score=0.9,
                domain="spam-site.org",
            ),
            WebSearchResult(
                url="https://example.com/page",
                title="Good",
                content="Good content",
                score=0.8,
                domain="example.com",
            ),
        ]

        result = await execute_web_fallback("test query")

        assert len(result.web_documents) == 1
        assert result.web_documents[0].domain == "example.com"
        assert "spam-site.org" in result.domains_excluded


class TestIntegration:
    """Integration tests for CRAG pipeline."""

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.grade_retrieval")
    @patch("app.agents.crag_agent.settings")
    async def test_crag_adds_scores_to_documents(self, mock_settings, mock_grade):
        """Test that CRAG adds scores to documents."""
        mock_settings.CRAG_ENABLED = True
        mock_settings.CRAG_FALLBACK_THRESHOLD = 0.5

        mock_grade.return_value = GradingResult(
            graded_documents=[
                GradedDocument("doc1", 0.85, RetrievalGrade.RELEVANT, "good", "local"),
                GradedDocument("doc2", 0.3, RetrievalGrade.IRRELEVANT, "bad", "local"),
            ],
            needs_web_fallback=False,
            relevant_count=1,
            partial_count=0,
            irrelevant_count=1,
        )

        state = {
            "rewritten_query": "test",
            "reranked_chunks": [
                {"id": "doc1", "content": "good content"},
                {"id": "doc2", "content": "bad content"},
            ],
        }

        result = await crag_agent(state)

        # Check scores are added
        assert "crag_score" in result["reranked_chunks"][0]
        assert "crag_grade" in result["reranked_chunks"][0]
        assert result["reranked_chunks"][0]["crag_grade"] == "relevant"
        assert result["reranked_chunks"][1]["crag_grade"] == "irrelevant"

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.execute_web_fallback")
    @patch("app.agents.crag_agent.grade_retrieval")
    @patch("app.agents.crag_agent.settings")
    async def test_crag_web_fallback_adds_weighted_scores(self, mock_settings, mock_grade, mock_fallback):
        """Test that web fallback applies source weighting."""
        mock_settings.CRAG_ENABLED = True
        mock_settings.CRAG_FALLBACK_THRESHOLD = 0.5
        mock_settings.CRAG_MAX_WEB_RESULTS = 10

        # First call: trigger fallback
        mock_grade.side_effect = [
            GradingResult(
                graded_documents=[
                    GradedDocument("doc1", 0.2, RetrievalGrade.IRRELEVANT, "bad", "local"),
                ],
                needs_web_fallback=True,
                relevant_count=0,
                partial_count=0,
                irrelevant_count=1,
            ),
            GradingResult(
                graded_documents=[
                    GradedDocument("doc1", 0.2, RetrievalGrade.IRRELEVANT, "bad", "local"),
                    GradedDocument("web_0", 0.7, RetrievalGrade.RELEVANT, "good", "web"),
                ],
                needs_web_fallback=False,
                relevant_count=1,
                partial_count=0,
                irrelevant_count=1,
            ),
        ]

        mock_fallback.return_value = MagicMock(
            merged_documents=[
                {"id": "web_0", "content": "web content", "metadata": {"source": "web"}},
            ],
            search_query_used="query",
            domains_included=["web.com"],
        )

        state = {
            "rewritten_query": "test",
            "reranked_chunks": [{"id": "doc1", "content": "local", "metadata": {"source": "local"}}],
        }

        result = await crag_agent(state)

        # Check web document was added
        assert len(result["reranked_chunks"]) == 2

        # Check web doc has lower score (0.7 * 0.8 = 0.56)
        web_doc = next(d for d in result["reranked_chunks"] if d["id"] == "web_0")
        assert web_doc["crag_score"] == pytest.approx(0.56, rel=0.01)


class TestCRAGWebFallbackBudget:
    """Regression: the web-fallback merge overwrote `reranked_chunks` AFTER
    context_merge_agent had enforced MAX_GROUNDING_CHUNKS / CONTEXT_CHAR_BUDGET,
    so the answer prompt could receive ~15 full-page web docs with zero budget
    enforcement. The merged set must respect the same caps."""

    @pytest.mark.asyncio
    @patch("app.agents.crag_agent.execute_web_fallback")
    @patch("app.agents.crag_agent.grade_retrieval")
    @patch("app.agents.crag_agent.settings")
    async def test_web_fallback_respects_chunk_cap_and_char_budget(
        self, mock_settings, mock_grade, mock_fallback
    ):
        from app.agents.context_merge_agent import MAX_GROUNDING_CHUNKS

        mock_settings.CRAG_ENABLED = True
        mock_settings.CRAG_FALLBACK_THRESHOLD = 0.5
        mock_settings.CRAG_MAX_WEB_RESULTS = 10
        mock_settings.CONTEXT_CHAR_BUDGET = 24000

        huge = "w" * 20000
        web_docs = [
            {"id": f"web_{i}", "content": huge, "metadata": {"source": "web"}}
            for i in range(10)
        ]
        mock_grade.side_effect = [
            GradingResult(
                graded_documents=[
                    GradedDocument("doc1", 0.2, RetrievalGrade.IRRELEVANT, "bad", "local"),
                ],
                needs_web_fallback=True,
                relevant_count=0,
                partial_count=0,
                irrelevant_count=1,
            ),
            GradingResult(
                graded_documents=[
                    GradedDocument("doc1", 0.24, RetrievalGrade.IRRELEVANT, "bad", "local"),
                ]
                + [
                    GradedDocument(f"web_{i}", 0.7, RetrievalGrade.RELEVANT, "good", "web")
                    for i in range(10)
                ],
                needs_web_fallback=False,
                relevant_count=10,
                partial_count=0,
                irrelevant_count=1,
            ),
        ]
        mock_fallback.return_value = MagicMock(
            merged_documents=web_docs,
            search_query_used="query",
            domains_included=["web.com"],
        )

        state = {
            "query": "test",
            "rewritten_query": "test",
            "reranked_chunks": [
                {"id": "doc1", "content": "local", "metadata": {"source": "local"}}
            ],
        }

        result = await crag_agent(state)

        merged = result["reranked_chunks"]
        assert len(merged) <= MAX_GROUNDING_CHUNKS
        total_chars = sum(len(d.get("content") or "") for d in merged)
        assert total_chars <= mock_settings.CONTEXT_CHAR_BUDGET
