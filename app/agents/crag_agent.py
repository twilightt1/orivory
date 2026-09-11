"""
Corrective-RAG (CRAG) Agent for Orivory v2.0

Implements the CRAG pattern from Yan et al. (arXiv 2401.15884):
- LLM-based retrieval quality assessment
- Automatic web search fallback when local retrieval fails
- Result re-ranking after fallback

Reference: https://arxiv.org/abs/2401.15884
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import httpx

# Shared client seam: tests patch <module>._get_client, which rebinds
# this module attribute and is picked up by all call sites below.
from app.agents.llm_client import get_llm_client as _get_client
from app.agents.llm_parsing import parse_llm_json_object
from app.config import settings
from app.observability.fallbacks import count_fallback

if TYPE_CHECKING:
    from app.agents.state import AgentState

log = logging.getLogger(__name__)

# ─── Grading Thresholds ───────────────────────────────────────────────────────

RELEVANT_SCORE_MIN = 0.7  # >= 0.7: RELEVANT
PARTIAL_SCORE_MIN = 0.4  # >= 0.4: PARTIAL
FALLBACK_THRESHOLD = 0.5  # % of docs that must be >= PARTIAL to avoid fallback

# ─── LLM Client ───────────────────────────────────────────────────────────────


# ─── Enums and Dataclasses ────────────────────────────────────────────────────


class RetrievalGrade(StrEnum):
    """Classification levels for retrieved document relevance."""
    RELEVANT = "relevant"  # >= 0.7 - Directly answers query
    PARTIAL = "partial"  # 0.4-0.69 - Contains useful context
    IRRELEVANT = "irrelevant"  # < 0.4 - Does not address query


@dataclass
class GradedDocument:
    """A single document with CRAG grading."""
    doc_id: str
    score: float  # 0.0 - 1.0
    grade: RetrievalGrade
    reasoning: str
    source: str  # "local" or "web"
    key_information: str | None = None  # Extracted relevant portion


@dataclass
class GradingResult:
    """Result from CRAG grading pipeline."""
    graded_documents: list[GradedDocument]
    needs_web_fallback: bool
    fallback_reason: str | None = None
    consensus_score: float = 0.0  # Agreement among grader on relevance
    relevant_count: int = 0
    partial_count: int = 0
    irrelevant_count: int = 0


@dataclass
class WebSearchResult:
    """Structured web search result."""
    url: str
    title: str
    content: str
    score: float
    published_date: str | None = None
    domain: str | None = None


@dataclass
class WebFallbackResult:
    """Result from web fallback pipeline."""
    web_documents: list[dict]
    merged_documents: list[dict]
    search_query_used: str
    domains_included: list[str]
    domains_excluded: list[str]


# ─── Prompt Templates ─────────────────────────────────────────────────────────

GRADE_DOCS_PROMPT = """You are a retrieval quality assessor for a research assistant.
Evaluate whether the retrieved document chunk helps answer the user's question.

Classification Definitions:
- "relevant": Document contains information DIRECTLY relevant to answering the question.
              Even if incomplete, the relevant portions can be used.
- "partial": Document MAY contain relevant information but:
              1. It's mixed with irrelevant content
              2. The relevant part requires extraction
              3. Confidence is low
- "irrelevant": Document CONTRADICTS the question's premise OR is about a
                completely different topic. Should NOT be used.

Question: {query}

Document Chunk:
{chunk_content}

Respond with JSON:
{{
    "score": 0.0-1.0,
    "classification": "relevant" | "partial" | "irrelevant",
    "reasoning": "brief explanation (1-2 sentences)",
    "key_information": "extracted relevant portion if partial/irrelevant" | null
}}"""

QUERY_EXPANSION_PROMPT = """Given this research query and the existing context, generate an optimized web search query.

Query: {query}

Existing context summary:
{existing_context}

Generate a search query that:
- Incorporates key terms from both query and context
- Uses precise terminology (not vague)
- Is suitable for web search (not too broad or narrow)
- Maximum 20 words

Return only the search query, nothing else."""


# ─── CRAG Grading Functions ───────────────────────────────────────────────────

def _classify_score(score: float) -> RetrievalGrade:
    """Classify a numeric score into a RetrievalGrade."""
    if score >= RELEVANT_SCORE_MIN:
        return RetrievalGrade.RELEVANT
    elif score >= PARTIAL_SCORE_MIN:
        return RetrievalGrade.PARTIAL
    else:
        return RetrievalGrade.IRRELEVANT


def _calculate_consensus(scores: list[float]) -> float:
    """
    Calculate consensus score based on score agreement.

    Uses coefficient of variation inverse: high agreement when scores are similar.
    Returns 0.0-1.0 where 1.0 = perfect agreement.
    """
    if len(scores) < 2:
        return 1.0

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std_dev = variance ** 0.5

    # Handle edge case where all scores are identical
    if std_dev < 1e-8:
        return 1.0

    # Coefficient of variation
    cv = std_dev / mean if mean > 0 else float('inf')

    # Convert CV to consensus: high CV = low consensus
    # Using exponential decay: consensus = exp(-cv)
    consensus = max(0.0, min(1.0, 1.0 / (1.0 + cv)))
    return consensus


async def grade_single_document(
    doc_id: str,
    content: str,
    query: str,
    source: str = "local",
) -> GradedDocument:
    """
    Grade a single document using LLM.

    Args:
        doc_id: Unique document identifier
        content: Document text content
        query: The user's search query
        source: Document source ("local" or "web")

    Returns:
        GradedDocument with score, grade, and reasoning
    """
    client = _get_client()

    # Truncate content to avoid token limits
    truncated_content = content[:2000] if len(content) > 2000 else content

    prompt = GRADE_DOCS_PROMPT.format(query=query, chunk_content=truncated_content)

    try:
        response = await client.chat.completions.create(
            model=settings.CRAG_GRADING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=500,
        )

        result = parse_llm_json_object(response.choices[0].message.content)

        if not result.ok:
            raise ValueError(f"Failed to parse LLM response: {result.error}")

        data = result.data
        score = float(data.get("score", 0.0) if data else 0.0)
        classification = data.get("classification", "irrelevant") if data else "irrelevant"

        # Map string classification to enum
        if classification == "relevant":
            grade = RetrievalGrade.RELEVANT
        elif classification == "partial":
            grade = RetrievalGrade.PARTIAL
        else:
            grade = RetrievalGrade.IRRELEVANT

        # Ensure score consistency with grade
        if grade == RetrievalGrade.RELEVANT and score < RELEVANT_SCORE_MIN:
            score = RELEVANT_SCORE_MIN
        elif grade == RetrievalGrade.PARTIAL and (score >= RELEVANT_SCORE_MIN or score < PARTIAL_SCORE_MIN):
            score = (RELEVANT_SCORE_MIN + PARTIAL_SCORE_MIN) / 2
        elif grade == RetrievalGrade.IRRELEVANT and score >= PARTIAL_SCORE_MIN:
            score = PARTIAL_SCORE_MIN - 0.1

        return GradedDocument(
            doc_id=doc_id,
            score=score,
            grade=grade,
            reasoning=data.get("reasoning", "") if data else "",
            source=source,
            key_information=data.get("key_information") if data else None,
        )

    except Exception as e:
        log.warning(f"CRAG grading failed for doc {doc_id}: {e}")
        count_fallback("crag.grading_failed")
        # Default to IRRELEVANT on error (fail-safe)
        return GradedDocument(
            doc_id=doc_id,
            score=0.0,
            grade=RetrievalGrade.IRRELEVANT,
            reasoning=f"Grading failed: {str(e)[:100]}",
            source=source,
        )


async def grade_retrieval(
    documents: list[dict],
    query: str,
    threshold: float = FALLBACK_THRESHOLD,
) -> GradingResult:
    """
    Grade retrieved documents and determine if web fallback is needed.

    This is the core CRAG function that evaluates each document's relevance
    to the query and decides whether to trigger a web search fallback.

    Args:
        documents: List of document dicts with 'id' and 'content' keys
        query: The user's search query
        threshold: Minimum ratio of PARTIAL+ documents required (default 0.5)

    Returns:
        GradingResult with graded documents and fallback decision
    """
    if not documents:
        return GradingResult(
            graded_documents=[],
            needs_web_fallback=False,
            consensus_score=1.0,
            relevant_count=0,
            partial_count=0,
            irrelevant_count=0,
        )

    # Prepare document summaries for grading
    doc_summaries = [
        {
            "id": doc.get("id", f"doc_{i}"),
            "content": doc.get("content", doc.get("text", ""))[:2000],
            "source": doc.get("source", doc.get("metadata", {}).get("source", "local")),
        }
        for i, doc in enumerate(documents)
    ]

    # Grade each document (parallelized LLM calls)
    grading_tasks = [
        grade_single_document(doc["id"], doc["content"], query, doc["source"])
        for doc in doc_summaries
    ]

    results = await asyncio.gather(*grading_tasks, return_exceptions=True)

    graded_docs = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            log.warning(f"CRAG task failed for doc {i}: {result}")
            graded_docs.append(GradedDocument(
                doc_id=doc_summaries[i]["id"],
                score=0.0,
                grade=RetrievalGrade.IRRELEVANT,
                reasoning=f"Grading failed: {str(result)[:100]}",
                source=doc_summaries[i]["source"],
            ))
        else:
            graded_docs.append(result)

    # Calculate statistics
    relevant_count = sum(1 for d in graded_docs if d.grade == RetrievalGrade.RELEVANT)
    partial_count = sum(1 for d in graded_docs if d.grade == RetrievalGrade.PARTIAL)
    irrelevant_count = sum(1 for d in graded_docs if d.grade == RetrievalGrade.IRRELEVANT)

    usable_count = relevant_count + partial_count
    usable_ratio = usable_count / len(graded_docs) if graded_docs else 0

    # Calculate consensus
    scores = [d.score for d in graded_docs]
    consensus = _calculate_consensus(scores)

    # Determine fallback reason
    fallback_reason = None
    if usable_ratio < threshold:
        fallback_reason = (
            f"{irrelevant_count}/{len(graded_docs)} docs irrelevant "
            f"(usable ratio: {usable_ratio:.1%}, threshold: {threshold:.1%})"
        )

    return GradingResult(
        graded_documents=graded_docs,
        needs_web_fallback=usable_ratio < threshold,
        fallback_reason=fallback_reason,
        consensus_score=consensus,
        relevant_count=relevant_count,
        partial_count=partial_count,
        irrelevant_count=irrelevant_count,
    )


# ─── Web Fallback Functions ───────────────────────────────────────────────────

# Blocked domains configuration
BLOCKED_DOMAINS = frozenset({
    "example-paywalled.com",
    "spam-site.org",
    "known-scraper.io",
})


async def tavily_search(
    query: str,
    api_key: str,
    max_results: int = 10,
) -> list[WebSearchResult]:
    """
    Execute web search via Tavily API.

    Args:
        query: Search query
        api_key: Tavily API key
        max_results: Maximum number of results

    Returns:
        List of WebSearchResult objects
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": True,
                    "include_images": False,
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for result in data.get("results", []):
                domain = _extract_domain(result["url"])
                results.append(WebSearchResult(
                    url=result["url"],
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.5),
                    published_date=result.get("published_date"),
                    domain=domain,
                ))

            return results

        except httpx.HTTPStatusError as e:
            log.error(f"Tavily API error: {e.response.status_code}")
            raise
        except Exception as e:
            log.error(f"Tavily search failed: {e}")
            raise


async def duckduckgo_search(
    query: str,
    max_results: int = 10,
) -> list[WebSearchResult]:
    """
    Execute web search via DuckDuckGo Instant Answer API.
    Fallback when Tavily is not available.
    """
    from urllib.parse import quote

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(
                f"https://api.duckduckgo.com/{quote(query)}",
                params={
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if "Text" in topic and "FirstURL" in topic:
                    results.append(WebSearchResult(
                        url=topic["FirstURL"],
                        title=data.get("Heading", query),
                        content=topic["Text"],
                        score=0.5,  # No relevance score from DDG
                        published_date=None,
                        domain=_extract_domain(topic["FirstURL"]),
                    ))

            return results

        except Exception as e:
            log.error(f"DuckDuckGo search failed: {e}")
            raise


def _extract_domain(url: str) -> str:
    """Extract domain from URL."""
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc
    except Exception:
        return url


async def expand_query(query: str, context: str = "") -> str:
    """
    Generate expanded search query using LLM.

    Args:
        query: Original query
        context: Existing context for query expansion

    Returns:
        Expanded search query
    """
    client = _get_client()

    prompt = QUERY_EXPANSION_PROMPT.format(
        query=query,
        existing_context=context[:1000] if context else "No additional context",
    )

    try:
        response = await client.chat.completions.create(
            model=settings.CRAG_GRADING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )

        expanded = response.choices[0].message.content.strip()
        # Validate response is reasonable
        if len(expanded) < 5 or len(expanded) > 200:
            log.warning(f"Query expansion returned unusual result: {expanded[:50]}")
            return query
        return expanded

    except Exception as e:
        log.warning(f"Query expansion failed: {e}")
        return query  # Fallback to original query


async def execute_web_fallback(
    query: str,
    existing_context: str = "",
    existing_doc_ids: set[str] | None = None,
    max_results: int = 10,
) -> WebFallbackResult:
    """
    Execute web fallback search with query expansion.

    Steps:
    1. Expand query using existing context
    2. Execute web search (Tavily primary, DDG fallback)
    3. Filter blocked domains
    4. Format results

    Args:
        query: Original query
        existing_context: Context from local retrieval
        existing_doc_ids: IDs of already retrieved docs (to avoid duplicates)
        max_results: Maximum web results to return

    Returns:
        WebFallbackResult with formatted web documents
    """
    # Step 1: Query expansion
    expanded_query = await expand_query(query, existing_context)
    log.info(f"CRAG: Expanded query: {query} -> {expanded_query}")

    # Step 2: Execute search
    web_results: list[WebSearchResult] = []
    tavily_key = settings.TAVILY_API_KEY

    try:
        if tavily_key:
            web_results = await tavily_search(expanded_query, tavily_key, max_results)
        else:
            # Fallback to DuckDuckGo if Tavily not configured
            log.warning("TAVILY_API_KEY not set, using DuckDuckGo fallback")
            web_results = await duckduckgo_search(expanded_query, max_results)
    except Exception as e:
        log.error(f"Web search failed: {e}")
        # Return empty result on failure
        return WebFallbackResult(
            web_documents=[],
            merged_documents=[],
            search_query_used=expanded_query,
            domains_included=[],
            domains_excluded=[],
        )

    # Step 3: Filter blocked domains
    domains_excluded = []
    filtered_results = []
    for result in web_results:
        if any(blocked in (result.domain or "") for blocked in BLOCKED_DOMAINS):
            domains_excluded.append(result.domain)
        else:
            filtered_results.append(result)

    # Step 4: Format results. Web page text is raw third-party content —
    # cap it per-doc at ingest so a few full pages can't blow the context
    # budget downstream (the merge below enforces the global budget too).
    max_chars = settings.CRAG_MAX_WEB_CHARS
    web_documents = [
        {
            "id": f"web_{i}",
            "content": (r.content or "")[:max_chars],
            "metadata": {
                "source": "web",
                "url": r.url,
                "title": r.title,
                "score": r.score,
                "published_date": r.published_date,
                "domain": r.domain,
            },
        }
        for i, r in enumerate(filtered_results)
    ]

    return WebFallbackResult(
        web_documents=filtered_results,
        merged_documents=web_documents,
        search_query_used=expanded_query,
        domains_included=list(set(r.domain for r in filtered_results if r.domain)),
        domains_excluded=domains_excluded,
    )


# ─── CRAG Agent Node for LangGraph ───────────────────────────────────────────

async def crag_agent(state: AgentState) -> AgentState:
    """
    CRAG agent node for LangGraph workflow.

    This node:
    1. Grades retrieved documents using LLM
    2. Determines if web fallback is needed
    3. Executes web fallback if threshold not met
    4. Merges local + web results

    Args:
        state: Current agent state

    Returns:
        Updated agent state with CRAG results
    """
    state.setdefault("agent_trace", {})
    state.setdefault("crag_trace", {})

    # Check if CRAG is enabled
    if not settings.CRAG_ENABLED:
        log.debug("CRAG disabled, skipping")
        state["crag_trace"]["enabled"] = False
        return state

    query = state.get("rewritten_query", state.get("query", ""))
    # Grade the SAME population the answer agent will consume
    # (reranked_chunks = post-merge, evaluator-filtered context). Grading the
    # upstream fused_chunks instead meant CRAG's verdicts — and, critically,
    # its web-fallback documents — never reached the answer.
    retrieved_docs = state.get("reranked_chunks", [])
    if not retrieved_docs:
        log.debug("No retrieved docs to grade")
        state["crag_trace"]["no_documents"] = True
        return state

    # Get existing context for query expansion
    existing_context = ""
    if state.get("graph_context_chunks"):
        existing_context = " ".join(
            c.get("content", "")[:500]
            for c in state["graph_context_chunks"][:3]
        )

    # Normalize doc identity: every graded doc must have an "id" so the
    # score/grade lookups below can never KeyError, and so graded ids match
    # the fallback ids grade_retrieval assigns (doc_{i}) for unkeyed chunks.
    for i, doc in enumerate(retrieved_docs):
        if not doc.get("id"):
            doc["id"] = f"doc_{i}"

    # Step 1: Grade retrieved documents
    log.info(f"CRAG: Grading {len(retrieved_docs)} retrieved documents")

    grading_result = await grade_retrieval(
        documents=retrieved_docs,
        query=query,
        threshold=settings.CRAG_FALLBACK_THRESHOLD,
    )

    # Store grading trace
    state["crag_trace"]["grade_result"] = {
        "relevant_count": grading_result.relevant_count,
        "partial_count": grading_result.partial_count,
        "irrelevant_count": grading_result.irrelevant_count,
        "consensus_score": grading_result.consensus_score,
        "needs_web_fallback": grading_result.needs_web_fallback,
        "fallback_reason": grading_result.fallback_reason,
    }

    # Step 2: Determine action based on grading
    if grading_result.needs_web_fallback:
        log.info(f"CRAG: Triggering web fallback - {grading_result.fallback_reason}")

        # Execute web fallback
        fallback_result = await execute_web_fallback(
            query=query,
            existing_context=existing_context,
            max_results=settings.CRAG_MAX_WEB_RESULTS,
        )

        # Store web fallback trace
        state["crag_trace"]["web_fallback"] = {
            "triggered": True,
            "search_query": fallback_result.search_query_used,
            "results_count": len(fallback_result.merged_documents),
            "domains": fallback_result.domains_included,
        }

        # Merge local and web documents
        local_docs = {doc.get("id"): doc for doc in retrieved_docs}
        for doc in fallback_result.merged_documents:
            if doc["id"] not in local_docs:
                local_docs[doc["id"]] = doc

        # Re-grade combined documents
        combined_docs = list(local_docs.values())
        combined_result = await grade_retrieval(
            documents=combined_docs,
            query=query,
            threshold=settings.CRAG_FALLBACK_THRESHOLD,
        )

        # Apply weighting: local × 1.2, web × 0.8
        weighted_docs = []
        for doc in combined_docs:
            # Find the grade for this doc
            grade_info = next(
                (g for g in combined_result.graded_documents if g.doc_id == doc["id"]),
                None
            )

            if grade_info:
                # Adjust score based on source
                if doc.get("metadata", {}).get("source") == "local":
                    adjusted_score = min(1.0, grade_info.score * 1.2)
                else:
                    adjusted_score = grade_info.score * 0.8

                # Update the document with adjusted score
                doc["crag_score"] = adjusted_score
                doc["crag_grade"] = grade_info.grade.value
                weighted_docs.append(doc)
            else:
                doc["crag_score"] = 0.0
                doc["crag_grade"] = "irrelevant"
                weighted_docs.append(doc)

        # Sort by adjusted score
        weighted_docs.sort(key=lambda x: x.get("crag_score", 0), reverse=True)

        # Re-apply the context budget AFTER the merge: context_merge_agent ran
        # earlier in the graph (merge_context -> grade_docs -> crag), so without
        # this the fallback overwrites the budgeted `reranked_chunks` with up
        # to ~5 local + CRAG_MAX_WEB_RESULTS untruncated web docs.
        from app.agents.context_merge_agent import merge_context_chunks

        state["reranked_chunks"] = weighted_docs
        budgeted, dropped = merge_context_chunks(state)
        state["reranked_chunks"] = budgeted
        state["crag_trace"]["merged_result"] = {
            "total_documents": len(budgeted),
            "local_weighted": sum(1 for d in budgeted if d.get("metadata", {}).get("source") == "local"),
            "web_added": len(fallback_result.merged_documents),
            "dropped_for_budget": dropped,
        }

        state["agent_trace"]["crag"] = {
            "mode": "web_fallback",
            "grade_summary": f"{combined_result.relevant_count}R/{combined_result.partial_count}P/{combined_result.irrelevant_count}I",
            "consensus": combined_result.consensus_score,
        }

    else:
        # No fallback needed - add scores to documents
        log.info(f"CRAG: Retrieval passed - {grading_result.relevant_count}R/{grading_result.partial_count}P")

        doc_scores = {g.doc_id: g for g in grading_result.graded_documents}
        for doc in retrieved_docs:
            doc["crag_score"] = doc_scores.get(doc["id"], GradedDocument(
                doc_id=doc["id"],
                score=0.0,
                grade=RetrievalGrade.IRRELEVANT,
                reasoning="Not graded",
                source="local",
            )).score
            doc["crag_grade"] = doc_scores.get(doc["id"], GradedDocument(
                doc_id=doc["id"],
                score=0.0,
                grade=RetrievalGrade.IRRELEVANT,
                reasoning="Not graded",
                source="local",
            )).grade.value

        state["reranked_chunks"] = retrieved_docs
        state["agent_trace"]["crag"] = {
            "mode": "local_pass",
            "grade_summary": f"{grading_result.relevant_count}R/{grading_result.partial_count}P/{grading_result.irrelevant_count}I",
            "consensus": grading_result.consensus_score,
        }

    state["crag_trace"]["enabled"] = True
    return state
