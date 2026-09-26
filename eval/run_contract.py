"""Run-compatibility seam: may a partial artifact be resumed under the current config?

Every key that can move a score is compared, not just the retrieval lane — a
resume re-answers and re-judges with the *current* answer/judge config, so a
changed prompt or token cap would silently merge two different measurements.
"""

_RETRIEVAL_CONTRACT_KEYS = (
    "embedding_backend",
    "embeddings_actual",
    "rerank",
    "recall_top_k",
    "graph_builds",
    "retrieval",
)
_RETRIEVAL_POLICY_KEYS = ("hybrid_enabled", "rerank_pool_multiplier", "rrf_k")
_CONTEXT_POLICY_KEYS = ("session_level", "chunk_chars", "fuse")
_ANSWER_KEYS = ("model", "temperature", "max_tokens", "prompt_version", "passes_reference_date")
_JUDGE_KEYS = ("model", "prompt_version", "temperature", "max_tokens")
_GRAPH_BUILD_POLICIES = frozenset(("on", "off (bench ingest)"))


def _keys_match(recorded: object, current: object, keys: tuple[str, ...]) -> bool:
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    return all(key in recorded and key in current for key in keys) and all(
        recorded[key] == current[key] for key in keys
    )


def run_contracts_match(recorded: object, current: object) -> bool:
    """Require matching embedding, retrieval, rerank, ingest, graph, answer, and judge config."""
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    recorded_graph_builds = recorded.get("graph_builds")
    current_graph_builds = current.get("graph_builds")
    if (
        not isinstance(recorded_graph_builds, str)
        or not isinstance(current_graph_builds, str)
        or recorded_graph_builds not in _GRAPH_BUILD_POLICIES
        or current_graph_builds not in _GRAPH_BUILD_POLICIES
    ):
        return False

    recorded_retrieval = recorded.get("retrieval")
    current_retrieval = current.get("retrieval")
    if not isinstance(recorded_retrieval, dict) or not isinstance(current_retrieval, dict):
        return False
    if any(key not in recorded_retrieval or key not in current_retrieval for key in _RETRIEVAL_POLICY_KEYS):
        return False
    if any(recorded.get(key) != current.get(key) for key in _RETRIEVAL_CONTRACT_KEYS):
        return False

    if not _keys_match(recorded.get("answer"), current.get("answer"), _ANSWER_KEYS):
        return False
    if not _keys_match(recorded.get("judge"), current.get("judge"), _JUDGE_KEYS):
        return False

    recorded_execution = recorded.get("execution")
    current_execution = current.get("execution")
    if not isinstance(recorded_execution, dict) or not isinstance(current_execution, dict):
        return False
    recorded_policy = recorded_execution.get("context_policy")
    current_policy = current_execution.get("context_policy")
    if not isinstance(recorded_policy, dict) or not isinstance(current_policy, dict):
        return False
    if any(key not in recorded_policy or key not in current_policy for key in _CONTEXT_POLICY_KEYS):
        return False
    return recorded_policy == current_policy
