_RETRIEVAL_CONTRACT_KEYS = (
    "embedding_backend",
    "embeddings_actual",
    "rerank",
    "recall_top_k",
)
_CONTEXT_POLICY_KEYS = ("session_level", "chunk_chars", "fuse")


def retrieval_contract_matches(recorded: object, current: object) -> bool:
    """Require the active embedding, rerank, recall, and ingestion contracts to match."""
    if not isinstance(recorded, dict) or not isinstance(current, dict):
        return False
    if any(recorded.get(key) != current.get(key) for key in _RETRIEVAL_CONTRACT_KEYS):
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
