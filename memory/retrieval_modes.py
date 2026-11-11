# memory/retrieval_modes.py
"""
Three ways to turn stored memory into the context handed to the model.
This is the main experimental axis: scratchpad vs. vector.

    scratchpad -> dump the most recent N items into context, every time.
                  No ranking, no query filtering. Poison is present until
                  newer memories push it out of the window.

    keyword    -> the project's original overlap scorer (top-k by the
                  fraction of shared words). A weak stand-in for retrieval.

    vector     -> real RAG: embed the query and each memory, rank by cosine
                  similarity, keep only items above sim_threshold, then top-k.
                  Because it can drop items below threshold, an off-topic
                  poison can fail to be retrieved at all — that exclusion is
                  the whole point of the comparison.

All three return the same shape as the original retrieve_memories():
    {"episodic": [...], "semantic": [...]}
so memory_retrieval.format_memories() consumes any of them unchanged.

Semantic facts are filtered to status == "active" in every mode, so the
comparison is about retrieval, not about the status filter.
"""

from memory.episodic_memory import get_episodes
from memory.semantic_memory import get_semantic_memories
from memory.memory_retrieval import retrieve_memories
from memory.embeddings import embed, cosine


def _episode_text(e: dict) -> str:
    return f"{e.get('event', '')} {e.get('context', '')}"


def _fact_text(f: dict) -> str:
    return f"{f.get('subject', '')} {f.get('predicate', '')} {f.get('object', '')}"


def _active_semantic(user_id: str):
    return [f for f in get_semantic_memories(user_id) if f.get("status") == "active"]


def scratchpad_context(user_id: str, query: str, scratchpad_size: int = 10, **_):
    """
    A bounded FIFO scratchpad: the most recent `scratchpad_size` episodes and
    the most recent `scratchpad_size` active facts, unranked and unfiltered
    by the query. Old items (including old poison) fall out of the window.
    """
    facts = sorted(_active_semantic(user_id), key=lambda f: f.get("id", 0), reverse=True)
    return {
        "episodic": get_episodes(user_id, limit=scratchpad_size),
        "semantic": facts[:scratchpad_size],
    }


def keyword_context(user_id: str, query: str,
                    episode_limit: int = 5, semantic_limit: int = 10, **_):
    """The original keyword-overlap retrieval."""
    return retrieve_memories(user_id, query, episode_limit, semantic_limit)


def vector_context(user_id: str, query: str,
                   episode_limit: int = 5, semantic_limit: int = 10,
                   sim_threshold: float = 0.35, **_):
    """Embedding similarity with a threshold, then top-k."""
    q = embed(query, "query")

    def rank(items, text_fn, limit):
        scored = []
        for it in items:
            score = cosine(q, embed(text_fn(it)))
            if score >= sim_threshold:
                scored.append((score, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:limit]]

    return {
        "episodic": rank(get_episodes(user_id, limit=100000), _episode_text, episode_limit),
        "semantic": rank(_active_semantic(user_id), _fact_text, semantic_limit),
    }


def similarity_report(user_id: str, query: str, marker: str):
    """
    Cosine similarity of the query to every stored item, flagged by whether
    the item contains `marker`. Used to calibrate sim_threshold: you want a
    cutoff that separates relevant from irrelevant items, not one that
    keeps everything.
    Returns a list of dicts: {store, is_poison, sim, text}.
    """
    q = embed(query, "query")
    m = marker.casefold()
    out = []
    for e in get_episodes(user_id, limit=100000):
        blob = " ".join(str(e.get(k, "")) for k in
                        ("event", "context", "entities", "actions", "outcome"))
        out.append({"store": "episodic", "is_poison": m in blob.casefold(),
                    "sim": cosine(q, embed(_episode_text(e))), "text": _episode_text(e)})
    for f in _active_semantic(user_id):
        out.append({"store": "semantic", "is_poison": m in _fact_text(f).casefold(),
                    "sim": cosine(q, embed(_fact_text(f))), "text": _fact_text(f)})
    return out


RETRIEVERS = {
    "scratchpad": scratchpad_context,
    "keyword": keyword_context,
    "vector": vector_context,
}


def get_context(user_id: str, query: str, mode: str, **kwargs):
    """Dispatch to the chosen retrieval mode."""
    return RETRIEVERS[mode](user_id, query, **kwargs)
