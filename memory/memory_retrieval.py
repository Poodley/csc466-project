import json

from memory.episodic_memory import get_episodes
from memory.semantic_memory import get_semantic_memories


def retrieve_memories(
    user_id: str,
    query: str,
    episode_limit: int = 5,
    semantic_limit: int = 10
):
    query_words = set(query.lower().split())

    episodes = get_episodes(user_id, limit=100)
    semantic = get_semantic_memories(user_id)

    def relevance(text):
        words = set(str(text).lower().split())

        if not query_words:
            return 0

        return len(query_words & words) / len(query_words)

    # Episodic retrieval
    ranked_episodes = sorted(
        episodes,
        key=lambda x: relevance(
            x["event"] + " " + str(x["context"])
        ),
        reverse=True
    )

    # Semantic retrieval
    active_semantic = [
        x for x in semantic
        if x["status"] == "active"
    ]

    ranked_semantic = sorted(
        active_semantic,
        key=lambda x: relevance(
            x["subject"] + " " +
            x["predicate"] + " " +
            x["object"]
        ),
        reverse=True
    )

    return {
        "episodic": ranked_episodes[:episode_limit],
        "semantic": ranked_semantic[:semantic_limit]
    }


def format_memories(memories):
    return f"""
SEMANTIC MEMORY:
{json.dumps(memories["semantic"], indent=2)}

EPISODIC MEMORY:
{json.dumps(memories["episodic"], indent=2)}
"""