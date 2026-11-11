# memory/embeddings.py
"""
Embedding + cosine similarity via the local Ollama server, for the
vector/RAG retrieval mode.

Requires an embedding model pulled in Ollama:

    ollama pull nomic-embed-text

Embeddings are cached in-process by exact text so repeated trials over
the same stored memories don't re-embed. Nothing here touches the
episodic/semantic schemas — retrieval embeds text on the fly.
"""

import math

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

EMBED_MODEL = "nomic-embed-text"

_cache: dict[str, list[float]] = {}


# nomic-embed-text is trained with task prefixes; without them, relevant and
# irrelevant items get much closer similarity scores.
PREFIXES = {"query": "search_query: ", "document": "search_document: "}


def embed(text: str, kind: str = "document"):
    """
    Return the embedding vector for text, or None for empty input.
    kind is "query" for the user's question, "document" for stored memories.
    """
    text = (text or "").strip()
    if not text:
        return None
    text = PREFIXES[kind] + text
    if text in _cache:
        return _cache[text]

    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = resp.data[0].embedding
    _cache[text] = vec
    return vec


def cosine(a, b) -> float:
    """Cosine similarity between two vectors; 0.0 if either is missing."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def embeddings_available() -> bool:
    """Preflight check: is the embedding model reachable and working?"""
    try:
        return embed("healthcheck") is not None
    except Exception as e:  # noqa: BLE001  (harness preflight, report and skip)
        print(f"[embeddings] unavailable: {e}")
        return False
