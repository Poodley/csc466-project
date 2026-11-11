# memory/semantic_memory.py

import sqlite3
import json

from llm import ask_qwen

DB_PATH = "database/semantic.db"
PROMPT_PATH = "prompts/semantic_memory.txt"

def initialize_semantic_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            status TEXT DEFAULT 'active',
            evidence_ids TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


SEMANTIC_PROMPT = "".join(open(PROMPT_PATH, "r", encoding="utf-8").readlines())


def extract_semantic_facts(episode: dict):
    raw = ask_qwen(
        SEMANTIC_PROMPT,
        f"<episode_data>\n{json.dumps(episode)}\n</episode_data>"
    )

    try:
        cleaned = raw.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]

        result = json.loads(cleaned)
        return result.get("facts", [])

    except (json.JSONDecodeError, AttributeError):
        print("Could not parse semantic memory JSON.")
        return []


def save_semantic_fact(
    user_id: str,
    fact: dict,
    episode_id: int
):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    subject = str(fact.get("subject", "user")).strip()
    predicate = str(fact.get("predicate", "")).strip()
    obj = str(fact.get("object", "")).strip()

    if not predicate or not obj:
        conn.close()
        return None

    confidence = max(
        0.0,
        min(1.0, float(fact.get("confidence", 0.5)))
    )

    existing = conn.execute("""
        SELECT * FROM semantic_facts
        WHERE user_id = ?
        AND subject = ?
        AND predicate = ?
        AND status = 'active'
        ORDER BY updated_at DESC
    """, (user_id, subject, predicate)).fetchall()

    # Check whether the fact already exists.
    for old in existing:
        if old["object"].casefold() == obj.casefold():

            evidence = json.loads(old["evidence_ids"] or "[]")

            if episode_id not in evidence:
                evidence.append(episode_id)

            # Corroborating evidence increases confidence.
            new_confidence = min(
                0.99,
                max(old["confidence"], confidence) + 0.05
            )

            conn.execute("""
                UPDATE semantic_facts
                SET confidence = ?,
                    evidence_ids = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                new_confidence,
                json.dumps(evidence),
                old["id"]
            ))

            conn.commit()
            conn.close()

            return old["id"]

    # A different value for the same subject/predicate
    # is a potential conflict, not an automatic replacement.
    status = "uncertain" if existing else "active"

    cursor = conn.execute("""
        INSERT INTO semantic_facts (
            user_id,
            subject,
            predicate,
            object,
            confidence,
            status,
            evidence_ids
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        subject,
        predicate,
        obj,
        confidence,
        status,
        json.dumps([episode_id])
    ))

    fact_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return fact_id


def process_semantic_memory(
    user_id: str,
    episode: dict,
    episode_id: int
):
    facts = extract_semantic_facts(episode)

    saved = []

    for fact in facts:
        if not isinstance(fact, dict):
            continue

        fact_id = save_semantic_fact(
            user_id,
            fact,
            episode_id
        )

        if fact_id is not None:
            saved.append(fact_id)

    return saved


def get_semantic_memories(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT * FROM semantic_facts
        WHERE user_id = ?
        ORDER BY confidence DESC
    """, (user_id,)).fetchall()

    conn.close()

    return [dict(row) for row in rows]