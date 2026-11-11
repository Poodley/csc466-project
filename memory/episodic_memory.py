import re
import sqlite3
import json
from datetime import datetime, timezone

from llm import ask_qwen

DB_PATH = "database/episodic.db"


def initialize_episodic_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            event TEXT NOT NULL,
            context TEXT,
            entities TEXT,
            actions TEXT,
            outcome TEXT,
            event_time TEXT,
            recorded_at TEXT NOT NULL,
            importance REAL DEFAULT 0.5
        )
    """)

    conn.commit()
    conn.close()


EPISODIC_PROMPT = "".join(open("prompts/episodic_memory.txt", "r", encoding="utf-8").readlines())


def extract_episodes(conversation: str):
    raw = ask_qwen(
        EPISODIC_PROMPT,
        f"<conversation_data>\n{conversation}\n</conversation_data>"
    )
    print("\n--- RAW QWEN EPISODIC RESPONSE ---")
    print(repr(raw))
    print("--- END RAW RESPONSE ---\n")
    try:
        data = parse_json_response(raw)
        return data.get("episodes", [])
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Could not parse episodic memory JSON: {e}")
        return []

def parse_json_response(raw: str):
    if not raw or not raw.strip():
        raise ValueError("Qwen returned an empty response")

    # Remove Markdown code fences
    cleaned = re.sub(
        r"```(?:json)?\s*",
        "",
        raw.strip(),
        flags=re.IGNORECASE
    )
    cleaned = cleaned.replace("```", "").strip()

    # Try parsing the complete response first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting a JSON object or array
    start_positions = [
        pos for pos in (cleaned.find("{"), cleaned.find("["))
        if pos != -1
    ]

    if not start_positions:
        raise ValueError(f"No JSON object or array found: {raw!r}")

    start = min(start_positions)
    opening = cleaned[start]
    closing = "}" if opening == "{" else "]"
    end = cleaned.rfind(closing)

    if end == -1:
        raise ValueError(f"Incomplete JSON response: {raw!r}")

    return json.loads(cleaned[start:end + 1])
def save_episode(user_id: str, episode: dict):
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        INSERT INTO episodes (
            user_id,
            event,
            context,
            entities,
            actions,
            outcome,
            event_time,
            recorded_at,
            importance
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        str(episode.get("event") or "Unspecified event"),
        episode.get("context"),
        json.dumps(episode.get("entities") or []),
        json.dumps(episode.get("actions") or []),
        episode.get("outcome"),
        episode.get("event_time"),
        datetime.now(timezone.utc).isoformat(),
        max(0.0, min(1.0, float(episode.get("importance", 0.5))))
    ))

    conn.commit()
    episode_id = conn.execute(
        "SELECT last_insert_rowid()"
    ).fetchone()[0]

    conn.close()

    return episode_id


def get_episodes(user_id: str, limit: int = 10):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT * FROM episodes
        WHERE user_id = ?
        ORDER BY recorded_at DESC
        LIMIT ?
    """, (user_id, limit)).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def process_episodic_memory(user_id: str, conversation: str):
    episodes = extract_episodes(conversation)

    saved_ids = []

    for episode in episodes:
        if not isinstance(episode, dict) or not episode.get("event"):
            continue

        episode_id = save_episode(user_id, episode)
        saved_ids.append(episode_id)

    return saved_ids