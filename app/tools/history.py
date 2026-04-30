import logging
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = "/tmp/sentiment_history.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY,
            text       TEXT,
            label      TEXT,
            score      REAL,
            reasoning  TEXT,
            source     TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    logger.info("History DB initialised at %s", DB_PATH)


def store_result(
    text: str,
    label: str,
    score: float,
    reasoning: str,
    source: str = "text",
) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (text, label, score, reasoning, source, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (text[:500], label, score, reasoning, source, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def analyze_history(topic: str) -> str:
    logger.info("Querying history for topic: %s", topic)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT text, label, score, reasoning, created_at FROM predictions "
        "WHERE text LIKE ? ORDER BY created_at DESC LIMIT 10",
        (f"%{topic}%",),
    ).fetchall()
    conn.close()

    if not rows:
        return f"No past predictions found for topic: {topic}"

    lines = [
        f"[{r[4]}] '{r[0][:60]}' → {r[1]} (score {r[2]:.2f}): {r[3]}"
        for r in rows
    ]
    avg = sum(r[2] for r in rows) / len(rows)
    return f"Found {len(rows)} past predictions:\n" + "\n".join(lines) + f"\nAverage score: {avg:.2f}"
