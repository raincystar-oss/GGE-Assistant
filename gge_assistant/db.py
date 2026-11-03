from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 2

STATUS_PENDING = "PENDING"
STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETE = "COMPLETE"
STATUS_ERROR = "ERROR"
STATUS_NEEDS_ATTENTION = "NEEDS_ATTENTION"
STATUSES = {
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_COMPLETE,
    STATUS_ERROR,
    STATUS_NEEDS_ATTENTION,
}


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(f"Database schema {version} is newer than this app supports ({SCHEMA_VERSION})")
    if version == 0:
        _apply_schema_v1(conn)
    elif version == 1:
        _migrate_v1_to_v2(conn)
    elif version < SCHEMA_VERSION:
        raise RuntimeError(f"No migration path from schema {version} to {SCHEMA_VERSION}")
    _repair_missing_stages(conn)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    try:
        conn.execute("BEGIN")
        yield
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def _apply_schema_v1(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        conn.executescript(
            """
            CREATE TABLE materials (
                material_id TEXT PRIMARY KEY,
                platform_key TEXT NOT NULL UNIQUE,
                discovery_url TEXT,
                media_url TEXT NOT NULL,
                title TEXT,
                webinar_date TEXT,
                published_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE material_stages (
                material_id TEXT NOT NULL REFERENCES materials(material_id) ON DELETE CASCADE,
                stage TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('PENDING','PROCESSING','COMPLETE','ERROR','NEEDS_ATTENTION')),
                attempts INTEGER NOT NULL DEFAULT 0,
                artifact_path TEXT,
                artifact_sha256 TEXT,
                params_json TEXT NOT NULL DEFAULT '{}',
                error_message TEXT,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (material_id, stage)
            );

            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                material_id TEXT NOT NULL REFERENCES materials(material_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                text_sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('PENDING','PROCESSING','COMPLETE','ERROR','NEEDS_ATTENTION')),
                document_id TEXT,
                document_title TEXT,
                document_location TEXT,
                workspace_doc_id TEXT,
                last_error TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE (material_id, chunk_index)
            );

            PRAGMA user_version = 2;
            """
        )


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
        }
        if "document_id" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN document_id TEXT")
        if "document_title" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN document_title TEXT")
        if "last_error" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN last_error TEXT")
        conn.execute("PRAGMA user_version = 2")


def _repair_missing_stages(conn: sqlite3.Connection) -> None:
    with transaction(conn):
        for material in conn.execute("SELECT material_id FROM materials").fetchall():
            for stage in ("transcript", "chunks"):
                exists = conn.execute(
                    """
                    SELECT 1 FROM material_stages
                    WHERE material_id = ? AND stage = ?
                    """,
                    (material["material_id"], stage),
                ).fetchone()
                if not exists:
                    conn.execute(
                        """
                        INSERT INTO material_stages (material_id, stage, status)
                        VALUES (?, ?, ?)
                        """,
                        (material["material_id"], stage, STATUS_PENDING),
                    )


def touch_material(conn: sqlite3.Connection, material_id: str) -> None:
    conn.execute(
        "UPDATE materials SET updated_at = datetime('now') WHERE material_id = ?",
        (material_id,),
    )
