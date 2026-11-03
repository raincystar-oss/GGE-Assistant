from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Iterable

from .db import (
    STATUS_COMPLETE,
    STATUS_ERROR,
    STATUS_NEEDS_ATTENTION,
    STATUS_PENDING,
    STATUS_PROCESSING,
    touch_material,
    transaction,
)

PIPELINE_STAGES = ("transcript", "chunks")


def add_material(
    conn: sqlite3.Connection,
    *,
    media_url: str,
    platform_key: str | None = None,
    discovery_url: str | None = None,
    title: str | None = None,
    webinar_date: str | None = None,
    published_at: str | None = None,
    stages: Iterable[str] = PIPELINE_STAGES,
) -> str:
    key = platform_key or infer_platform_key(media_url)
    existing = conn.execute(
        "SELECT material_id FROM materials WHERE platform_key = ?",
        (key,),
    ).fetchone()
    if existing:
        return existing["material_id"]

    material_id = str(uuid.uuid4())
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO materials
            (material_id, platform_key, discovery_url, media_url, title, webinar_date, published_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (material_id, key, discovery_url, media_url, title, webinar_date, published_at),
        )
        for stage in stages:
            conn.execute(
                """
                INSERT INTO material_stages (material_id, stage, status)
                VALUES (?, ?, ?)
                """,
                (material_id, stage, STATUS_PENDING),
            )
    return material_id


def infer_platform_key(url: str) -> str:
    rutube = re.search(r"rutube\.ru/(?:video|play/embed)/([a-f0-9]{32})", url, re.IGNORECASE)
    if rutube:
        return f"rutube:{rutube.group(1).lower()}"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return f"url:{digest}"


def list_material_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT material_id FROM materials ORDER BY created_at, material_id").fetchall()
    return [row["material_id"] for row in rows]


def get_material(conn: sqlite3.Connection, material_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM materials WHERE material_id = ?",
        (material_id,),
    ).fetchone()


def get_stage(conn: sqlite3.Connection, material_id: str, stage: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM material_stages WHERE material_id = ? AND stage = ?",
        (material_id, stage),
    ).fetchone()


def stage_artifact_path(data_root: Path, material_id: str, stage: str) -> Path:
    suffix = "transcript.json" if stage == "transcript" else f"{stage}.json"
    return data_root / "materials" / material_id / suffix


def relative_to_data_root(data_root: Path, path: Path) -> str:
    return path.resolve().relative_to(data_root.resolve()).as_posix()


def path_from_db(data_root: Path, artifact_path: str | None) -> Path | None:
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if path.is_absolute():
        return path
    return data_root / path


def mark_processing(conn: sqlite3.Connection, material_id: str, stage: str) -> None:
    with transaction(conn):
        conn.execute(
            """
            UPDATE material_stages
            SET status = ?, attempts = attempts + 1, error_message = NULL,
                started_at = datetime('now'), completed_at = NULL, updated_at = datetime('now')
            WHERE material_id = ? AND stage = ?
            """,
            (STATUS_PROCESSING, material_id, stage),
        )
        touch_material(conn, material_id)


def mark_complete(
    conn: sqlite3.Connection,
    *,
    data_root: Path,
    material_id: str,
    stage: str,
    artifact_path: Path,
    params: dict,
) -> None:
    digest = sha256_file(artifact_path)
    rel_path = relative_to_data_root(data_root, artifact_path)
    with transaction(conn):
        conn.execute(
            """
            UPDATE material_stages
            SET status = ?, artifact_path = ?, artifact_sha256 = ?, params_json = ?,
                error_message = NULL, completed_at = datetime('now'), updated_at = datetime('now')
            WHERE material_id = ? AND stage = ?
            """,
            (STATUS_COMPLETE, rel_path, digest, json.dumps(params, sort_keys=True), material_id, stage),
        )
        touch_material(conn, material_id)


def mark_error(conn: sqlite3.Connection, material_id: str, stage: str, message: str) -> None:
    with transaction(conn):
        conn.execute(
            """
            UPDATE material_stages
            SET status = ?, error_message = ?, updated_at = datetime('now'), completed_at = NULL
            WHERE material_id = ? AND stage = ?
            """,
            (STATUS_ERROR, message[:2000], material_id, stage),
        )
        touch_material(conn, material_id)


def mark_needs_attention(conn: sqlite3.Connection, material_id: str, stage: str, message: str) -> None:
    with transaction(conn):
        conn.execute(
            """
            UPDATE material_stages
            SET status = ?, error_message = ?, updated_at = datetime('now'), completed_at = NULL
            WHERE material_id = ? AND stage = ?
            """,
            (STATUS_NEEDS_ATTENTION, message[:2000], material_id, stage),
        )
        touch_material(conn, material_id)


def reset_stage_for_retry(conn: sqlite3.Connection, material_id: str, stage: str) -> bool:
    current = get_stage(conn, material_id, stage)
    if current is None:
        raise ValueError(f"Unknown stage {stage!r} for material {material_id}")
    if current["status"] == STATUS_COMPLETE:
        return False
    with transaction(conn):
        conn.execute(
            """
            UPDATE material_stages
            SET status = ?, error_message = NULL, started_at = NULL, completed_at = NULL,
                updated_at = datetime('now')
            WHERE material_id = ? AND stage = ?
            """,
            (STATUS_PENDING, material_id, stage),
        )
        touch_material(conn, material_id)
    return True


def recover_processing_stages(conn: sqlite3.Connection, data_root: Path, validators: dict[str, object]) -> None:
    rows = conn.execute(
        "SELECT * FROM material_stages WHERE status = ? ORDER BY updated_at",
        (STATUS_PROCESSING,),
    ).fetchall()
    for row in rows:
        artifact = path_from_db(data_root, row["artifact_path"]) or stage_artifact_path(
            data_root, row["material_id"], row["stage"]
        )
        validator = validators.get(row["stage"])
        if artifact.exists() and validator and validator(artifact, row["material_id"]):
            mark_complete(
                conn,
                data_root=data_root,
                material_id=row["material_id"],
                stage=row["stage"],
                artifact_path=artifact,
                params=json.loads(row["params_json"] or "{}"),
            )
        else:
            mark_error(conn, row["material_id"], row["stage"], "Interrupted previous run before completion")


def overall_status(conn: sqlite3.Connection, material_id: str) -> str:
    rows = conn.execute(
        "SELECT status FROM material_stages WHERE material_id = ?",
        (material_id,),
    ).fetchall()
    statuses = [row["status"] for row in rows]
    if not statuses:
        return STATUS_PENDING
    if any(status == STATUS_NEEDS_ATTENTION for status in statuses):
        return STATUS_NEEDS_ATTENTION
    if any(status == STATUS_ERROR for status in statuses):
        return STATUS_ERROR
    if any(status == STATUS_PROCESSING for status in statuses):
        return STATUS_PROCESSING
    if all(status == STATUS_COMPLETE for status in statuses):
        return STATUS_COMPLETE
    return STATUS_PENDING


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
