from __future__ import annotations

import json
import logging
import sqlite3
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .anythingllm import AnythingLLMClient, AnythingLLMNeedsAttention
from .artifacts import atomic_write_json, validate_chunks, validate_transcript
from .chunking import ChunkingConfig, build_chunks_artifact, build_document_text, document_title, safe_source_url
from .config import AppConfig
from .db import STATUS_COMPLETE, STATUS_ERROR, STATUS_NEEDS_ATTENTION, STATUS_PENDING, STATUS_PROCESSING, transaction
from .state import (
    PIPELINE_STAGES,
    get_material,
    get_stage,
    mark_complete,
    mark_error,
    mark_needs_attention,
    mark_processing,
    overall_status,
    path_from_db,
    recover_processing_stages,
    stage_artifact_path,
)
from .transcription import transcribe_material, transcript_paths, write_transcript_txt_from_json

LOGGER = logging.getLogger(__name__)


class StageFailed(RuntimeError):
    pass


class StageNeedsAttention(RuntimeError):
    pass


class CrashAfterArtifact(BaseException):
    pass


@dataclass
class ProcessingContext:
    data_root: Path
    config: AppConfig | None = None
    use_test_handlers: bool = True
    crash_after_file: bool = False
    crash_after_json_before_txt: bool = False
    fail_stage_for_title: str | None = None
    test_lose_upload_response_once: bool = False
    progress: Callable[[str], None] | None = None
    preserve_workspace_settings: bool = False


StageHandler = Callable[[sqlite3.Connection, sqlite3.Row, ProcessingContext], Path]


VALIDATORS = {
    "transcript": validate_transcript,
    "chunks": validate_chunks,
}


def process_all(conn: sqlite3.Connection, context: ProcessingContext) -> dict[str, str]:
    recover_processing_stages(conn, context.data_root, VALIDATORS)
    _recover_transcript_txt_files(conn, context.data_root)
    rows = conn.execute("SELECT material_id FROM materials ORDER BY created_at, material_id").fetchall()
    results = {}
    for row in rows:
        material_id = row["material_id"]
        process_material(conn, material_id, context)
        results[material_id] = overall_status(conn, material_id)
    return results


def process_material(conn: sqlite3.Connection, material_id: str, context: ProcessingContext) -> str:
    recover_processing_stages(conn, context.data_root, VALIDATORS)
    _recover_transcript_txt_files(conn, context.data_root, material_id)
    material = get_material(conn, material_id)
    if material is None:
        raise ValueError(f"Unknown material_id: {material_id}")

    for stage in PIPELINE_STAGES:
        row = get_stage(conn, material_id, stage)
        if row is None:
            continue
        status = row["status"]
        if status == STATUS_COMPLETE:
            continue
        if status in {STATUS_ERROR, STATUS_NEEDS_ATTENTION}:
            LOGGER.info("Skipping %s %s because status is %s", material_id, stage, status)
            break
        if status != STATUS_PENDING:
            continue
        try:
            _process_stage(conn, material, stage, context)
        except StageNeedsAttention as exc:
            mark_needs_attention(conn, material_id, stage, str(exc))
            LOGGER.error("Stage %s needs attention for %s: %s", stage, material_id, exc)
            break
        except Exception as exc:
            mark_error(conn, material_id, stage, str(exc))
            LOGGER.error("Stage %s failed for %s: %s", stage, material_id, exc)
            break
    return overall_status(conn, material_id)


def retry_stage(conn: sqlite3.Connection, material_id: str, stage: str, context: ProcessingContext) -> str:
    from .state import reset_stage_for_retry

    reset = reset_stage_for_retry(conn, material_id, stage)
    if not reset:
        LOGGER.info("Retry requested for complete stage %s %s; keeping existing result", material_id, stage)
        return overall_status(conn, material_id)
    return process_material(conn, material_id, context)


def _process_stage(
    conn: sqlite3.Connection,
    material: sqlite3.Row,
    stage: str,
    context: ProcessingContext,
) -> None:
    handler = _handler_for(stage, context)
    material_id = material["material_id"]

    artifact = stage_artifact_path(context.data_root, material_id, stage)
    validator = VALIDATORS[stage]
    if artifact.exists() and validator(artifact, material_id) and stage != "chunks":
        LOGGER.info("Accepting existing valid %s artifact for %s", stage, material_id)
        if stage == "transcript":
            _ensure_transcript_txt(context.data_root, material_id, artifact)
        mark_complete(
            conn,
            data_root=context.data_root,
            material_id=material_id,
            stage=stage,
            artifact_path=artifact,
            params={"handler": _handler_name(context, stage), "recovered_existing_file": True},
        )
        return

    mark_processing(conn, material_id, stage)
    artifact = handler(conn, material, context)
    if not validator(artifact, material_id):
        raise StageFailed(f"{stage} handler produced an invalid artifact")
    if stage == "transcript":
        if context.crash_after_json_before_txt:
            raise CrashAfterArtifact("Intentional crash after transcript JSON before TXT/SQLite COMPLETE")
        _ensure_transcript_txt(context.data_root, material_id, artifact)
    if context.crash_after_file:
        raise CrashAfterArtifact("Intentional crash after artifact write before SQLite COMPLETE")
    mark_complete(
        conn,
        data_root=context.data_root,
        material_id=material_id,
        stage=stage,
        artifact_path=artifact,
        params={"handler": _handler_name(context, stage)},
    )
    if stage == "chunks":
        _record_chunks(conn, material_id, artifact, reset_changed=False)


def _test_transcript_handler(
    conn: sqlite3.Connection,
    material: sqlite3.Row,
    context: ProcessingContext,
) -> Path:
    if context.fail_stage_for_title and material["title"] == context.fail_stage_for_title:
        raise StageFailed("Intentional test failure")
    material_id = material["material_id"]
    artifact = stage_artifact_path(context.data_root, material_id, "transcript")
    payload = {
        "schema_version": 1,
        "material_id": material_id,
        "metadata": {
            "platform_key": material["platform_key"],
            "title": material["title"],
            "webinar_date": material["webinar_date"],
            "source_url_kind": "media_url",
        },
        "duration": 12.0,
        "model": {
            "name": "test-handler",
            "device": "cpu",
            "language": "ru",
            "parameters": {},
        },
        "segments": [
            {"start": 0.0, "end": 6.0, "text": "Тестовый фрагмент расшифровки."},
            {"start": 6.0, "end": 12.0, "text": "Второй фрагмент для проверки восстановления."},
        ],
    }
    atomic_write_json(artifact, payload)
    return artifact


def _test_chunks_handler(
    conn: sqlite3.Connection,
    material: sqlite3.Row,
    context: ProcessingContext,
) -> Path:
    material_id = material["material_id"]
    transcript_path = path_from_db(
        context.data_root,
        get_stage(conn, material_id, "transcript")["artifact_path"],
    )
    if transcript_path is None or not transcript_path.exists():
        raise StageFailed("Transcript is required before chunks")
    transcript_hash = _sha256(transcript_path)
    with transcript_path.open("r", encoding="utf-8") as fh:
        transcript = json.load(fh)
    text = " ".join(segment["text"] for segment in transcript["segments"])
    payload = {
        "schema_version": 1,
        "material_id": material_id,
        "chunking": {
            "version": "test-v1",
            "target_seconds": 150,
            "overlap_seconds": 20,
        },
        "source_transcript": {
            "path": get_stage(conn, material_id, "transcript")["artifact_path"],
            "sha256": transcript_hash,
        },
        "chunks": [
            {
                "chunk_id": f"{material_id}:test-v1:0000",
                "index": 0,
                "start": 0.0,
                "end": transcript["duration"],
                "text": text,
            }
        ],
    }
    artifact = stage_artifact_path(context.data_root, material_id, "chunks")
    atomic_write_json(artifact, payload)
    return artifact


def _real_transcript_handler(
    conn: sqlite3.Connection,
    material: sqlite3.Row,
    context: ProcessingContext,
) -> Path:
    if context.config is None:
        raise StageFailed("Real transcript handler requires AppConfig")
    material_id = material["material_id"]
    artifact = stage_artifact_path(context.data_root, material_id, "transcript")
    if context.progress:
        return transcribe_material(context.config, material, artifact, progress=context.progress)
    return transcribe_material(context.config, material, artifact)


def _real_chunks_handler(
    conn: sqlite3.Connection,
    material: sqlite3.Row,
    context: ProcessingContext,
) -> Path:
    if context.config is None:
        raise StageFailed("Real chunks handler requires AppConfig")
    material_id = material["material_id"]
    transcript_stage = get_stage(conn, material_id, "transcript")
    if transcript_stage is None or transcript_stage["status"] != STATUS_COMPLETE:
        raise StageFailed("Transcript is required before chunks")
    transcript_path = path_from_db(context.data_root, transcript_stage["artifact_path"])
    if transcript_path is None or not transcript_path.exists():
        raise StageFailed("Transcript artifact is missing")

    artifact = stage_artifact_path(context.data_root, material_id, "chunks")
    if context.progress:
        context.progress("chunking")
    chunk_config = ChunkingConfig(
        target_seconds=context.config.chunk_target_seconds,
        overlap_seconds=context.config.chunk_overlap_seconds,
        max_document_chars=context.config.chunk_max_document_chars,
    )
    if not artifact.exists() or not validate_chunks(artifact, material_id):
        build_chunks_artifact(
            transcript_path=transcript_path,
            output_path=artifact,
            data_root=context.data_root,
            material=_material_dict(material),
            config=chunk_config,
        )
    _record_chunks(conn, material_id, artifact, reset_changed=True)
    _ingest_chunks(conn, material, context, artifact, transcript_path)
    return artifact


def _sha256(path: Path) -> str:
    from .state import sha256_file

    return sha256_file(path)


def _record_chunks(conn: sqlite3.Connection, material_id: str, artifact: Path, *, reset_changed: bool) -> None:
    with artifact.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    with transaction(conn):
        for chunk in payload["chunks"]:
            digest = _text_sha256(chunk["text"])
            existing = conn.execute(
                "SELECT text_sha256, status FROM chunks WHERE chunk_id = ?",
                (chunk["chunk_id"],),
            ).fetchone()
            status = STATUS_PENDING
            if existing and existing["text_sha256"] == digest:
                status = existing["status"]
            elif existing and existing["text_sha256"] != digest and not reset_changed:
                status = existing["status"]
            conn.execute(
                """
                INSERT INTO chunks
                (chunk_id, material_id, chunk_index, start, end, text_sha256, status, document_id, document_title, document_location, workspace_doc_id, last_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    start = excluded.start,
                    end = excluded.end,
                    text_sha256 = excluded.text_sha256,
                    status = CASE
                        WHEN chunks.text_sha256 = excluded.text_sha256 THEN chunks.status
                        ELSE excluded.status
                    END,
                    document_id = CASE WHEN chunks.text_sha256 = excluded.text_sha256 THEN chunks.document_id ELSE NULL END,
                    document_title = CASE WHEN chunks.text_sha256 = excluded.text_sha256 THEN chunks.document_title ELSE NULL END,
                    document_location = CASE WHEN chunks.text_sha256 = excluded.text_sha256 THEN chunks.document_location ELSE NULL END,
                    workspace_doc_id = CASE WHEN chunks.text_sha256 = excluded.text_sha256 THEN chunks.workspace_doc_id ELSE NULL END,
                    last_error = CASE WHEN chunks.text_sha256 = excluded.text_sha256 THEN chunks.last_error ELSE NULL END,
                    updated_at = datetime('now')
                """,
                (
                    chunk["chunk_id"],
                    material_id,
                    chunk["index"],
                    chunk["start"],
                    chunk["end"],
                    digest,
                    status,
                ),
            )


def _ingest_chunks(
    conn: sqlite3.Connection,
    material: sqlite3.Row,
    context: ProcessingContext,
    chunks_path: Path,
    transcript_path: Path,
) -> None:
    config = context.config
    if config is None:
        raise StageFailed("AppConfig is required for AnythingLLM ingestion")
    with chunks_path.open("r", encoding="utf-8") as fh:
        chunks_payload = json.load(fh)
    with transcript_path.open("r", encoding="utf-8") as fh:
        transcript = json.load(fh)
    material_dict = _material_dict(material)
    prompt = (
        "Отвечай только по найденным источникам ГГЭ. Указывай название, дату и диапазон времени источника. "
        "Если источники относятся к соседним временным диапазонам одного вебинара, объединяй их в один ответ. "
        "Если найденные источники похожи по словам, но не подтверждают вопрос, прямо скажи, что данных недостаточно."
    )
    lose_once = context.test_lose_upload_response_once
    with AnythingLLMClient(
        base_url=config.anythingllm_base_url,
        api_key_path=config.anythingllm_api_key_path,
        workspace_slug=config.anythingllm_workspace_slug,
    ) as client:
        if context.preserve_workspace_settings:
            client.workspace()
        else:
            client.ensure_workspace("gge", prompt)
        if context.progress:
            context.progress("indexing")
        for chunk in chunks_payload["chunks"]:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk["chunk_id"],)).fetchone()
            if row and row["status"] == STATUS_COMPLETE:
                continue
            document_text = build_document_text(
                chunk,
                transcript,
                material_dict,
                max_chars=config.chunk_max_document_chars,
            )
            digest = hashlib.sha256(document_text.encode("utf-8")).hexdigest()
            title = document_title(material_dict, chunk, document_text)
            _mark_chunk_processing(conn, chunk["chunk_id"], title)
            try:
                document = client.upload_or_locate_document(
                    title=title,
                    text=document_text,
                    sha256=digest,
                    url=safe_source_url(material_dict.get("media_url") or ""),
                    published=_published_ms(transcript, material_dict),
                    lose_response_for_test=lose_once,
                )
                lose_once = False
                workspace_doc_id = client.ensure_workspace_document(document.location)
            except AnythingLLMNeedsAttention as exc:
                _mark_chunk_attention(conn, chunk["chunk_id"], str(exc))
                raise StageNeedsAttention(str(exc)) from exc
            except Exception as exc:
                _mark_chunk_error(conn, chunk["chunk_id"], str(exc))
                raise
            _mark_chunk_complete(conn, chunk["chunk_id"], document, workspace_doc_id)
    unfinished = conn.execute(
        "SELECT count(*) AS n FROM chunks WHERE material_id = ? AND status != ?",
        (material["material_id"], STATUS_COMPLETE),
    ).fetchone()["n"]
    if unfinished:
        raise StageFailed(f"{unfinished} chunks are not indexed")


def _text_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ensure_transcript_txt(data_root: Path, material_id: str, transcript_path: Path) -> None:
    paths = transcript_paths(data_root, material_id)
    write_transcript_txt_from_json(transcript_path, paths.txt_path, material_id)


def _handler_for(stage: str, context: ProcessingContext) -> StageHandler:
    if context.use_test_handlers:
        return TEST_HANDLERS[stage]
    if stage == "transcript":
        return _real_transcript_handler
    if stage == "chunks":
        return _real_chunks_handler
    return TEST_HANDLERS[stage]


def _handler_name(context: ProcessingContext, stage: str) -> str:
    if context.use_test_handlers:
        return "test"
    if stage == "transcript":
        return "faster-whisper"
    if stage == "chunks":
        return "anythingllm"
    return "unknown"


def _material_dict(material: sqlite3.Row) -> dict:
    return {key: material[key] for key in material.keys()}


def _mark_chunk_processing(conn: sqlite3.Connection, chunk_id: str, title: str) -> None:
    with transaction(conn):
        conn.execute(
            """
            UPDATE chunks
            SET status = ?, document_title = ?, last_error = NULL, updated_at = datetime('now')
            WHERE chunk_id = ?
            """,
            (STATUS_PROCESSING, title, chunk_id),
        )


def _mark_chunk_complete(conn: sqlite3.Connection, chunk_id: str, document, workspace_doc_id: str) -> None:
    with transaction(conn):
        conn.execute(
            """
            UPDATE chunks
            SET status = ?, document_id = ?, document_title = ?, document_location = ?,
                workspace_doc_id = ?, last_error = NULL, updated_at = datetime('now')
            WHERE chunk_id = ?
            """,
            (
                STATUS_COMPLETE,
                document.document_id,
                document.title,
                document.location,
                workspace_doc_id,
                chunk_id,
            ),
        )


def _mark_chunk_error(conn: sqlite3.Connection, chunk_id: str, message: str) -> None:
    with transaction(conn):
        conn.execute(
            """
            UPDATE chunks
            SET status = ?, last_error = ?, updated_at = datetime('now')
            WHERE chunk_id = ?
            """,
            (STATUS_ERROR, message[:2000], chunk_id),
        )


def _mark_chunk_attention(conn: sqlite3.Connection, chunk_id: str, message: str) -> None:
    with transaction(conn):
        conn.execute(
            """
            UPDATE chunks
            SET status = ?, last_error = ?, updated_at = datetime('now')
            WHERE chunk_id = ?
            """,
            (STATUS_NEEDS_ATTENTION, message[:2000], chunk_id),
        )


def _published_ms(transcript: dict, material: dict) -> int | None:
    raw = transcript.get("metadata", {}).get("webinar_date") or material.get("webinar_date") or material.get("published_at")
    if not raw:
        return None
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


def _recover_transcript_txt_files(
    conn: sqlite3.Connection,
    data_root: Path,
    material_id: str | None = None,
) -> None:
    sql = """
        SELECT * FROM material_stages
        WHERE stage = 'transcript' AND status = ?
    """
    params: tuple[object, ...] = (STATUS_COMPLETE,)
    if material_id is not None:
        sql += " AND material_id = ?"
        params = (STATUS_COMPLETE, material_id)
    rows = conn.execute(sql, params).fetchall()
    for row in rows:
        artifact = path_from_db(data_root, row["artifact_path"]) or stage_artifact_path(
            data_root, row["material_id"], "transcript"
        )
        paths = transcript_paths(data_root, row["material_id"])
        if artifact.exists() and not paths.txt_path.exists():
            _ensure_transcript_txt(data_root, row["material_id"], artifact)


TEST_HANDLERS: dict[str, StageHandler] = {
    "transcript": _test_transcript_handler,
    "chunks": _test_chunks_handler,
}
