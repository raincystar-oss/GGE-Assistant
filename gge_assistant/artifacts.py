from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    validate_json_file(tmp_path)
    os.replace(tmp_path, path)


def validate_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_transcript(path: Path, material_id: str) -> bool:
    try:
        payload = validate_json_file(path)
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("schema_version") != 1 or payload.get("material_id") != material_id:
        return False
    if not isinstance(payload.get("metadata"), dict):
        return False
    duration = payload.get("duration")
    if not isinstance(duration, (int, float)) or duration < 0:
        return False
    if not isinstance(payload.get("model"), dict):
        return False
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return False
    previous_end = 0.0
    for segment in segments:
        if not _valid_segment(segment):
            return False
        if segment["start"] + 0.001 < previous_end:
            return False
        if segment["end"] > duration + 5:
            return False
        previous_end = segment["end"]
    return True


def validate_chunks(path: Path, material_id: str) -> bool:
    try:
        payload = validate_json_file(path)
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("schema_version") != 1 or payload.get("material_id") != material_id:
        return False
    if not isinstance(payload.get("chunking"), dict):
        return False
    if not isinstance(payload.get("source_transcript"), dict):
        return False
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        return False
    return all(_valid_chunk(chunk, material_id) for chunk in chunks)


def _valid_segment(segment: object) -> bool:
    if not isinstance(segment, dict):
        return False
    return (
        isinstance(segment.get("start"), (int, float))
        and isinstance(segment.get("end"), (int, float))
        and segment["end"] >= segment["start"]
        and isinstance(segment.get("text"), str)
        and bool(segment["text"].strip())
    )


def _valid_chunk(chunk: object, material_id: str) -> bool:
    if not isinstance(chunk, dict):
        return False
    return (
        isinstance(chunk.get("chunk_id"), str)
        and chunk["chunk_id"].startswith(f"{material_id}:")
        and isinstance(chunk.get("index"), int)
        and isinstance(chunk.get("start"), (int, float))
        and isinstance(chunk.get("end"), (int, float))
        and chunk["end"] >= chunk["start"]
        and isinstance(chunk.get("text"), str)
    )
