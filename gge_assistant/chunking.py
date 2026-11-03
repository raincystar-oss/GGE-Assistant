from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .artifacts import atomic_write_json
from .state import relative_to_data_root, sha256_file

CHUNKING_VERSION = "v1"


@dataclass(frozen=True)
class ChunkingConfig:
    target_seconds: float = 240.0
    overlap_seconds: float = 45.0
    max_document_chars: int = 1800


def build_chunks_artifact(
    *,
    transcript_path: Path,
    output_path: Path,
    data_root: Path,
    material: dict,
    config: ChunkingConfig,
) -> Path:
    with transcript_path.open("r", encoding="utf-8") as fh:
        transcript = json.load(fh)
    chunks = chunk_transcript(transcript, material, config)
    payload = {
        "schema_version": 1,
        "material_id": transcript["material_id"],
        "chunking": {
            "version": CHUNKING_VERSION,
            "target_seconds": config.target_seconds,
            "overlap_seconds": config.overlap_seconds,
            "max_document_chars": config.max_document_chars,
        },
        "source_transcript": {
            "path": relative_to_data_root(data_root, transcript_path),
            "sha256": sha256_file(transcript_path),
        },
        "chunks": chunks,
    }
    atomic_write_json(output_path, payload)
    return output_path


def chunk_transcript(transcript: dict, material: dict, config: ChunkingConfig) -> list[dict]:
    raw_segments = []
    for segment in transcript.get("segments", []):
        text = normalize_text(segment.get("text", ""))
        if text:
            raw_segments.append(
                {
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "text": text,
                }
            )
    segments = remove_known_noise_runs(raw_segments)
    chunks: list[dict] = []
    index = 0
    i = 0
    while i < len(segments):
        start = segments[i]["start"]
        body_limit = body_char_limit(transcript, material, config, start, segments[i]["end"])
        selected: list[dict] = []
        text_len = 0
        j = i
        while j < len(segments):
            candidate = segments[j]
            candidate_text_len = len(candidate["text"]) + (1 if selected else 0)
            over_target = bool(selected) and candidate["end"] > start + config.target_seconds
            over_chars = bool(selected) and text_len + candidate_text_len > body_limit
            if over_target or over_chars:
                break
            if candidate_text_len > body_limit and not selected:
                for part in split_long_text(candidate["text"], body_limit):
                    chunks.append(_chunk(transcript["material_id"], index, candidate["start"], candidate["end"], part))
                    index += 1
                j += 1
                break
            selected.append(candidate)
            text_len += candidate_text_len
            j += 1
        if selected:
            chunks.append(
                _chunk(
                    transcript["material_id"],
                    index,
                    selected[0]["start"],
                    selected[-1]["end"],
                    " ".join(segment["text"] for segment in selected),
                )
            )
            index += 1

        next_i = j
        if selected and j < len(segments):
            overlap_start = max(selected[-1]["end"] - config.overlap_seconds, selected[0]["start"])
            for candidate_i in range(i + 1, j + 1):
                if candidate_i >= len(segments) or segments[candidate_i]["end"] > overlap_start:
                    next_i = candidate_i
                    break
        if next_i <= i:
            next_i = i + 1
        i = next_i
    return chunks


def build_document_text(chunk: dict, transcript: dict, material: dict, max_chars: int | None = None) -> str:
    metadata = transcript.get("metadata", {})
    title = metadata.get("title") or material.get("title") or "ГГЭ вебинар"
    webinar_date = metadata.get("webinar_date") or material.get("webinar_date") or "не указана"
    source_url = safe_source_url(material.get("media_url") or "")
    time_range = f"{format_time(chunk['start'])}-{format_time(chunk['end'])}"
    header = (
        f"Источник: {title} | {webinar_date} | {time_range} | "
        f"{source_url} | {chunk['chunk_id']}\n"
        "Текст:\n"
    )
    text = header + "\n".join(_source_anchored_blocks(normalize_text(chunk["text"]), time_range)) + "\n"
    if max_chars is not None and len(text) > max_chars:
        raise ValueError(f"Document text exceeds max_document_chars: {len(text)} > {max_chars}")
    return text


def document_title(material: dict, chunk: dict, document_text: str) -> str:
    digest = hashlib.sha256(document_text.encode("utf-8")).hexdigest()[:12]
    source = material.get("platform_key", "gge")
    source_id = source.split(":", 1)[-1][:8]
    return f"gge-{source_id}-{CHUNKING_VERSION}-{chunk['index']:04d}-{digest}"


def body_char_limit(transcript: dict, material: dict, config: ChunkingConfig, start: float, end: float) -> int:
    probe = {"chunk_id": "probe", "index": 0, "start": start, "end": end, "text": ""}
    header_len = len(build_document_text(probe, transcript, material))
    return max(120, config.max_document_chars - header_len - 260)


def split_long_text(text: str, limit: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > limit:
            parts.append(" ".join(current))
            current = [word]
            current_len = len(word)
        elif len(word) > limit:
            if current:
                parts.append(" ".join(current))
                current = []
                current_len = 0
            parts.extend(word[i : i + limit] for i in range(0, len(word), limit))
        else:
            current.append(word)
            current_len += extra
    if current:
        parts.append(" ".join(current))
    return parts


def _source_anchored_blocks(text: str, time_range: str) -> list[str]:
    if not text:
        return []
    return [f"Фрагмент {time_range}: {part}" for part in split_long_text(text, 650)]


def remove_known_noise_runs(segments: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    i = 0
    while i < len(segments):
        text = segments[i]["text"].casefold()
        if text == "humid":
            for j in range(i + 1, min(i + 7, len(segments))):
                if segments[j]["text"].casefold() == "texts":
                    i = j + 1
                    break
            else:
                cleaned.append(segments[i])
                i += 1
            continue
        cleaned.append(segments[i])
        i += 1
    return cleaned


def normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if re.fullmatch(r"(?i)Продолжение следует[.!?…]*", normalized):
        return ""
    normalized = re.sub(r"(?i)(?:\bПродолжение следует[.!?…]*\s*){2,}", "", normalized).strip()
    normalized = re.sub(r"humid\s+чем\s+потребитель\s+\S+\s+texts", "", normalized, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", normalized).strip()


def format_time(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def safe_source_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _chunk(material_id: str, index: int, start: float, end: float, text: str) -> dict:
    return {
        "chunk_id": f"{material_id}:{CHUNKING_VERSION}:{index:04d}",
        "index": index,
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text,
    }
