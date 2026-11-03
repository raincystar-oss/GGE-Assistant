from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from .artifacts import atomic_write_json, validate_transcript
from .config import AppConfig

LOGGER = logging.getLogger(__name__)
_DLL_DIRECTORY_HANDLES: list[object] = []
# Prefer an audio-only track; otherwise avoid downloading HD video solely for ASR.
DOWNLOAD_FORMAT = "bestaudio/best[height<=360]/best"


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptPaths:
    material_dir: Path
    json_path: Path
    txt_path: Path
    audio_path: Path
    download_dir: Path


def transcript_paths(data_root: Path, material_id: str) -> TranscriptPaths:
    material_dir = data_root / "materials" / material_id
    return TranscriptPaths(
        material_dir=material_dir,
        json_path=material_dir / "transcript.json",
        txt_path=material_dir / "transcript.txt",
        audio_path=material_dir / "audio.wav",
        download_dir=material_dir / "download",
    )


def write_transcript_txt_from_json(json_path: Path, txt_path: Path, material_id: str) -> None:
    if not validate_transcript(json_path, material_id):
        raise TranscriptionError("Cannot build TXT from invalid transcript.json")
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    lines = []
    for segment in payload["segments"]:
        text = " ".join(segment["text"].split())
        if text:
            lines.append(f"[{_hms(segment['start'])}-{_hms(segment['end'])}] {text}")
    if not lines:
        raise TranscriptionError("Transcript has no text segments")
    _atomic_write_text(txt_path, "\n".join(lines) + "\n")


def transcribe_material(config: AppConfig, material: Any, json_path: Path, *,
                        progress: Callable[[str], None] | None = None) -> Path:
    material_id = material["material_id"]
    paths = transcript_paths(config.data_root, material_id)
    paths.material_dir.mkdir(parents=True, exist_ok=True)

    if validate_transcript(json_path, material_id):
        write_transcript_txt_from_json(json_path, paths.txt_path, material_id)
        return json_path

    if progress:
        progress("audio")
    audio_path = _ensure_audio(config, material, paths)
    duration = _probe_duration(config.ffprobe_path, audio_path)
    if duration < config.min_audio_duration_seconds:
        raise TranscriptionError(f"Audio duration is implausibly short: {duration:.2f}s")

    started = time.perf_counter()
    model_kwargs: dict[str, Any] = {
        "device": config.whisper_device,
        "compute_type": config.whisper_compute_type,
    }
    if config.whisper_model_cache_dir is not None:
        model_kwargs["download_root"] = str(config.whisper_model_cache_dir)

    if progress:
        progress("transcribing")
    _add_nvidia_dll_directories()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError("Missing dependency: faster-whisper") from exc

    model = WhisperModel(config.whisper_model, **model_kwargs)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=config.whisper_language,
        beam_size=config.whisper_beam_size,
        condition_on_previous_text=True,
    )
    segments = [
        {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "text": segment.text.strip(),
        }
        for segment in segments_iter
        if segment.text and segment.text.strip()
    ]
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "material_id": material_id,
        "metadata": {
            "platform_key": material["platform_key"],
            "title": material["title"],
            "webinar_date": material["webinar_date"],
            "published_at": material["published_at"],
            "source_url_kind": "media_url",
            "language_probability": getattr(info, "language_probability", None),
            "transcription_elapsed_seconds": round(elapsed, 3),
        },
        "duration": round(duration, 3),
        "model": {
            "name": config.whisper_model,
            "device": config.whisper_device,
            "language": config.whisper_language,
            "parameters": {
                "compute_type": config.whisper_compute_type,
                "beam_size": config.whisper_beam_size,
                "condition_on_previous_text": True,
            },
        },
        "segments": segments,
    }
    # Preserve the raw pass even when a single bad timestamp prevents acceptance.
    atomic_write_json(paths.material_dir / 'transcript_primary_raw.json', payload)
    payload, primary_timing = _screen_timestamps(payload)
    _validate_segments_for_audio(payload['segments'], duration, config.min_transcript_chars)
    if config.asr_quality_check:
        if progress:
            progress("quality")
        from .asr_quality import screen_passes
        import wave

        import numpy as np

        primary = payload
        second_iter, _ = model.transcribe(
            str(audio_path), language=config.whisper_language,
            beam_size=config.whisper_beam_size, condition_on_previous_text=False,
        )
        secondary = dict(primary)
        secondary["model"] = dict(primary["model"])
        secondary["model"]["parameters"] = {
            **primary["model"]["parameters"], "condition_on_previous_text": False,
        }
        secondary["segments"] = [
            {"start": round(float(s.start), 3), "end": round(float(s.end), 3), "text": s.text.strip()}
            for s in second_iter if s.text and s.text.strip()
        ]
        atomic_write_json(paths.material_dir / 'transcript_second_pass_raw.json', secondary)
        secondary, secondary_timing = _screen_timestamps(secondary)
        _validate_segments_for_audio(secondary["segments"], duration, config.min_transcript_chars)

        with wave.open(str(audio_path), "rb") as audio:
            if audio.getframerate() != 16000 or audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise TranscriptionError("ASR quality check requires 16 kHz mono PCM audio")

            def recognize_clip(lo: float, hi: float) -> list[dict[str, Any]]:
                audio.setpos(int(lo * audio.getframerate()))
                raw = audio.readframes(int((hi - lo) * audio.getframerate()))
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                iterator, _ = model.transcribe(
                    samples, language=config.whisper_language,
                    beam_size=config.whisper_beam_size, condition_on_previous_text=False,
                )
                return [{"start": round(lo + float(s.start), 3),
                         "end": round(lo + float(s.end), 3), "text": s.text.strip()}
                        for s in iterator if s.text and s.text.strip()]

            screened, quality_report = screen_passes(primary, secondary, audio_path, recognize_clip)
        atomic_write_json(paths.material_dir / 'transcript_screened_raw.json', screened)
        screened, candidate_timing = _screen_timestamps(screened)
        _validate_segments_for_audio(screened["segments"], duration, config.min_transcript_chars)
        screened["metadata"] = dict(screened["metadata"])
        screened["metadata"]["transcription_elapsed_seconds"] = round(time.perf_counter() - started, 3)
        screened["model"] = dict(screened["model"])
        screened["model"]["parameters"] = {
            **screened["model"]["parameters"], "automatic_quality_check": True,
        }
        atomic_write_json(paths.material_dir / "transcript_primary.json", primary)
        atomic_write_json(paths.material_dir / "transcript_second_pass.json", secondary)
        payload = screened
        quality_report['timestamp_checks'] = {'primary': primary_timing, 'secondary': secondary_timing,
                                               'screened': candidate_timing}
        if primary_timing or secondary_timing or candidate_timing:
            payload['quality']['status'] = 'partial'
            payload['quality']['timestamp_exclusions'] = len(primary_timing) + len(secondary_timing) + len(candidate_timing)
            quality_report['quality'] = payload['quality']
        atomic_write_json(paths.material_dir / 'asr_quality.json', quality_report)
    elif primary_timing:
        payload['quality'] = {'status': 'partial', 'timestamp_exclusions': len(primary_timing)}
        atomic_write_json(paths.material_dir / 'asr_quality.json', {'timestamp_checks': {'primary': primary_timing},
                          'quality': payload['quality']})
    atomic_write_json(json_path, payload)
    write_transcript_txt_from_json(json_path, paths.txt_path, material_id)

    if not config.keep_audio and payload.get("quality", {}).get("status") != "partial":
        _remove_if_exists(audio_path)
    _remove_tree_if_exists(paths.download_dir)
    LOGGER.info("Transcribed %s in %.3fs with %s/%s", material_id, elapsed, config.whisper_model, config.whisper_device)
    return json_path


def extract_url_from_text(text: str) -> str:
    match = re.search(r"https?://[^\s)>\]]+", text)
    if not match:
        raise ValueError("No URL found")
    return match.group(0)


def _ensure_audio(config: AppConfig, material: Any, paths: TranscriptPaths) -> Path:
    if paths.audio_path.exists():
        duration = _probe_duration(config.ffprobe_path, paths.audio_path)
        if duration >= config.min_audio_duration_seconds:
            return paths.audio_path
        _remove_if_exists(paths.audio_path)

    paths.download_dir.mkdir(parents=True, exist_ok=True)
    _remove_download_partials(paths.download_dir)
    source_path = _download_media(material["media_url"], paths.download_dir)
    try:
        _extract_audio(config.ffmpeg_path, source_path, paths.audio_path)
        duration = _probe_duration(config.ffprobe_path, paths.audio_path)
        if duration < config.min_audio_duration_seconds:
            raise TranscriptionError(f"Extracted audio duration is implausibly short: {duration:.2f}s")
    finally:
        _remove_if_exists(source_path)
    return paths.audio_path


def _download_media(url: str, download_dir: Path) -> Path:
    local_path = _local_file_from_url(url)
    if local_path is not None:
        if not local_path.exists():
            raise TranscriptionError("Local media file does not exist")
        destination = download_dir / f"source{local_path.suffix or '.media'}"
        if local_path.resolve() != destination.resolve():
            shutil.copy2(local_path, destination)
        return destination

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise TranscriptionError("Missing dependency: yt-dlp") from exc

    outtmpl = str(download_dir / "source.%(ext)s")
    options = {
        "format": DOWNLOAD_FORMAT,
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
        "socket_timeout": 60,
    }
    before = set(download_dir.glob("source.*"))
    try:
        with YoutubeDL(options) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        raise TranscriptionError(_redact_secrets(str(exc))) from exc
    candidates = [path for path in download_dir.glob("source.*") if not path.name.endswith(".part")]
    candidates = [path for path in candidates if path not in before or path.exists()]
    if not candidates:
        raise TranscriptionError("yt-dlp did not produce a media file")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _extract_audio(ffmpeg_path: str, source_path: Path, audio_path: Path) -> None:
    tmp_path = audio_path.with_name(f".{audio_path.name}.{os.getpid()}.tmp.wav")
    _remove_if_exists(tmp_path)
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(tmp_path),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        _remove_if_exists(tmp_path)
        raise TranscriptionError(f"FFmpeg failed to extract audio: {result.stderr.strip()[:500]}")
    os.replace(tmp_path, audio_path)


def _probe_duration(ffprobe_path: str, path: Path) -> float:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise TranscriptionError(f"ffprobe could not read audio: {result.stderr.strip()[:500]}")
    try:
        duration = float(result.stdout.strip())
    except ValueError as exc:
        raise TranscriptionError("ffprobe did not return a numeric duration") from exc
    if duration <= 0:
        raise TranscriptionError("Audio duration is zero")
    return duration


def _screen_timestamps(payload: dict) -> tuple[dict, list[dict]]:
    """Quarantine invalid timing; never invent or move a spoken phrase in time."""
    kept, issues = [], []
    previous_end = 0.0
    for i, segment in enumerate(payload['segments']):
        start, end = segment['start'], segment['end']
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0
                or end < start or end > payload['duration'] + 5 or start + .001 < previous_end):
            issues.append({'segment_index': i, 'reason': 'invalid_timing'})
            continue
        kept.append(segment)
        previous_end = end
    return {**payload, 'segments': kept}, issues


def _validate_segments_for_audio(segments: list[dict[str, Any]], duration: float, min_chars: int) -> None:
    if not segments:
        raise TranscriptionError("Whisper returned no text segments")
    previous_end = 0.0
    total_text = 0
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        if start < 0 or end < start or end > duration + 5:
            raise TranscriptionError("Whisper returned invalid segment boundaries")
        if start + 0.001 < previous_end:
            raise TranscriptionError("Whisper segments are not ordered")
        previous_end = end
        total_text += len(text)
    if total_text < min_chars:
        raise TranscriptionError("Whisper result is too short for a webinar recording")


def _hms(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def _remove_download_partials(path: Path) -> None:
    for partial in path.glob("*.part"):
        _remove_if_exists(partial)


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _remove_tree_if_exists(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _redact_secrets(text: str) -> str:
    text = re.sub(r"([?&]p=)[^&\s'\"<>]+", r"\1<redacted>", text)
    text = re.sub(r"(https?://[^\s'\"<>?]+)\?[^\s'\"<>]+", r"\1?<redacted>", text)
    return text


def _local_file_from_url(url: str) -> Path | None:
    parsed = urlparse(url)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path.lstrip("/"))) if os.name == "nt" else Path(unquote(parsed.path))
    if parsed.scheme == "" and Path(url).exists():
        return Path(url)
    return None


def _add_nvidia_dll_directories() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates: list[Path] = []
    for module_name, parts in (
        ("nvidia.cublas", ("bin",)),
        ("nvidia.cudnn", ("bin",)),
        ("nvidia.cudnn", ("lib",)),
    ):
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.submodule_search_locations is None:
            continue
        for location in spec.submodule_search_locations:
            candidates.append(Path(location).resolve().joinpath(*parts))
    for directory in candidates:
        if directory.exists():
            handle = os.add_dll_directory(str(directory))
            _DLL_DIRECTORY_HANDLES.append(handle)
            current_path = os.environ.get("PATH", "")
            directory_text = str(directory)
            if directory_text.lower() not in {part.lower() for part in current_path.split(os.pathsep)}:
                os.environ["PATH"] = directory_text + os.pathsep + current_path
