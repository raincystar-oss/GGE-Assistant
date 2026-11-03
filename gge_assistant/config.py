from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    data_root: Path
    db_path: Path
    lock_path: Path
    log_path: Path
    anythingllm_base_url: str
    anythingllm_api_key_path: Path
    anythingllm_workspace_slug: str
    chunk_target_seconds: float
    chunk_overlap_seconds: float
    chunk_max_document_chars: int
    ffmpeg_path: str
    ffprobe_path: str
    whisper_device: str
    whisper_model: str
    whisper_compute_type: str
    whisper_language: str
    whisper_beam_size: int
    whisper_model_cache_dir: Path | None
    keep_audio: bool
    min_audio_duration_seconds: float
    min_transcript_chars: int
    use_test_handlers: bool
    asr_quality_check: bool = True


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path) if config_path else Path("config.ini")
    parser = configparser.ConfigParser()
    if path.exists():
        parser.read(path, encoding="utf-8")

    data_root = Path(parser.get("app", "data_root", fallback="data")).expanduser()
    if not data_root.is_absolute():
        data_root = (Path.cwd() / data_root).resolve()

    db_path = _path_value(parser, "app", "db_path", data_root / "gge.sqlite3", data_root)
    lock_path = _path_value(parser, "app", "lock_path", data_root / "gge.lock", data_root)
    log_path = _path_value(parser, "app", "log_path", data_root / "logs" / "gge.log", data_root)

    return AppConfig(
        data_root=data_root,
        db_path=db_path,
        lock_path=lock_path,
        log_path=log_path,
        anythingllm_base_url=parser.get("anythingllm", "base_url", fallback="http://127.0.0.1:3001/api/v1"),
        anythingllm_api_key_path=_path_value(
            parser,
            "anythingllm",
            "api_key_path",
            Path.cwd() / ".private" / "anythingllm_api_key.txt",
            Path.cwd(),
        ),
        anythingllm_workspace_slug=parser.get("anythingllm", "workspace_slug", fallback="gge"),
        chunk_target_seconds=parser.getfloat("chunks", "target_seconds", fallback=240.0),
        chunk_overlap_seconds=parser.getfloat("chunks", "overlap_seconds", fallback=45.0),
        chunk_max_document_chars=parser.getint("chunks", "max_document_chars", fallback=1800),
        ffmpeg_path=parser.get("tools", "ffmpeg_path", fallback="ffmpeg"),
        ffprobe_path=parser.get("tools", "ffprobe_path", fallback="ffprobe"),
        whisper_device=parser.get("whisper", "device", fallback="cuda"),
        whisper_model=parser.get("whisper", "model", fallback="large-v3"),
        whisper_compute_type=parser.get("whisper", "compute_type", fallback="float16"),
        whisper_language=parser.get("whisper", "language", fallback="ru"),
        whisper_beam_size=parser.getint("whisper", "beam_size", fallback=5),
        whisper_model_cache_dir=_optional_path_value(parser, "whisper", "model_cache_dir", Path.cwd()),
        keep_audio=parser.getboolean("audio", "keep_audio", fallback=False),
        min_audio_duration_seconds=parser.getfloat("audio", "min_audio_duration_seconds", fallback=30.0),
        min_transcript_chars=parser.getint("audio", "min_transcript_chars", fallback=100),
        use_test_handlers=parser.getboolean("app", "use_test_handlers", fallback=False),
        asr_quality_check=parser.getboolean("audio", "quality_check", fallback=True),
    )


def _path_value(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    fallback: Path,
    data_root: Path,
) -> Path:
    raw = parser.get(section, option, fallback=str(fallback))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


def _optional_path_value(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    base: Path,
) -> Path | None:
    if not parser.has_option(section, option):
        return None
    raw = parser.get(section, option).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
