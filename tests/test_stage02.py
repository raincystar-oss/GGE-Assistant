from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from gge_assistant.config import AppConfig
from gge_assistant.db import STATUS_COMPLETE, STATUS_ERROR, STATUS_PROCESSING, connect, init_db
from gge_assistant.processing import CrashAfterArtifact, ProcessingContext, process_all, process_material
from gge_assistant.state import add_material, get_stage, overall_status

ROOT = Path(__file__).resolve().parents[1]


class Stage02Tests(unittest.TestCase):
    def make_app(self, tmp: Path) -> tuple[AppConfig, object]:
        config = AppConfig(
            data_root=tmp / "data",
            db_path=tmp / "data" / "gge.sqlite3",
            lock_path=tmp / "data" / "gge.lock",
            log_path=tmp / "data" / "logs" / "gge.log",
            anythingllm_base_url="http://127.0.0.1:3001/api/v1",
            ffmpeg_path="ffmpeg",
            ffprobe_path="ffprobe",
            whisper_device="cuda",
            whisper_model="large-v3",
            whisper_compute_type="float16",
            whisper_language="ru",
            whisper_beam_size=5,
            whisper_model_cache_dir=None,
            keep_audio=False,
            min_audio_duration_seconds=30.0,
            min_transcript_chars=100,
            use_test_handlers=True,
            anythingllm_api_key_path=tmp / ".private" / "anythingllm_api_key.txt",
            anythingllm_workspace_slug="gge-test",
            chunk_target_seconds=150.0,
            chunk_overlap_seconds=20.0,
            chunk_max_document_chars=950,
        )
        conn = connect(config.db_path)
        init_db(conn)
        return config, conn

    def test_completed_stages_are_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config, conn = self.make_app(Path(raw))
            material_id = add_material(
                conn,
                media_url="https://rutube.ru/video/fd02dbec883d824524affbcd27f6720f/",
                title="ok",
            )
            context = ProcessingContext(data_root=config.data_root)
            self.assertEqual(process_material(conn, material_id, context), STATUS_COMPLETE)
            first_attempts = [
                row["attempts"]
                for row in conn.execute(
                    "SELECT attempts FROM material_stages WHERE material_id = ? ORDER BY stage",
                    (material_id,),
                )
            ]
            self.assertEqual(process_material(conn, material_id, context), STATUS_COMPLETE)
            second_attempts = [
                row["attempts"]
                for row in conn.execute(
                    "SELECT attempts FROM material_stages WHERE material_id = ? ORDER BY stage",
                    (material_id,),
                )
            ]
            self.assertEqual(first_attempts, second_attempts)
            conn.close()

    def test_crash_after_file_is_recovered_without_repeating_stage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config, conn = self.make_app(Path(raw))
            material_id = add_material(conn, media_url="https://example.test/video-a", title="recover")
            with self.assertRaises(CrashAfterArtifact):
                process_material(
                    conn,
                    material_id,
                    ProcessingContext(data_root=config.data_root, crash_after_file=True),
                )
            self.assertEqual(get_stage(conn, material_id, "transcript")["status"], STATUS_PROCESSING)
            self.assertEqual(process_material(conn, material_id, ProcessingContext(config.data_root)), STATUS_COMPLETE)
            self.assertEqual(get_stage(conn, material_id, "transcript")["attempts"], 1)
            txt_path = config.data_root / "materials" / material_id / "transcript.txt"
            self.assertTrue(txt_path.exists())
            conn.close()

    def test_crash_after_json_before_txt_is_recovered_without_repeating_whisper(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config, conn = self.make_app(Path(raw))
            material_id = add_material(conn, media_url="https://example.test/video-txt", title="recover-txt")
            with self.assertRaises(CrashAfterArtifact):
                process_material(
                    conn,
                    material_id,
                    ProcessingContext(data_root=config.data_root, crash_after_json_before_txt=True),
                )
            txt_path = config.data_root / "materials" / material_id / "transcript.txt"
            self.assertFalse(txt_path.exists())
            self.assertEqual(process_material(conn, material_id, ProcessingContext(config.data_root)), STATUS_COMPLETE)
            self.assertTrue(txt_path.exists())
            self.assertEqual(get_stage(conn, material_id, "transcript")["attempts"], 1)
            conn.close()

    def test_one_material_error_does_not_block_others(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config, conn = self.make_app(Path(raw))
            bad_id = add_material(conn, media_url="https://example.test/bad", title="bad")
            good_id = add_material(conn, media_url="https://example.test/good", title="good")
            results = process_all(
                conn,
                ProcessingContext(data_root=config.data_root, fail_stage_for_title="bad"),
            )
            self.assertEqual(results[bad_id], STATUS_ERROR)
            self.assertEqual(results[good_id], STATUS_COMPLETE)
            self.assertEqual(overall_status(conn, good_id), STATUS_COMPLETE)
            conn.close()

    def test_database_enforces_unique_platform_key(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            _config, conn = self.make_app(Path(raw))
            first = add_material(conn, media_url="https://example.test/same", platform_key="rutube:same")
            second = add_material(conn, media_url="https://example.test/same-again", platform_key="rutube:same")
            count = conn.execute("SELECT count(*) AS n FROM materials").fetchone()["n"]
            self.assertEqual(first, second)
            self.assertEqual(count, 1)
            conn.close()

    def test_second_process_cannot_start_with_lock_held(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            config_path = tmp / "config.ini"
            data_root = tmp / "data"
            config_path.write_text(f"[app]\ndata_root = {data_root}\n", encoding="utf-8")
            holder = textwrap.dedent(
                f"""
                import time
                from pathlib import Path
                from gge_assistant.lock import AppLock
                with AppLock(Path(r"{data_root}") / "gge.lock"):
                    print("ready", flush=True)
                    time.sleep(5)
                """
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", holder],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(proc.stdout.readline().strip(), "ready")
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "gge_assistant",
                        "--config",
                        str(config_path),
                        "add-url",
                        "https://example.test/locked",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("already active", result.stderr)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
