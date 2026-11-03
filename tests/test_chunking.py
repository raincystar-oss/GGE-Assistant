from __future__ import annotations

import unittest

from gge_assistant.chunking import ChunkingConfig, build_document_text, chunk_transcript, normalize_text


class ChunkingTests(unittest.TestCase):
    def test_default_config_uses_larger_context_chunks(self) -> None:
        config = ChunkingConfig()

        self.assertEqual(config.target_seconds, 240.0)
        self.assertEqual(config.overlap_seconds, 45.0)
        self.assertEqual(config.max_document_chars, 1800)

    def test_chunks_cover_segments_and_progress(self) -> None:
        transcript = {
            "material_id": "mat-1",
            "metadata": {"title": "Title", "webinar_date": "2026-09-17"},
            "segments": [
                {"start": 0.0, "end": 30.0, "text": "alpha"},
                {"start": 30.0, "end": 60.0, "text": "bravo"},
                {"start": 60.0, "end": 90.0, "text": "charlie"},
                {"start": 90.0, "end": 120.0, "text": "delta"},
            ],
        }
        chunks = chunk_transcript(
            transcript,
            {"media_url": "https://rutube.ru/video/private/abcdef/?p=secret"},
            ChunkingConfig(target_seconds=60, overlap_seconds=20, max_document_chars=950),
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual([chunk["index"] for chunk in chunks], list(range(len(chunks))))
        joined = " ".join(chunk["text"] for chunk in chunks)
        for word in ["alpha", "bravo", "charlie", "delta"]:
            self.assertIn(word, joined)
        self.assertTrue(all(chunks[i + 1]["start"] > chunks[i]["start"] for i in range(len(chunks) - 1)))

    def test_document_text_keeps_safe_url_and_size_limit(self) -> None:
        transcript = {"metadata": {"title": "Title", "webinar_date": "2026-09-17"}}
        chunk = {"chunk_id": "mat-1:v1:0000", "index": 0, "start": 0, "end": 10, "text": "short text"}
        text = build_document_text(
            chunk,
            transcript,
            {"media_url": "https://rutube.ru/video/private/abcdef/?p=secret"},
            max_chars=950,
        )
        self.assertIn("00:00:00-00:00:10", text)
        self.assertIn("Источник: Title | 2026-09-17", text)
        self.assertIn("Фрагмент 00:00:00-00:00:10: short text", text)
        self.assertIn("https://rutube.ru/video/private/abcdef/", text)
        self.assertNotIn("secret", text)
        self.assertLessEqual(len(text), 1000)

    def test_normalize_text_removes_known_transcript_noise(self) -> None:
        text = normalize_text(
            "Продолжение следует... Продолжение следует... Важный текст "
            "humid чем потребитель mogę texts продолжается."
        )

        self.assertEqual(text, "Важный текст продолжается.")
        self.assertEqual(normalize_text("Продолжение следует..."), "")

    def test_chunking_removes_known_noise_segment_run(self) -> None:
        transcript = {
            "material_id": "mat-1",
            "metadata": {"title": "Title", "webinar_date": "2026-09-17"},
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "нормальный текст"},
                {"start": 1.0, "end": 2.0, "text": "humid"},
                {"start": 2.0, "end": 3.0, "text": "чем"},
                {"start": 3.0, "end": 4.0, "text": "потребитель"},
                {"start": 4.0, "end": 5.0, "text": "mogę"},
                {"start": 5.0, "end": 6.0, "text": "texts"},
                {"start": 6.0, "end": 7.0, "text": "важное продолжение"},
            ],
        }

        chunks = chunk_transcript(transcript, {}, ChunkingConfig())
        joined = " ".join(chunk["text"] for chunk in chunks)

        self.assertEqual(joined, "нормальный текст важное продолжение")


if __name__ == "__main__":
    unittest.main()
