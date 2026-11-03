import unittest
from pathlib import Path
import tempfile
import wave

from gge_assistant.asr_quality import compare_passes, merge_intervals, quarantine_segments, screen_passes


class AsrQualityTests(unittest.TestCase):
    def source(self, segments):
        return {"material_id": "m", "duration": 60.0, "segments": segments}

    def test_finds_recovered_speech_inside_primary_gap(self):
        primary = self.source([
            {"start": 0, "end": 10, "text": "Первый раздел заканчивается."},
            {"start": 40, "end": 50, "text": "Теперь перейдём ко второму разделу."},
        ])
        secondary = self.source([
            {"start": 0, "end": 10, "text": "Первый раздел заканчивается."},
            {"start": 15, "end": 35, "text": "В пропущенном месте описано важное условие и срок его применения."},
            {"start": 40, "end": 50, "text": "Теперь перейдём ко второму разделу."},
        ])
        issues = compare_passes(primary, secondary)["issues"]
        self.assertEqual([(x["start"], x["end"]) for x in issues if x["kind"] == "missing_speech"], [(10, 40)])

    def test_finds_meaning_changing_negation_and_excludes_both_variants(self):
        base = "В рамках такой оценки проводится проверка достоверности определения сметной стоимости по установленному порядку."
        primary = self.source([{"start": 16, "end": 29, "text": base}])
        secondary = self.source([{"start": 16, "end": 29, "text": base.replace("проводится", "не проводится")}])
        issues = compare_passes(primary, secondary)["issues"]
        self.assertTrue(any(x["kind"] == "negation_disagreement" for x in issues))
        screened = quarantine_segments(secondary, merge_intervals(issues))
        self.assertEqual(screened["segments"], [])
        self.assertEqual(screened["quality"]["status"], "partial")
        self.assertEqual(primary["segments"][0]["text"], base)

    def test_agreement_keeps_text(self):
        source = self.source([{"start": 0, "end": 14,
                               "text": "Эти документы подписываются усиленной квалифицированной электронной подписью."}])
        report = compare_passes(source, source)
        self.assertEqual(report["issues"], [])
        self.assertEqual(quarantine_segments(source, [])["segments"], source["segments"])

    def test_local_retries_resolve_missing_speech_without_human_review(self):
        first = self.source([
            {"start": 0, "end": 10, "text": "Первый раздел документа завершён."},
            {"start": 40, "end": 60, "text": "Последующий раздел документа начинается здесь."},
        ])
        recovered = {"start": 12, "end": 38,
                     "text": "В этом месте говорится о порядке представления документов и о сроках проверки каждого раздела проекта."}
        second = self.source([first["segments"][0], recovered, first["segments"][1]])
        with tempfile.TemporaryDirectory() as raw:
            audio = Path(raw) / "audio.wav"
            with wave.open(str(audio), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes((1000).to_bytes(2, "little", signed=True) * (60 * 16000))
            screened, report = screen_passes(first, second, audio, lambda _lo, _hi: [recovered])
        self.assertEqual(report["quality"]["status"], "screened")
        self.assertEqual(screened["segments"], second["segments"])
        self.assertTrue(any(x["kind"] == "missing_speech" for x in report["resolved"]))

    def test_unresolved_negation_is_not_searchable(self):
        first = self.source([
            {"start": 0, "end": 10, "text": "Вводная часть встречи и постановка вопроса."},
            {"start": 10, "end": 20, "text": "В рамках такой оценки проводится проверка достоверности определения сметной стоимости по установленному порядку."},
            {"start": 20, "end": 60, "text": "Далее рассматриваются иные правила подачи документов."},
        ])
        second = self.source([first["segments"][0],
            {**first["segments"][1], "text": first["segments"][1]["text"].replace("проводится", "не проводится")},
            first["segments"][2]])
        with tempfile.TemporaryDirectory() as raw:
            audio = Path(raw) / "audio.wav"
            with wave.open(str(audio), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes((1000).to_bytes(2, "little", signed=True) * (60 * 16000))
            screened, report = screen_passes(first, second, audio,
                                              lambda _lo, _hi: [first["segments"][1]])
        self.assertEqual(report["quality"]["status"], "partial")
        self.assertFalse(any("сметной стоимости" in s["text"] for s in screened["segments"]))


if __name__ == "__main__":
    unittest.main()
