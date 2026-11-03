"""Conditional citation cards preserve evidence and stay within the context cap."""
import unittest

from gge_assistant.rag_context import render_context
from gge_assistant.rag_scopes import scoped_citations


def context_for(text):
    block = {"source_id": "S1", "material_id": "one", "title": "Запись", "date": "2026-10-01",
             "url": "https://edu.gge.ru/example", "start": 100, "end": 160,
             "text": text, "segments": [1, 2]}
    return {"blocks": [block], "context": render_context([block]),
            "context_chars": len(render_context([block]))}


class ScopeCitationTests(unittest.TestCase):
    def test_distinct_conditions_get_distinct_ids_without_text_loss(self):
        source = ("Правила повторной экспертизы. "
                  "Если ранее было получено отрицательное заключение, подают полный комплект "
                  "технической части и смету на дату передачи заказчику. "
                  "Если ранее было получено положительное заключение, допустима только "
                  "изменяемая часть, а смета на дату подачи заявления.")
        result = scoped_citations(context_for(source),
                                  "После отрицательного заключения полный комплект или изменяемая часть?")
        self.assertGreaterEqual(result["citation_scope_splits"], 1)
        self.assertEqual([b["source_id"] for b in result["blocks"]],
                         [f"S{i}" for i in range(1, len(result["blocks"]) + 1)])
        self.assertEqual(" ".join(" ".join(b["text"] for b in result["blocks"]).split()),
                         " ".join(source.split()))
        negative = [b for b in result["blocks"] if "полный комплект" in b["text"]]
        positive = [b for b in result["blocks"] if "изменяемая часть" in b["text"]]
        self.assertEqual(len(negative), 1)
        self.assertEqual(len(positive), 1)
        self.assertNotEqual(negative[0]["source_id"], positive[0]["source_id"])
        self.assertEqual({(b["start"], b["end"]) for b in result["blocks"]}, {(100, 160)})

    def test_unrelated_or_single_condition_stays_whole(self):
        source = ("Если заявка получена, проверьте подпись. "
                  "Если документ заверен, отправьте ответ.")
        unrelated = scoped_citations(context_for(source), "Как оплатить отчёт?")
        self.assertEqual(unrelated["citation_scope_splits"], 0)
        single = scoped_citations(context_for("Если заявка получена, проверьте подпись."),
                                  "Когда проверять подпись заявки?")
        self.assertEqual(single["citation_scope_splits"], 0)

    def test_header_overhead_never_evicts_source_text(self):
        source = ("Если решение отрицательное, подают полный комплект. "
                  "Если решение положительное, подают изменённую часть.")
        original = context_for(source)
        result = scoped_citations(original, "Что подают при отрицательном решении?",
                                  max_chars=original["context_chars"])
        self.assertEqual(result["context"], original["context"])
        self.assertEqual(result["citation_scope_splits"], 0)


if __name__ == "__main__":
    unittest.main()
