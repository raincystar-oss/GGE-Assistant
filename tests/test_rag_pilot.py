from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from gge_assistant.rag_budget import BudgetError, BudgetLedger, BudgetedDeepSeek, LIMIT_NANO, input_upper_bound
from gge_assistant.rag_context import ContextConfig, ContextIndex, make_units, query_aspects, render_context, safe_card, safe_text, terms
from gge_assistant.rag_pilot import answer, answer_consistency_warnings, citation_scope_warnings
from gge_assistant.spending import configure


MESSAGES = [{"role": "system", "content": "Answer from sources."}, {"role": "user", "content": "Вопрос"}]


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "budget.json"
        self.ledger = BudgetLedger(self.path, "test-account")
        # Regression checks for archived, now inactive grant rules.
        self.ledger.reserve = self.ledger._reserve_legacy

    def tearDown(self):
        self.temp.cleanup()

    def test_reserve_before_request_survives_restart_and_blocks_retry(self):
        _, reserve = self.ledger.reserve(MESSAGES, 100, "test")
        self.assertGreater(reserve, 0)
        restarted = BudgetLedger(self.path, "test-account")
        with self.assertRaises(BudgetError):
            restarted.reserve(MESSAGES, 100, "retry")

    def test_usage_releases_only_unused_reservation(self):
        entry, reserve = self.ledger.reserve(MESSAGES, 100, "test")
        cost = self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        self.assertLess(cost, reserve)
        self.assertEqual(self.ledger.used(self.ledger.read()), cost)
        self.ledger.reserve(MESSAGES, 100, "next")

    def test_money_limit_checked_before_reserving_not_after_charging(self):
        data = self.ledger.read()
        data["entries"] = [{"status": "complete", "cost_nano": LIMIT_NANO - 1, "reserved_nano": LIMIT_NANO - 1}]
        self.path.write_text(json.dumps(data))
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 1, "over")
        self.assertEqual(len(self.ledger.read()["entries"]), 1)

    def test_missing_or_excess_usage_stops_future_calls(self):
        entry, _ = self.ledger.reserve(MESSAGES, 100, "test")
        with self.assertRaises(BudgetError):
            self.ledger.complete(entry, {"prompt_tokens": 10, "completion_tokens": 101})
        self.assertEqual(self.ledger.read()["entries"][0]["status"], "uncertain")
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "retry")

    def test_output_limit_and_plain_text_contract(self):
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 2049, "too long")
        with self.assertRaises(BudgetError):
            input_upper_bound(MESSAGES + [{"role": "user", "content": "history"}])
        self.assertGreaterEqual(input_upper_bound(MESSAGES), sum(len(m["content"].encode()) for m in MESSAGES))

    def test_thirty_attempts_is_a_separate_limit(self):
        data = self.ledger.read()
        data["entries"] = [{"status": "complete", "cost_nano": 0, "reserved_nano": 0} for _ in range(30)]
        self.path.write_text(json.dumps(data))
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 1, "attempt 31")

    def test_new_grant_preserves_history_and_restricts_requests(self):
        data = self.ledger.read()
        data["entries"] = [{"id": i, "status": "complete", "cost_nano": 1000,
                            "reserved_nano": 1000} for i in range(30)]
        self.path.write_text(json.dumps(data))
        historic = json.loads(self.path.read_text())["entries"]
        allowed = {f"answer_eval_2026_10_27_{i}": self.ledger.request_hash(MESSAGES)
                   for i in range(10)}
        self.ledger.authorize_answer_eval(allowed)
        with self.assertRaises(BudgetError):
            self.ledger.authorize_answer_eval(allowed)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "ordinary_answer")
        entry, _ = self.ledger.reserve(MESSAGES, 100, "answer_eval_2026_10_27_0")
        self.assertEqual(entry, 30)
        self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "answer_eval_2026_10_27_0")
        self.assertEqual(self.ledger.read()["entries"][:30], historic)

    def test_new_grant_stops_at_its_own_cost_and_attempt_limits(self):
        data = self.ledger.read()
        data["entries"] = [{"id": i, "status": "complete", "cost_nano": 0,
                            "reserved_nano": 0} for i in range(30)]
        self.path.write_text(json.dumps(data))
        allowed = {f"answer_eval_2026_10_27_{i}": self.ledger.request_hash(MESSAGES)
                   for i in range(10)}
        self.ledger.authorize_answer_eval(allowed)
        data = self.ledger.read()
        data["entries"].append({"id": 30, "purpose": "answer_eval_2026_10_27_0",
                                "status": "complete", "cost_nano": 199_999_999,
                                "reserved_nano": 199_999_999})
        self.path.write_text(json.dumps(data))
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 1, "answer_eval_2026_10_27_1")
        data["entries"] = data["entries"][:30] + [
            {"id": 30+i, "purpose": f"answer_eval_2026_10_27_{i}",
             "status": "complete", "cost_nano": 0, "reserved_nano": 0}
            for i in range(10)]
        self.path.write_text(json.dumps(data))
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 1, "answer_eval_2026_10_27_0")

    def test_answer_guard_grant_keeps_both_previous_grants_and_exact_scope(self):
        data = self.ledger.read()
        data["entries"] = [{"id": i, "status": "complete", "cost_nano": 1000,
                            "reserved_nano": 1000} for i in range(30)]
        self.path.write_text(json.dumps(data))
        eval_allowed = {f"answer_eval_2026_10_27_{i}": self.ledger.request_hash(MESSAGES)
                        for i in range(10)}
        self.ledger.authorize_answer_eval(eval_allowed)
        data = self.ledger.read()
        data["entries"].extend({"id": 30+i, "purpose": f"answer_eval_2026_10_27_{i}",
                                "status": "complete", "cost_nano": 1000,
                                "reserved_nano": 1000} for i in range(10))
        self.path.write_text(json.dumps(data))
        before = json.loads(self.path.read_text())["entries"]
        guard_allowed = {f"answer_guard_2026_10_27_{i}": self.ledger.request_hash(MESSAGES)
                         for i in range(6)}
        self.ledger.authorize_answer_guard(guard_allowed)
        with self.assertRaises(BudgetError):
            self.ledger.authorize_answer_guard(guard_allowed)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "answer_eval_2026_10_27_0")
        entry, _ = self.ledger.reserve(MESSAGES, 100, "answer_guard_2026_10_27_0")
        self.assertEqual(entry, 40)
        self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "answer_guard_2026_10_27_0")
        self.assertEqual(self.ledger.read()["entries"][:40], before)

    def test_answer_guard_stops_at_new_money_and_attempt_caps(self):
        data = self.ledger.read()
        data["entries"] = [{"id": i, "status": "complete", "cost_nano": 0,
                            "reserved_nano": 0} for i in range(30)]
        self.path.write_text(json.dumps(data))
        self.ledger.authorize_answer_eval({f"answer_eval_2026_10_27_{i}": self.ledger.request_hash(MESSAGES)
                                          for i in range(10)})
        data = self.ledger.read()
        data["entries"].extend({"id": 30+i, "status": "complete", "cost_nano": 0,
                                "reserved_nano": 0} for i in range(10))
        self.path.write_text(json.dumps(data))
        self.ledger.authorize_answer_guard({f"answer_guard_2026_10_27_{i}": self.ledger.request_hash(MESSAGES)
                                           for i in range(6)})
        data = self.ledger.read()
        data["entries"].append({"id": 40, "purpose": "answer_guard_2026_10_27_0",
                                "status": "complete", "cost_nano": 99_999_999,
                                "reserved_nano": 99_999_999})
        self.path.write_text(json.dumps(data))
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 1, "answer_guard_2026_10_27_1")
        data["entries"] = data["entries"][:40] + [
            {"id": 40+i, "purpose": f"answer_guard_2026_10_27_{i}",
             "status": "complete", "cost_nano": 0, "reserved_nano": 0}
            for i in range(6)]
        self.path.write_text(json.dumps(data))
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 1, "answer_guard_2026_10_27_0")

    def test_asr_context_grant_preserves_45_entries_and_stops_after_two(self):
        data = self.ledger.read()
        data["entries"] = [{"id": i, "status": "complete", "cost_nano": 1000,
                            "reserved_nano": 1000} for i in range(30)]
        self.path.write_text(json.dumps(data))
        self.ledger.authorize_answer_eval({f"answer_eval_2026_10_27_{i}":
            self.ledger.request_hash(MESSAGES) for i in range(10)})
        data = self.ledger.read()
        data["entries"].extend({"id": i, "status": "complete", "cost_nano": 1000,
                                "reserved_nano": 1000} for i in range(30, 40))
        self.path.write_text(json.dumps(data))
        self.ledger.authorize_answer_guard({f"answer_guard_2026_10_27_{i}":
            self.ledger.request_hash(MESSAGES) for i in range(6)})
        data = self.ledger.read()
        data["entries"].extend({"id": i, "status": "complete", "cost_nano": 1000,
                                "reserved_nano": 1000} for i in range(40, 45))
        self.path.write_text(json.dumps(data))
        historic = json.loads(self.path.read_text())["entries"]
        allowed = {f"asr_context_2026_10_28_{i}": self.ledger.request_hash(MESSAGES)
                   for i in range(2)}
        self.ledger.authorize_asr_context(allowed)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "answer_guard_2026_10_27_5")
        for i in range(2):
            entry, _ = self.ledger.reserve(MESSAGES, 100, f"asr_context_2026_10_28_{i}")
            self.assertEqual(entry, 45 + i)
            self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "asr_context_2026_10_28_1")
        self.assertEqual(self.ledger.read()["entries"][:45], historic)

    def test_citation_scopes_grant_preserves_47_entries_and_exact_scope(self):
        data = self.ledger.read()
        data["entries"] = [{"id": i, "status": "complete", "cost_nano": 1000,
                            "reserved_nano": 1000} for i in range(30)]
        self.path.write_text(json.dumps(data))
        self.ledger.authorize_answer_eval({f"answer_eval_2026_10_27_{i}":
            self.ledger.request_hash(MESSAGES) for i in range(10)})
        data = self.ledger.read()
        data["entries"].extend({"id": i, "status": "complete", "cost_nano": 1000,
                                "reserved_nano": 1000} for i in range(30, 40))
        self.path.write_text(json.dumps(data))
        self.ledger.authorize_answer_guard({f"answer_guard_2026_10_27_{i}":
            self.ledger.request_hash(MESSAGES) for i in range(6)})
        data = self.ledger.read()
        data["entries"].extend({"id": i, "status": "complete", "cost_nano": 1000,
                                "reserved_nano": 1000} for i in range(40, 45))
        self.path.write_text(json.dumps(data))
        self.ledger.authorize_asr_context({f"asr_context_2026_10_28_{i}":
            self.ledger.request_hash(MESSAGES) for i in range(2)})
        data = self.ledger.read()
        data["entries"].extend({"id": i, "status": "complete", "cost_nano": 1000,
                                "reserved_nano": 1000} for i in range(45, 47))
        self.path.write_text(json.dumps(data))
        historic = json.loads(self.path.read_text())["entries"]
        allowed = {f"citation_scopes_2026_10_28_{i}": self.ledger.request_hash(MESSAGES)
                   for i in range(2)}
        self.ledger.authorize_citation_scopes(allowed)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "asr_context_2026_10_28_0")
        for i in range(2):
            entry, _ = self.ledger.reserve(MESSAGES, 100, f"citation_scopes_2026_10_28_{i}")
            self.assertEqual(entry, 47 + i)
            self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "citation_scopes_2026_10_28_1")
        self.assertEqual(self.ledger.read()["entries"][:47], historic)
        prior_49 = self.ledger.read()["entries"][:]
        allowed_search = {f"search_fix_2026_10_28_{i}": self.ledger.request_hash(MESSAGES)
                          for i in range(2)}
        self.ledger.authorize_search_fix(allowed_search)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "citation_scopes_2026_10_28_0")
        for i in range(2):
            entry, _ = self.ledger.reserve(MESSAGES, 100, f"search_fix_2026_10_28_{i}")
            self.assertEqual(entry, 49 + i)
            self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "search_fix_2026_10_28_1")
        self.assertEqual(self.ledger.read()["entries"][:49], prior_49)
        prior_51 = self.ledger.read()["entries"][:]
        allowed_answers = {f"search_answer_2026_10_28_{i:02d}": self.ledger.request_hash(MESSAGES)
                           for i in range(10)}
        self.ledger.authorize_search_answer(allowed_answers)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "search_fix_2026_10_28_0")
        for i in range(10):
            entry, _ = self.ledger.reserve(MESSAGES, 100, f"search_answer_2026_10_28_{i:02d}")
            self.assertEqual(entry, 51 + i)
            self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "search_answer_2026_10_28_00")
        self.assertEqual(self.ledger.read()["entries"][:51], prior_51)
        prior_61 = self.ledger.read()["entries"][:]
        allowed_compact = {f"search_compact_2026_10_28_{i}": self.ledger.request_hash(MESSAGES)
                           for i in range(2)}
        self.ledger.authorize_search_compact(allowed_compact)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "search_answer_2026_10_28_00")
        for i in range(2):
            entry, _ = self.ledger.reserve(MESSAGES, 100, f"search_compact_2026_10_28_{i}")
            self.assertEqual(entry, 61 + i)
            self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "search_compact_2026_10_28_1")
        self.assertEqual(self.ledger.read()["entries"][:61], prior_61)
        prior_63 = self.ledger.read()["entries"][:]
        allowed_prompt = {f"search_prompt_2026_10_28_{i}": self.ledger.request_hash(MESSAGES)
                          for i in range(6)}
        self.ledger.authorize_search_prompt(allowed_prompt)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "search_compact_2026_10_28_0")
        for i in range(6):
            entry, _ = self.ledger.reserve(MESSAGES, 100, f"search_prompt_2026_10_28_{i}")
            self.assertEqual(entry, 63 + i)
            self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "search_prompt_2026_10_28_0")
        self.assertEqual(self.ledger.read()["entries"][:63], prior_63)
        prior_69 = self.ledger.read()["entries"][:]
        final_allowed = {f"search_final_2026_10_29_{i}": self.ledger.request_hash(MESSAGES)
                         for i in range(2)}
        self.ledger.authorize_search_final(final_allowed)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "search_prompt_2026_10_28_1")
        for i in range(2):
            entry, _ = self.ledger.reserve(MESSAGES, 100, f"search_final_2026_10_29_{i}")
            self.assertEqual(entry, 69 + i)
            self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "search_final_2026_10_29_0")
        self.assertEqual(self.ledger.read()["entries"][:69], prior_69)
        prior_71 = self.ledger.read()["entries"][:]
        guard_v4_allowed = {f"answer_guard_v4_2026_10_29_{i}": self.ledger.request_hash(MESSAGES)
                            for i in range(4)}
        self.ledger.authorize_answer_guard_v4(guard_v4_allowed)
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "search_final_2026_10_29_1")
        for i in range(4):
            entry, _ = self.ledger.reserve(MESSAGES, 100, f"answer_guard_v4_2026_10_29_{i}")
            self.assertEqual(entry, 71 + i)
            self.ledger.complete(entry, {"prompt_tokens": 30, "completion_tokens": 50})
        with self.assertRaises(BudgetError):
            self.ledger.reserve(MESSAGES, 100, "answer_guard_v4_2026_10_29_0")
        self.assertEqual(self.ledger.read()["entries"][:71], prior_71)

    def test_balance_floor_is_checked_before_paid_post(self):
        def handle(request):
            self.assertEqual(request.method, "GET")
            return httpx.Response(200, json={"balance_infos": [{"currency": "USD", "total_balance": "3.5001"}]})
        client = httpx.Client(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handle))
        paid = BudgetedDeepSeek(self.path, client=client, key="test-key")
        configure(self.path, '0.50', '30', True, '3.50')
        with self.assertRaises(BudgetError):
            paid.complete(MESSAGES)
        self.assertEqual(paid.ledger.read()['entries'], [])
        paid.close()

    def test_balance_read_retries_once_without_paid_post(self):
        calls = []
        def handle(request):
            calls.append(request.method)
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"balance_infos": [
                {"currency": "USD", "total_balance": "4.00"}]})
        client = httpx.Client(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handle))
        paid = BudgetedDeepSeek(self.path, client=client, key="test-key")
        self.assertEqual(paid.balance(), 4.0)
        self.assertEqual(calls, ["GET", "GET"])
        self.assertFalse(self.path.exists())
        paid.close()

    def test_http_error_is_one_attempt_and_no_secret_logged(self):
        calls = []
        def handle(request):
            calls.append(request.method)
            if request.method == "GET":
                return httpx.Response(200, json={"balance_infos": [{"currency": "USD", "total_balance": "4.82"}]})
            return httpx.Response(500, text="secret-test-key")
        client = httpx.Client(base_url="https://api.deepseek.com", transport=httpx.MockTransport(handle))
        paid = BudgetedDeepSeek(self.path, client=client, key="secret-test-key")
        configure(self.path, '0.50', '30', True, '3.50')
        with self.assertRaises(BudgetError) as error:
            paid.complete(MESSAGES)
        self.assertNotIn("secret-test-key", str(error.exception))
        with self.assertRaises(BudgetError):
            paid.complete(MESSAGES)
        self.assertEqual(calls.count("POST"), 1)
        paid.close()


class CitationScopeTests(unittest.TestCase):
    def test_blanket_refusal_followed_by_cited_explanation_is_flagged(self):
        draft = ("Недостаточно данных в источниках.\n\n"
                 "После отрицательного заключения нужен полный комплект [S5].")
        self.assertEqual(answer_consistency_warnings(draft),
                         ["blanket_refusal_followed_by_cited_evidence"])
        self.assertEqual(answer_consistency_warnings("Недостаточно данных в источниках."), [])

    def test_explicit_opposite_condition_is_flagged(self):
        blocks = [{"source_id": "S2", "text": "Если получено положительное заключение, допустима изменяемая часть."}]
        question = "После отрицательного заключения что подать?"
        answer = "После отрицательного заключения достаточно изменяемой части [S2]."
        self.assertEqual(citation_scope_warnings(question, answer, blocks), [
            {"source_id": "S2", "question_condition": "отрицательн", "source_condition": "положительн"}])

    def test_valid_contrast_and_uncertain_source_are_not_flagged(self):
        question = "После отрицательного заключения что подать?"
        positive = [{"source_id": "S2", "text": "Если получено положительное заключение, допустима изменяемая часть."}]
        contrast = ("Для отрицательного заключения данных недостаточно. "
                    "Правило относится к положительному заключению [S2].")
        self.assertEqual(citation_scope_warnings(question, contrast, positive), [])
        both = [{"source_id": "S2", "text": "Положительное заключение; отрицательное заключение."}]
        self.assertEqual(citation_scope_warnings(question,
            "После отрицательного заключения нужен полный комплект [S2].", both), [])

    def test_question_without_single_clear_status_abstains(self):
        blocks = [{"source_id": "S2", "text": "Положительное заключение."}]
        answer = "После отрицательного заключения достаточно изменяемой части [S2]."
        self.assertEqual(citation_scope_warnings("Как подать документацию?", answer, blocks), [])
        self.assertEqual(citation_scope_warnings("Сравни положительное и отрицательное заключения", answer, blocks), [])

    def test_mixed_source_keeps_cases_separate(self):
        source = ("Если ранее было получено отрицательное заключение по технической части и сметам, "
                  "на повторную экспертизу представляют полный комплект технической части. "
                  "Смета в уровне цен на дату передачи документации заказчику. "
                  "Если ранее было получено положительное заключение, допускается только изменяемая "
                  "часть проектной документации. Смета в уровне цен на дату подачи заявления.")
        blocks = [{"source_id": "S2", "text": source}]
        question = "После отрицательного заключения что подать и на какую дату цены?"
        bad = ("После отрицательного заключения нужно подать изменяемую часть проектной "
               "документации, а смету в уровне цен на дату подачи заявления [S2].")
        good = ("После отрицательного заключения нужен полный комплект технической части, "
                "а уровень цен — на дату передачи документации заказчику [S2].")
        self.assertEqual(citation_scope_warnings(question, bad, blocks)[0]["reason"], "opposite_case_only")
        self.assertEqual(citation_scope_warnings(question, good, blocks), [])

    def test_answer_reports_warning_without_a_new_paid_call(self):
        block = {"source_id": "S1", "material_id": "one", "title": "Запись", "date": "2026-10-01",
                 "url": "https://edu.gge.ru/example", "start": 0, "end": 20, "segments": [0],
                 "text": "Если получено положительное заключение, допустима изменяемая часть."}
        context = {"blocks": [block], "context": render_context([block])}
        draft = "После отрицательного заключения достаточно изменяемой части [S1]."
        with patch("gge_assistant.rag_pilot.cached_complete", return_value={"text": draft, "cached": True}):
            result = answer("После отрицательного заключения что подать?", context, None, "local_test")
        self.assertEqual(result["text"], draft)
        self.assertEqual(result["unknown_citations"], [])
        self.assertEqual(result["condition_warnings"][0]["source_id"], "S1")

    def test_answer_uses_scoped_citation_ids(self):
        block = {"source_id": "S1", "material_id": "one", "title": "Запись", "date": "2026-10-01",
                 "url": "https://edu.gge.ru/example", "start": 0, "end": 60, "segments": [0, 1],
                 "text": ("Если ранее было получено отрицательное заключение, нужен полный комплект "
                          "технической части и смета на дату передачи заказчику. "
                          "Если ранее было получено положительное заключение, достаточно изменяемой "
                          "части проектной документации, а смета на дату подачи заявления.")}
        context = {"blocks": [block], "context": render_context([block])}
        draft = "После положительного заключения достаточно изменяемой части [S2]."
        with patch("gge_assistant.rag_pilot.cached_complete", return_value={"text": draft, "cached": True}) as complete:
            result = answer("Что после положительного заключения: полный комплект или изменяемая часть?",
                            context, None, "local_test")
        self.assertEqual(result["unknown_citations"], [])
        self.assertEqual(result["citation_scope_splits"], 1)
        self.assertEqual([s["source_id"] for s in result["sources"]], ["S1", "S2"])
        self.assertIn("[S2]", complete.call_args.args[1][1]["content"])

class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.index = ContextIndex(Path(self.temp.name) / "context.db")

    def tearDown(self):
        self.index.close()
        self.temp.cleanup()

    def add(self, material, segments, title="Материал", doc="doc"):
        transcript = {"material_id": material, "metadata": {"title": title}, "segments": segments}
        documents = [{"title": doc, "start": segments[0]["start"], "end": segments[-1]["end"]}]
        return self.index.upsert(transcript, {"discovery_url": "https://edu.gge.ru/test/?p=secret"}, documents)

    def test_every_segment_retained_and_long_segment_not_truncated(self):
        segments = [{"start": i * 5, "end": i * 5 + 4, "text": f"Строка {i}."} for i in range(100)]
        segments.append({"start": 501, "end": 900, "text": "Ю" * 5000})
        units = make_units(segments, ContextConfig())
        present = {i for u in units for i in range(u["first_segment"], u["last_segment"] + 1)}
        self.assertEqual(present, set(range(101)))
        self.assertEqual(sum(u["text"].count("Ю") for u in units), 5000)
        self.assertTrue(all(len(u["text"]) <= 1200 for u in units))

    def test_retrieved_fragment_recovers_later_condition(self):
        segments = [
            {"start": 0, "end": 30, "text": "Для экспорта отчета запустите задание."},
            {"start": 32, "end": 60, "text": "Отчет хранится семь суток, затем удаляется."},
        ]
        self.add("alpha", segments)
        result = self.index.assemble(["экспорт отчета"], [[{"metadata": {"title": "doc.txt"}, "text": segments[0]["text"]}]])
        self.assertIn("семь суток", result["context"])
        self.assertEqual(result["blocks"][0]["segments"], [0, 1])

    def test_materials_are_not_merged_even_with_identical_timestamps(self):
        self.add("alpha", [{"start": 0, "end": 30, "text": "Документ альфа действует семь суток."}], doc="a")
        self.add("beta", [{"start": 0, "end": 30, "text": "Документ бета действует девять суток."}], doc="b")
        result = self.index.assemble(["Документ"], [[]])
        self.assertEqual(len(result["blocks"]), 2)
        self.assertEqual({b["material_id"] for b in result["blocks"]}, {"alpha", "beta"})

    def test_incremental_rebuild_and_removed_material_leave_no_fts_ghosts(self):
        source = [{"start": 0, "end": 20, "text": "Старыймаркер."}]
        self.assertTrue(self.add("alpha", source))
        self.assertFalse(self.add("alpha", source))
        self.assertTrue(self.add("alpha", [{**source[0], "text": "Новыймаркер."}]))
        self.assertEqual(self.index.lexical("Старыймаркер"), [])
        self.assertTrue(self.index.lexical("Новыймаркер"))
        self.index.restrict_to(set())
        self.assertEqual(self.index.lexical("Новыймаркер"), [])

    def test_overlapping_hits_do_not_duplicate_context(self):
        self.add("alpha", [{"start": 0, "end": 40, "text": "Уникальное предложение про экспорт."}])
        hit = {"metadata": {"title": "doc.txt"}, "text": "экспорт"}
        result = self.index.assemble(["экспорт", "экспорт"], [[hit] * 8, [hit] * 8])
        self.assertEqual(result["context"].count("Уникальное предложение"), 1)
        self.assertLessEqual(len(result["context"]), self.index.cfg.context_chars)

    def test_unknown_document_is_not_mapped_by_title_guess(self):
        self.add("alpha", [{"start": 0, "end": 30, "text": "Исходник."}])
        self.assertIsNone(self.index.map_vector_hit({"metadata": {"title": "unrelated.txt"}, "text": "Исходник"}))

    def test_citations_keep_separate_parent_ranges(self):
        cfg = ContextConfig(parent_seconds=30, parent_max_seconds=60)
        self.index.cfg = cfg
        segments = [{"start": i*40, "end": i*40+35, "text": f"Маркер экспорт номер {i}."} for i in range(3)]
        self.add("alpha", segments)
        result = self.index.assemble(["экспорт"], [[]])
        self.assertGreaterEqual(len(result["blocks"]), 2)
        self.assertTrue(all(b["end"]-b["start"] <= 60 for b in result["blocks"]))

    def test_question_keeps_its_immediate_answer_when_parent_budget_allows(self):
        segments = [{"start": 0, "end": 35, "text": "Как отменить операцию?"},
                    {"start": 37, "end": 55, "text": "Выберите отмену в меню."}]
        units = make_units(segments, ContextConfig(parent_seconds=30, parent_max_seconds=120))
        self.assertEqual(units[0]["parent_index"], units[-1]["parent_index"])

    def test_ten_independent_materials_only_changed_one_is_rebuilt(self):
        for n in range(10):
            self.add(f"material-{n}", [{"start": 0, "end": 40, "text": f"Инструкция {n}: экспорт данных."}], doc=f"doc-{n}")
        before = dict(self.index.db.execute("SELECT id,fingerprint FROM materials"))
        self.add("material-4", [{"start": 0, "end": 40, "text": "Инструкция 4: экспорт исправлен."}], doc="doc-4")
        after = dict(self.index.db.execute("SELECT id,fingerprint FROM materials"))
        self.assertEqual(len(after), 10)
        self.assertEqual([k for k in before if before[k] != after[k]], ["material-4"])
        result = self.index.assemble(["экспорт"], [[]])
        self.assertLessEqual(len(result["context"]), self.index.cfg.context_chars)
        self.assertLessEqual(len(result["selected_parents"]), self.index.cfg.max_parents)

    def test_private_links_never_enter_outbound_context(self):
        url = "https://rutube.ru/video/private/abcdef/?p=private-secret&r=x"
        self.assertEqual(safe_card(url), "")
        self.assertNotIn("private-secret", safe_text("текст " + url))
        self.assertEqual(safe_card("https://edu.gge.ru/a/?p=private-secret"), "https://edu.gge.ru/a/")
        self.assertEqual(safe_card("https://edu.gge.ru.evil.test/a"), "")

    def test_short_abbreviations_and_numbers_are_not_prefix_matches(self):
        self.add("target", [{"start": 0, "end": 20, "text": "ПД. Пункт 1.3 приказа 115."}], doc="target")
        self.add("lookalike", [{"start": 0, "end": 20, "text": "ПДН. Пункт 1.30 приказа 1150."}], doc="lookalike")
        self.assertEqual(terms("Как найти ПД по пункту 1.3 приказа 115?"),
                         ["найти", "пд", "пункту", "1.3", "приказа", "115"])
        for query in ("ПД", "пд", "115", "1.3"):
            self.assertEqual({u['material_id'] for u in self.index.lexical(query)}, {"target"})

    def test_exact_number_outweighs_first_vector_hit_with_different_number(self):
        self.index.cfg = ContextConfig(max_parents=1, continuation_units=0)
        self.add("lookalike", [{"start": 0, "end": 20, "text": "Порядок приказа 1150."}], doc="lookalike")
        self.add("target", [{"start": 0, "end": 20, "text": "Порядок приказа 115."}], doc="target")
        hits = [{"metadata": {"title": name}, "text": text} for name, text in
                [("lookalike", "Порядок приказа 1150."), ("target", "Порядок приказа 115.")]]
        result = self.index.assemble(["Порядок приказа 115"], [hits])
        self.assertEqual(result['selected_parents'][0]['material_id'], 'target')
        self.assertNotIn('1150', result['context'])

    def test_hybrid_result_replaces_unrelated_first_vector_hit(self):
        self.index.cfg = ContextConfig(max_parents=1, continuation_units=0)
        self.add("intro", [{"start": 0, "end": 20, "text": "Сегодня обсудим общие возможности системы."}], doc="intro")
        self.add("steps", [{"start": 0, "end": 20, "text": "Экспорт отчета запускается кнопкой. Скачайте результат."}], doc="steps")
        hits = [{"metadata": {"title": name}, "text": text} for name, text in
                [("intro", "Сегодня обсудим общие возможности системы."), ("steps", "Экспорт отчета запускается кнопкой.")]]
        result = self.index.assemble(["Экспорт отчета"], [hits])
        self.assertIn('Скачайте результат', result['context'])
        self.assertNotIn('общие возможности', result['context'])

    def test_local_aspects_keep_original_scope_and_add_only_literal_clauses(self):
        question = "Как открыть доступ организации и подключить нового работника, где найти его права?"
        queries = query_aspects(question)
        self.assertEqual(queries[0], question)
        self.assertEqual(len(queries), 3)
        self.assertTrue(all(query in question for query in queries))
        self.assertIn("подключить нового работника", queries)
        self.assertEqual(query_aspects("Что означает ПД?"), ["Что означает ПД?"])
        self.assertEqual(query_aspects("Поиск по СП и ПД"), ["Поиск по СП и ПД"])

    def test_full_procedure_keeps_middle_condition_even_without_a_search_hit(self):
        self.index.cfg = ContextConfig(parent_seconds=30, parent_max_seconds=60,
                                       max_parents=3, continuation_units=0)
        self.add("procedure", [
            {"start": 0, "end": 35, "text": "Начальная операция экспорта."},
            {"start": 40, "end": 75, "text": "Разрешение действует сутки. Проверяйте подпись."},
            {"start": 80, "end": 115, "text": "Завершение загрузки данных."},
        ], doc="procedure")
        self.add("other", [{"start": 0, "end": 35, "text": "Чужая процедура. Срок равен месяцу."}], doc="other")
        queries = ["Опиши весь порядок экспорта и загрузки", "Начальная операция экспорта", "Завершение загрузки данных"]
        result = self.index.assemble(queries, [[], [], []])
        self.assertIn('Разрешение действует сутки', result['context'])
        self.assertEqual({b['material_id'] for b in result['blocks']}, {'procedure'})
        self.assertEqual({n for b in result['blocks'] for n in b['segments']}, {0, 1, 2})
        self.assertLessEqual(len(result['context']), self.index.cfg.context_chars)

    def test_distant_procedure_parts_do_not_force_unbounded_continuation(self):
        self.index.cfg = ContextConfig(parent_seconds=30, parent_max_seconds=60,
                                       max_parents=3, continuation_units=0)
        segments = [{"start": n*40, "end": n*40+35, "text": "Промежуточная посторонняя тема."} for n in range(10)]
        segments[0]['text'] = "Начальная операция экспорта."
        segments[-1]['text'] = "Завершение загрузки данных."
        self.add("procedure", segments)
        result = self.index.assemble(["Опиши весь порядок экспорта и загрузки",
            "Начальная операция экспорта", "Завершение загрузки данных"], [[], [], []])
        self.assertIn('Начальная операция', result['context'])
        self.assertIn('Завершение загрузки', result['context'])
        self.assertNotIn('посторонняя тема', result['context'])

    def test_empty_full_query_does_not_skip_first_successful_subquestion(self):
        self.index.cfg = ContextConfig(max_parents=1, continuation_units=0)
        self.add('first', [{'start': 0, 'end': 30, 'text': 'Экспорт отчета.'}], doc='first')
        self.add('second', [{'start': 0, 'end': 30, 'text': 'Загрузка проекта.'}], doc='second')
        result = self.index.assemble(['Несуществующиймаркер', 'Экспорт отчета', 'Загрузка проекта'], [[], [], []])
        self.assertEqual(result['selected_parents'][0]['material_id'], 'first')


if __name__ == "__main__":
    unittest.main()
