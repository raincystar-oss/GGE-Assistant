"""Small, fail-closed DeepSeek client. Every paid attempt is durably reserved first."""
from __future__ import annotations

import hashlib
import json
import os
import time
import math
from decimal import Decimal
from pathlib import Path
from datetime import datetime, timezone

import httpx

from .artifacts import atomic_write_json
from .lock import AppLock
from .request_trace import RequestTrace
from . import spending

# Peak, cache-miss USD rates verified against official pricing on 2026-10-28.
# Integer nano-dollars avoid floating-point accounting. No cache discount assumed.
INPUT_NANO = 1320
OUTPUT_NANO = 3960
# Archived pilot constants validate historic accounting only. Runtime requests
# require an enabled user policy (ui_budget); no implicit spending allowance.
LIMIT_NANO = 500_000_000
MAX_CALLS = 30
ANSWER_EVAL_GRANT_NANO = 200_000_000
ANSWER_EVAL_GRANT_CALLS = 10
ANSWER_GUARD_GRANT_NANO = 100_000_000
ANSWER_GUARD_GRANT_CALLS = 6
ASR_CONTEXT_GRANT_NANO = 60_000_000
ASR_CONTEXT_GRANT_CALLS = 2
ASR_CONTEXT_START_ENTRY = 45
CITATION_SCOPES_GRANT_NANO = 70_000_000
CITATION_SCOPES_GRANT_CALLS = 2
CITATION_SCOPES_START_ENTRY = 47
SEARCH_FIX_GRANT_NANO = 70_000_000
SEARCH_FIX_GRANT_CALLS = 2
SEARCH_FIX_START_ENTRY = 49
SEARCH_ANSWER_GRANT_NANO = 750_000_000
SEARCH_ANSWER_GRANT_CALLS = 10
SEARCH_ANSWER_START_ENTRY = 51
SEARCH_COMPACT_GRANT_NANO = 150_000_000
SEARCH_COMPACT_GRANT_CALLS = 2
SEARCH_COMPACT_START_ENTRY = 61
SEARCH_PROMPT_GRANT_NANO = 500_000_000
SEARCH_PROMPT_GRANT_CALLS = 6
SEARCH_PROMPT_START_ENTRY = 63
SEARCH_FINAL_GRANT_NANO = 150_000_000
SEARCH_FINAL_GRANT_CALLS = 2
SEARCH_FINAL_START_ENTRY = 69
ANSWER_GUARD_V4_GRANT_NANO = 100_000_000
ANSWER_GUARD_V4_GRANT_CALLS = 4
ANSWER_GUARD_V4_START_ENTRY = 71
MODEL = "deepseek-v4-pro"
BASE_URL = "https://api.deepseek.com"


class BudgetError(RuntimeError):
    pass


def usd(nano: int) -> str:
    return f"{nano / 1_000_000_000:.6f}"


def input_upper_bound(messages: list[dict]) -> int:
    """Conservative byte-level bound, NOT chars/4 or an E5 tokenizer estimate.

    Requests contain only two plain-text messages, no tools/history/attachments.
    UTF-8 bytes bound byte-BPE tokens; extra 1024 tokens cover the chat template.
    Actual provider usage is checked against the bound on every response.
    """
    if len(messages) != 2 or any(set(m) != {"role", "content"} for m in messages):
        raise BudgetError("Only two plain-text messages are supported by the budget guard")
    if [m["role"] for m in messages] != ["system", "user"]:
        raise BudgetError("Unexpected message roles")
    return 1024 + sum(len(m["content"].encode("utf-8")) for m in messages)


def load_deepseek_key(explicit_path: Path | None = None) -> str:
    """Use existing credentials without copying them or logging their value."""
    if explicit_path is not None:
        key = explicit_path.read_text(encoding="utf-8").strip()
    elif os.environ.get("DEEPSEEK_API_KEY"):
        key = os.environ["DEEPSEEK_API_KEY"].strip()
    elif Path(".private/deepseek_api_key.txt").exists():
        key = Path(".private/deepseek_api_key.txt").read_text(encoding="utf-8").strip()
    else:
        env_path = Path(os.environ.get("APPDATA", "__unavailable__")) / "anythingllm-desktop/storage/.env"
        key = ""
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise BudgetError("DeepSeek key not found locally; no paid request made")
    return key


class BudgetLedger:
    def __init__(self, path: Path, account_id: str | None):
        self.path = path
        self.account_id = account_id

    def read(self) -> dict:
        if not self.path.exists():
            return {"schema": 1, "limit_nano": LIMIT_NANO, "account_id": self.account_id,
                    "pricing_date": "2026-10-04", "model": MODEL, "entries": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if (data.get("schema") != 1 or data.get("limit_nano") != LIMIT_NANO
                or (data.get("account_id") != self.account_id
                    and not (data.get("account_id") is None and not data.get("entries") and data.get("ui_budget")))
                or data.get("model") != MODEL):
            raise BudgetError("Budget ledger configuration/account mismatch")
        grant = data.get("answer_eval_grant")
        if grant is not None:
            if (grant.get("id") != "answer_eval_2026_10_27" or grant.get("start_entry") != MAX_CALLS
                    or grant.get("limit_nano") != ANSWER_EVAL_GRANT_NANO
                    or grant.get("max_calls") != ANSWER_EVAL_GRANT_CALLS
                    or not isinstance(grant.get("allowed_requests"), dict)
                    or len(grant["allowed_requests"]) != ANSWER_EVAL_GRANT_CALLS):
                raise BudgetError("Answer evaluation grant configuration mismatch")
        guard = data.get("answer_guard_grant")
        if guard is not None:
            if (grant is None or guard.get("id") != "answer_guard_2026_10_27"
                    or guard.get("start_entry") != MAX_CALLS + ANSWER_EVAL_GRANT_CALLS
                    or guard.get("limit_nano") != ANSWER_GUARD_GRANT_NANO
                    or guard.get("max_calls") != ANSWER_GUARD_GRANT_CALLS
                    or not isinstance(guard.get("allowed_requests"), dict)
                    or len(guard["allowed_requests"]) != ANSWER_GUARD_GRANT_CALLS):
                raise BudgetError("Answer guard grant configuration mismatch")
        context_grant = data.get("asr_context_grant")
        if context_grant is not None:
            if (guard is None or context_grant.get("id") != "asr_context_2026_10_28"
                    or context_grant.get("start_entry") != ASR_CONTEXT_START_ENTRY
                    or context_grant.get("limit_nano") != ASR_CONTEXT_GRANT_NANO
                    or context_grant.get("max_calls") != ASR_CONTEXT_GRANT_CALLS
                    or not isinstance(context_grant.get("allowed_requests"), dict)
                    or len(context_grant["allowed_requests"]) != ASR_CONTEXT_GRANT_CALLS
                    or context_grant.get("previous_entries_sha256") != hashlib.sha256(json.dumps(
                        data["entries"][:ASR_CONTEXT_START_ENTRY], ensure_ascii=False,
                        sort_keys=True).encode()).hexdigest()):
                raise BudgetError("ASR context grant configuration or history mismatch")
        citations_grant = data.get("citation_scopes_grant")
        if citations_grant is not None:
            if (context_grant is None or citations_grant.get("id") != "citation_scopes_2026_10_28"
                    or citations_grant.get("start_entry") != CITATION_SCOPES_START_ENTRY
                    or citations_grant.get("limit_nano") != CITATION_SCOPES_GRANT_NANO
                    or citations_grant.get("max_calls") != CITATION_SCOPES_GRANT_CALLS
                    or not isinstance(citations_grant.get("allowed_requests"), dict)
                    or len(citations_grant["allowed_requests"]) != CITATION_SCOPES_GRANT_CALLS
                    or citations_grant.get("previous_entries_sha256") != hashlib.sha256(json.dumps(
                        data["entries"][:CITATION_SCOPES_START_ENTRY], ensure_ascii=False,
                        sort_keys=True).encode()).hexdigest()):
                raise BudgetError("Citation scopes grant configuration or history mismatch")
        search_grant = data.get("search_fix_grant")
        if search_grant is not None:
            if (citations_grant is None or search_grant.get("id") != "search_fix_2026_10_28"
                    or search_grant.get("start_entry") != SEARCH_FIX_START_ENTRY
                    or search_grant.get("limit_nano") != SEARCH_FIX_GRANT_NANO
                    or search_grant.get("max_calls") != SEARCH_FIX_GRANT_CALLS
                    or not isinstance(search_grant.get("allowed_requests"), dict)
                    or len(search_grant["allowed_requests"]) != SEARCH_FIX_GRANT_CALLS
                    or search_grant.get("previous_entries_sha256") != hashlib.sha256(json.dumps(
                        data["entries"][:SEARCH_FIX_START_ENTRY], ensure_ascii=False,
                        sort_keys=True).encode()).hexdigest()):
                raise BudgetError("Search-fix grant configuration or history mismatch")
        answer_search_grant = data.get("search_answer_grant")
        if answer_search_grant is not None:
            if (search_grant is None or answer_search_grant.get("id") != "search_answer_2026_10_28"
                    or answer_search_grant.get("start_entry") != SEARCH_ANSWER_START_ENTRY
                    or answer_search_grant.get("limit_nano") != SEARCH_ANSWER_GRANT_NANO
                    or answer_search_grant.get("max_calls") != SEARCH_ANSWER_GRANT_CALLS
                    or not isinstance(answer_search_grant.get("allowed_requests"), dict)
                    or len(answer_search_grant["allowed_requests"]) != SEARCH_ANSWER_GRANT_CALLS
                    or answer_search_grant.get("previous_entries_sha256") != hashlib.sha256(json.dumps(
                        data["entries"][:SEARCH_ANSWER_START_ENTRY], ensure_ascii=False,
                        sort_keys=True).encode()).hexdigest()):
                raise BudgetError("Search-answer grant configuration or history mismatch")
        compact_grant = data.get("search_compact_grant")
        if compact_grant is not None:
            if (answer_search_grant is None or compact_grant.get("id") != "search_compact_2026_10_28"
                    or compact_grant.get("start_entry") != SEARCH_COMPACT_START_ENTRY
                    or compact_grant.get("limit_nano") != SEARCH_COMPACT_GRANT_NANO
                    or compact_grant.get("max_calls") != SEARCH_COMPACT_GRANT_CALLS
                    or not isinstance(compact_grant.get("allowed_requests"), dict)
                    or len(compact_grant["allowed_requests"]) != SEARCH_COMPACT_GRANT_CALLS
                    or compact_grant.get("previous_entries_sha256") != hashlib.sha256(json.dumps(
                        data["entries"][:SEARCH_COMPACT_START_ENTRY], ensure_ascii=False,
                        sort_keys=True).encode()).hexdigest()):
                raise BudgetError("Search-compact grant configuration or history mismatch")
        prompt_grant = data.get("search_prompt_grant")
        if prompt_grant is not None:
            if (compact_grant is None or prompt_grant.get("id") != "search_prompt_2026_10_28"
                    or prompt_grant.get("start_entry") != SEARCH_PROMPT_START_ENTRY
                    or prompt_grant.get("limit_nano") != SEARCH_PROMPT_GRANT_NANO
                    or prompt_grant.get("max_calls") != SEARCH_PROMPT_GRANT_CALLS
                    or not isinstance(prompt_grant.get("allowed_requests"), dict)
                    or len(prompt_grant["allowed_requests"]) != SEARCH_PROMPT_GRANT_CALLS
                    or prompt_grant.get("previous_entries_sha256") != hashlib.sha256(json.dumps(
                        data["entries"][:SEARCH_PROMPT_START_ENTRY], ensure_ascii=False,
                        sort_keys=True).encode()).hexdigest()):
                raise BudgetError("Search-prompt grant configuration or history mismatch")
        final_grant = data.get("search_final_grant")
        if final_grant is not None:
            if (prompt_grant is None or final_grant.get("id") != "search_final_2026_10_29"
                    or final_grant.get("start_entry") != SEARCH_FINAL_START_ENTRY
                    or final_grant.get("limit_nano") != SEARCH_FINAL_GRANT_NANO
                    or final_grant.get("max_calls") != SEARCH_FINAL_GRANT_CALLS
                    or not isinstance(final_grant.get("allowed_requests"), dict)
                    or len(final_grant["allowed_requests"]) != SEARCH_FINAL_GRANT_CALLS
                    or final_grant.get("previous_entries_sha256") != hashlib.sha256(json.dumps(
                        data["entries"][:SEARCH_FINAL_START_ENTRY], ensure_ascii=False,
                        sort_keys=True).encode()).hexdigest()):
                raise BudgetError("Search-final grant configuration or history mismatch")
        guard_v4 = data.get("answer_guard_v4_grant")
        if guard_v4 is not None:
            if (final_grant is None or guard_v4.get("id") != "answer_guard_v4_2026_10_29"
                    or guard_v4.get("start_entry") != ANSWER_GUARD_V4_START_ENTRY
                    or guard_v4.get("limit_nano") != ANSWER_GUARD_V4_GRANT_NANO
                    or guard_v4.get("max_calls") != ANSWER_GUARD_V4_GRANT_CALLS
                    or not isinstance(guard_v4.get("allowed_requests"), dict)
                    or len(guard_v4["allowed_requests"]) != ANSWER_GUARD_V4_GRANT_CALLS
                    or guard_v4.get("previous_entries_sha256") != hashlib.sha256(json.dumps(
                        data["entries"][:ANSWER_GUARD_V4_START_ENTRY], ensure_ascii=False,
                        sort_keys=True).encode()).hexdigest()):
                raise BudgetError("Answer-guard v4 grant configuration or history mismatch")
        demo = data.get("demo_grant")
        if demo is not None:
            if (guard_v4 is None or demo.get("id") != "demo_2026_11_03"
                    or demo.get("start_entry") != 75 or demo.get("limit_nano") != 100_000_000
                    or demo.get("max_calls") != 8 or len(data["entries"]) < 75
                    or demo.get("previous_entries_sha256") != hashlib.sha256(json.dumps(
                        data["entries"][:75], ensure_ascii=False, sort_keys=True).encode()).hexdigest()):
                raise BudgetError("Demo grant configuration or history mismatch")
        diagnostic = data.get("diagnostic_grant")
        if diagnostic is not None:
            if (demo is None or diagnostic.get("id") != "diagnostic_2026_11_03"
                    or diagnostic.get("start_entry") != 77 or diagnostic.get("limit_nano") != 60_000_000
                    or diagnostic.get("max_calls") != 2 or len(data["entries"]) < 77
                    or diagnostic.get("previous_entries_sha256") != self.request_hash(data["entries"][:77])):
                raise BudgetError("Diagnostic grant or previous history mismatch")
        chat_test = data.get("chat_test_grant")
        if chat_test is not None:
            if (diagnostic is None or chat_test.get("id") != "chat_test_2026_11_03"
                    or chat_test.get("start_entry") != 79 or chat_test.get("limit_nano") != 100_000_000
                    or chat_test.get("max_calls") != 3 or len(data["entries"]) < 79
                    or chat_test.get("previous_entries_sha256") != self.request_hash(data["entries"][:79])):
                raise BudgetError("Chat test grant or previous history mismatch")
        try:
            spending.validate(data)
        except spending.SpendingError as exc:
            raise BudgetError(str(exc)) from None
        return data

    def authorize_chat_test(self):
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("chat_test_grant" in data or len(data["entries"]) != 79
                    or any(e["status"] != "complete" for e in data["entries"][77:])):
                raise BudgetError("Unexpected history for chat test grant")
            data["chat_test_grant"] = {"id": "chat_test_2026_11_03", "start_entry": 79,
                "limit_nano": 100_000_000, "max_calls": 3,
                "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "previous_entries_sha256": self.request_hash(data["entries"])}
            atomic_write_json(self.path, data)

    def authorize_diagnostic(self):
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if "diagnostic_grant" in data or len(data["entries"]) != 77:
                raise BudgetError("Unexpected history for diagnostic grant")
            data["diagnostic_grant"] = {"id": "diagnostic_2026_11_03", "start_entry": 77,
                "limit_nano": 60_000_000, "max_calls": 2,
                "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "previous_entries_sha256": self.request_hash(data["entries"]),
                "note": "User authorized continuation with full prior reservations retained."}
            atomic_write_json(self.path, data)

    def authorize_demo(self) -> None:
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("demo_grant" in data or len(data["entries"]) != 75
                    or any(e["status"] != "complete" for e in data["entries"])):
                raise BudgetError("Prior budget history must be complete")
            data["demo_grant"] = {"id": "demo_2026_11_03", "start_entry": 75,
                "limit_nano": 100_000_000, "max_calls": 8,
                "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "previous_entries_sha256": hashlib.sha256(json.dumps(data["entries"],
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            atomic_write_json(self.path, data)

    @staticmethod
    def used(data: dict) -> int:
        return sum(e.get("cost_nano", e["reserved_nano"]) for e in data["entries"])

    @staticmethod
    def request_hash(messages: list[dict]) -> str:
        return hashlib.sha256(json.dumps(messages, ensure_ascii=False,
                                         sort_keys=True).encode()).hexdigest()

    def authorize_answer_eval(self, allowed_requests: dict[str, str]) -> None:
        """Append the user's 2026-10-27 grant without changing the 30 historic entries."""
        if len(allowed_requests) != ANSWER_EVAL_GRANT_CALLS or any(
                not purpose.startswith("answer_eval_2026_10_27_") or len(digest) != 64
                for purpose, digest in allowed_requests.items()):
            raise BudgetError("Expected exactly ten frozen answer-evaluation requests")
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("answer_eval_grant" in data or len(data["entries"]) != MAX_CALLS
                    or any(e["status"] != "complete" for e in data["entries"])
                    or self.used(data) > LIMIT_NANO):
                raise BudgetError("Historic pilot must be complete and unextended")
            data["answer_eval_grant"] = {
                "id": "answer_eval_2026_10_27", "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "start_entry": MAX_CALLS, "limit_nano": ANSWER_EVAL_GRANT_NANO,
                "max_calls": ANSWER_EVAL_GRANT_CALLS, "allowed_requests": allowed_requests,
                "historic_cost_nano": self.used(data)}
            atomic_write_json(self.path, data)

    def authorize_answer_guard(self, allowed_requests: dict[str, str]) -> None:
        """Append the separate user-approved $0.10 / six-attempt grant."""
        if len(allowed_requests) != ANSWER_GUARD_GRANT_CALLS or any(
                not purpose.startswith("answer_guard_2026_10_27_") or len(digest) != 64
                for purpose, digest in allowed_requests.items()):
            raise BudgetError("Expected exactly six frozen answer-guard requests")
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("answer_guard_grant" in data
                    or len(data["entries"]) != MAX_CALLS + ANSWER_EVAL_GRANT_CALLS
                    or any(e["status"] != "complete" for e in data["entries"])
                    or sum(e["cost_nano"] for e in data["entries"][:MAX_CALLS]) > LIMIT_NANO
                    or sum(e["cost_nano"] for e in data["entries"][MAX_CALLS:]) > ANSWER_EVAL_GRANT_NANO):
                raise BudgetError("Previous grants must be complete and within budget")
            data["answer_guard_grant"] = {
                "id": "answer_guard_2026_10_27", "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "start_entry": MAX_CALLS + ANSWER_EVAL_GRANT_CALLS,
                "limit_nano": ANSWER_GUARD_GRANT_NANO,
                "max_calls": ANSWER_GUARD_GRANT_CALLS, "allowed_requests": allowed_requests,
                "previous_entries_sha256": hashlib.sha256(json.dumps(data["entries"],
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            atomic_write_json(self.path, data)

    def authorize_asr_context(self, allowed_requests: dict[str, str]) -> None:
        """Append the separately approved $0.06 / two-attempt grant."""
        if len(allowed_requests) != ASR_CONTEXT_GRANT_CALLS or any(
                not purpose.startswith("asr_context_2026_10_28_") or len(digest) != 64
                for purpose, digest in allowed_requests.items()):
            raise BudgetError("Expected two frozen ASR-context requests")
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("asr_context_grant" in data or "answer_guard_grant" not in data
                    or len(data["entries"]) != ASR_CONTEXT_START_ENTRY
                    or any(e["status"] != "complete" for e in data["entries"])):
                raise BudgetError("Previous grants must be complete before new grant")
            data["asr_context_grant"] = {
                "id": "asr_context_2026_10_28", "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "start_entry": ASR_CONTEXT_START_ENTRY, "limit_nano": ASR_CONTEXT_GRANT_NANO,
                "max_calls": ASR_CONTEXT_GRANT_CALLS, "allowed_requests": allowed_requests,
                "previous_entries_sha256": hashlib.sha256(json.dumps(data["entries"],
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            atomic_write_json(self.path, data)

    def authorize_citation_scopes(self, allowed_requests: dict[str, str]) -> None:
        """Append the separately approved $0.07 / two-attempt grant."""
        if len(allowed_requests) != CITATION_SCOPES_GRANT_CALLS or any(
                not purpose.startswith("citation_scopes_2026_10_28_") or len(digest) != 64
                for purpose, digest in allowed_requests.items()):
            raise BudgetError("Expected two frozen citation-scope requests")
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("citation_scopes_grant" in data or "asr_context_grant" not in data
                    or len(data["entries"]) != CITATION_SCOPES_START_ENTRY
                    or any(e["status"] != "complete" for e in data["entries"])):
                raise BudgetError("Previous grants must be complete before citation-scope grant")
            data["citation_scopes_grant"] = {
                "id": "citation_scopes_2026_10_28", "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "start_entry": CITATION_SCOPES_START_ENTRY,
                "limit_nano": CITATION_SCOPES_GRANT_NANO,
                "max_calls": CITATION_SCOPES_GRANT_CALLS,
                "allowed_requests": allowed_requests,
                "previous_entries_sha256": hashlib.sha256(json.dumps(data["entries"],
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            atomic_write_json(self.path, data)

    def authorize_search_fix(self, allowed_requests: dict[str, str]) -> None:
        """Append the separately approved $0.07 / two-attempt search evaluation grant."""
        if len(allowed_requests) != SEARCH_FIX_GRANT_CALLS or any(
                not purpose.startswith("search_fix_2026_10_28_") or len(digest) != 64
                for purpose, digest in allowed_requests.items()):
            raise BudgetError("Expected two frozen search evaluation requests")
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("search_fix_grant" in data or "citation_scopes_grant" not in data
                    or len(data["entries"]) != SEARCH_FIX_START_ENTRY
                    or any(e["status"] != "complete" for e in data["entries"])):
                raise BudgetError("Previous grants must be complete before search evaluation grant")
            data["search_fix_grant"] = {
                "id": "search_fix_2026_10_28", "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "start_entry": SEARCH_FIX_START_ENTRY, "limit_nano": SEARCH_FIX_GRANT_NANO,
                "max_calls": SEARCH_FIX_GRANT_CALLS, "allowed_requests": allowed_requests,
                "previous_entries_sha256": hashlib.sha256(json.dumps(data["entries"],
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            atomic_write_json(self.path, data)

    def authorize_search_answer(self, allowed_requests: dict[str, str]) -> None:
        """Append ten frozen answer requests under the user's $1.50 / 20-call ceiling."""
        if len(allowed_requests) != SEARCH_ANSWER_GRANT_CALLS or any(
                not purpose.startswith("search_answer_2026_10_28_") or len(digest) != 64
                for purpose, digest in allowed_requests.items()):
            raise BudgetError("Expected ten frozen search-answer requests")
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("search_answer_grant" in data or "search_fix_grant" not in data
                    or len(data["entries"]) != SEARCH_ANSWER_START_ENTRY
                    or any(e["status"] != "complete" for e in data["entries"])):
                raise BudgetError("Previous grants must be complete before answer evaluation")
            data["search_answer_grant"] = {
                "id": "search_answer_2026_10_28", "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "start_entry": SEARCH_ANSWER_START_ENTRY, "limit_nano": SEARCH_ANSWER_GRANT_NANO,
                "max_calls": SEARCH_ANSWER_GRANT_CALLS, "allowed_requests": allowed_requests,
                "user_total_limit_nano": 1_500_000_000,
                "previous_entries_sha256": hashlib.sha256(json.dumps(data["entries"],
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            atomic_write_json(self.path, data)

    def authorize_search_compact(self, allowed_requests: dict[str, str]) -> None:
        """Append two frozen compact-context diagnostics under the same user ceiling."""
        if len(allowed_requests) != SEARCH_COMPACT_GRANT_CALLS or any(
                not purpose.startswith("search_compact_2026_10_28_") or len(digest) != 64
                for purpose, digest in allowed_requests.items()):
            raise BudgetError("Expected two frozen compact-context requests")
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("search_compact_grant" in data or "search_answer_grant" not in data
                    or len(data["entries"]) != SEARCH_COMPACT_START_ENTRY
                    or any(e["status"] != "complete" for e in data["entries"])):
                raise BudgetError("Previous answer grant must be complete before compact diagnostics")
            data["search_compact_grant"] = {
                "id": "search_compact_2026_10_28", "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "start_entry": SEARCH_COMPACT_START_ENTRY, "limit_nano": SEARCH_COMPACT_GRANT_NANO,
                "max_calls": SEARCH_COMPACT_GRANT_CALLS, "allowed_requests": allowed_requests,
                "previous_entries_sha256": hashlib.sha256(json.dumps(data["entries"],
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            atomic_write_json(self.path, data)

    def authorize_search_prompt(self, allowed_requests: dict[str, str]) -> None:
        """Append six frozen prompt comparisons under the remaining user ceiling."""
        if len(allowed_requests) != SEARCH_PROMPT_GRANT_CALLS or any(
                not purpose.startswith("search_prompt_2026_10_28_") or len(digest) != 64
                for purpose, digest in allowed_requests.items()):
            raise BudgetError("Expected six frozen prompt requests")
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("search_prompt_grant" in data or "search_compact_grant" not in data
                    or len(data["entries"]) != SEARCH_PROMPT_START_ENTRY
                    or any(e["status"] != "complete" for e in data["entries"])):
                raise BudgetError("Previous compact grant must be complete before prompt comparison")
            data["search_prompt_grant"] = {
                "id": "search_prompt_2026_10_28", "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "start_entry": SEARCH_PROMPT_START_ENTRY, "limit_nano": SEARCH_PROMPT_GRANT_NANO,
                "max_calls": SEARCH_PROMPT_GRANT_CALLS, "allowed_requests": allowed_requests,
                "previous_entries_sha256": hashlib.sha256(json.dumps(data["entries"],
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            atomic_write_json(self.path, data)

    def authorize_search_final(self, allowed_requests: dict[str, str]) -> None:
        """Freeze the last two calls within the existing $1.50 / 20-call user grant."""
        if len(allowed_requests) != SEARCH_FINAL_GRANT_CALLS or any(
                not purpose.startswith("search_final_2026_10_29_") or len(digest) != 64
                for purpose, digest in allowed_requests.items()):
            raise BudgetError("Expected two frozen final-answer requests")
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("search_final_grant" in data or "search_prompt_grant" not in data
                    or len(data["entries"]) != SEARCH_FINAL_START_ENTRY
                    or any(e["status"] != "complete" for e in data["entries"])):
                raise BudgetError("Previous answer checks must be complete")
            data["search_final_grant"] = {
                "id": "search_final_2026_10_29", "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "start_entry": SEARCH_FINAL_START_ENTRY, "limit_nano": SEARCH_FINAL_GRANT_NANO,
                "max_calls": SEARCH_FINAL_GRANT_CALLS, "allowed_requests": allowed_requests,
                "previous_entries_sha256": hashlib.sha256(json.dumps(data["entries"],
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            atomic_write_json(self.path, data)

    def authorize_answer_guard_v4(self, allowed_requests: dict[str, str]) -> None:
        """Grant four frozen requests under the user's new $0.10 / eight-call ceiling."""
        if len(allowed_requests) != ANSWER_GUARD_V4_GRANT_CALLS or any(
                not purpose.startswith("answer_guard_v4_2026_10_29_") or len(digest) != 64
                for purpose, digest in allowed_requests.items()):
            raise BudgetError("Expected four frozen answer-guard requests")
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if ("answer_guard_v4_grant" in data or "search_final_grant" not in data
                    or len(data["entries"]) != ANSWER_GUARD_V4_START_ENTRY
                    or any(e["status"] != "complete" for e in data["entries"])):
                raise BudgetError("Prior budget history must be complete")
            data["answer_guard_v4_grant"] = {
                "id": "answer_guard_v4_2026_10_29", "authorized_utc": datetime.now(timezone.utc).isoformat(),
                "start_entry": ANSWER_GUARD_V4_START_ENTRY,
                "limit_nano": ANSWER_GUARD_V4_GRANT_NANO,
                "max_calls": ANSWER_GUARD_V4_GRANT_CALLS,
                "user_max_calls": 8, "allowed_requests": allowed_requests,
                "previous_entries_sha256": hashlib.sha256(json.dumps(data["entries"],
                    ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            atomic_write_json(self.path, data)

    def check_request(self, messages, max_output):
        if not 1 <= max_output <= 2048:
            raise BudgetError("Output limit must be between 1 and 2048 tokens")
        bound = input_upper_bound(messages)
        if bound > 55_000:
            raise BudgetError("Request exceeds the conservative input budget")
        try:
            return spending.check(self.read(), bound * INPUT_NANO + max_output * OUTPUT_NANO)
        except spending.SpendingError as exc:
            raise BudgetError(str(exc)) from None

    def reserve(self, messages: list[dict], max_output: int, purpose: str, *, balance_nano=None) -> tuple[int, int]:
        with AppLock(self.path.with_suffix('.lock')):
            settings = self.check_request(messages, max_output)
            data = self.read()
            bound = input_upper_bound(messages)
            reserve = bound * INPUT_NANO + max_output * OUTPUT_NANO
            if balance_nano is not None and balance_nano - reserve < settings['min_balance_nano']:
                raise BudgetError('Недостаточно средств с учётом минимального остатка на счёте. Проверьте настройки расходов.')
            entry = dict(id=len(data['entries']), status='pending', purpose=purpose,
                         created_utc=datetime.now(timezone.utc).isoformat(), input_bound=bound,
                         output_bound=max_output, reserved_nano=reserve,
                         request_sha256=self.request_hash(messages))
            data['account_id'] = self.account_id
            data['entries'].append(entry)
            atomic_write_json(self.path, data)
            return entry['id'], reserve

    def _reserve_legacy(self, messages: list[dict], max_output: int, purpose: str) -> tuple[int, int]:
        """Historical grant rules retained for regression checks, not runtime calls."""
        if not 1 <= max_output <= 2048:
            raise BudgetError("Output limit must be between 1 and 2048 tokens")
        bound = input_upper_bound(messages)
        if bound > 55_000:
            raise BudgetError("Request exceeds the conservative input budget")
        reserve = bound * INPUT_NANO + max_output * OUTPUT_NANO
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            if any(not (e["status"] == "complete" or (
                    e.get("id") == 75 and e.get("status") == "reserved_unknown"
                    and e.get("reserved_nano") == 43_339_560
                    and e.get("request_sha256") == "1d6f19613780af2a35d88a640c8d753029c3a910d3de9232d9f747df2a6b10ae"
                    and "cost_nano" not in e and data.get("demo_grant")) or (
                    e.get("id") == 76 and e.get("status") == "pending"
                    and data.get("diagnostic_grant") and "cost_nano" not in e)) for e in data["entries"]):
                raise BudgetError("Earlier request has uncertain cost; paid calls stopped")
            request_sha = self.request_hash(messages)
            if len(data["entries"]) < MAX_CALLS:
                if self.used(data) + reserve > LIMIT_NANO:
                    raise BudgetError("Pilot budget reached; paid calls stopped")
            elif len(data["entries"]) < MAX_CALLS + ANSWER_EVAL_GRANT_CALLS:
                grant = data.get("answer_eval_grant")
                if grant is None:
                    raise BudgetError("Pilot call limit reached; paid calls stopped")
                entries = data["entries"][grant["start_entry"]:]
                if (len(entries) >= grant["max_calls"]
                        or purpose not in grant["allowed_requests"]
                        or grant["allowed_requests"][purpose] != request_sha
                        or any(e["purpose"] == purpose for e in entries)
                        or sum(e.get("cost_nano", e["reserved_nano"]) for e in entries) + reserve > grant["limit_nano"]):
                    raise BudgetError("Answer evaluation grant limit or request check failed")
            else:
                if (SEARCH_ANSWER_START_ENTRY <= len(data["entries"]) < ANSWER_GUARD_V4_START_ENTRY
                        and (len(data["entries"]) - SEARCH_ANSWER_START_ENTRY >= 20
                             or sum(e.get("cost_nano", e["reserved_nano"])
                                    for e in data["entries"][SEARCH_ANSWER_START_ENTRY:]) + reserve
                             > 1_500_000_000)):
                    raise BudgetError("User's $1.50 / 20-call ceiling reached")
                if len(data["entries"]) >= 79:
                    grant = data.get("chat_test_grant")
                elif len(data["entries"]) >= 77:
                    grant = data.get("diagnostic_grant")
                elif len(data["entries"]) >= 75:
                    grant = data.get("demo_grant")
                elif len(data["entries"]) >= ANSWER_GUARD_V4_START_ENTRY:
                    grant = data.get("answer_guard_v4_grant")
                elif len(data["entries"]) >= SEARCH_FINAL_START_ENTRY:
                    grant = data.get("search_final_grant")
                elif len(data["entries"]) >= SEARCH_PROMPT_START_ENTRY:
                    grant = data.get("search_prompt_grant")
                elif len(data["entries"]) >= SEARCH_COMPACT_START_ENTRY:
                    grant = data.get("search_compact_grant")
                elif len(data["entries"]) >= SEARCH_ANSWER_START_ENTRY:
                    grant = data.get("search_answer_grant")
                elif len(data["entries"]) >= SEARCH_FIX_START_ENTRY:
                    grant = data.get("search_fix_grant")
                elif len(data["entries"]) >= CITATION_SCOPES_START_ENTRY:
                    grant = data.get("citation_scopes_grant")
                elif len(data["entries"]) >= ASR_CONTEXT_START_ENTRY:
                    grant = data.get("asr_context_grant")
                else:
                    grant = data.get("answer_guard_grant")
                if grant is None:
                    raise BudgetError("Answer evaluation call limit reached; paid calls stopped")
                entries = data["entries"][grant["start_entry"]:]
                if grant.get("id") in ("demo_2026_11_03", "diagnostic_2026_11_03", "chat_test_2026_11_03"):
                    allowed = ("diagnostic_short", "interactive_answer") if grant["id"] == "diagnostic_2026_11_03" else ("interactive_answer",)
                    if (len(entries) >= grant["max_calls"] or purpose not in allowed
                            or sum(e.get("cost_nano", e["reserved_nano"]) for e in entries) + reserve > grant["limit_nano"]):
                        raise BudgetError("Demo budget or call limit reached")
                elif (len(entries) >= grant["max_calls"]
                        or purpose not in grant["allowed_requests"]
                        or grant["allowed_requests"][purpose] != request_sha
                        or any(e["purpose"] == purpose for e in entries)
                        or sum(e.get("cost_nano", e["reserved_nano"]) for e in entries) + reserve > grant["limit_nano"]):
                    raise BudgetError("Additional grant limit or request check failed")
            entry = {"id": len(data["entries"]), "status": "pending", "purpose": purpose,
                     "created_utc": datetime.now(timezone.utc).isoformat(),
                     "input_bound": bound, "output_bound": max_output,
                     "reserved_nano": reserve,
                     "request_sha256": request_sha}
            data["entries"].append(entry)
            atomic_write_json(self.path, data)
            return entry["id"], reserve

    def complete(self, entry_id: int, usage: dict) -> int:
        with AppLock(self.path.with_suffix(".lock")):
            data = self.read()
            entry = data["entries"][entry_id]
            if entry["status"] != "pending":
                raise BudgetError("Invalid reservation state")
            incoming = usage.get("prompt_tokens")
            outgoing = usage.get("completion_tokens")
            if (type(incoming) is not int or type(outgoing) is not int or incoming < 0 or outgoing < 0
                    or incoming > entry["input_bound"] or outgoing > entry["output_bound"]):
                entry["status"] = "uncertain"
                atomic_write_json(self.path, data)
                raise BudgetError("Usage is missing or exceeds the reservation; paid calls stopped")
            entry.update(status="complete", prompt_tokens=incoming, completion_tokens=outgoing,
                         cost_nano=incoming * INPUT_NANO + outgoing * OUTPUT_NANO)
            atomic_write_json(self.path, data)
            return entry["cost_nano"]


class BudgetedDeepSeek:
    def __init__(self, ledger_path: Path, key_path: Path | None = None, *, client=None, key=None):
        credential = key or load_deepseek_key(key_path)
        self.ledger = BudgetLedger(ledger_path, hashlib.sha256(credential.encode()).hexdigest())
        self.client = client or httpx.Client(base_url=BASE_URL,
                    headers={"Authorization": "Bearer " + credential},
                    timeout=httpx.Timeout(90, connect=15), trust_env=False, follow_redirects=False)

    def close(self) -> None:
        self.client.close()

    def balance(self) -> float:
        # A failed balance read has no model charge. Retry it once, but never
        # retry a paid completion whose outcome may be uncertain.
        for _ in range(2):
            try:
                response = self.client.get("/user/balance")
                response.raise_for_status()
                balances = response.json().get("balance_infos", [])
                balance = float(next(b["total_balance"] for b in balances if b["currency"] == "USD"))
                if not math.isfinite(balance) or balance < 0:
                    raise ValueError('Invalid balance')
                return balance
            except Exception:
                pass
        raise BudgetError("Could not verify USD balance; no paid request made")

    def complete(self, messages: list[dict], *, max_output: int = 1536, purpose: str = "answer") -> dict:
        trace = RequestTrace(self.ledger.path.parent / "request_traces")
        trace.event("started")
        maximum = input_upper_bound(messages) * INPUT_NANO + max_output * OUTPUT_NANO
        try:
            trace.event("expense_settings_check")
            settings = self.ledger.check_request(messages, max_output)
            trace.event("balance_check_started")
            balance = self.balance()
            trace.event("balance_check_complete")
            if balance * 1_000_000_000 - maximum < settings['min_balance_nano']:
                raise BudgetError("Недостаточно средств с учётом минимального остатка на счёте. Проверьте настройки расходов.")
            entry_id, reserved = self.ledger.reserve(messages, max_output, purpose,
                balance_nano=int(Decimal(str(balance)) * spending.NANO))
            trace.event("reserved", entry_id=entry_id)
        except Exception:
            trace.event("stopped_before_send")
            raise
        # Intentionally no automatic retries. A lost response leaves the reservation pending.
        try:
            trace.event("send_started")
            with self.client.stream("POST", "/chat/completions", json={
                "model": MODEL, "messages": messages, "max_tokens": max_output,
                "thinking": {"type": "disabled"}, "reasoning_effort": "none",
                "stream": False, "temperature": 0.1,
            }, extensions={"trace": trace.network}) as response:
                trace.event("response_headers", http_status=response.status_code)
                response.raise_for_status()
                chunks = []
                total = 0
                last_logged = time.monotonic()
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    total += len(chunk)
                    if len(chunks) == 1 or time.monotonic() - last_logged >= 15:
                        trace.event("response_bytes", bytes_received=total)
                        last_logged = time.monotonic()
                trace.event("response_body_complete", bytes_received=total)
                data = json.loads(b"".join(chunks))
                trace.event("response_json_parsed")
        except KeyboardInterrupt:
            trace.event("interrupted")
            raise
        except httpx.TimeoutException:
            trace.event("timeout")
            raise BudgetError("DeepSeek request timed out; reservation retained, paid calls stopped") from None
        except Exception:
            trace.event("request_failed")
            raise BudgetError("DeepSeek request failed; reservation retained, paid calls stopped") from None
        try:
            cost = self.ledger.complete(entry_id, data.get("usage", {}))
            trace.event("usage_recorded")
        except Exception:
            trace.event("usage_failed")
            raise
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError()
        except (KeyError, IndexError, TypeError, ValueError):
            trace.event("answer_invalid")
            raise BudgetError("DeepSeek returned no text answer; usage was recorded") from None
        trace.event("complete")
        return {"text": content, "finish_reason": choice.get("finish_reason"),
                "usage": data["usage"], "cost_upper_usd": usd(cost), "budget_entry": entry_id}
