"""Conservative, local checks for ASR omissions and meaning-changing disagreement.

The checks identify *uncertain* audio spans. They do not certify the remaining words.
No external model, account, or domain-specific vocabulary is used.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
import re
from pathlib import Path
import statistics
from typing import Any, Callable
import wave

import numpy as np

TOKEN = re.compile(r"[а-яёa-z]+|\d+[.,]?\d*", re.IGNORECASE)
NEGATIONS = {"не", "нет", "нельзя", "невозможно", "отсутствует", "отсутствуют"}


def audio_gap_candidates(transcript: dict[str, Any], audio_path: Path, *,
                         min_gap: float = 8.0, min_relative_rms: float = 0.5) -> list[dict]:
    """Locate text-free spans containing signal; signal is not proof of speech."""
    duration = float(transcript["duration"])
    segments = transcript["segments"]
    boundaries = [(0.0, float(segments[0]["start"]))] if segments else [(0.0, duration)]
    boundaries += [(float(a["end"]), float(b["start"])) for a, b in zip(segments, segments[1:])]
    if segments:
        boundaries.append((float(segments[-1]["end"]), duration))
    with wave.open(str(audio_path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("ASR QA requires mono 16-bit PCM audio")
        rate = w.getframerate()

        def rms(lo: float, hi: float) -> float:
            start = max(0, min(w.getnframes(), int(lo * rate)))
            stop = max(start, min(w.getnframes(), int(hi * rate)))
            w.setpos(start)
            raw = w.readframes(stop - start)
            if not raw:
                return 0.0
            values = np.frombuffer(raw, dtype="<i2").astype(np.float64)
            return float(np.sqrt(np.mean(values ** 2)))

        references = [rms(x * duration, min(duration, x * duration + 15)) for x in (0.15, 0.45, 0.75)]
        positive = [x for x in references if x > 0]
        baseline = statistics.median(positive) if positive else 0.0
        if baseline <= 0:
            return []
        results = []
        for lo, hi in boundaries:
            if hi - lo < min_gap:
                continue
            level = rms(lo, hi)
            relative = level / baseline
            if relative >= min_relative_rms:
                results.append({"start": round(lo, 2), "end": round(hi, 2),
                                "relative_rms": round(relative, 2)})
        return results


def _timed_tokens(segments: list[dict], start: float, end: float) -> list[tuple[str, float]]:
    words: list[tuple[str, float]] = []
    for segment in segments:
        if segment["end"] <= start or segment["start"] >= end:
            continue
        tokens = TOKEN.findall(segment["text"].casefold())
        span = float(segment["end"]) - float(segment["start"])
        words.extend((token, moment) for i, token in enumerate(tokens)
                     if start <= (moment := float(segment["start"]) + span * (i + 0.5) / len(tokens)) < end)
    return words


def _anchored_conflicts(a: list[str], b: list[str]) -> list[tuple[str, int, int, int, int]]:
    """Report meaning-bearing changes only when adjacent words align."""
    found: list[tuple[str, int, int, int, int]] = []
    opcodes = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    for index, (tag, a0, a1, b0, b1) in enumerate(opcodes):
        if tag == "equal":
            continue
        left, right = a[a0:a1], b[b0:b1]
        before = opcodes[index - 1] if index else None
        after = opcodes[index + 1] if index + 1 < len(opcodes) else None
        anchored_before = before is not None and before[0] == "equal" and before[2] - before[1] >= 2
        anchored_after = after is not None and after[0] == "equal" and after[2] - after[1] >= 2
        if not (anchored_before and anchored_after):
            continue
        if any(w in NEGATIONS for w in left + right):
            found.append(("negation_disagreement", a0, a1, b0, b1))
        if any(w[0].isdigit() for w in left + right):
            found.append(("number_disagreement", a0, a1, b0, b1))
    return found


def _unique_issues(issues: list[dict]) -> list[dict]:
    issues.sort(key=lambda x: (x["start"], x["end"], x["kind"]))
    unique = []
    seen = set()
    for issue in issues:
        key = (issue["kind"], round((issue["start"] + issue["end"]) / 10))
        if key not in seen:
            unique.append(issue)
            seen.add(key)
    return unique


def _repetition_runs(segments: list[dict], kind: str) -> list[dict]:
    issues = []
    run_start = 0
    while run_start < len(segments):
        normalized = " ".join(TOKEN.findall(segments[run_start]["text"].casefold()))
        run_end = run_start + 1
        while run_end < len(segments) and " ".join(TOKEN.findall(segments[run_end]["text"].casefold())) == normalized:
            run_end += 1
        if len(normalized) >= 15 and run_end - run_start >= 3:
            lo, hi = float(segments[run_start]["start"]), float(segments[run_end - 1]["end"])
            if hi - lo >= 30:
                issues.append({"kind": kind, "start": round(lo, 2),
                               "end": round(hi, 2), "repetitions": run_end - run_start})
        run_start = run_end
    return issues


def gap_agreement(a: list[dict], b: list[dict], start: float, end: float) -> dict[str, Any]:
    """Check whether two clipped ASR retries agree inside a missing-text span."""
    first = [w for w, _ in _timed_tokens(a, start, end)]
    second = [w for w, _ in _timed_tokens(b, start, end)]
    ratio = SequenceMatcher(None, first, second, autojunk=False).ratio() if first and second else 0.0
    critical_first = [w for w in first if w in NEGATIONS or w[0].isdigit()]
    critical_second = [w for w in second if w in NEGATIONS or w[0].isdigit()]
    return {"ratio": round(ratio, 3), "words": [len(first), len(second)],
            "critical_agree": critical_first == critical_second,
            "accept": min(len(first), len(second)) >= 10 and ratio >= 0.75
                      and critical_first == critical_second}


def best_gap_agreement(variants: list[list[dict]], start: float, end: float) -> dict[str, Any]:
    """Require two context variants to agree before accepting gap text."""
    candidates = [(i, j, gap_agreement(variants[i], variants[j], start, end))
                  for i, j in combinations(range(len(variants)), 2)]
    if not candidates:
        return {"accept": False, "reason": "fewer_than_two_retries"}
    i, j, best = max(candidates, key=lambda item: (item[2]["accept"], item[2]["ratio"]))
    return {"accept": best["accept"], "pair": [i, j], **best}


def compare_passes(primary: dict[str, Any], secondary: dict[str, Any], *,
                   window_seconds: float = 30.0, gap_seconds: float = 8.0,
                   min_recovered_chars: int = 25) -> dict[str, Any]:
    """Compare two ASR passes of the same audio, returning bounded risk intervals.

    Window counts intentionally prefer false alarms over silently accepting a
    missing negation or number. An interval is a candidate for retry/exclusion,
    never proof that either version is correct.
    """
    if primary["material_id"] != secondary["material_id"]:
        raise ValueError("ASR passes refer to different materials")
    duration = float(primary["duration"])
    if abs(duration - float(secondary["duration"])) > 1:
        raise ValueError("ASR passes have different durations")
    a, b = primary["segments"], secondary["segments"]
    issues: list[dict] = []
    issues.extend(_repetition_runs(a, "repetition_run"))
    issues.extend(_repetition_runs(b, "secondary_repetition_run"))
    gap_edges = [(0.0, float(a[0]["start"]))] if a else [(0.0, duration)]
    gap_edges += [(float(before["end"]), float(after["start"])) for before, after in zip(a, a[1:])]
    if a:
        gap_edges.append((float(a[-1]["end"]), duration))
    for lo, hi in gap_edges:
        if hi - lo < gap_seconds:
            continue
        recovered = " ".join(s["text"] for s in b if s["end"] > lo + 1 and s["start"] < hi - 1)
        if len(recovered.strip()) >= min_recovered_chars:
            issues.append({"kind": "missing_speech", "start": round(lo, 2),
                           "end": round(hi, 2), "recovered_chars": len(recovered)})
    step = window_seconds / 2
    for i in range(int((duration + step - 1) // step)):
        lo, hi = i * step, min(duration, i * step + window_seconds)
        timed_a, timed_b = _timed_tokens(a, lo, hi), _timed_tokens(b, lo, hi)
        aw, bw = [w for w, _ in timed_a], [w for w, _ in timed_b]
        if not aw or not bw:
            continue
        # A one-word border shift should not quarantine a whole passage.
        if len(aw) < 8 or len(bw) < 8:
            continue
        similarity = SequenceMatcher(None, aw, bw, autojunk=False).ratio()
        if similarity < 0.65:
            continue
        for kind, a0, a1, b0, b1 in _anchored_conflicts(aw, bw):
            moments = [t for _, t in timed_a[a0:a1] + timed_b[b0:b1]]
            if not moments:
                continue
            center = sum(moments) / len(moments)
            issues.append({"kind": kind, "start": round(max(0, center - 2), 2),
                           "end": round(min(duration, center + 2), 2),
                           "similarity": round(similarity, 3)})
    return {"schema": 1, "material_id": primary["material_id"],
            "duration": duration, "issues": _unique_issues(issues)}


def merge_intervals(issues: list[dict], *, padding: float = 0.0) -> list[tuple[float, float]]:
    spans = sorted((max(0.0, x["start"] - padding), x["end"] + padding) for x in issues)
    merged: list[list[float]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(round(lo, 2), round(hi, 2)) for lo, hi in merged]


def quarantine_segments(transcript: dict[str, Any], intervals: list[tuple[float, float]]) -> dict[str, Any]:
    """Return a copy without disputed segments; source transcript stays intact."""
    result = dict(transcript)
    result["segments"] = [s for s in transcript["segments"]
                          if not any(s["end"] > lo and s["start"] < hi for lo, hi in intervals)]
    result["quality"] = {"status": "partial" if intervals else "screened",
                         "excluded_intervals": [{"start": lo, "end": hi} for lo, hi in intervals],
                         "removed_segments": len(transcript["segments"]) - len(result["segments"])}
    return result


def screen_passes(primary: dict[str, Any], secondary: dict[str, Any], audio_path: Path,
                  recognize_clip: Callable[[float, float], list[dict]], *,
                  max_checks: int = 80) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a searchable transcript while quarantining unresolved risky words.

    `recognize_clip` returns segments with absolute times. Rechecks are local;
    unresolved spans stay out of the searchable transcript.
    """
    comparison = compare_passes(primary, secondary)
    duration = float(secondary["duration"])
    selected = [x for x in comparison["issues"] if x["kind"] != "repetition_run"]
    priority = {"negation_disagreement": 0, "missing_speech": 1, "number_disagreement": 2}
    selected.sort(key=lambda x: (priority.get(x["kind"], 3), x["start"]))
    unresolved = []
    resolved = []
    calls = 0
    cache: dict[tuple[float, float], list[dict]] = {}

    def retry(start: float, end: float, padding: float) -> list[dict]:
        nonlocal calls
        lo, hi = round(max(0.0, start - padding), 2), round(min(duration, end + padding), 2)
        key = (lo, hi)
        if key not in cache:
            cache[key] = recognize_clip(lo, hi)
            calls += 1
        return cache[key]

    for issue in selected:
        if issue["kind"] == "secondary_repetition_run":
            unresolved.append({**issue, "resolution": "repeated_candidate_text"})
            continue
        if calls + 2 > max_checks:
            unresolved.append({**issue, "resolution": "retry_limit"})
            continue
        lo, hi = issue["start"], issue["end"]
        variants = [retry(lo, hi, 5), retry(lo, hi, 10)]
        agreements = [gap_agreement(secondary["segments"], variant, lo, hi) for variant in variants]
        if all(a["accept"] for a in agreements):
            resolved.append({**issue, "resolution": "two_retries_agree"})
        else:
            unresolved.append({**issue, "resolution": "disagreement"})

    masked = merge_intervals(unresolved)
    candidate = quarantine_segments(secondary, masked)
    gap_candidates = audio_gap_candidates(secondary, audio_path)
    recovered = []
    pending_gaps = []
    for gap in gap_candidates:
        lo, hi = gap["start"], gap["end"]
        if calls + 3 > max_checks:
            pending_gaps.append({**gap, "resolution": "retry_limit"})
            continue
        variants = [retry(lo, hi, padding) for padding in (5, 10, 15)]
        agreement = best_gap_agreement(variants, lo, hi)
        if not agreement["accept"]:
            pending_gaps.append({**gap, "resolution": "unstable_speech", "agreement": agreement})
            continue
        chosen = variants[agreement["pair"][0]]
        words = [word for word, _ in _timed_tokens(chosen, lo, hi)]
        if words and not any(hi > bad_lo and lo < bad_hi for bad_lo, bad_hi in masked):
            recovered.append({"start": lo, "end": hi, "text": " ".join(words)})
        else:
            pending_gaps.append({**gap, "resolution": "overlapping_dispute"})
    candidate["segments"] = sorted(candidate["segments"] + recovered, key=lambda s: (s["start"], s["end"]))
    candidate["quality"] = {
        "status": "partial" if masked or pending_gaps else "screened",
        "excluded_intervals": [{"start": lo, "end": hi} for lo, hi in masked],
        "removed_segments": len(secondary["segments"]) - len(quarantine_segments(secondary, masked)["segments"]),
        "recovered_gaps": len(recovered), "unresolved_gaps": len(pending_gaps),
    }
    report = {"schema": 1, "material_id": secondary["material_id"],
              "comparison": comparison, "resolved": resolved, "unresolved": unresolved,
              "audio_gap_candidates": gap_candidates, "recovered": [{"start": x["start"], "end": x["end"]} for x in recovered],
              "pending_gaps": pending_gaps, "targeted_calls": calls,
              "quality": candidate["quality"]}
    return candidate, report
