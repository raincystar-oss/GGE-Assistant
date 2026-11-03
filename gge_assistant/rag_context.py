"""Local, incremental context index. It never writes to the ingestion/AnythingLLM DBs."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .chunking import format_time

VERSION = "context-v1"
RETRIEVAL_VERSION = "retrieval-v5-cross-source-questions"
URL_RE = re.compile(r"https?://[^\s<>\]\)]+", re.I)
STOP = set("а без бы был была были быть в вам вас ваш ваша ваши весь во вот вы где да для до его ее если есть ещё же за зачем здесь и из или им их к как какой какая какие когда кто ли мне может можно мы на над нам нас не него нее нет но о об один она они оно от по под при про с со так такой там то того тоже только ты у уже чего чем что чтобы это этой этот я the a an and or of to in for is are how what why when".split())
STOP.update("хотя нужно нужны необходимо пожалуйста расскажи опиши нашей наша наше наши наш моего моей мой мне ему ей своего своей свой какой каких каким какую которое которой которые которых всем всех весь это этого этому этим эту эти почему сколько он ну ещё еще потом such please describe explain".split())
STOP.update("них каждое каждый каждую каждом перед укажи отдельно назван".split())


def safe_card(url: str) -> str:
    try:
        parts = urlsplit(url or "")
        if parts.scheme == "https" and parts.hostname in {"edu.gge.ru", "kb.gge.ru"} and not parts.username:
            return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except ValueError:
        pass
    return ""


def safe_text(text: str) -> str:
    # Source URLs and signed links can occur inside an arbitrary transcript/question.
    return URL_RE.sub(lambda m: safe_card(m.group()) or "[ссылка скрыта]", str(text))


def terms(text: str) -> list[str]:
    # Keep short abbreviations and exact numeric identifiers (including 1.3 / 3.0).
    words = re.findall(r"\d+(?:[.,]\d+)+|[^\W_]+", text.casefold().replace("ё", "е"))
    return list(dict.fromkeys(w for w in words if (len(w) >= 2 or w.isdigit()) and w not in STOP))


def search_prefix(term: str) -> str:
    # Only long alphabetic words get the existing, bounded inflection fallback.
    # Never turn 115 into 115* or ПД into ПД*: these denote different identifiers.
    return term[:6] if term.isalpha() and len(term) >= 7 else term


def fts_term(term: str, *, prefix: bool = False) -> str:
    stem = search_prefix(term) if prefix else term
    return '"' + stem.replace('"', '""') + '"' + ('*' if prefix and term.isalpha() and len(term) >= 4 else '')


def query_aspects(question: str) -> list[str]:
    """Up to two clauses, with explicit shared abbreviations carried forward.

    The full question is always retained. No names or assumptions are generated.
    """
    parts = re.split(r"[?;!]\s+|,\s*(?=(?:где|когда|сколько|почему|какие|кто|как)\b)"
                     r"|,\s*(?=а\s+в\s+каком\b)"
                     r"|\s+и\s+(?=(?:как|где|когда|сколько|почему|какие|кто|что|можно)\b|[а-яё-]+(?:ть|ти|чь)\b)"
                     r"|,?\s+(?:хотя|если)\s+", question, flags=re.I)
    parts = [p.strip(' ,;.!?') for p in parts if len(terms(p)) >= 2]
    if len(parts) < 2:
        return [question]
    # Reuse only abbreviations explicitly present in the question. A clause
    # without a subject inherits them if other clauses don't name different ones.
    named = [re.findall(r'\b[А-ЯЁA-Z]{2,}[А-ЯЁA-Z\d]*\b', p) for p in parts]
    groups = {tuple(n) for n in named if n}
    shared = next(iter(groups)) if len(groups) == 1 else ()
    parts = [p + (' (' + ' '.join(shared) + ')' if shared and not n else '')
             for p, n in zip(parts, named)]
    queries = list(dict.fromkeys([question] + parts))[:3]
    # A dependent clause such as "почему это важно" has no subject of its own.
    # The full question already retains it; searching it alone can pull in an
    # unrelated section and displace the section that answers the real question.
    if len(queries) >= 3:
        queries = [queries[0]] + [q for q in queries[1:] if not re.match(
            r"^\s*(?:почему|зачем|для чего)\s+это\s+(?:важно|нужно|необходимо)\b",
            q, flags=re.I)]
    return queries


def coverage_terms(text: str) -> set[str]:
    """Bounded inflection matching for scoring only; never alter source text.

    This is an approximate suffix/prefix match, not full morphology. Numeric
    identifiers and short abbreviations retain exact matching.
    """
    result = set()
    for word in terms(text):
        stem = word
        if re.fullmatch(r'[а-я]{5,}', word):
            candidate = re.sub(
                r'(?:иями|ями|ами|ого|ему|ому|ыми|ими|ий|ый|ая|яя|ое|ее|ые|ие|ов|ев|ей|ам|ям|ах|ях|ом|ем|ой|ы|и|а|я|у|ю|е|о|ь)$',
                '', word)
            if len(candidate) >= 4:
                stem = candidate
        result.add(search_prefix(stem))
    return result


@dataclass(frozen=True)
class ContextConfig:
    child_chars: int = 650
    child_max_chars: int = 1200
    parent_seconds: float = 180
    parent_max_seconds: float = 360
    parent_chars: int = 3000
    parent_max_chars: int = 4800
    pause_seconds: float = 1.5
    context_chars: int = 16000
    max_parents: int = 6
    candidates: int = 40
    continuation_units: int = 3

    def validate(self) -> None:
        if not (100 <= self.child_chars <= self.child_max_chars <= self.parent_max_chars
                and self.child_chars <= self.parent_chars <= self.parent_max_chars
                and 30 <= self.parent_seconds <= self.parent_max_seconds
                and 0 <= self.pause_seconds <= 30 and 1 <= self.max_parents <= 12
                and self.parent_max_chars + 1000 <= self.context_chars <= 20000
                and 1 <= self.candidates <= 100 and 0 <= self.continuation_units <= 3):
            raise ValueError("Invalid context settings")


def make_units(segments: list[dict], cfg: ContextConfig) -> list[dict]:
    """Group original ASR segments at sentence/pause boundaries; no topic vocabulary.

    Long segments are split without dropping text and retain the original time range.
    These are structural windows, not claimed to be perfect semantic chapters.
    """
    cfg.validate()
    pieces = []
    last_start = -1.0
    for index, segment in enumerate(segments):
        start, end = float(segment["start"]), float(segment["end"])
        if not 0 <= start <= end or start < last_start:
            raise ValueError("Invalid or unordered transcript timing")
        last_start = start
        text = re.sub(r"\s+", " ", safe_text(segment["text"])).strip()
        while text:
            n = min(len(text), cfg.child_max_chars)
            if n < len(text):
                boundary = max(text.rfind(". ", 0, n), text.rfind("? ", 0, n), text.rfind("! ", 0, n))
                n = boundary + 1 if boundary >= n // 2 else (text.rfind(" ", 0, n) or n)
                if n < 1:
                    n = cfg.child_max_chars
            pieces.append(dict(start=start, end=end, first_segment=index, last_segment=index,
                               text=text[:n].strip()))
            text = text[n:].strip()
    units = []
    current = None
    for piece in pieces:
        if current is not None:
            pause = piece["start"] - current["end"]
            sentence_end = bool(re.search(r"[.!?…][\"»)]?$", current["text"]))
            if (len(current["text"]) + len(piece["text"]) + 1 > cfg.child_max_chars
                    or pause >= cfg.pause_seconds
                    or (len(current["text"]) >= cfg.child_chars and sentence_end)):
                units.append(current)
                current = None
        if current is None:
            current = dict(piece)
        else:
            current.update(end=max(current["end"], piece["end"]), last_segment=piece["last_segment"],
                           text=current["text"] + " " + piece["text"])
    if current:
        units.append(current)
    parent = 0
    parent_start = 0.0
    length = 0
    previous = None
    for unit in units:
        if previous is not None:
            hard_limit = (unit["end"] - parent_start > cfg.parent_max_seconds
                          or length + len(unit["text"]) > cfg.parent_max_chars)
            target = (previous["end"] - parent_start >= cfg.parent_seconds or length >= cfg.parent_chars)
            # Keep an explicit question with its immediate continuation when the hard cap permits.
            question = bool(re.search(r"\?[\"»)]?$", previous["text"]))
            if hard_limit or (target and not question):
                parent += 1
                length = 0
        if length == 0:
            parent_start = unit["start"]
        unit["parent_index"] = parent
        length += len(unit["text"]) + 1
        previous = unit
    return units


SCHEMA = """
CREATE TABLE IF NOT EXISTS materials(
 id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, title TEXT, webinar_date TEXT, url TEXT);
CREATE TABLE IF NOT EXISTS units(
 id INTEGER PRIMARY KEY, material_id TEXT REFERENCES materials(id) ON DELETE CASCADE,
 unit_index INTEGER, parent_index INTEGER, start REAL, end REAL,
 first_segment INTEGER, last_segment INTEGER, text TEXT,
 UNIQUE(material_id,unit_index));
CREATE INDEX IF NOT EXISTS units_material_time ON units(material_id,start,end);
CREATE INDEX IF NOT EXISTS units_parent ON units(material_id,parent_index);
CREATE TABLE IF NOT EXISTS docmap(
 title TEXT PRIMARY KEY, material_id TEXT REFERENCES materials(id) ON DELETE CASCADE,
 start REAL, end REAL);
CREATE VIRTUAL TABLE IF NOT EXISTS unit_fts USING fts5(
 text, content='units', content_rowid='id', tokenize='unicode61', prefix='3 4 5 6');
CREATE TRIGGER IF NOT EXISTS unit_insert AFTER INSERT ON units BEGIN
 INSERT INTO unit_fts(rowid,text) VALUES (new.id,new.text); END;
CREATE TRIGGER IF NOT EXISTS unit_delete AFTER DELETE ON units BEGIN
 INSERT INTO unit_fts(unit_fts,rowid,text) VALUES ('delete',old.id,old.text); END;
"""


class ContextIndex:
    def __init__(self, path: Path, cfg: ContextConfig | None = None):
        self.cfg = cfg or ContextConfig()
        self.cfg.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def upsert(self, transcript: dict, material: dict, documents: list[dict]) -> bool:
        material_id = transcript["material_id"]
        meta = transcript.get("metadata", {})
        title = safe_text(meta.get("title") or material.get("title") or "Без названия")
        date = safe_text(meta.get("webinar_date") or material.get("webinar_date") or "не указана")
        url = safe_card(material.get("discovery_url", ""))
        structure = {k: v for k, v in asdict(self.cfg).items() if k not in {"context_chars", "max_parents", "candidates", "continuation_units"}}
        fingerprint = hashlib.sha256(json.dumps([VERSION, structure, transcript,
                    title, date, url, documents], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        previous = self.db.execute("SELECT fingerprint FROM materials WHERE id=?", (material_id,)).fetchone()
        if previous and previous[0] == fingerprint:
            return False
        units = make_units(transcript["segments"], self.cfg)
        with self.db:
            self.db.execute("DELETE FROM materials WHERE id=?", (material_id,))
            self.db.execute("INSERT INTO materials VALUES (?,?,?,?,?)", (material_id, fingerprint, title, date, url))
            for i, u in enumerate(units):
                self.db.execute("INSERT INTO units(material_id,unit_index,parent_index,start,end,first_segment,last_segment,text) VALUES (?,?,?,?,?,?,?,?)",
                     (material_id, i, u["parent_index"], u["start"], u["end"], u["first_segment"], u["last_segment"], u["text"]))
            for d in documents:
                self.db.execute("INSERT INTO docmap VALUES (?,?,?,?)",
                                (d["title"].removesuffix(".txt"), material_id, d["start"], d["end"]))
        return True

    def restrict_to(self, material_ids: set[str]) -> None:
        """Remove stale entries only from the expendable pilot index, never the main DB."""
        with self.db:
            for row in self.db.execute("SELECT id FROM materials").fetchall():
                if row[0] not in material_ids:
                    self.db.execute("DELETE FROM materials WHERE id=?", (row[0],))

    def lexical(self, question: str) -> list[dict]:
        tokens = terms(question)[:32]
        if not tokens:
            return []
        # Union exact and inflection candidates so a broad prefix cannot crowd out
        # a rare exact identifier before parent selection gets to inspect it.
        found = {}
        for prefix in (False, True):
            query = " OR ".join(fts_term(t, prefix=prefix) for t in tokens)
            rows = self.db.execute("SELECT u.*, bm25(unit_fts) AS lexical_score FROM unit_fts "
                        "JOIN units u ON u.id=unit_fts.rowid WHERE unit_fts MATCH ? ORDER BY lexical_score,u.id LIMIT ?",
                        (query, self.cfg.candidates)).fetchall()
            for row in rows:
                unit = dict(row)
                old = found.get(unit['id'])
                if old is None or unit['lexical_score'] < old['lexical_score']:
                    found[unit['id']] = unit
        return sorted(found.values(), key=lambda u: (u['lexical_score'], u['id']))

    def map_vector_hit(self, hit: dict) -> dict | None:
        metadata = hit.get("metadata") or {}
        title = str(metadata.get("title") or hit.get("title") or "").removesuffix(".txt")
        doc = self.db.execute("SELECT * FROM docmap WHERE title=?", (title,)).fetchone()
        if not doc:
            return None
        rows = self.db.execute("SELECT * FROM units WHERE material_id=? AND end>? AND start<? ORDER BY unit_index",
                               (doc["material_id"], doc["start"], doc["end"])).fetchall()
        if not rows:
            return None
        hit_terms = set(terms(hit.get("text", "")))
        def similarity(row):
            own = set(terms(row["text"]))
            return len(own & hit_terms) / max(1, len(own | hit_terms))
        return dict(max(rows, key=similarity))

    def assemble(self, queries: list[str], vector_results: list[list[dict]], priority_parents: list[tuple] | None = None) -> dict:
        scores = defaultdict(float)
        hybrid_scores = defaultdict(float)
        diagnostics = {"mapped_vector_hits": 0, "unmapped_vector_hits": 0}
        aspect_seeds = []
        hybrid_seeds = []
        primary_material = None
        named_parts = [set(re.findall(r'\b[А-ЯЁA-Z]{2,}[А-ЯЁA-Z\d]*\b', q)) for q in queries[1:]]
        named_parts = [s for s in named_parts if s]
        multiple_scopes = (len(named_parts) > 1 and not set.intersection(*named_parts)) or bool(
            len(queries) > 1 and re.search(r"\bв\s+каком\b.+\bа\s+в\s+каком\b", queries[0], re.I))
        parent_rows = {}
        titles = {r['id']: r['title'] for r in self.db.execute('SELECT id,title FROM materials')}

        def rows_for(key):
            if key not in parent_rows:
                parent_rows[key] = self.db.execute(
                    'SELECT * FROM units WHERE material_id=? AND parent_index=? ORDER BY unit_index', key).fetchall()
            return parent_rows[key]

        for query_index, (query, hits) in enumerate(zip(queries, vector_results)):
            vector_units = []
            query_scores = defaultdict(float)
            for hit in hits:
                unit = self.map_vector_hit(hit)
                diagnostics["mapped_vector_hits" if unit else "unmapped_vector_hits"] += 1
                if unit:
                    vector_units.append(unit)
            for ranked in (vector_units, self.lexical(query)):
                seen = set()
                for rank, unit in enumerate(ranked, 1):
                    key = (unit["material_id"], unit["parent_index"])
                    if key not in seen:
                        query_scores[key] += 1 / (40 + rank)
                        seen.add(key)
            # Exact identifiers outrank lookalike prefixes without rewarding every
            # generic word that happens to match an introductory paragraph.
            anchors = [t for t in terms(query) if any(c.isdigit() for c in t)]
            if anchors:
                for key in query_scores:
                    text = ' '.join(r[0] for r in self.db.execute(
                        "SELECT text FROM units WHERE material_id=? AND parent_index=?", key))
                    exact = set(terms(text))
                    query_scores[key] *= 1 + 0.25 * sum(t in exact for t in anchors) / len(anchors)
            original_scores = dict(query_scores)
            for key, score in original_scores.items():
                hybrid_scores[key] += score
            # Hybrid ranks alone reward general mentions. Inspect the actual
            # words of each bounded parent, weighting rare matching query terms.
            wanted = coverage_terms(query)
            matched = {key: wanted & set().union(*(coverage_terms(r['text']) for r in rows_for(key)))
                       for key in query_scores}
            weights = {t: math.log(1 + (len(matched) + 1) / (1 + sum(t in value for value in matched.values())))
                       for t in sorted(wanted)}
            total = sum(weights.values()) or 1
            names = set(re.findall(r'\b[А-ЯЁA-Z]{2,}[А-ЯЁA-Z\d]*\b', query))
            for key in query_scores:
                coverage = sum(weights[t] for t in sorted(matched[key])) / total
                named = sum(n.casefold() in set(terms(titles[key[0]])) for n in names) / max(1, len(names))
                query_scores[key] *= (0.5 + 1.5 * coverage) * (1 + 0.6 * named)
            if query_index == 0 and query_scores:
                primary_material = max(query_scores, key=lambda k: (query_scores[k], k))[0]
            for key, score in query_scores.items():
                scores[key] += score * (2 if query_index == 0 else 1)
            if query_index > 0 and query_scores:
                # Each literal subquestion gets its best hybrid result, not its
                # first vector hit (which can describe an unrelated access issue).
                scope = {k for k in query_scores if multiple_scopes or primary_material is None or k[0] == primary_material}
                scope = scope or set(query_scores)
                aspect_seeds.append(max(scope, key=lambda k: (query_scores[k], k)))
                hybrid_seeds.append(max(scope, key=lambda k: (original_scores[k], k)))
        selected_units = {}
        selected_parents = []
        # Give each subquestion a chance before common introductory passages exhaust the budget.
        ranked = sorted(scores, key=lambda k: (-scores[k], k))
        hybrid_ranked = sorted(hybrid_scores, key=lambda k: (-hybrid_scores[k], k))
        # Preserve the complementary hybrid view: exact-word coverage alone
        # can prefer a general mention over a paraphrased answer.
        seeds = list(dict.fromkeys(hybrid_ranked[:1] + ranked[:1] + hybrid_seeds + aspect_seeds))
        if len(seeds) == 1:
            key = seeds[0]
            first = rows_for(key)[0]['text']
            dependency = re.match(r'\s*(?:а\s+также|также|далее|после\s+этого|при\s+этом|соответственно|данн(?:ая|ый|ое)|таким\s+образом)\b', first, re.I)
            previous = (key[0], key[1] - 1)
            if len(queries) > 1 and dependency and key[1] > 0 and rows_for(previous):
                seeds.append(previous)
                scores.setdefault(previous, 0.0)
                diagnostics['preceding_section_for_dependency'] = previous
            else:
                # Agreement on one hit does not prove completeness. Keep one
                # alternative, especially when the hit is a summary or Q&A.
                seeds = list(dict.fromkeys(seeds + hybrid_ranked[:2] + ranked[:2]))[:2]
        broad = re.search(r"\b(?:(?:весь|полный)\s+(?:порядок|процесс|алгоритм|цикл)|все\s+(?:шаги|этапы)|пошагово|поэтапно|all\s+steps|entire\s+process)\b", queries[0].casefold()) if queries else None
        if broad and not priority_parents and len({k[0] for k in aspect_seeds}) <= 1:
            window = self._procedure_window(scores, aspect_seeds)
            if window:
                seeds = window
                for key in window:
                    scores.setdefault(key, 0.0)
        ranked_keys = list(dict.fromkeys((priority_parents or []) + seeds))
        limit = self.cfg.max_parents if broad or priority_parents else min(self.cfg.max_parents, 3 + (len(queries) > 1))
        # Reserve complete core sections for distinct parts first. Expanding one
        # early hit must not evict the core of the next part of the question.
        for key in ranked_keys:
            if key not in scores:
                continue
            score = scores[key]
            if len(selected_parents) >= limit:
                break
            rows = rows_for(key)
            trial = dict(selected_units)
            trial.update({(r["material_id"], r["unit_index"]): dict(r) for r in rows})
            blocks = self._blocks(trial)
            if len(render_context(blocks)) <= self.cfg.context_chars:
                selected_units = trial
                selected_parents.append({"material_id": key[0], "parent_index": key[1], "score": score})
        for distance in range(1, self.cfg.continuation_units + 1):
            for parent in selected_parents:
                key = (parent['material_id'], parent['parent_index'])
                core = rows_for(key)
                for unit_number in (core[0]['unit_index'] - distance, core[-1]['unit_index'] + distance):
                    inner = unit_number + (1 if unit_number < core[0]['unit_index'] else -1)
                    if distance > 1 and (key[0], inner) not in selected_units:
                        continue
                    row = self.db.execute('SELECT * FROM units WHERE material_id=? AND unit_index=?', (key[0], unit_number)).fetchone()
                    if row is None:
                        continue
                    trial = dict(selected_units)
                    trial[(key[0], unit_number)] = dict(row)
                    if len(render_context(self._blocks(trial))) <= self.cfg.context_chars:
                        selected_units = trial
        blocks = self._blocks(selected_units)
        return {"retrieval_version": RETRIEVAL_VERSION,
                "blocks": blocks, "context": render_context(blocks), "selected_parents": selected_parents,
                "diagnostics": diagnostics, "queries": queries, "context_chars": len(render_context(blocks)),
                "evidence_status": "retrieved_unverified" if blocks else "no_candidates",
                "candidate_parents": [{"material_id": k[0], "parent_index": k[1], "score": scores[k]} for k in ranked]}

    def _procedure_window(self, scores: dict, anchors: list[tuple]) -> list[tuple]:
        """Fit a continuous explanation for an explicitly requested full procedure.

        Evaluate bounded windows of candidate parents, not an entire transcript.
        This prevents a distant Q&A or introduction from displacing the middle steps.
        """
        best, best_score = [], 0.0
        by_material = defaultdict(list)
        for material, parent in sorted(scores):
            by_material[material].append(parent)
        for material, parents in by_material.items():
            for first in parents:
                keys = []
                for parent in range(first, first + self.cfg.max_parents):
                    key = (material, parent)
                    rows = self.db.execute('SELECT * FROM units WHERE material_id=? AND parent_index BETWEEN ? AND ? ORDER BY unit_index',
                                           (material, first, parent)).fetchall()
                    if not rows or rows[-1]['parent_index'] != parent:
                        break
                    keys.append(key)
                    lo = max(0, rows[0]['unit_index'] - self.cfg.continuation_units)
                    hi = rows[-1]['unit_index'] + self.cfg.continuation_units
                    units = {(material, r['unit_index']): dict(r) for r in self.db.execute(
                        'SELECT * FROM units WHERE material_id=? AND unit_index BETWEEN ? AND ? ORDER BY unit_index', (material, lo, hi))}
                    if len(render_context(self._blocks(units))) > self.cfg.context_chars:
                        break
                    represented = {(u['material_id'], u['parent_index']) for u in units.values()}
                    if not set(anchors) <= represented:
                        continue
                    score = sum(scores.get(k, 0.0) for k in keys)
                    if score > best_score:
                        best, best_score = list(keys), score
        return best

    def _blocks(self, units: dict) -> list[dict]:
        groups = []
        for key in sorted(units):
            u = units[key]
            if (groups and groups[-1][-1]["material_id"] == u["material_id"]
                    and groups[-1][-1]["parent_index"] == u["parent_index"]
                    and groups[-1][-1]["unit_index"] + 1 == u["unit_index"]):
                groups[-1].append(u)
            else:
                groups.append([u])
        blocks = []
        for group in groups:
            meta = dict(self.db.execute("SELECT * FROM materials WHERE id=?", (group[0]["material_id"],)).fetchone())
            blocks.append({"source_id": f"S{len(blocks)+1}", "material_id": meta["id"], "title": meta["title"],
                 "date": meta["webinar_date"], "url": meta["url"], "start": group[0]["start"],
                 "end": max(u["end"] for u in group), "text": "\n\n".join(u["text"] for u in group),
                 "segments": sorted({n for u in group for n in range(u["first_segment"],u["last_segment"]+1)})})
        return blocks


def render_context(blocks: list[dict]) -> str:
    return "\n\n".join(f"[{b['source_id']}] {b['title']} | {b['date']} | "
            f"{format_time(b['start'])}–{format_time(b['end'])}"
            + (f" | {b['url']}" if b.get("url") else "") + "\n" + b["text"] for b in blocks)
