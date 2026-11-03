"""Split retrieved source blocks into bounded conditional citation cards.

This is a presentation step after retrieval. It never changes selected evidence,
edits source words, or claims more precise timestamps than the parent block has.
"""
from __future__ import annotations

import re

from .rag_context import coverage_terms, render_context

VERSION = "conditional-citations-v1"
_CONDITION = re.compile(r"(?<!\w)(?:в\s+случае,?\s+)?если\s+", re.IGNORECASE)


def _parts(text: str) -> list[str]:
    anchors = list(_CONDITION.finditer(text))
    if len(anchors) < 2 or len(anchors) > 8:
        return [text]
    positions = [m.start() for m in anchors]
    prefix = text[:positions[0]]
    spans = [text[start:positions[i + 1] if i + 1 < len(positions) else len(text)]
             for i, start in enumerate(positions)]
    if prefix.strip():
        # A short lead-in normally names the topic of the first condition.
        if len(prefix.strip()) < 120:
            spans[0] = prefix + spans[0]
        else:
            spans.insert(0, prefix)
    parts = []
    for part in spans:
        part = part.strip()
        if not part:
            continue
        if parts and len(part) < 120 and not re.search(r"[.!?]", part):
            parts[-1] += " " + part
        else:
            parts.append(part)
    return parts


def scoped_citations(context: dict, question: str, max_chars: int = 20_000) -> dict:
    """Give distinct conditions separate citation IDs without losing any text.

    Every card retains the parent time range. If repeated headers would breach
    the context limit, that block stays whole. Callers must not treat the card
    range as a new sentence-level timestamp.
    """
    original = context["blocks"]
    chosen = [[dict(block)] for block in original]

    def renumber(groups: list[list[dict]]) -> list[dict]:
        flat = []
        for group in groups:
            for block in group:
                flat.append({**block, "source_id": f"S{len(flat) + 1}"})
        return flat

    wanted = coverage_terms(question)
    candidates = []
    for i, block in enumerate(original):
        parts = _parts(block["text"])
        if len(parts) < 2:
            continue
        relevance = max(len(wanted & coverage_terms(part[:240])) for part in parts)
        if relevance >= 3:
            candidates.append((relevance, i, parts))
    # Split only the two most relevant conditional blocks; other text remains
    # available unchanged, avoiding a large increase in citation headers.
    for _, i, parts in sorted(candidates, key=lambda x: (-x[0], x[1]))[:2]:
        block = original[i]
        cards = [{**block, "text": part, "parent_source_id": block["source_id"],
                  "scope_part": j + 1, "scope_parts": len(parts),
                  "time_precision": "parent_range"}
                 for j, part in enumerate(parts)]
        trial = [*chosen[:i], cards, *chosen[i + 1:]]
        if len(render_context(renumber(trial))) <= max_chars:
            chosen[i] = cards
    blocks = renumber(chosen)
    result = {**context, "blocks": blocks, "context": render_context(blocks),
              "citation_scope_version": VERSION,
              "citation_scope_splits": sum(len(group) - 1 for group in chosen)}
    result["context_chars"] = len(result["context"])
    return result
