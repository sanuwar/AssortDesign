from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?\n]?")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
MAX_QUOTE_CHARS = 320


@dataclass
class CitationResult:
    claim_text: str
    quote_text: str
    source_start: Optional[int]
    source_end: Optional[int]
    confidence: float


@dataclass
class RiskFlag:
    severity: str
    category: str
    text_span: str
    suggested_fix: str


def _sentences_with_offsets(text: str) -> List[tuple[str, int, int]]:
    sentences: List[tuple[str, int, int]] = []
    index = 0
    for match in _SENTENCE_RE.finditer(text):
        sentence = match.group(0)
        start = match.start()
        end = match.end()
        if sentence.strip():
            sentences.append((sentence.strip(), start, end))
        index = end
    if index < len(text):
        tail = text[index:].strip()
        if tail:
            sentences.append((tail, index, len(text)))
    return sentences


def _tokenize(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def _find_focus(sentence: str, claim_tokens: set[str]) -> int:
    lower = sentence.lower()
    indices = []
    for token in claim_tokens:
        if not token:
            continue
        idx = lower.find(token)
        if idx != -1:
            indices.append(idx)
    if indices:
        return min(indices)
    return max(0, len(sentence) // 2)


def _trim_span(text: str, focus_index: int, max_len: int) -> tuple[str, int, int]:
    if len(text) <= max_len:
        return text, 0, len(text)
    half = max_len // 2
    start = max(0, focus_index - half)
    end = min(len(text), start + max_len)
    start = max(0, end - max_len)
    return text[start:end], start, end


def citation_finder(source_text: str, claims: Iterable[str]) -> List[CitationResult]:
    source_text = source_text or ""
    sentences = _sentences_with_offsets(source_text)
    results: List[CitationResult] = []
    claims_list = [c for c in (claims or []) if str(c).strip()]
    if not claims_list:
        return results

    for claim in claims_list[:8]:
        claim_text = str(claim).strip()
        if not claim_text:
            continue
        claim_tokens = _tokenize(claim_text)
        best = None
        best_score = 0.0

        # Exact substring match first.
        idx = source_text.lower().find(claim_text.lower())
        if idx != -1:
            quote = source_text[idx : idx + len(claim_text)]
            trimmed, t_start, t_end = _trim_span(
                quote, max(0, len(quote) // 2), MAX_QUOTE_CHARS
            )
            results.append(
                CitationResult(
                    claim_text=claim_text,
                    quote_text=trimmed.strip(),
                    source_start=idx + t_start,
                    source_end=idx + t_end,
                    confidence=0.9,
                )
            )
            continue

        for sentence, start, end in sentences:
            sentence_tokens = _tokenize(sentence)
            if not sentence_tokens:
                continue
            overlap = claim_tokens.intersection(sentence_tokens)
            score = len(overlap) / max(1, len(claim_tokens))
            if score > best_score:
                best_score = score
                best = (sentence, start, end)

        if best and best_score >= 0.25:
            quote, start, end = best
            focus = _find_focus(quote, claim_tokens)
            trimmed, t_start, t_end = _trim_span(quote, focus, MAX_QUOTE_CHARS)
            results.append(
                CitationResult(
                    claim_text=claim_text,
                    quote_text=trimmed.strip(),
                    source_start=start + t_start,
                    source_end=start + t_end,
                    confidence=min(0.85, 0.4 + best_score),
                )
            )
        else:
            results.append(
                CitationResult(
                    claim_text=claim_text,
                    quote_text="",
                    source_start=None,
                    source_end=None,
                    confidence=0.2,
                )
            )

    return results[:8]


def count_supported_citations(citations: Iterable[CitationResult]) -> int:
    count = 0
    for item in citations:
        if item.quote_text and item.confidence >= 0.4:
            count += 1
    return count


def risk_checker(text: str) -> List[RiskFlag]:
    output = text or ""
    flags: List[RiskFlag] = []

    absolute_re = re.compile(r"\b(will|guarantee|guarantees|always|never|proves?)\b", re.I)
    comparative_re = re.compile(r"\b(best|superior|outperforms?|better than)\b", re.I)
    caution_re = re.compile(r"\b(may|might|could|suggests?|limited|preliminary|potential)\b", re.I)

    for match in absolute_re.finditer(output):
        span = match.group(0)
        flags.append(
            RiskFlag(
                severity="high",
                category="over-certain language",
                text_span=span,
                suggested_fix="Add uncertainty (e.g., may/might) or cite evidence.",
            )
        )

    for match in comparative_re.finditer(output):
        span = match.group(0)
        flags.append(
            RiskFlag(
                severity="medium",
                category="unsupported comparison",
                text_span=span,
                suggested_fix="Clarify comparison basis or soften the claim.",
            )
        )

    if output and not caution_re.search(output):
        flags.append(
            RiskFlag(
                severity="low",
                category="missing limitation",
                text_span="No limitation language detected.",
                suggested_fix="Add a limitation or uncertainty note.",
            )
        )

    return flags[:8]
