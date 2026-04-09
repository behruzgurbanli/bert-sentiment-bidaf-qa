#!/usr/bin/env python3
"""
Shared helpers for Project 4.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence


TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


@dataclass(frozen=True)
class QAExample:
    example_id: str
    context: str
    question: str
    answer_texts: tuple[str, ...]
    answer_start_char: int
    answer_end_char: int
    title: str | None = None


@dataclass(frozen=True)
class SentimentExample:
    example_id: str
    text: str
    label: str
    split: str
    rating: int | None = None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tokenize_with_spans(text: str) -> List[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def answer_span_from_chars(
    token_spans: Sequence[tuple[str, int, int]],
    start_char: int,
    end_char: int,
) -> tuple[int, int] | None:
    start_tok = None
    end_tok = None
    for idx, (_, s0, s1) in enumerate(token_spans):
        if start_tok is None and s0 <= start_char < s1:
            start_tok = idx
        if s0 < end_char <= s1:
            end_tok = idx
            break
        if start_tok is not None and s1 >= end_char:
            end_tok = idx
            break
    if start_tok is None or end_tok is None or start_tok > end_tok:
        return None
    return start_tok, end_tok


def load_squad_examples(path: Path) -> List[QAExample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[QAExample] = []
    fallback_id = 0
    for article in data.get("data", []):
        title = article.get("title")
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context", ""))
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                if not answers:
                    continue
                first = answers[0]
                answer_text = str(first.get("text", ""))
                answer_start = int(first.get("answer_start", -1))
                if answer_start < 0 or not answer_text:
                    continue
                out.append(
                    QAExample(
                        example_id=str(qa.get("id") or f"qa_{fallback_id}"),
                        context=context,
                        question=str(qa.get("question", "")),
                        answer_texts=tuple(str(a.get("text", "")) for a in answers if a.get("text")),
                        answer_start_char=answer_start,
                        answer_end_char=answer_start + len(answer_text),
                        title=title,
                    )
                )
                fallback_id += 1
    return out


def load_imdb_split(split_dir: Path, split_name: str) -> List[SentimentExample]:
    examples: List[SentimentExample] = []
    for label in ("pos", "neg"):
        label_dir = split_dir / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.glob("*.txt")):
            rating = None
            parts = path.stem.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                rating = int(parts[1])
            examples.append(
                SentimentExample(
                    example_id=f"{split_name}/{label}/{path.stem}",
                    text=path.read_text(encoding="utf-8").strip(),
                    label=label,
                    split=split_name,
                    rating=rating,
                )
            )
    return examples


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match_score(prediction: str, gold_answers: Iterable[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(gold) for gold in gold_answers))


def f1_score_span(prediction: str, gold_answers: Iterable[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    pred_counts = {tok: pred_tokens.count(tok) for tok in set(pred_tokens)}
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        gold_counts = {tok: gold_tokens.count(tok) for tok in set(gold_tokens)}
        overlap = sum(min(pred_counts.get(tok, 0), gold_counts.get(tok, 0)) for tok in pred_counts)
        if overlap <= 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return float(best)
