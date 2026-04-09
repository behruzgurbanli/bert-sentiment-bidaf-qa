#!/usr/bin/env python3
"""
Project 4 Task 2 using SQuAD 1.1 as the active dataset.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from nlp_project.p4.common import (
    QAExample,
    answer_span_from_chars,
    ensure_parent,
    exact_match_score,
    f1_score_span,
    load_squad_examples,
    tokenize_with_spans,
)


DEFAULT_QA_MODEL_NAME = "distilbert-base-cased-distilled-squad"
DEFAULT_BERT_EMBEDDING_MODEL = "bert-base-uncased"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class P4Task2QAConfig:
    train_path: Path = Path("dataset/squad/train-v1.1.json")
    dev_path: Path = Path("dataset/squad/dev-v1.1.json")
    runtime_model_name: str = DEFAULT_QA_MODEL_NAME
    bert_embedding_model_name: str = DEFAULT_BERT_EMBEDDING_MODEL

    max_context_tokens: int = 160
    max_question_tokens: int = 32
    sample_count: int = 5
    min_word_count: int = 2

    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 0.001
    bert_learning_rate: float = 2e-5
    hidden_size: int = 64
    word_embedding_dim: int = 100
    dropout: float = 0.2
    freeze_bert: bool = True
    max_answer_len: int = 30

    prepare_only: bool = True
    train_word_bidaf: bool = True
    train_bert_bidaf: bool = False
    max_train_examples: int = 0
    max_dev_examples: int = 0
    max_eval_examples: int = 0

    try_runtime_inference: bool = False
    local_files_only: bool = False
    demo_question: str = "When did Beyonce start becoming popular?"
    demo_context: str = (
        "Beyonce Giselle Knowles-Carter became popular in the late 1990s as the lead singer "
        "of Destiny's Child."
    )
    out_dir: Path = Path("data/reports/p4_task2_qa")
    out_json: Path = Path("data/reports/p4_task2_qa_prepare_report.json")


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
        from torch import nn
        from torch.utils.data import DataLoader
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Task 2 training requires PyTorch at runtime. Run training inside your conda environment."
        ) from e
    return torch, nn, F, DataLoader


def _require_transformers_auto():
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "BERT-embedding training requires transformers at runtime. Use your conda environment."
        ) from e
    return AutoTokenizer, AutoModel


@dataclass(frozen=True)
class TokenizedQAExample:
    example_id: str
    title: str | None
    context: str
    question: str
    context_tokens: tuple[str, ...]
    question_tokens: tuple[str, ...]
    start_idx: int
    end_idx: int
    answer_texts: tuple[str, ...]


def _prepare_example_view(example: QAExample, *, max_context_tokens: int, max_question_tokens: int) -> dict[str, Any] | None:
    tokenized = _tokenize_example(
        example,
        max_context_tokens=max_context_tokens,
        max_question_tokens=max_question_tokens,
    )
    if tokenized is None:
        return None
    return {
        "example_id": tokenized.example_id,
        "title": tokenized.title,
        "question": tokenized.question,
        "gold_answers": list(tokenized.answer_texts),
        "context_char_length": len(tokenized.context),
        "context_token_count": len(tokenized.context_tokens),
        "question_token_count": len(tokenized.question_tokens),
        "answer_token_span": {"start": tokenized.start_idx, "end": tokenized.end_idx},
        "question_preview_tokens": list(tokenized.question_tokens[:max_question_tokens]),
        "context_preview_tokens": list(tokenized.context_tokens[:max_context_tokens]),
    }


def _summarize_split(
    name: str,
    examples: Sequence[QAExample],
    *,
    max_context_tokens: int,
    max_question_tokens: int,
    sample_count: int,
) -> dict[str, Any]:
    prepared = []
    dropped = 0
    context_lengths = []
    question_lengths = []
    answer_lengths = []
    titles = Counter()

    for example in examples:
        row = _prepare_example_view(
            example,
            max_context_tokens=max_context_tokens,
            max_question_tokens=max_question_tokens,
        )
        if row is None:
            dropped += 1
            continue
        prepared.append(row)
        context_lengths.append(row["context_token_count"])
        question_lengths.append(row["question_token_count"])
        answer_lengths.append(row["answer_token_span"]["end"] - row["answer_token_span"]["start"] + 1)
        if row["title"]:
            titles[row["title"]] += 1

    return {
        "split": name,
        "raw_examples": len(examples),
        "usable_examples": len(prepared),
        "dropped_examples": dropped,
        "avg_context_tokens": round(sum(context_lengths) / len(context_lengths), 2) if context_lengths else 0.0,
        "avg_question_tokens": round(sum(question_lengths) / len(question_lengths), 2) if question_lengths else 0.0,
        "avg_answer_tokens": round(sum(answer_lengths) / len(answer_lengths), 2) if answer_lengths else 0.0,
        "top_titles": [{"title": title, "count": count} for title, count in titles.most_common(5)],
        "samples": prepared[:sample_count],
    }


def _normalize_token(token: str) -> str:
    return re.sub(r"[^\w]", "", token.lower())


def _extract_when_phrase(context_tokens: list[str], start_idx: int) -> tuple[int, int] | None:
    joined = " ".join(context_tokens)
    patterns = [
        r"\b(?:in|on|during|around|by|after|before)\s+(?:the\s+)?(?:late|early|mid)?\s*\d{4}s?\b",
        r"\b(?:in|on|during|around|by|after|before)\s+\d{4}\b",
        r"\b(?:the\s+)?(?:late|early|mid)\s+\d{4}s\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, joined, flags=re.IGNORECASE)
        if not match:
            continue
        phrase = match.group(0).split()
        for idx in range(len(context_tokens)):
            if context_tokens[idx:idx + len(phrase)] == phrase:
                return idx, idx + len(phrase) - 1
    for idx in range(max(0, start_idx - 3), min(len(context_tokens), start_idx + 6)):
        norm = _normalize_token(context_tokens[idx])
        if re.fullmatch(r"\d{4}s?", norm):
            left = idx
            while left > 0 and _normalize_token(context_tokens[left - 1]) in {"the", "late", "early", "mid", "in", "on", "during"}:
                left -= 1
            return left, idx
    return None


def _heuristic_extract_answer(question: str, context: str) -> dict[str, Any]:
    question_spans = tokenize_with_spans(question)
    context_spans = tokenize_with_spans(context)
    question_terms = [_normalize_token(tok) for tok, _, _ in question_spans]
    context_tokens = [tok for tok, _, _ in context_spans]
    normalized_context = [_normalize_token(tok) for tok in context_tokens]
    keywords = [tok for tok in question_terms if tok and tok not in STOPWORDS]

    best = None
    for idx in range(len(context_tokens)):
        score = 0
        for keyword in keywords:
            if keyword and normalized_context[idx] == keyword:
                score += 2
            for back in range(max(0, idx - 6), idx):
                if normalized_context[back] == keyword:
                    score += 1
        if best is None or score > best["score"]:
            best = {"score": score, "start": idx}

    if not context_spans:
        return {
            "answer": "",
            "start_char": -1,
            "end_char": -1,
            "start_token": -1,
            "end_token": -1,
            "score": 0.0,
            "method": "heuristic_keyword_overlap",
        }

    start_idx = 0 if best is None else best["start"]
    q_lower = question.lower()
    if "when" in q_lower:
        when_span = _extract_when_phrase(context_tokens, start_idx)
        if when_span is not None:
            start_idx, end_idx = when_span
        else:
            end_idx = min(len(context_tokens) - 1, start_idx + 3)
    elif "where" in q_lower:
        end_idx = min(len(context_tokens) - 1, start_idx + 4)
    elif "who" in q_lower:
        end_idx = min(len(context_tokens) - 1, start_idx + 3)
    else:
        end_idx = min(len(context_tokens) - 1, start_idx + 5)
    start_char = context_spans[start_idx][1]
    end_char = context_spans[end_idx][2]
    return {
        "answer": context[start_char:end_char].strip(),
        "start_char": start_char,
        "end_char": end_char,
        "start_token": start_idx,
        "end_token": end_idx,
        "score": float(0.0 if best is None else best["score"]),
        "method": "heuristic_keyword_overlap",
    }


def _try_runtime_qa(question: str, context: str, *, model_name: str, local_files_only: bool) -> dict[str, Any]:
    try:
        from transformers import pipeline
    except Exception as e:  # pragma: no cover
        return {
            "available": False,
            "model_loaded": False,
            "error": f"transformers import failed: {type(e).__name__}: {e}",
        }

    try:
        qa = pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            local_files_only=local_files_only,
        )
        out = qa(question=question, context=context)
    except Exception as e:  # pragma: no cover
        return {
            "available": True,
            "model_loaded": False,
            "error": f"qa model load/inference failed: {type(e).__name__}: {e}",
        }

    answer = str(out.get("answer", "")).strip()
    start_char = int(out.get("start", -1))
    end_char = int(out.get("end", -1))
    context_spans = tokenize_with_spans(context)
    span = None
    if start_char >= 0 and end_char >= 0:
        span = answer_span_from_chars(context_spans, start_char, end_char)
    return {
        "available": True,
        "model_loaded": True,
        "answer": answer,
        "start_char": start_char,
        "end_char": end_char,
        "start_token": None if span is None else span[0],
        "end_token": None if span is None else span[1],
        "score": float(out.get("score", 0.0)),
        "method": "transformers_question_answering_pipeline",
    }


def _truncate_answerable_window(
    context_tokens: Sequence[str],
    start_idx: int,
    end_idx: int,
    max_context_tokens: int,
) -> tuple[tuple[str, ...], int, int] | None:
    if len(context_tokens) <= max_context_tokens:
        return tuple(context_tokens), start_idx, end_idx
    answer_len = end_idx - start_idx + 1
    if answer_len > max_context_tokens:
        return None
    half = max(0, (max_context_tokens - answer_len) // 2)
    window_start = max(0, start_idx - half)
    window_end = min(len(context_tokens), window_start + max_context_tokens)
    window_start = max(0, window_end - max_context_tokens)
    return (
        tuple(context_tokens[window_start:window_end]),
        start_idx - window_start,
        end_idx - window_start,
    )


def _tokenize_example(
    example: QAExample,
    *,
    max_context_tokens: int,
    max_question_tokens: int,
) -> TokenizedQAExample | None:
    context_spans = tokenize_with_spans(example.context)
    question_spans = tokenize_with_spans(example.question)
    if not context_spans or not question_spans:
        return None
    span = answer_span_from_chars(context_spans, example.answer_start_char, example.answer_end_char)
    if span is None:
        return None
    context_tokens = [tok for tok, _, _ in context_spans]
    question_tokens = [tok for tok, _, _ in question_spans[:max_question_tokens]]
    truncated = _truncate_answerable_window(context_tokens, span[0], span[1], max_context_tokens)
    if truncated is None or not question_tokens:
        return None
    trimmed_context, start_idx, end_idx = truncated
    return TokenizedQAExample(
        example_id=example.example_id,
        title=example.title,
        context=example.context,
        question=example.question,
        context_tokens=trimmed_context,
        question_tokens=tuple(question_tokens),
        start_idx=int(start_idx),
        end_idx=int(end_idx),
        answer_texts=example.answer_texts,
    )


def _prepare_examples(
    examples: Sequence[QAExample],
    *,
    max_context_tokens: int,
    max_question_tokens: int,
    limit: int,
) -> list[TokenizedQAExample]:
    out: list[TokenizedQAExample] = []
    for ex in examples:
        tok = _tokenize_example(
            ex,
            max_context_tokens=max_context_tokens,
            max_question_tokens=max_question_tokens,
        )
        if tok is not None:
            out.append(tok)
        if limit and len(out) >= limit:
            break
    return out


def _build_vocab(examples: Sequence[TokenizedQAExample], *, min_word_count: int) -> dict[str, int]:
    counts = Counter()
    for ex in examples:
        counts.update(ex.context_tokens)
        counts.update(ex.question_tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counts.items():
        if count >= min_word_count:
            vocab[token] = len(vocab)
    return vocab


class QADataset:
    def __init__(self, examples: Sequence[TokenizedQAExample], vocab: dict[str, int]):
        self.examples = list(examples)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        return {
            "example_id": ex.example_id,
            "context_tokens": list(ex.context_tokens),
            "question_tokens": list(ex.question_tokens),
            "context_ids": [self.vocab.get(tok, 1) for tok in ex.context_tokens],
            "question_ids": [self.vocab.get(tok, 1) for tok in ex.question_tokens],
            "start_idx": ex.start_idx,
            "end_idx": ex.end_idx,
            "answer_texts": list(ex.answer_texts),
        }


def _collate_fn(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    torch, _, _, _ = _require_torch()
    pad_id = 0
    max_c = max(len(x["context_ids"]) for x in batch)
    max_q = max(len(x["question_ids"]) for x in batch)

    context_ids = []
    question_ids = []
    context_mask = []
    question_mask = []
    start_idx = []
    end_idx = []
    example_ids = []
    answer_texts = []
    context_tokens = []
    question_tokens = []

    for item in batch:
        c_ids = item["context_ids"]
        q_ids = item["question_ids"]
        context_ids.append(c_ids + [pad_id] * (max_c - len(c_ids)))
        question_ids.append(q_ids + [pad_id] * (max_q - len(q_ids)))
        context_mask.append([1] * len(c_ids) + [0] * (max_c - len(c_ids)))
        question_mask.append([1] * len(q_ids) + [0] * (max_q - len(q_ids)))
        start_idx.append(item["start_idx"])
        end_idx.append(item["end_idx"])
        example_ids.append(item["example_id"])
        answer_texts.append(item["answer_texts"])
        context_tokens.append(item["context_tokens"])
        question_tokens.append(item["question_tokens"])

    return {
        "context_ids": torch.tensor(context_ids, dtype=torch.long),
        "question_ids": torch.tensor(question_ids, dtype=torch.long),
        "context_mask": torch.tensor(context_mask, dtype=torch.bool),
        "question_mask": torch.tensor(question_mask, dtype=torch.bool),
        "start_idx": torch.tensor(start_idx, dtype=torch.long),
        "end_idx": torch.tensor(end_idx, dtype=torch.long),
        "example_ids": example_ids,
        "answer_texts": answer_texts,
        "context_tokens": context_tokens,
        "question_tokens": question_tokens,
    }


def _masked_softmax(logits, mask, dim: int):
    torch, _, F, _ = _require_torch()
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return F.softmax(masked, dim=dim)


def _replace_masked_positions(logits, mask):
    torch, _, _, _ = _require_torch()
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class BiDAFModel:
    def __init__(
        self,
        *,
        vocab_size: int,
        word_embedding_dim: int,
        hidden_size: int,
        dropout: float,
        use_bert_embeddings: bool,
        bert_hidden_size: int | None = None,
    ):
        _, nn, _, _ = _require_torch()
        emb_dim = bert_hidden_size if use_bert_embeddings else word_embedding_dim
        self.use_bert_embeddings = use_bert_embeddings
        self.embedding = None
        if not use_bert_embeddings:
            self.embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)

        self.context_encoder = nn.LSTM(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        self.question_encoder = nn.LSTM(emb_dim, hidden_size, batch_first=True, bidirectional=True)
        self.similarity = nn.Linear(hidden_size * 6, 1, bias=False)
        self.modeling = nn.LSTM(hidden_size * 8, hidden_size, batch_first=True, bidirectional=True)
        self.output_start = nn.Linear(hidden_size * 10, 1)
        self.output_end = nn.Linear(hidden_size * 10, 1)
        self.dropout = nn.Dropout(dropout)

    def parameters(self):
        modules = [
            self.context_encoder,
            self.question_encoder,
            self.similarity,
            self.modeling,
            self.output_start,
            self.output_end,
            self.dropout,
        ]
        if self.embedding is not None:
            modules.append(self.embedding)
        params = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    def to(self, device):
        modules = [
            self.context_encoder,
            self.question_encoder,
            self.similarity,
            self.modeling,
            self.output_start,
            self.output_end,
            self.dropout,
        ]
        if self.embedding is not None:
            modules.append(self.embedding)
        for module in modules:
            module.to(device)
        return self

    def train(self):
        for module in [
            self.context_encoder,
            self.question_encoder,
            self.similarity,
            self.modeling,
            self.output_start,
            self.output_end,
            self.dropout,
        ]:
            module.train()
        if self.embedding is not None:
            self.embedding.train()
        return self

    def eval(self):
        for module in [
            self.context_encoder,
            self.question_encoder,
            self.similarity,
            self.modeling,
            self.output_start,
            self.output_end,
            self.dropout,
        ]:
            module.eval()
        if self.embedding is not None:
            self.embedding.eval()
        return self

    def state_dict(self):
        state = {}
        named = {
            "context_encoder": self.context_encoder,
            "question_encoder": self.question_encoder,
            "similarity": self.similarity,
            "modeling": self.modeling,
            "output_start": self.output_start,
            "output_end": self.output_end,
        }
        if self.embedding is not None:
            named["embedding"] = self.embedding
        for prefix, module in named.items():
            for key, value in module.state_dict().items():
                state[f"{prefix}.{key}"] = value
        return state

    def load_state_dict(self, state):
        groups = {
            "context_encoder": self.context_encoder,
            "question_encoder": self.question_encoder,
            "similarity": self.similarity,
            "modeling": self.modeling,
            "output_start": self.output_start,
            "output_end": self.output_end,
        }
        if self.embedding is not None:
            groups["embedding"] = self.embedding
        for prefix, module in groups.items():
            sub = {k[len(prefix) + 1:]: v for k, v in state.items() if k.startswith(prefix + ".")}
            module.load_state_dict(sub)

    def __call__(self, *, context_ids=None, question_ids=None, context_emb=None, question_emb=None, context_mask=None, question_mask=None):
        import torch

        if self.use_bert_embeddings:
            if context_emb is None or question_emb is None:
                raise ValueError("BERT mode requires context_emb and question_emb.")
            c_in = self.dropout(context_emb)
            q_in = self.dropout(question_emb)
        else:
            if self.embedding is None or context_ids is None or question_ids is None:
                raise ValueError("Word mode requires context_ids and question_ids.")
            c_in = self.dropout(self.embedding(context_ids))
            q_in = self.dropout(self.embedding(question_ids))

        c_enc, _ = self.context_encoder(c_in)
        q_enc, _ = self.question_encoder(q_in)

        c_len = c_enc.size(1)
        q_len = q_enc.size(1)
        c_exp = c_enc.unsqueeze(2).expand(-1, c_len, q_len, -1)
        q_exp = q_enc.unsqueeze(1).expand(-1, c_len, q_len, -1)
        sim = self.similarity(torch.cat([c_exp, q_exp, c_exp * q_exp], dim=-1)).squeeze(-1)

        if question_mask is None or context_mask is None:
            raise ValueError("Masks are required for BiDAF forward.")

        q_mask_exp = question_mask.unsqueeze(1).expand(-1, c_len, -1)
        attn_c2q = _masked_softmax(sim, q_mask_exp, dim=2)
        attended_q = torch.bmm(attn_c2q, q_enc)

        max_sim, _ = sim.max(dim=2)
        attn_q2c = _masked_softmax(max_sim, context_mask, dim=1).unsqueeze(1)
        attended_c = torch.bmm(attn_q2c, c_enc).expand(-1, c_len, -1)

        fused_input = torch.cat([c_enc, attended_q, c_enc * attended_q, c_enc * attended_c], dim=-1)
        modeled, _ = self.modeling(self.dropout(fused_input))
        fused = torch.cat([fused_input, modeled], dim=-1)
        start_logits = self.output_start(self.dropout(fused)).squeeze(-1)
        end_logits = self.output_end(self.dropout(fused)).squeeze(-1)
        start_logits = _replace_masked_positions(start_logits, context_mask)
        end_logits = _replace_masked_positions(end_logits, context_mask)
        return start_logits, end_logits


class BertWordEmbedder:
    def __init__(self, model_name: str, *, freeze: bool, local_files_only: bool):
        import torch

        AutoTokenizer, AutoModel = _require_transformers_auto()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 768))
        self.freeze = freeze
        self.torch = torch
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def to(self, device):
        self.model.to(device)
        return self

    def train(self):
        self.model.train(not self.freeze)
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return [] if self.freeze else list(self.model.parameters())

    def encode(self, batch_tokens: Sequence[Sequence[str]], *, max_words: int, device):
        torch = self.torch
        batch_enc = self.tokenizer(
            [list(tokens) for tokens in batch_tokens],
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=max_words + 2,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(device) for k, v in batch_enc.items()}
        outputs = self.model(**model_inputs)
        hidden = outputs.last_hidden_state
        batch_size = hidden.size(0)
        pooled = torch.zeros((batch_size, max_words, self.hidden_size), dtype=hidden.dtype, device=device)
        mask = torch.zeros((batch_size, max_words), dtype=torch.bool, device=device)

        for i in range(batch_size):
            word_ids = batch_enc.word_ids(batch_index=i)
            bucket: dict[int, list[Any]] = {}
            for j, wid in enumerate(word_ids):
                if wid is None or wid >= max_words:
                    continue
                bucket.setdefault(wid, []).append(hidden[i, j])
            for wid, vectors in bucket.items():
                pooled[i, wid] = torch.stack(vectors, dim=0).mean(dim=0)
                mask[i, wid] = True
        return pooled, mask


def _select_best_span(start_logits, end_logits, max_answer_len: int) -> tuple[int, int]:
    best_score = None
    best_pair = (0, 0)
    start_vals = start_logits.tolist()
    end_vals = end_logits.tolist()
    for i, sv in enumerate(start_vals):
        max_j = min(len(end_vals), i + max_answer_len)
        for j in range(i, max_j):
            score = sv + end_vals[j]
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (i, j)
    return best_pair


def _evaluate_predictions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"exact_match": 0.0, "f1": 0.0, "count": 0}
    em = sum(exact_match_score(r["prediction"], r["gold_answers"]) for r in rows) / len(rows)
    f1 = sum(f1_score_span(r["prediction"], r["gold_answers"]) for r in rows) / len(rows)
    return {"exact_match": float(em), "f1": float(f1), "count": len(rows)}


def _predict_batches(model, bert_embedder, loader, device, *, max_answer_len: int) -> list[dict[str, Any]]:
    import torch

    model.eval()
    if bert_embedder is not None:
        bert_embedder.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            context_mask = batch["context_mask"].to(device)
            question_mask = batch["question_mask"].to(device)
            if bert_embedder is None:
                start_logits, end_logits = model(
                    context_ids=batch["context_ids"].to(device),
                    question_ids=batch["question_ids"].to(device),
                    context_mask=context_mask,
                    question_mask=question_mask,
                )
            else:
                c_emb, c_mask_bert = bert_embedder.encode(
                    batch["context_tokens"],
                    max_words=batch["context_ids"].shape[1],
                    device=device,
                )
                q_emb, q_mask_bert = bert_embedder.encode(
                    batch["question_tokens"],
                    max_words=batch["question_ids"].shape[1],
                    device=device,
                )
                start_logits, end_logits = model(
                    context_emb=c_emb,
                    question_emb=q_emb,
                    context_mask=context_mask & c_mask_bert,
                    question_mask=question_mask & q_mask_bert,
                )

            for i in range(start_logits.shape[0]):
                s_idx, e_idx = _select_best_span(
                    start_logits[i].cpu(),
                    end_logits[i].cpu(),
                    max_answer_len=max_answer_len,
                )
                tokens = batch["context_tokens"][i]
                pred = " ".join(tokens[s_idx:e_idx + 1]).strip()
                rows.append(
                    {
                        "example_id": batch["example_ids"][i],
                        "prediction": pred,
                        "gold_answers": batch["answer_texts"][i],
                        "predicted_start": int(s_idx),
                        "predicted_end": int(e_idx),
                    }
                )
    return rows


def _train_model(
    *,
    train_data: QADataset,
    dev_data: QADataset,
    cfg: P4Task2QAConfig,
    use_bert_embeddings: bool,
) -> dict[str, Any]:
    torch, _, F, DataLoader = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate_fn)

    bert_embedder = None
    bert_hidden = None
    if use_bert_embeddings:
        bert_embedder = BertWordEmbedder(
            cfg.bert_embedding_model_name,
            freeze=cfg.freeze_bert,
            local_files_only=cfg.local_files_only,
        ).to(device)
        bert_hidden = bert_embedder.hidden_size

    model = BiDAFModel(
        vocab_size=len(train_data.vocab),
        word_embedding_dim=cfg.word_embedding_dim,
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
        use_bert_embeddings=use_bert_embeddings,
        bert_hidden_size=bert_hidden,
    ).to(device)

    params = list(model.parameters())
    if bert_embedder is not None:
        params.extend(bert_embedder.parameters())
    lr = cfg.bert_learning_rate if use_bert_embeddings and not cfg.freeze_bert else cfg.learning_rate
    optimizer = torch.optim.Adam(params, lr=lr)

    best = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if bert_embedder is not None:
            bert_embedder.train()
        total_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            context_mask = batch["context_mask"].to(device)
            question_mask = batch["question_mask"].to(device)
            start_gold = batch["start_idx"].to(device)
            end_gold = batch["end_idx"].to(device)

            if bert_embedder is None:
                start_logits, end_logits = model(
                    context_ids=batch["context_ids"].to(device),
                    question_ids=batch["question_ids"].to(device),
                    context_mask=context_mask,
                    question_mask=question_mask,
                )
            else:
                c_emb, c_mask_bert = bert_embedder.encode(
                    batch["context_tokens"],
                    max_words=batch["context_ids"].shape[1],
                    device=device,
                )
                q_emb, q_mask_bert = bert_embedder.encode(
                    batch["question_tokens"],
                    max_words=batch["question_ids"].shape[1],
                    device=device,
                )
                start_logits, end_logits = model(
                    context_emb=c_emb,
                    question_emb=q_emb,
                    context_mask=context_mask & c_mask_bert,
                    question_mask=question_mask & q_mask_bert,
                )

            loss = F.cross_entropy(start_logits, start_gold) + F.cross_entropy(end_logits, end_gold)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        dev_rows = _predict_batches(
            model,
            bert_embedder,
            dev_loader,
            device,
            max_answer_len=cfg.max_answer_len,
        )
        dev_metrics = _evaluate_predictions(dev_rows)
        epoch_info = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_batches, 1),
            "dev_exact_match": dev_metrics["exact_match"],
            "dev_f1": dev_metrics["f1"],
        }
        history.append(epoch_info)
        if best is None or dev_metrics["f1"] > best["dev_metrics"]["f1"]:
            best = {
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "dev_metrics": dev_metrics,
            }

    if best is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best["model_state"])
    return {
        "model": model,
        "bert_embedder": bert_embedder,
        "device": device,
        "history": history,
        "best_dev_metrics": best["dev_metrics"],
    }


def _artifact_path(cfg: P4Task2QAConfig, name: str) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    return cfg.out_dir / name


def _summary_markdown_path(out_json: Path) -> Path:
    return out_json.with_suffix(".md")


def _build_task2_ui_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "headline_metrics": {
            "train_usable": int(report["splits"]["train"]["usable_examples"]),
            "dev_usable": int(report["splits"]["dev"]["usable_examples"]),
            "prepared_train": int(report["prepared_training_data"]["train_examples"]),
            "prepared_eval": int(report["prepared_training_data"]["eval_examples"]),
            "runtime_model_loaded": bool(report["runtime"]["model_loaded"]),
        },
        "model_rows": [],
        "comparison": report.get("comparison"),
        "sample_predictions": {},
    }
    for model_name, model_info in (report.get("models") or {}).items():
        row = {
            "model": model_name,
            "status": model_info.get("status", ""),
            "use_bert_embeddings": bool(model_info.get("use_bert_embeddings", False)),
            "eval_exact_match": None,
            "eval_f1": None,
        }
        if model_info.get("status") == "trained":
            row["eval_exact_match"] = float(model_info["eval_metrics"]["exact_match"])
            row["eval_f1"] = float(model_info["eval_metrics"]["f1"])
            summary["sample_predictions"][model_name] = model_info.get("sample_predictions", [])[:3]
        summary["model_rows"].append(row)
    return summary


def _build_task2_markdown(report: dict[str, Any]) -> str:
    summary = _build_task2_ui_summary(report)
    lines = [
        "# Project 4 Task 2 Summary",
        "",
        f"- Dataset: {report['dataset']['name']}",
        f"- Train usable examples: {summary['headline_metrics']['train_usable']}",
        f"- Dev usable examples: {summary['headline_metrics']['dev_usable']}",
        f"- Prepared train examples: {summary['headline_metrics']['prepared_train']}",
        f"- Prepared eval examples: {summary['headline_metrics']['prepared_eval']}",
        f"- Runtime model loaded: {summary['headline_metrics']['runtime_model_loaded']}",
        "",
        "## Model Results",
    ]
    for row in summary["model_rows"]:
        line = f"- {row['model']}: {row['status']}"
        if row["eval_exact_match"] is not None and row["eval_f1"] is not None:
            line += f" | EM={row['eval_exact_match']:.4f} | F1={row['eval_f1']:.4f}"
        lines.append(line)
    if summary.get("comparison"):
        comp = summary["comparison"]
        lines.extend(
            [
                "",
                "## Comparison",
                f"- EM delta (BERT - Word): {float(comp['exact_match_delta_bert_minus_word']):+.4f}",
                f"- F1 delta (BERT - Word): {float(comp['f1_delta_bert_minus_word']):+.4f}",
                f"- Interpretation: {comp['interpretation']}",
            ]
        )
    return "\n".join(lines) + "\n"


def _run_single_setting(
    *,
    cfg: P4Task2QAConfig,
    train_examples: Sequence[TokenizedQAExample],
    dev_examples: Sequence[TokenizedQAExample],
    eval_examples: Sequence[TokenizedQAExample],
    use_bert_embeddings: bool,
) -> dict[str, Any]:
    vocab = _build_vocab(train_examples, min_word_count=cfg.min_word_count)
    train_data = QADataset(train_examples, vocab)
    dev_data = QADataset(dev_examples, vocab)
    eval_data = QADataset(eval_examples, vocab)

    if cfg.prepare_only:
        return {
            "status": "prepared_only",
            "use_bert_embeddings": use_bert_embeddings,
            "vocab_size": len(vocab),
            "train_examples": len(train_data),
            "dev_examples": len(dev_data),
            "eval_examples": len(eval_data),
        }

    _, _, _, DataLoader = _require_torch()
    trained = _train_model(
        train_data=train_data,
        dev_data=dev_data,
        cfg=cfg,
        use_bert_embeddings=use_bert_embeddings,
    )
    eval_loader = DataLoader(eval_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate_fn)
    rows = _predict_batches(
        trained["model"],
        trained["bert_embedder"],
        eval_loader,
        trained["device"],
        max_answer_len=cfg.max_answer_len,
    )
    metrics = _evaluate_predictions(rows)

    artifact_name = "bert_bidaf_predictions.jsonl" if use_bert_embeddings else "word_bidaf_predictions.jsonl"
    sample_path = _artifact_path(cfg, artifact_name)
    with sample_path.open("w", encoding="utf-8") as f:
        for row in rows[:100]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "status": "trained",
        "use_bert_embeddings": use_bert_embeddings,
        "vocab_size": len(vocab),
        "train_examples": len(train_data),
        "dev_examples": len(dev_data),
        "eval_examples": len(eval_data),
        "best_dev_metrics": trained["best_dev_metrics"],
        "eval_metrics": metrics,
        "training_history": trained["history"],
        "artifacts": {"predictions_jsonl": sample_path.as_posix()},
        "sample_predictions": rows[:5],
    }


def _prepare_training_splits(cfg: P4Task2QAConfig) -> tuple[list[TokenizedQAExample], list[TokenizedQAExample], list[TokenizedQAExample]]:
    train_raw = load_squad_examples(cfg.train_path)
    dev_raw = load_squad_examples(cfg.dev_path)
    train_examples = _prepare_examples(
        train_raw,
        max_context_tokens=cfg.max_context_tokens,
        max_question_tokens=cfg.max_question_tokens,
        limit=cfg.max_train_examples,
    )
    dev_examples = _prepare_examples(
        dev_raw,
        max_context_tokens=cfg.max_context_tokens,
        max_question_tokens=cfg.max_question_tokens,
        limit=cfg.max_dev_examples,
    )
    eval_examples = _prepare_examples(
        dev_raw,
        max_context_tokens=cfg.max_context_tokens,
        max_question_tokens=cfg.max_question_tokens,
        limit=cfg.max_eval_examples,
    )
    if not train_examples:
        raise ValueError("No usable SQuAD train examples remained after tokenization/truncation.")
    if not dev_examples:
        raise ValueError("No usable SQuAD dev examples remained after tokenization/truncation.")
    if not eval_examples:
        eval_examples = list(dev_examples)
    return train_examples, dev_examples, eval_examples


def run_p4_task2_qa(cfg: P4Task2QAConfig) -> dict[str, Any]:
    train_raw = load_squad_examples(cfg.train_path)
    dev_raw = load_squad_examples(cfg.dev_path)

    runtime = {
        "attempted": cfg.try_runtime_inference,
        "available": False,
        "model_loaded": False,
        "notes": [],
    }
    demo_prediction = None

    if cfg.try_runtime_inference:
        runtime_res = _try_runtime_qa(
            cfg.demo_question,
            cfg.demo_context,
            model_name=cfg.runtime_model_name,
            local_files_only=cfg.local_files_only,
        )
        runtime["available"] = bool(runtime_res.get("available"))
        runtime["model_loaded"] = bool(runtime_res.get("model_loaded"))
        if runtime_res.get("model_loaded"):
            demo_prediction = {
                "question": cfg.demo_question,
                "context": cfg.demo_context,
                "prediction": runtime_res,
            }
            runtime["notes"].append("Runtime QA inference succeeded in the current Python environment.")
        else:
            runtime["notes"].append(runtime_res.get("error", "Runtime QA inference failed."))
    else:
        runtime["notes"].append(
            "Runtime QA inference is disabled by config. Enable it when running inside your conda environment."
        )

    train_examples, dev_examples, eval_examples = _prepare_training_splits(cfg)

    result: dict[str, Any] = {
        "task": "project4_task2_qa",
        "dataset": {
            "name": "SQuAD 1.1",
            "train_path": cfg.train_path.as_posix(),
            "dev_path": cfg.dev_path.as_posix(),
            "active_project_dataset": True,
        },
        "settings": {
            "runtime_model_name": cfg.runtime_model_name,
            "bert_embedding_model_name": cfg.bert_embedding_model_name,
            "max_context_tokens": cfg.max_context_tokens,
            "max_question_tokens": cfg.max_question_tokens,
            "sample_count": cfg.sample_count,
            "min_word_count": cfg.min_word_count,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "bert_learning_rate": cfg.bert_learning_rate,
            "hidden_size": cfg.hidden_size,
            "word_embedding_dim": cfg.word_embedding_dim,
            "dropout": cfg.dropout,
            "freeze_bert": cfg.freeze_bert,
            "prepare_only": cfg.prepare_only,
            "train_word_bidaf": cfg.train_word_bidaf,
            "train_bert_bidaf": cfg.train_bert_bidaf,
            "max_train_examples": cfg.max_train_examples,
            "max_dev_examples": cfg.max_dev_examples,
            "max_eval_examples": cfg.max_eval_examples,
        },
        "splits": {
            "train": _summarize_split(
                "train",
                train_raw,
                max_context_tokens=cfg.max_context_tokens,
                max_question_tokens=cfg.max_question_tokens,
                sample_count=cfg.sample_count,
            ),
            "dev": _summarize_split(
                "dev",
                dev_raw,
                max_context_tokens=cfg.max_context_tokens,
                max_question_tokens=cfg.max_question_tokens,
                sample_count=cfg.sample_count,
            ),
        },
        "prepared_training_data": {
            "train_examples": len(train_examples),
            "dev_examples": len(dev_examples),
            "eval_examples": len(eval_examples),
        },
        "model_plan": {
            "baseline": "BiDAF with trainable word embeddings",
            "comparison_model": "BiDAF with BERT-base contextual embeddings",
            "demo_runtime_model": cfg.runtime_model_name,
            "expected_output": "answer start and end token positions within context",
            "evaluation_metrics": ["exact_match", "f1"],
        },
        "runtime": runtime,
        "demo_prediction": demo_prediction,
        "models": {},
        "ui_summary": {},
        "status": {
            "stage": "dataset_ready_and_training_optional",
            "ready_for_model_work": True,
            "notes": [
                "SQuAD 1.1 paths are validated from the current project root.",
                "This report now covers data preparation, QA demo behavior, and optional BiDAF training runs.",
            ],
        },
    }

    if cfg.train_word_bidaf:
        result["models"]["bidaf_word_embeddings"] = _run_single_setting(
            cfg=cfg,
            train_examples=train_examples,
            dev_examples=dev_examples,
            eval_examples=eval_examples,
            use_bert_embeddings=False,
        )
    if cfg.train_bert_bidaf:
        result["models"]["bidaf_bert_embeddings"] = _run_single_setting(
            cfg=cfg,
            train_examples=train_examples,
            dev_examples=dev_examples,
            eval_examples=eval_examples,
            use_bert_embeddings=True,
        )

    if (
        "bidaf_word_embeddings" in result["models"]
        and "bidaf_bert_embeddings" in result["models"]
        and result["models"]["bidaf_word_embeddings"]["status"] == "trained"
        and result["models"]["bidaf_bert_embeddings"]["status"] == "trained"
    ):
        base = result["models"]["bidaf_word_embeddings"]["eval_metrics"]
        bert = result["models"]["bidaf_bert_embeddings"]["eval_metrics"]
        result["comparison"] = {
            "exact_match_delta_bert_minus_word": float(bert["exact_match"] - base["exact_match"]),
            "f1_delta_bert_minus_word": float(bert["f1"] - base["f1"]),
            "interpretation": (
                "Positive deltas indicate that contextual BERT embeddings improved BiDAF over the word-embedding baseline."
            ),
        }

    result["ui_summary"] = _build_task2_ui_summary(result)

    ensure_parent(cfg.out_json)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _summary_markdown_path(cfg.out_json).write_text(_build_task2_markdown(result), encoding="utf-8")
    return result


def run_p4_task2_qa_inference(
    cfg: P4Task2QAConfig,
    *,
    question: str,
    context: str,
    gold_answers: list[str] | None = None,
) -> dict[str, Any]:
    heuristic = _heuristic_extract_answer(question, context)
    result: dict[str, Any] = {
        "question": question,
        "context": context,
        "runtime": {
            "attempted": cfg.try_runtime_inference,
            "available": False,
            "model_loaded": False,
            "notes": [],
        },
        "heuristic_prediction": heuristic,
        "runtime_prediction": None,
        "selected_prediction": heuristic,
    }

    if cfg.try_runtime_inference:
        runtime_res = _try_runtime_qa(
            question,
            context,
            model_name=cfg.runtime_model_name,
            local_files_only=cfg.local_files_only,
        )
        result["runtime"]["available"] = bool(runtime_res.get("available"))
        result["runtime"]["model_loaded"] = bool(runtime_res.get("model_loaded"))
        if runtime_res.get("model_loaded"):
            result["runtime_prediction"] = runtime_res
            result["selected_prediction"] = runtime_res
            result["runtime"]["notes"].append("Runtime QA inference succeeded.")
        else:
            result["runtime"]["notes"].append(runtime_res.get("error", "Runtime QA inference failed."))
    else:
        result["runtime"]["notes"].append(
            "Runtime QA inference is disabled, so the heuristic fallback was used."
        )

    if gold_answers:
        selected_answer = str(result["selected_prediction"].get("answer", ""))
        result["evaluation"] = {
            "gold_answers": gold_answers,
            "exact_match": exact_match_score(selected_answer, gold_answers),
            "f1": f1_score_span(selected_answer, gold_answers),
        }
    return result


def format_p4_task2_qa_report(report: dict[str, Any], out_json: Path) -> str:
    lines = []
    lines.append("=== PROJECT 4 - TASK 2 REPORT ===\n")
    lines.append(f"Dataset: {report['dataset']['name']}\n")
    lines.append(
        f"Prepared examples: train={report['prepared_training_data']['train_examples']} "
        f"dev={report['prepared_training_data']['dev_examples']} "
        f"eval={report['prepared_training_data']['eval_examples']}\n"
    )
    lines.append(f"Runtime model loaded: {report['runtime']['model_loaded']}\n")
    for name, info in report["models"].items():
        lines.append(f"{name}: {info['status']}\n")
        if info["status"] == "trained":
            lines.append(
                f"  eval EM={info['eval_metrics']['exact_match']:.4f} "
                f"eval F1={info['eval_metrics']['f1']:.4f}\n"
            )
    lines.append(f"Saved JSON report: {out_json.resolve()}\n")
    return "".join(lines)
