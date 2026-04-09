#!/usr/bin/env python3
"""
Project 4 Task 1 using IMDb as the active dataset.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nlp_project.p4.common import SentimentExample, ensure_parent, load_imdb_split


DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


@dataclass(frozen=True)
class P4Task1SentimentConfig:
    dataset_dir: Path = Path("dataset/aclImdb")
    active_model_name: str = DEFAULT_MODEL_NAME
    expected_num_classes: int = 2
    sample_per_split: int = 3
    try_runtime_inference: bool = False
    local_files_only: bool = False
    demo_texts: tuple[str, ...] = field(
        default_factory=lambda: (
            "This movie was surprisingly good and emotionally rich.",
            "The plot was dull and the acting felt weak.",
            "It starts slowly, but the ending is strong.",
        )
    )
    out_json: Path = Path("data/reports/p4_task1_sentiment_prepare_report.json")


def _split_stats(rows: list[SentimentExample]) -> dict[str, Any]:
    label_counts = Counter(x.label for x in rows)
    lengths = [len(x.text.split()) for x in rows if x.text]
    return {
        "count": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "avg_word_count": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
    }


def _sample_rows(rows: list[SentimentExample], limit: int) -> list[dict[str, Any]]:
    return [
        {
            "example_id": x.example_id,
            "label": x.label,
            "rating": x.rating,
            "text_preview": x.text[:240],
        }
        for x in rows[:limit]
    ]


def _default_analysis(cfg: P4Task1SentimentConfig) -> dict[str, Any]:
    return {
        "model_name": cfg.active_model_name,
        "task_type": "sequence_classification",
        "inputs": [
            "Single review text string",
            "Tokenizer output such as input_ids and attention_mask",
        ],
        "outputs": [
            "Class logits",
            "Predicted sentiment label",
            "Confidence scores after softmax",
        ],
        "num_classes": cfg.expected_num_classes,
        "class_labels": ["NEGATIVE", "POSITIVE"],
        "max_input_tokens": 512,
        "case_sensitive": False,
        "case_sensitivity_effect": (
            "The selected model is uncased, so capitalization is normalized during tokenization. "
            "That usually improves robustness on noisy text, but removes some case-based signals."
        ),
        "imdb_alignment_note": (
            "IMDb is a binary sentiment dataset, so a binary sequence-classification BERT model is the cleanest fit "
            "for demos, evaluation, and report discussion."
        ),
        "agglutinative_language_note": (
            "The English IMDb setup is not sufficient evidence for Azerbaijani performance. "
            "Azerbaijani is agglutinative, so subword fragmentation and domain mismatch can hurt results unless "
            "the model is multilingual and additionally evaluated or fine-tuned on relevant data."
        ),
    }


def _summary_markdown_path(out_json: Path) -> Path:
    return out_json.with_suffix(".md")


def _build_task1_ui_summary(report: dict[str, Any]) -> dict[str, Any]:
    dataset = report["dataset"]
    analysis = report["model_analysis"]
    runtime = report["runtime"]
    demo_items = []
    for row in report.get("demo_predictions", []):
        pred = row["prediction"][0]
        demo_items.append(
            {
                "text": row["text"],
                "label": pred.get("label", ""),
                "score": float(pred.get("score", 0.0)),
            }
        )
    return {
        "headline_metrics": {
            "train_reviews": int(dataset["full_counts"]["train"]),
            "test_reviews": int(dataset["full_counts"]["test"]),
            "num_classes": int(analysis["num_classes"]),
            "max_input_tokens": int(analysis["max_input_tokens"]),
            "case_sensitive": bool(analysis["case_sensitive"]),
            "runtime_model_loaded": bool(runtime["model_loaded"]),
        },
        "key_points": [
            f"Input: {analysis['inputs'][0]}",
            f"Output: {analysis['outputs'][1]}",
            f"IMDb labels: {', '.join(dataset['class_labels'])}",
            analysis["imdb_alignment_note"],
            analysis["agglutinative_language_note"],
        ],
        "demo_predictions": demo_items,
        "sample_reviews": report["samples"],
    }


def _build_task1_markdown(report: dict[str, Any]) -> str:
    summary = _build_task1_ui_summary(report)
    lines = [
        "# Project 4 Task 1 Summary",
        "",
        f"- Dataset: {report['dataset']['name']}",
        f"- Model: {report['model_analysis']['model_name']}",
        f"- Train reviews: {summary['headline_metrics']['train_reviews']}",
        f"- Test reviews: {summary['headline_metrics']['test_reviews']}",
        f"- Classes: {summary['headline_metrics']['num_classes']}",
        f"- Max input tokens: {summary['headline_metrics']['max_input_tokens']}",
        f"- Case sensitive: {summary['headline_metrics']['case_sensitive']}",
        f"- Runtime model loaded: {summary['headline_metrics']['runtime_model_loaded']}",
        "",
        "## Key Points",
    ]
    lines.extend(f"- {item}" for item in summary["key_points"])
    if summary["demo_predictions"]:
        lines.extend(["", "## Demo Predictions"])
        for item in summary["demo_predictions"]:
            lines.append(f"- `{item['label']}` ({item['score']:.4f}): {item['text']}")
    return "\n".join(lines) + "\n"


def _try_runtime_inference(model_name: str, texts: list[str], *, local_files_only: bool) -> dict[str, Any]:
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except Exception as e:  # pragma: no cover
        return {
            "available": False,
            "model_loaded": False,
            "error": f"transformers import failed: {type(e).__name__}: {e}",
            "predictions": [],
        }

    try:
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        clf = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:  # pragma: no cover
        return {
            "available": True,
            "model_loaded": False,
            "error": f"model load failed: {type(e).__name__}: {e}",
            "predictions": [],
        }

    id2label = getattr(cfg, "id2label", {}) or {}
    label_order = [
        str(label)
        for _, label in sorted(
            id2label.items(),
            key=lambda item: int(item[0]) if str(item[0]).isdigit() else str(item[0]),
        )
    ]
    do_lower_case = getattr(tokenizer, "do_lower_case", None)
    case_sensitive = None if do_lower_case is None else not bool(do_lower_case)
    preds = []
    for text in texts:
        try:
            out = clf(text)
        except Exception as e:  # pragma: no cover
            out = [{"label": f"ERROR: {type(e).__name__}", "score": 0.0}]
        preds.append({"text": text, "prediction": out})

    return {
        "available": True,
        "model_loaded": True,
        "num_labels": int(getattr(cfg, "num_labels", 0)),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "case_sensitive": case_sensitive,
        "class_labels": label_order,
        "predictions": preds,
    }


def run_p4_task1_sentiment(cfg: P4Task1SentimentConfig) -> dict:
    train_rows = load_imdb_split(cfg.dataset_dir / "train", "train")
    test_rows = load_imdb_split(cfg.dataset_dir / "test", "test")
    if not train_rows or not test_rows:
        raise FileNotFoundError(
            f"IMDb dataset not found or incomplete under {cfg.dataset_dir.resolve()}"
        )

    analysis = _default_analysis(cfg)
    runtime = {
        "attempted": cfg.try_runtime_inference,
        "available": False,
        "model_loaded": False,
        "notes": [],
    }
    demo_predictions: list[dict[str, Any]] = []

    if cfg.try_runtime_inference:
        runtime_res = _try_runtime_inference(
            cfg.active_model_name,
            list(cfg.demo_texts),
            local_files_only=cfg.local_files_only,
        )
        runtime["available"] = bool(runtime_res.get("available"))
        runtime["model_loaded"] = bool(runtime_res.get("model_loaded"))
        if runtime_res.get("model_loaded"):
            if runtime_res.get("num_labels"):
                analysis["num_classes"] = runtime_res["num_labels"]
            if runtime_res.get("class_labels"):
                analysis["class_labels"] = runtime_res["class_labels"]
            if runtime_res.get("max_position_embeddings"):
                analysis["max_input_tokens"] = int(runtime_res["max_position_embeddings"])
            if runtime_res.get("case_sensitive") is not None:
                analysis["case_sensitive"] = bool(runtime_res["case_sensitive"])
            demo_predictions = runtime_res.get("predictions", [])
            runtime["notes"].append("Runtime inference succeeded in the current Python environment.")
        else:
            runtime["notes"].append(runtime_res.get("error", "Runtime inference failed."))
    else:
        runtime["notes"].append(
            "Runtime inference is disabled by config. Enable it when running inside your Conda environment."
        )

    report = {
        "task": "project4_task1_sentiment",
        "dataset": {
            "name": "IMDb Large Movie Review Dataset",
            "dataset_dir": cfg.dataset_dir.as_posix(),
            "active_project_dataset": True,
            "splits": {
                "train": _split_stats(train_rows),
                "test": _split_stats(test_rows),
            },
            "full_counts": {
                "train": len(train_rows),
                "test": len(test_rows),
                "total_labeled": len(train_rows) + len(test_rows),
            },
            "class_labels": ["neg", "pos"],
        },
        "model_analysis": analysis,
        "samples": {
            "train": _sample_rows(train_rows, cfg.sample_per_split),
            "test": _sample_rows(test_rows, cfg.sample_per_split),
        },
        "runtime": runtime,
        "demo_predictions": demo_predictions,
        "ui_summary": {},
        "status": {
            "stage": "dataset_ready_and_runtime_optional",
            "ready_for_model_work": True,
            "notes": [
                "IMDb is the active Task 1 dataset in this project root.",
                "This report supports both static instructor-facing analysis and optional runtime demos.",
            ],
        },
    }
    report["ui_summary"] = _build_task1_ui_summary(report)

    ensure_parent(cfg.out_json)
    cfg.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _summary_markdown_path(cfg.out_json).write_text(_build_task1_markdown(report), encoding="utf-8")
    return report


def run_p4_task1_sentiment_inference(
    cfg: P4Task1SentimentConfig,
    texts: list[str],
) -> dict[str, Any]:
    payload = {
        "model_name": cfg.active_model_name,
        "texts": texts,
        "predictions": [],
        "runtime": {
            "attempted": True,
            "available": False,
            "model_loaded": False,
            "notes": [],
        },
    }
    runtime_res = _try_runtime_inference(
        cfg.active_model_name,
        texts,
        local_files_only=cfg.local_files_only,
    )
    payload["runtime"]["available"] = bool(runtime_res.get("available"))
    payload["runtime"]["model_loaded"] = bool(runtime_res.get("model_loaded"))
    if runtime_res.get("model_loaded"):
        payload["predictions"] = runtime_res.get("predictions", [])
        payload["runtime"]["notes"].append("Custom-text inference succeeded.")
    else:
        payload["runtime"]["notes"].append(
            runtime_res.get(
                "error",
                "Custom-text inference failed in the current Python environment.",
            )
        )
    return payload


def format_p4_task1_sentiment_report(report: dict, out_json: Path) -> str:
    ds = report["dataset"]
    analysis = report["model_analysis"]
    runtime = report["runtime"]
    return (
        "=== PROJECT 4 - TASK 1 REPORT ===\n"
        f"Dataset: {ds['name']}\n"
        f"Train examples: {ds['full_counts']['train']}\n"
        f"Test examples: {ds['full_counts']['test']}\n"
        f"Model: {analysis['model_name']}\n"
        f"Classes: {analysis['num_classes']}\n"
        f"Case sensitive: {analysis['case_sensitive']}\n"
        f"Runtime model loaded: {runtime['model_loaded']}\n"
        f"Saved JSON report: {out_json.resolve()}\n"
    )
