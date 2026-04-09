#!/usr/bin/env python3
"""
Project 4 CLI entrypoint.
"""

from __future__ import annotations

import argparse

from nlp_project.common.config import load_config_as


def cmd_task_p4_sentiment(args: argparse.Namespace) -> int:
    from nlp_project.p4.task1_sentiment import (
        P4Task1SentimentConfig,
        format_p4_task1_sentiment_report,
        run_p4_task1_sentiment,
    )

    cfg = load_config_as(args.config, P4Task1SentimentConfig)
    report = run_p4_task1_sentiment(cfg)
    print(format_p4_task1_sentiment_report(report, cfg.out_json))
    return 0


def cmd_task_p4_qa(args: argparse.Namespace) -> int:
    from nlp_project.p4.task2_qa import (
        P4Task2QAConfig,
        format_p4_task2_qa_report,
        run_p4_task2_qa,
    )

    cfg = load_config_as(args.config, P4Task2QAConfig)
    report = run_p4_task2_qa(cfg)
    print(format_p4_task2_qa_report(report, cfg.out_json))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nlp_project", description="Project 4 NLP workflows")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_task = sub.add_parser("task", help="Run Project 4 tasks")
    sub_task = p_task.add_subparsers(dest="task_cmd", required=True)

    p_sent = sub_task.add_parser("p4-sentiment", help="Prepare IMDb-backed Task 1 report")
    p_sent.add_argument("--config", required=True, help="Path to task_p4_sentiment YAML config")
    p_sent.set_defaults(func=cmd_task_p4_sentiment)

    p_qa = sub_task.add_parser("p4-qa", help="Prepare SQuAD-backed Task 2 report")
    p_qa.add_argument("--config", required=True, help="Path to task_p4_qa YAML config")
    p_qa.set_defaults(func=cmd_task_p4_qa)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
