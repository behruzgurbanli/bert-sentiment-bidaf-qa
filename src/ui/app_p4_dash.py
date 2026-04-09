#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from nlp_project.common.config import load_yaml
from nlp_project.p4.task1_sentiment import (
    P4Task1SentimentConfig,
    run_p4_task1_sentiment,
    run_p4_task1_sentiment_inference,
)
from nlp_project.p4.task2_qa import (
    P4Task2QAConfig,
    run_p4_task2_qa,
    run_p4_task2_qa_inference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
ASSETS_DIR = Path(__file__).resolve().with_name("assets")

PLOT_BG = "#f6f7fb"
CARD_BG = "#ffffff"
TEXT = "#162033"
MUTED = "#637089"
ACCENT = "#1f6feb"
ACCENT_2 = "#31c48d"
ACCENT_3 = "#f59e0b"


def _to_path(value: Any) -> Any:
    if isinstance(value, str) and ("/" in value or value.endswith(".json") or value.endswith(".yaml")):
        return Path(value)
    return value


def build_cfg(cls, cfg_dict: dict[str, Any]):
    kwargs = {k: _to_path(v) for k, v in cfg_dict.items()}
    return cls(**kwargs)


def load_json_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pick_report_files(pattern: str) -> list[Path]:
    return sorted(REPORTS_DIR.glob(pattern))


def pretty_model_name(name: str) -> str:
    mapping = {
        "bidaf_word_embeddings": "BiDAF Baseline",
        "bidaf_bert_embeddings": "BiDAF + BERT Embeddings",
    }
    return mapping.get(name, name.replace("_", " ").title())


def report_options() -> list[dict[str, str]]:
    return [{"label": path.name, "value": path.name} for path in pick_report_files("p4_task2_qa*_report.json")]


def task1_report_default() -> dict[str, Any] | None:
    cfg = load_yaml(CONFIG_DIR / "task_p4_sentiment.yaml")
    return load_json_report(Path(cfg["out_json"]))


def task2_report_default() -> dict[str, Any] | None:
    options = report_options()
    if not options:
        cfg = load_yaml(CONFIG_DIR / "task_p4_qa.yaml")
        return load_json_report(Path(cfg["out_json"]))
    return load_json_report(REPORTS_DIR / options[-1]["value"])


def card(title: str, value: str, subtitle: str | None = None, tone: str = "default") -> html.Div:
    return html.Div(
        [
            html.Div(title, className="metric-label"),
            html.Div(value, className=f"metric-value tone-{tone}"),
            html.Div(subtitle or "", className="metric-subtitle"),
        ],
        className="metric-card",
    )


def section_header(eyebrow: str, title: str, subtitle: str) -> html.Div:
    return html.Div(
        [
            html.Div(eyebrow, className="section-eyebrow"),
            html.H2(title, className="section-title"),
            html.P(subtitle, className="section-subtitle"),
        ],
        className="section-header",
    )


def story_point(text: str) -> html.Li:
    return html.Li(text, className="story-point")


def note_box(title: str, body: str) -> html.Div:
    return html.Div(
        [html.Div(title, className="note-title"), html.P(body, className="note-body")],
        className="note-box",
    )


def simple_table(df: pd.DataFrame) -> html.Table:
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in df.columns])),
            html.Tbody(
                [
                    html.Tr([html.Td("" if pd.isna(val) else str(val)) for val in row])
                    for row in df.itertuples(index=False, name=None)
                ]
            ),
        ],
        className="simple-table",
    )


def plotly_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        template="plotly_white",
        font={"family": "Inter, Arial, sans-serif", "color": TEXT},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(22,32,51,0.08)", zeroline=False)
    return fig


def task1_figures(report: dict[str, Any]) -> tuple[go.Figure, go.Figure, go.Figure]:
    dataset = report["dataset"]
    train_count = int(dataset["full_counts"]["train"])
    test_count = int(dataset["full_counts"]["test"])
    total_count = train_count + test_count
    train_avg = float(dataset["splits"]["train"]["avg_word_count"])
    test_avg = float(dataset["splits"]["test"]["avg_word_count"])

    fig_splits = go.Figure(
        data=[
            go.Pie(
                labels=["Train", "Test"],
                values=[train_count, test_count],
                hole=0.62,
                marker={"colors": [ACCENT, ACCENT_2]},
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} reviews (%{percent})<extra></extra>",
            )
        ]
    )
    plotly_layout(fig_splits)
    fig_splits.update_layout(showlegend=False, annotations=[{
        "text": f"{total_count:,}<br>reviews",
        "showarrow": False,
        "font": {"size": 22, "color": TEXT},
    }])

    fig_lengths = go.Figure()
    fig_lengths.add_trace(
        go.Scatter(
            x=[train_avg, test_avg],
            y=["Average review length", "Average review length"],
            mode="markers+text",
            marker={"size": [26, 26], "color": [ACCENT_3, ACCENT], "line": {"color": "#ffffff", "width": 2}},
            text=[f"Train {train_avg:.1f}", f"Test {test_avg:.1f}"],
            textposition=["top center", "bottom center"],
            hovertemplate="%{text} words<extra></extra>",
        )
    )
    fig_lengths.add_shape(
        type="line",
        x0=min(train_avg, test_avg),
        x1=max(train_avg, test_avg),
        y0=0,
        y1=0,
        line={"color": "#b5c0d4", "width": 4},
    )
    plotly_layout(fig_lengths)
    fig_lengths.update_yaxes(showticklabels=False, range=[-1, 1], title=None)
    fig_lengths.update_xaxes(title="Average number of words", showgrid=True, gridcolor="rgba(22,32,51,0.08)")

    train_neg = int(dataset["splits"]["train"]["label_counts"].get("neg", 0))
    train_pos = int(dataset["splits"]["train"]["label_counts"].get("pos", 0))
    test_neg = int(dataset["splits"]["test"]["label_counts"].get("neg", 0))
    test_pos = int(dataset["splits"]["test"]["label_counts"].get("pos", 0))
    fig_balance = go.Figure()
    fig_balance.add_trace(
        go.Bar(
            y=["Train", "Test"],
            x=[-train_neg, -test_neg],
            name="Negative",
            orientation="h",
            marker={"color": ACCENT},
            hovertemplate="Negative: %{customdata:,}<extra></extra>",
            customdata=[train_neg, test_neg],
        )
    )
    fig_balance.add_trace(
        go.Bar(
            y=["Train", "Test"],
            x=[train_pos, test_pos],
            name="Positive",
            orientation="h",
            marker={"color": ACCENT_3},
            hovertemplate="Positive: %{x:,}<extra></extra>",
        )
    )
    plotly_layout(fig_balance)
    max_count = max(train_neg, train_pos, test_neg, test_pos)
    fig_balance.update_layout(barmode="relative")
    fig_balance.update_xaxes(
        title="Review count mirrored around zero",
        tickvals=[-max_count, -max_count / 2, 0, max_count / 2, max_count],
        ticktext=[f"{max_count:,}", f"{int(max_count/2):,}", "0", f"{int(max_count/2):,}", f"{max_count:,}"],
    )
    return fig_splits, fig_lengths, fig_balance


def task2_figures(report: dict[str, Any]) -> tuple[go.Figure, go.Figure, go.Figure | None]:
    rows = []
    for model_name, model_info in (report.get("models") or {}).items():
        metrics = model_info.get("eval_metrics") or model_info.get("test_metrics")
        if not metrics:
            continue
        rows.append({"Model": pretty_model_name(model_name), "EM": float(metrics["exact_match"]), "F1": float(metrics["f1"])})
    fig_metrics = go.Figure()
    if rows:
        for idx, row in enumerate(rows):
            fig_metrics.add_trace(
                go.Scatter(
                    x=[row["EM"], row["F1"]],
                    y=[row["Model"], row["Model"]],
                    mode="lines+markers+text",
                    line={"color": ACCENT if idx == 0 else ACCENT_2, "width": 5},
                    marker={"size": 14, "color": [ACCENT, ACCENT_2]},
                    text=["EM", "F1"],
                    textposition="top center",
                    name=row["Model"],
                    hovertemplate="%{y}<br>%{text}: %{x:.4f}<extra></extra>",
                )
            )
    plotly_layout(fig_metrics)
    fig_metrics.update_xaxes(title="Score", range=[0, max([0.35] + [row["F1"] for row in rows]) + 0.05], showgrid=True, gridcolor="rgba(22,32,51,0.08)")
    fig_metrics.update_yaxes(title=None)

    split_rows = []
    for split_name in ("train", "dev"):
        split = report["splits"][split_name]
        split_rows.append(
            {
                "Split": split_name.title(),
                "Usable examples": int(split["usable_examples"]),
                "Average context tokens": float(split["avg_context_tokens"]),
                "Average question tokens": float(split["avg_question_tokens"]),
                "Average answer tokens": float(split["avg_answer_tokens"]),
            }
        )
    split_df = pd.DataFrame(split_rows)
    fig_dataset = go.Figure()
    for idx, row in split_df.iterrows():
        fig_dataset.add_trace(
            go.Scatter(
                x=[row["Average context tokens"]],
                y=[row["Average question tokens"]],
                mode="markers+text",
                marker={
                    "size": max(28, row["Average answer tokens"] * 12),
                    "color": ACCENT if row["Split"] == "Train" else ACCENT_3,
                    "line": {"color": "#ffffff", "width": 2},
                },
                text=[f"{row['Split']}<br>{int(row['Usable examples']):,}"],
                textposition="top center",
                name=row["Split"],
                hovertemplate=(
                    f"{row['Split']}<br>"
                    f"Usable examples: {int(row['Usable examples']):,}<br>"
                    f"Context tokens: {row['Average context tokens']:.2f}<br>"
                    f"Question tokens: {row['Average question tokens']:.2f}<br>"
                    f"Answer tokens: {row['Average answer tokens']:.2f}<extra></extra>"
                ),
            )
        )
    plotly_layout(fig_dataset)
    fig_dataset.update_xaxes(title="Average context tokens", showgrid=True, gridcolor="rgba(22,32,51,0.08)")
    fig_dataset.update_yaxes(title="Average question tokens")

    history_rows = []
    for model_name, model_info in (report.get("models") or {}).items():
        for row in model_info.get("training_history", []) or []:
            dev_f1 = row.get("dev_f1")
            if dev_f1 is None:
                continue
            history_rows.append(
                {
                    "Model": pretty_model_name(model_name),
                    "Epoch": int(row["epoch"]),
                    "Train Loss": None if row.get("train_loss") is None else float(row["train_loss"]),
                    "Dev F1": float(dev_f1),
                }
            )
    fig_history = None
    if history_rows:
        hist_df = pd.DataFrame(history_rows)
        fig_history = go.Figure()
        for idx, model_name in enumerate(hist_df["Model"].unique()):
            sub = hist_df[hist_df["Model"] == model_name]
            fig_history.add_trace(
                go.Scatter(
                    x=sub["Epoch"],
                    y=sub["Dev F1"],
                    mode="lines+markers",
                    name=model_name,
                    line={"width": 4, "color": ACCENT if idx == 0 else ACCENT_3},
                    marker={"size": 10},
                    hovertemplate=f"{model_name}<br>Epoch %{{x}}<br>Dev F1: %{{y:.4f}}<extra></extra>",
                )
            )
        plotly_layout(fig_history)
    return fig_metrics, fig_dataset, fig_history


def render_task1(report: dict[str, Any] | None, inference: dict[str, Any] | None) -> html.Div:
    if not report:
        return html.Div("Task 1 report not found. Generate it first from the CLI or UI.", className="empty-state")
    dataset = report["dataset"]
    analysis = report["model_analysis"]
    runtime = report["runtime"]
    summary = report.get("ui_summary") or {}
    fig_splits, fig_lengths, fig_balance = task1_figures(report)

    sample_reviews = summary.get("sample_reviews") or report.get("samples") or {}
    first_train = (sample_reviews.get("train") or [{}])[0]
    first_test = (sample_reviews.get("test") or [{}])[0]

    answer_df = pd.DataFrame(
        [
            {"Question": "Model input", "Answer": "A review text string tokenized into BERT inputs."},
            {"Question": "Model output", "Answer": "Sentiment label with confidence score."},
            {"Question": "Number of classes", "Answer": str(analysis["num_classes"])},
            {"Question": "Input size", "Answer": f"Up to {analysis['max_input_tokens']} tokens"},
            {"Question": "Case sensitivity", "Answer": "Case-insensitive (uncased model)"},
            {"Question": "Azerbaijani note", "Answer": analysis["agglutinative_language_note"]},
        ]
    )

    inference_block: list[Any] = []
    if inference:
        preds = inference.get("predictions") or []
        if preds:
            pred = preds[0]["prediction"][0]
            inference_block = [
                html.Div(
                    [
                        card("Predicted label", str(pred.get("label", "")), "Current custom review result", tone="accent"),
                        card("Confidence", f"{float(pred.get('score', 0.0)):.4f}", "Model confidence"),
                    ],
                    className="card-grid two-up",
                )
            ]
        else:
            inference_block = [note_box("Inference status", "Runtime inference could not be completed in the current environment.")]

    return html.Div(
        [
            section_header(
                "Task 1",
                "IMDb sentiment analysis with clear model interpretation",
                "This page summarizes the model inputs, outputs, dataset grounding, and custom-text inference behavior.",
            ),
            html.Div(
                [
                    card("Train reviews", str(int(dataset["full_counts"]["train"])), "IMDb training examples"),
                    card("Test reviews", str(int(dataset["full_counts"]["test"])), "IMDb held-out examples"),
                    card("Classes", str(int(analysis["num_classes"])), "NEGATIVE / POSITIVE"),
                    card("Runtime", "Available" if runtime["model_loaded"] else "Optional", "Live inference availability"),
                ],
                className="card-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Task summary", className="panel-title"),
                            html.Ul(
                                [
                                    story_point("IMDb is the active Task 1 dataset and it is a clean binary sentiment benchmark."),
                                    story_point("The selected BERT classifier returns a sentiment label and confidence score."),
                                    story_point("Because the model is uncased, capitalization is normalized rather than treated as a signal."),
                                    story_point("Azerbaijani would require multilingual or task-specific evaluation before making strong claims."),
                                ],
                                className="story-list",
                            ),
                            note_box("Interpretation", report["model_analysis"]["imdb_alignment_note"]),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H3("Required answers", className="panel-title"),
                            simple_table(answer_df),
                        ],
                        className="panel",
                    ),
                ],
                className="layout-two",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Train/Test composition", className="panel-title"),
                            html.P("A compact overview of how the full IMDb corpus is split.", className="panel-copy"),
                            dcc.Graph(figure=fig_splits, config={"displayModeBar": False}),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H3("Average review length", className="panel-title"),
                            html.P("This emphasizes the small difference between train and test review lengths.", className="panel-copy"),
                            dcc.Graph(figure=fig_lengths, config={"displayModeBar": False}),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H3("Sentiment symmetry", className="panel-title"),
                            html.P("Mirrored bars make the binary class balance visible without pretending there is a large gap.", className="panel-copy"),
                            dcc.Graph(figure=fig_balance, config={"displayModeBar": False}),
                        ],
                        className="panel",
                    ),
                ],
                className="layout-three",
            ),
            html.Div(
                [
                    note_box(
                        "Reading the charts",
                        "IMDb is intentionally balanced, so the important message is not that one class dominates. "
                        "The useful insight is that train and test are consistently structured, which makes evaluation cleaner.",
                    ),
                ],
                className="layout-two",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Sample training review", className="panel-title"),
                            html.Div(first_train.get("label", "").upper(), className="sample-chip"),
                            html.P(first_train.get("text_preview", "No sample available."), className="sample-copy"),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H3("Sample test review", className="panel-title"),
                            html.Div(first_test.get("label", "").upper(), className="sample-chip alt"),
                            html.P(first_test.get("text_preview", "No sample available."), className="sample-copy"),
                        ],
                        className="panel",
                    ),
                ],
                className="layout-two",
            ),
            html.Div(
                [
                    html.H3("Live review test", className="panel-title"),
                    html.P("Enter a custom review in the left panel and run inference. The result appears here after execution.", className="panel-copy"),
                    *inference_block,
                ],
                className="panel",
            ),
        ]
    )


def render_task2(report: dict[str, Any] | None, inference: dict[str, Any] | None) -> html.Div:
    if not report:
        return html.Div("Task 2 report not found. Generate it first from the CLI or UI.", className="empty-state")
    plan = report["model_plan"]
    runtime = report["runtime"]
    comparison = report.get("comparison")
    fig_metrics, fig_dataset, fig_history = task2_figures(report)

    best_model = None
    best_f1 = -1.0
    for model_name, model_info in (report.get("models") or {}).items():
        metrics = model_info.get("eval_metrics") or model_info.get("test_metrics")
        if metrics and float(metrics["f1"]) > best_f1:
            best_f1 = float(metrics["f1"])
            best_model = (pretty_model_name(model_name), metrics)

    sample = None
    for split_name in ("train", "dev"):
        samples = report["splits"][split_name]["samples"]
        if samples:
            sample = samples[0]
            break

    model_rows = []
    for model_name, model_info in (report.get("models") or {}).items():
        metrics = model_info.get("eval_metrics") or model_info.get("test_metrics") or {}
        model_rows.append(
            {
                "Model": pretty_model_name(model_name),
                "Status": model_info.get("status", ""),
                "EM": "" if "exact_match" not in metrics else f"{float(metrics['exact_match']):.4f}",
                "F1": "" if "f1" not in metrics else f"{float(metrics['f1']):.4f}",
            }
        )

    answer_df = pd.DataFrame(
        [
            {"Question": "Model input", "Answer": "A question and a context passage."},
            {"Question": "Model output", "Answer": plan["expected_output"]},
            {"Question": "Baseline", "Answer": plan["baseline"]},
            {"Question": "BERT variant", "Answer": plan["comparison_model"]},
            {"Question": "Metrics", "Answer": ", ".join(plan["evaluation_metrics"])},
            {"Question": "Dataset", "Answer": report["dataset"]["name"]},
        ]
    )

    inference_block: list[Any] = []
    if inference:
        pred = inference.get("selected_prediction") or {}
        eval_block = inference.get("evaluation")
        row_cards = [
            card("Predicted answer", str(pred.get("answer", "")), "Current QA output", tone="accent"),
            card("Method", str(pred.get("method", "")), "Inference path used"),
        ]
        if eval_block:
            row_cards.append(card("Exact Match", f"{float(eval_block['exact_match']):.4f}", "Against provided gold"))
            row_cards.append(card("F1", f"{float(eval_block['f1']):.4f}", "Against provided gold"))
        inference_block = [html.Div(row_cards, className="card-grid")]

    history_panel: list[Any] = []
    if fig_history is not None:
        history_panel = [html.Div([html.H3("Validation trend across epochs", className="panel-title"), dcc.Graph(figure=fig_history, config={"displayModeBar": False})], className="panel")]

    return html.Div(
        [
            section_header(
                "Task 2",
                "Reading comprehension as a direct model comparison",
                "This page presents the BiDAF baseline, the BERT-enhanced variant, and the measured difference between them.",
            ),
            html.Div(
                [
                    card("Runtime QA", "Available" if runtime["model_loaded"] else "Optional", "Live QA inference"),
                    card("Prepared train", str(int(report["prepared_training_data"]["train_examples"])), "Training spans"),
                    card("Prepared eval", str(int(report["prepared_training_data"]["eval_examples"])), "Evaluation spans"),
                    card("Best F1", f"{best_f1:.4f}" if best_model else "N/A", best_model[0] if best_model else "No trained model yet", tone="accent"),
                ],
                className="card-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Experiment narrative", className="panel-title"),
                            html.Ul(
                                [
                                    story_point("SQuAD 1.1 is a span-extraction dataset, so predictions are start and end answer positions in context."),
                                    story_point("The baseline uses BiDAF with word embeddings, while the comparison injects contextual BERT embeddings."),
                                    story_point("Exact Match and F1 expose whether contextual representations improve answer quality."),
                                ],
                                className="story-list",
                            ),
                            note_box(
                                "Main takeaway",
                                comparison["interpretation"] if comparison else "Run both models on the same subset to expose the effect of contextual embeddings.",
                            ),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H3("Required answers", className="panel-title"),
                            simple_table(answer_df),
                        ],
                        className="panel",
                    ),
                ],
                className="layout-two",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Baseline vs BERT", className="panel-title"),
                            html.P("Each line connects EM to F1 for the same model, so the gap between the two variants is easier to see.", className="panel-copy"),
                            dcc.Graph(figure=fig_metrics, config={"displayModeBar": False}),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H3("Dataset geometry", className="panel-title"),
                            html.P("This view compares how long SQuAD contexts and questions are, with answer length reflected in point size.", className="panel-copy"),
                            dcc.Graph(figure=fig_dataset, config={"displayModeBar": False}),
                        ],
                        className="panel",
                    ),
                ],
                className="layout-two",
            ),
            *history_panel,
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Model summary", className="panel-title"),
                            simple_table(pd.DataFrame(model_rows)),
                        ],
                        className="panel",
                    ),
                    html.Div(
                        [
                            html.H3("Sample QA example", className="panel-title"),
                            html.P(f"Question: {sample['question']}" if sample else "No sample available.", className="sample-copy"),
                            html.Div(f"Gold answer: {(sample['gold_answers'][0] if sample and sample['gold_answers'] else '')}", className="sample-chip"),
                            html.P(" ".join(sample["context_preview_tokens"][:70]) if sample else "", className="sample-copy"),
                        ],
                        className="panel",
                    ),
                ],
                className="layout-two",
            ),
            html.Div(
                [
                    html.H3("Live QA test", className="panel-title"),
                    html.P("Enter a question and context in the left panel. This area updates with the predicted answer and optional EM/F1 if you provide gold answers.", className="panel-copy"),
                    *inference_block,
                ],
                className="panel",
            ),
        ]
    )


app = Dash(__name__, title="Project 4 Demo", assets_folder=str(ASSETS_DIR))

initial_task1_report = task1_report_default()
initial_task2_report = task2_report_default()

app.layout = html.Div(
    [
        dcc.Store(id="task1-store", data=initial_task1_report),
        dcc.Store(id="task2-store", data=initial_task2_report),
        dcc.Store(id="task1-inference-store"),
        dcc.Store(id="task2-inference-store"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Project 4", className="hero-eyebrow"),
                        html.H1("Sentiment analysis and question answering with comparative results and interactive inference.", className="hero-title"),
                        html.P(
                            "This interface focuses on readable results, selected visual summaries, and interactive text inputs for model inspection.",
                            className="hero-copy",
                        ),
                    ],
                    className="hero-text",
                ),
                html.Div(
                    [
                        html.Div([html.Div("IMDb"), html.Strong("Task 1")], className="hero-pill"),
                        html.Div([html.Div("SQuAD 1.1"), html.Strong("Task 2")], className="hero-pill"),
                    ],
                    className="hero-pills",
                ),
            ],
            className="hero",
        ),
        dcc.Tabs(
            id="page-tabs",
            value="task1",
            className="main-tabs",
            children=[
                dcc.Tab(
                    label="Task 1",
                    value="task1",
                    className="main-tab",
                    selected_className="main-tab-selected",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("Controls", className="sidebar-title"),
                                        html.Button("Regenerate Task 1 report", id="task1-run-button", className="action-button"),
                                        html.Label("Custom review", className="control-label"),
                                        dcc.Textarea(
                                            id="task1-review-input",
                                            value="The acting was excellent, but the story was too predictable.",
                                            className="text-area",
                                        ),
                                        html.Button("Run sentiment inference", id="task1-infer-button", className="action-button secondary"),
                                    ],
                                    className="control-panel",
                                ),
                                html.Div(id="task1-content", className="content-panel"),
                            ],
                            className="page-layout",
                        )
                    ],
                ),
                dcc.Tab(
                    label="Task 2",
                    value="task2",
                    className="main-tab",
                    selected_className="main-tab-selected",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("Controls", className="sidebar-title"),
                                        html.Label("Saved Task 2 report", className="control-label"),
                                        dcc.Dropdown(id="task2-report-select", options=report_options(), value=(report_options()[-1]["value"] if report_options() else None), clearable=False, className="dropdown"),
                                        html.Button("Regenerate Task 2 report", id="task2-run-button", className="action-button"),
                                        html.Label("Question", className="control-label"),
                                        dcc.Input(
                                            id="task2-question-input",
                                            value="When did Beyonce start becoming popular?",
                                            className="text-input",
                                            type="text",
                                        ),
                                        html.Label("Context", className="control-label"),
                                        dcc.Textarea(
                                            id="task2-context-input",
                                            value="Beyonce Giselle Knowles-Carter became popular in the late 1990s as the lead singer of Destiny's Child.",
                                            className="text-area tall",
                                        ),
                                        html.Label("Gold answers (optional, separated by |)", className="control-label"),
                                        dcc.Input(id="task2-gold-input", value="", className="text-input", type="text"),
                                        html.Button("Run QA inference", id="task2-infer-button", className="action-button secondary"),
                                    ],
                                    className="control-panel",
                                ),
                                html.Div(id="task2-content", className="content-panel"),
                            ],
                            className="page-layout",
                        )
                    ],
                ),
            ],
        ),
    ],
    className="app-shell",
)


@app.callback(Output("task1-store", "data"), Input("task1-run-button", "n_clicks"), prevent_initial_call=True)
def regenerate_task1(_: int):
    cfg_dict = load_yaml(CONFIG_DIR / "task_p4_sentiment.yaml")
    report = run_p4_task1_sentiment(build_cfg(P4Task1SentimentConfig, cfg_dict))
    return report


@app.callback(
    Output("task1-inference-store", "data"),
    Input("task1-infer-button", "n_clicks"),
    State("task1-review-input", "value"),
    prevent_initial_call=True,
)
def run_task1_inference(_: int, review_text: str):
    cfg_dict = load_yaml(CONFIG_DIR / "task_p4_sentiment.yaml")
    cfg = build_cfg(P4Task1SentimentConfig, cfg_dict)
    return run_p4_task1_sentiment_inference(cfg, [review_text or ""])


@app.callback(Output("task1-content", "children"), Input("task1-store", "data"), Input("task1-inference-store", "data"))
def update_task1_content(report: dict[str, Any] | None, inference: dict[str, Any] | None):
    return render_task1(report, inference)


@app.callback(
    Output("task2-store", "data"),
    Input("task2-report-select", "value"),
    Input("task2-run-button", "n_clicks"),
    State("task2-store", "data"),
    prevent_initial_call=False,
)
def update_task2_store(selected_report: str | None, n_clicks: int | None, current: dict[str, Any] | None):
    from dash import ctx

    trigger = ctx.triggered_id
    if trigger == "task2-run-button":
        cfg_dict = load_yaml(CONFIG_DIR / "task_p4_qa.yaml")
        return run_p4_task2_qa(build_cfg(P4Task2QAConfig, cfg_dict))
    if selected_report:
        report = load_json_report(REPORTS_DIR / selected_report)
        if report is not None:
            return report
    return current


@app.callback(
    Output("task2-inference-store", "data"),
    Input("task2-infer-button", "n_clicks"),
    State("task2-question-input", "value"),
    State("task2-context-input", "value"),
    State("task2-gold-input", "value"),
    prevent_initial_call=True,
)
def run_task2_inference(_: int, question: str, context: str, gold_text: str):
    cfg_dict = load_yaml(CONFIG_DIR / "task_p4_qa.yaml")
    cfg = build_cfg(P4Task2QAConfig, cfg_dict)
    gold_answers = [item.strip() for item in (gold_text or "").split("|") if item.strip()]
    return run_p4_task2_qa_inference(
        cfg,
        question=question or "",
        context=context or "",
        gold_answers=gold_answers or None,
    )


@app.callback(Output("task2-content", "children"), Input("task2-store", "data"), Input("task2-inference-store", "data"))
def update_task2_content(report: dict[str, Any] | None, inference: dict[str, Any] | None):
    return render_task2(report, inference)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
