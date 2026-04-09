# NLP Project 4: BERT Sentiment Analysis and BiDAF/BERT QA

Course project implementing:
- Task 1: sentiment analysis model analysis on IMDb using a fine-tuned BERT-family classifier
- Task 2: extractive question answering on SQuAD 1.1 using a BiDAF-style PyTorch model and a BERT-enhanced variant

The project is built around exactly two datasets:
- IMDb for Task 1
- SQuAD 1.1 for Task 2

## Project Scope

### Task 1
Analyzes the open-source model `distilbert-base-uncased-finetuned-sst-2-english` and exposes:
- model input and output structure
- class count
- maximum input size
- case sensitivity behavior
- discussion of applicability to Azerbaijani as an agglutinative language

It also supports optional runtime inference on custom review text.

### Task 2
Implements a BiDAF-style extractive QA system in PyTorch:
- input: question + context
- output: answer start and end positions

Two settings are supported:
- baseline with trainable non-contextual word embeddings
- BERT-enhanced version using `bert-base-uncased` contextual embeddings

Evaluation is reported with:
- Exact Match (EM)
- F1

### UI
The repo includes a Dash-based interface for:
- browsing saved reports
- comparing baseline vs BERT results
- testing custom sentiment inference
- testing custom QA inference

## Repository Structure

```text
.
├── configs/
│   ├── task_p4_sentiment.yaml
│   └── task_p4_qa.yaml
├── dataset/
│   ├── aclImdb/
│   └── squad/
├── report/
│   └── project4_report.tex
├── src/
│   ├── nlp_project/
│   │   ├── cli.py
│   │   ├── common/
│   │   │   └── config.py
│   │   └── p4/
│   │       ├── common.py
│   │       ├── task1_sentiment.py
│   │       └── task2_qa.py
│   ├── scripts/
│   │   └── run_ui_p4_dash.sh
│   └── ui/
│       ├── app_p4_dash.py
│       └── assets/
│           └── p4_dash.css
├── environment-p4.yml
├── requirements-p4.txt
└── README.md
```

## Technical Design

### Task 1 backend
`src/nlp_project/p4/task1_sentiment.py`

Responsibilities:
- load IMDb splits
- summarize dataset statistics
- generate structured JSON and Markdown reports
- run optional Hugging Face sentiment inference

### Task 2 backend
`src/nlp_project/p4/task2_qa.py`

Responsibilities:
- load and preprocess SQuAD 1.1
- align answer spans from character offsets to token offsets
- build tokenized QA examples
- train a BiDAF-style baseline
- train a BERT-enhanced BiDAF variant
- evaluate EM/F1
- generate structured JSON and Markdown reports

### Shared helpers
`src/nlp_project/p4/common.py`

Responsibilities:
- IMDb loading
- SQuAD loading
- tokenization with spans
- answer normalization
- EM/F1 scoring

### UI
`src/ui/app_p4_dash.py`

Responsibilities:
- load saved reports
- display experiment summaries and charts
- run custom inference from the browser

## Environment Setup

```bash
python -m pip install -r requirements-p4.txt
```

## Dependencies

Main packages:
- PyYAML
- Dash
- Plotly
- PyTorch
- Torchvision
- Transformers
- SentencePiece
- Accelerate

## How to Run

### 1. Generate Task 1 report

```bash
PYTHONPATH=src python -m nlp_project.cli task p4-sentiment --config configs/task_p4_sentiment.yaml
```

### 2. Generate Task 2 prepare report

```bash
PYTHONPATH=src python -m nlp_project.cli task p4-qa --config configs/task_p4_qa.yaml
```

### 3. Run a small Task 2 training experiment

Example:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from nlp_project.p4.task2_qa import P4Task2QAConfig, run_p4_task2_qa

cfg = P4Task2QAConfig(
    train_path=Path("dataset/squad/train-v1.1.json"),
    dev_path=Path("dataset/squad/dev-v1.1.json"),
    prepare_only=False,
    train_word_bidaf=True,
    train_bert_bidaf=True,
    max_train_examples=1000,
    max_dev_examples=200,
    max_eval_examples=200,
    epochs=1,
    batch_size=4,
    freeze_bert=True,
    out_json=Path("data/reports/p4_task2_qa_word_vs_bert_report.json"),
    out_dir=Path("data/reports/p4_task2_qa_word_vs_bert"),
)
report = run_p4_task2_qa(cfg)
print(report.get("comparison"))
PY
```

### 4. Run the Dash UI

```bash
bash src/scripts/run_ui_p4_dash.sh
```

Then open:
- `http://127.0.0.1:8050`

## Main Saved Results

From the stronger saved comparison run:
- Word baseline: `EM = 0.1075`, `F1 = 0.1469`
- BERT-enhanced model: `EM = 0.2600`, `F1 = 0.3158`
- Improvement:
  - `EM +0.1525`
  - `F1 +0.1689`

This is the main empirical result of the project.


## Notes

- The project assumes the datasets already exist locally under `dataset/`.
- Generated report artifacts under `data/reports/` are ignored by `.gitignore`.
- The Task 2 baseline uses trainable non-contextual word embeddings, not pretrained GloVe or Word2Vec.
- The strongest QA comparison was run on controlled subsets for practical local training time.

## License

This repository is intended for academic course-project use.
