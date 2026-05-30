# RAG Hallucination Research

This repository contains a small reproducible experiment for studying answer accuracy, citation accuracy, retrieval quality, and hallucination in retrieval-augmented generation (RAG).

Working paper title:

**Evaluating Hallucination and Citation Accuracy in RAG-Based Question Answering Using Gold and Retrieved Evidence**

## What The Project Tests

The experiment asks models HotpotQA-style multi-hop questions using source-document context. It compares:

- gold evidence vs. retrieved evidence
- vector retrieval quality at different `k` values
- strict citation prompting
- cross-model behavior across OpenAI, Claude, and Gemini

The scoring script does not automatically decide correctness. Model answers are generated first, then human labels are filled in the answer CSVs:

- `answer_correct`
- `citation_correct`
- `hallucination`
- `refusal_correct`
- `notes`

Metrics are computed from those labels.

## Current Results

Full available labeled runs:

| Experiment | Rows Scored | Answer Accuracy | Citation Accuracy | Hallucination Rate |
|---|---:|---:|---:|---:|
| Gold evidence, `gpt-4.1-mini` | 25 | 88.0% | 87.5% | 0.0% |
| Vector top-8, `gpt-4.1-mini` | 50 | 60.0% | 67.5% | 6.0% |
| Vector top-8, `gpt-4.1` | 50 | 54.0% | 60.0% | 2.0% |
| Vector top-8, Claude | 50 | 56.0% | 60.0% | 2.0% |
| Vector top-8, Gemini | 23 | 47.8% | 20.0% | 4.3% |

Fair first-20 cross-model comparison:

| Model | Rows Scored | Answer Accuracy | Citation Accuracy | Hallucination Rate |
|---|---:|---:|---:|---:|
| `gpt-4.1-mini` | 20 | 55.0% | 73.3% | 10.0% |
| `gpt-4.1` | 20 | 55.0% | 58.3% | 0.0% |
| Claude | 20 | 55.0% | 55.0% | 0.0% |
| Gemini | 20 | 50.0% | 23.1% | 5.0% |

Vector retrieval quality:

| k | Any Gold Retrieved | All Gold Retrieved | Mean Citation Recall |
|---:|---:|---:|---:|
| 1 | 84.0% | 0.0% | 36.7% |
| 3 | 94.0% | 26.0% | 58.9% |
| 5 | 98.0% | 38.0% | 69.9% |
| 8 | 100.0% | 52.0% | 78.9% |

## Repository Structure

```text
data/
  hotpotqa/
    questions.csv
    source_documents/
  *_answers_raw.csv
outputs/
  model_comparison.csv
  model_comparison_first20.csv
  proper_metrics_all_available.csv
  retrieval_quality_vector_top8_summary.csv
src/
  main.py
  build_vector_index.py
  rag_experiment.py
  retrieval_quality.py
  compare_answer_files.py
scripts/
  run_all_test_cases.sh
  run_all_with_keys.sh
```

The vector index is intentionally ignored by git because it is a generated embedding file. Rebuild it locally when needed.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Optional API keys for model generation:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export GEMINI_API_KEY="your_gemini_key"
```

Do not commit real API keys. Use `.env.example` only as a template.

## Build Vector Index

```bash
python3 src/build_vector_index.py \
  --source-dir data/hotpotqa/source_documents \
  --output data/hotpotqa/vector_index.jsonl
```

## Generate Answers

Baseline vector retrieval with strict citation:

```bash
python3 src/main.py \
  --questions data/hotpotqa/questions.csv \
  --source-dir data/hotpotqa/source_documents \
  --retrieval vector \
  --vector-index data/hotpotqa/vector_index.jsonl \
  --top-k 8 \
  --limit 50 \
  --settings strict_citation_rag \
  --output data/hotpotqa_vector_answers_raw.csv
```

Cross-model runs reuse the saved vector context so each model receives the same evidence:

```bash
python3 src/main.py \
  --questions data/hotpotqa/questions.csv \
  --source-dir data/hotpotqa/source_documents \
  --retrieval saved \
  --saved-retrieval-answers data/hotpotqa_vector_answers_raw.csv \
  --top-k 8 \
  --limit 50 \
  --settings strict_citation_rag \
  --provider openai \
  --model gpt-4.1 \
  --output data/hotpotqa_vector_gpt41_answers_raw.csv
```

Use `--provider anthropic --model claude-sonnet-4-5` for Claude and `--provider gemini --model gemini-2.5-flash` for Gemini.

## Evaluate Metrics

Evaluate a labeled answer file:

```bash
python3 src/rag_experiment.py \
  --questions data/hotpotqa/questions.csv \
  --source-dir data/hotpotqa/source_documents \
  --retrieval saved \
  --answers data/hotpotqa_vector_answers_raw.csv \
  --skip-prompts
```

Compare answer files:

```bash
python3 src/compare_answer_files.py \
  --questions data/hotpotqa/questions.csv \
  --answer-file gpt-4.1-mini=data/hotpotqa_vector_answers_raw.csv \
  --answer-file gpt-4.1=data/hotpotqa_vector_gpt41_answers_raw.csv \
  --answer-file Claude=data/hotpotqa_vector_claude_answers_raw.csv \
  --answer-file Gemini=data/hotpotqa_vector_gemini_answers_raw.csv \
  --output outputs/model_comparison.csv
```

Measure retrieval quality:

```bash
python3 src/retrieval_quality.py \
  --questions data/hotpotqa/questions.csv \
  --source-dir data/hotpotqa/source_documents \
  --retrieval saved \
  --answers data/hotpotqa_vector_answers_raw.csv \
  --ks 1 3 5 8
```

Run all available checks and summaries:

```bash
bash scripts/run_all_test_cases.sh
```

Run API generation plus checks:

```bash
bash scripts/run_all_with_keys.sh
```

## Key Output Files

- `outputs/model_comparison.csv`
- `outputs/model_comparison_first20.csv`
- `outputs/proper_metrics_all_available.csv`
- `outputs/retrieval_quality_vector_top8_summary.csv`
- `outputs/results_summary_vector_top8_gpt41mini_50.csv`
- `outputs/results_summary_vector_top8_gpt41_50.csv`
- `outputs/results_summary_vector_top8_claude_50.csv`
- `outputs/results_summary_vector_top8_gemini_23.csv`

## Notes

Unanswerable-question metrics are present in the code but were not used in the current reported experiment. Those fields remain `n/a` when no unanswerable rows are evaluated.
