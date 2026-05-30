# Benchmark Plan

## First-Paper Direction

Use an existing benchmark dataset plus existing AI APIs plus our evaluation methodology.

This avoids training a new model, scraping random websites, using copyrighted internal documents, or creating a huge dataset.

## Recommended Dataset

Start with **HotpotQA** because it includes:

- Questions
- Gold answers
- Context passages
- Sentence-level supporting facts

This maps naturally to hallucination and citation evaluation:

- `question` becomes the user query
- `context` becomes the RAG source document
- `answer` becomes the expected answer
- `supporting_facts` become expected citation evidence

See `docs/dataset_selection.md` for the broader dataset decision table.

## First Experiment

Use 50 HotpotQA validation examples next, then scale to 100.

Compare:

- `strict_citation_rag`
- `gold` evidence retrieval
- automatically retrieved top-k evidence

Measure:

- Answer accuracy
- Partial-credit answer score
- Citation accuracy
- Partial-credit citation score
- Hallucination rate
- Refusal behavior where applicable

Scoring rule:

- `yes` = 1.0 point
- `partial` = 0.5 point
- `no` = 0.0 points

The evaluator reports both strict accuracy, where only `yes` counts as correct, and partial-credit score.

## Experimental Conditions

| Setting | Purpose |
|---|---|
| Gold supporting facts | Best-case RAG condition |
| Retrieved top-k passages | Realistic RAG condition |
| Distractor passages | Tests robustness |
| Missing evidence | Tests refusal behavior |
| Conflicting evidence | Tests citation and reasoning reliability |

The main first-paper comparison is:

1. Gold Evidence RAG: the model receives HotpotQA supporting facts.
2. Retrieved Evidence RAG: the system retrieves passages automatically.
3. Vector Evidence RAG: the system retrieves passages using a reusable embedding index and cosine similarity.

Research question:

> How does hallucination and citation accuracy change when RAG systems use gold supporting evidence versus automatically retrieved evidence?

## Cross-Model Comparison

After the retrieval experiments, compare multiple models while keeping the dataset, retrieval method, top-k value, and prompt identical.

Fixed setup:

- Dataset: HotpotQA 50-question sample
- Retrieval: vector retrieval
- Top-k: 8
- Prompt: `strict_citation_rag`

| Model | Retrieval | Prompt | Answer Accuracy | Citation Accuracy | Hallucination Rate | Status | Answer File |
|---|---|---|---:|---:|---:|---|---|
| `gpt-4.1-mini` | Vector top-8 | Strict citation | 60.0% | 67.5% | 6.0% | Done | `data/hotpotqa_vector_answers_raw.csv` |
| `gpt-4.1` | Vector top-8 | Strict citation | not_run_yet | not_run_yet | not_run_yet | Planned | `data/hotpotqa_vector_gpt41_answers_raw.csv` |
| Claude | Vector top-8 | Strict citation | not_run_yet | not_run_yet | not_run_yet | Planned cross-model comparison | `data/hotpotqa_vector_claude_answers_raw.csv` |
| Gemini | Vector top-8 | Strict citation | not_run_yet | not_run_yet | not_run_yet | Planned cross-model comparison | `data/hotpotqa_vector_gemini_answers_raw.csv` |
| Llama | Vector top-8 | Strict citation | not_run_yet | not_run_yet | not_run_yet | Planned open/local model comparison | `data/hotpotqa_vector_llama_answers_raw.csv` |

Purpose:

> This test checks whether stronger or different model families reduce hallucination and citation errors when retrieval quality is held constant.

Important control:

> Only the model should change. Retrieval, top-k, source context, prompt, question set, and scoring labels should stay the same.

## Milestones

| Milestone | Target |
|---|---:|
| Pilot | 5 questions |
| Current expanded pilot | 25 questions |
| Small experiment | 50 questions |
| First paper version | 100 questions |
| Stronger paper | 200-300 questions |

## Current Pilot Result

The current 25-question gold-evidence pilot produced:

| Metric | Result |
|---|---:|
| Questions tested | 25 |
| Strict answer accuracy | 88.0% |
| Partial-credit answer score | 90.0% |
| Strict citation accuracy | 87.5% |
| Partial-credit citation score | 91.7% |
| Hallucination rate | 0.0% |

Suggested wording:

> As an initial pilot, we evaluated 25 HotpotQA examples using a strict citation RAG prompt and gold supporting facts as evidence. The model achieved 88.0% strict answer accuracy, 90.0% partial-credit answer score, and 87.5% strict citation accuracy, with no hallucinated responses observed in this small sample. These preliminary results suggest that strict evidence-grounding prompts can support citation-faithful responses when gold evidence is available, but larger-scale evaluation and retrieved-evidence comparisons are required before drawing general conclusions.

Important limitation:

> The pilot uses only 25 examples and gold supporting facts, so it represents a best-case evidence setting rather than a realistic retrieval scenario.

HotpotQA is mostly answerable QA, so refusal accuracy is less central for the first HotpotQA experiment. A later extension can add unanswerable questions from another benchmark.

## Commands

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Prepare 25 HotpotQA examples:

```bash
python3 src/prepare_hotpotqa.py --limit 25
```

Dry run:

```bash
python3 src/main.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --retrieval gold --dry-run --verbose --limit 1
```

Run OpenAI on a tiny sample:

```bash
export OPENAI_API_KEY="your_api_key_here"
python3 src/main.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --retrieval gold --limit 3 --settings strict_citation_rag --verbose
```

Run a retrieval stress test:

```bash
python3 src/main.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --retrieval keyword --top-k 8 --dry-run --verbose --limit 1
```

Build and use vector retrieval:

```bash
python3 src/build_vector_index.py --source-dir data/hotpotqa/source_documents --output data/hotpotqa/vector_index.jsonl
python3 src/main.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --retrieval vector --vector-index data/hotpotqa/vector_index.jsonl --top-k 8 --dry-run --verbose --limit 1
```

After human labeling:

```bash
python3 src/rag_experiment.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --answers data/main_openai_answers_raw.csv --verbose
```

Run the `gpt-4.1` cross-model comparison:

```bash
python3 src/main.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --retrieval vector --vector-index data/hotpotqa/vector_index.jsonl --top-k 8 --limit 50 --settings strict_citation_rag --model gpt-4.1 --output data/hotpotqa_vector_gpt41_answers_raw.csv
```

Run the other model families with the same retrieval and prompt:

```bash
python3 src/main.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --retrieval vector --vector-index data/hotpotqa/vector_index.jsonl --top-k 8 --limit 50 --settings strict_citation_rag --provider anthropic --model claude-sonnet-4-5 --output data/hotpotqa_vector_claude_answers_raw.csv
python3 src/main.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --retrieval vector --vector-index data/hotpotqa/vector_index.jsonl --top-k 8 --limit 50 --settings strict_citation_rag --provider gemini --model gemini-2.5-flash --output data/hotpotqa_vector_gemini_answers_raw.csv
python3 src/main.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --retrieval vector --vector-index data/hotpotqa/vector_index.jsonl --top-k 8 --limit 50 --settings strict_citation_rag --provider ollama --model llama3.1 --output data/hotpotqa_vector_llama_answers_raw.csv
```

After labeling the new answer files, compare them with the baseline:

```bash
python3 src/compare_answer_files.py --questions data/hotpotqa/questions.csv --answer-file gpt-4.1-mini=data/hotpotqa_vector_answers_raw.csv --answer-file gpt-4.1=data/hotpotqa_vector_gpt41_answers_raw.csv --answer-file Claude=data/hotpotqa_vector_claude_answers_raw.csv --answer-file Gemini=data/hotpotqa_vector_gemini_answers_raw.csv --answer-file Llama=data/hotpotqa_vector_llama_answers_raw.csv --output outputs/model_comparison.csv
```
