#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GENERATE="no"
if [[ "${1:-}" == "--generate" ]]; then
  GENERATE="yes"
fi

mkdir -p outputs

copy_if_exists() {
  local source="$1"
  local target="$2"
  if [[ -f "$source" ]]; then
    cp "$source" "$target"
  fi
}

run_labeled_eval() {
  local label="$1"
  local answers="$2"
  local output="outputs/results_summary_${label}.csv"

  if [[ ! -f "$answers" ]]; then
    echo "Skipping ${label}: missing ${answers}"
    return
  fi

  echo
  echo "== Evaluating ${label} =="
  python3 src/rag_experiment.py \
    --questions data/hotpotqa/questions.csv \
    --source-dir data/hotpotqa/source_documents \
    --retrieval saved \
    --answers "$answers" \
    --skip-prompts
  copy_if_exists outputs/results_summary.csv "$output"
}

run_generation_if_possible() {
  local label="$1"
  local provider="$2"
  local model="$3"
  local output="$4"
  local required_env="${5:-}"

  if [[ "$GENERATE" != "yes" ]]; then
    echo "Skipping generation for ${label}: pass --generate to run API/model calls."
    return
  fi

  if [[ -n "$required_env" && -z "${!required_env:-}" ]]; then
    echo "Skipping generation for ${label}: ${required_env} is not set."
    return
  fi

  if [[ "$provider" == "ollama" ]]; then
    if ! curl -fsS "${OLLAMA_BASE_URL:-http://localhost:11434}/api/tags" >/dev/null 2>&1; then
      echo "Skipping generation for ${label}: Ollama is not running at ${OLLAMA_BASE_URL:-http://localhost:11434}."
      return
    fi
  fi

  echo
  echo "== Generating ${label} =="
  command=(
    python3 src/main.py
    --questions data/hotpotqa/questions.csv \
    --source-dir data/hotpotqa/source_documents \
    --retrieval saved \
    --saved-retrieval-answers data/hotpotqa_vector_answers_raw.csv \
    --top-k 8 \
    --limit 50 \
    --settings strict_citation_rag \
    --provider "$provider" \
    --model "$model" \
    --output "$output"
  )
  if [[ "$provider" == "gemini" ]]; then
    command+=(--sleep 15)
  fi
  "${command[@]}"
}

echo "== Syntax checks =="
python3 -m py_compile \
  src/main.py \
  src/rag_experiment.py \
  src/retrieval_quality.py \
  src/compare_answer_files.py

echo
echo "== Retrieval quality: saved vector top-k =="
python3 src/retrieval_quality.py \
  --questions data/hotpotqa/questions.csv \
  --source-dir data/hotpotqa/source_documents \
  --retrieval saved \
  --answers data/hotpotqa_vector_answers_raw.csv \
  --ks 1 3 5 8

run_labeled_eval "gold_25" "data/main_openai_answers_raw.csv"
run_labeled_eval "vector_top8_gpt41mini_50" "data/hotpotqa_vector_answers_raw.csv"

run_generation_if_possible \
  "gpt-4.1" \
  "openai" \
  "gpt-4.1" \
  "data/hotpotqa_vector_gpt41_answers_raw.csv" \
  "OPENAI_API_KEY"

run_generation_if_possible \
  "Claude" \
  "anthropic" \
  "claude-sonnet-4-5" \
  "data/hotpotqa_vector_claude_answers_raw.csv" \
  "ANTHROPIC_API_KEY"

run_generation_if_possible \
  "Gemini" \
  "gemini" \
  "gemini-2.5-flash" \
  "data/hotpotqa_vector_gemini_answers_raw.csv" \
  "GEMINI_API_KEY"

run_generation_if_possible \
  "Llama/Ollama" \
  "ollama" \
  "llama3.1" \
  "data/hotpotqa_vector_llama_answers_raw.csv"

echo
echo "== Model comparison from existing labeled files =="
compare_args=(
  --questions data/hotpotqa/questions.csv
  --answer-file gpt-4.1-mini=data/hotpotqa_vector_answers_raw.csv
)

for item in \
  "gpt-4.1=data/hotpotqa_vector_gpt41_answers_raw.csv" \
  "Claude=data/hotpotqa_vector_claude_answers_raw.csv" \
  "Gemini=data/hotpotqa_vector_gemini_answers_raw.csv" \
  "Llama=data/hotpotqa_vector_llama_answers_raw.csv"
do
  file="${item#*=}"
  if [[ -f "$file" ]]; then
    compare_args+=(--answer-file "$item")
  else
    echo "Not adding missing comparison file: $file"
  fi
done

python3 src/compare_answer_files.py \
  "${compare_args[@]}" \
  --output outputs/model_comparison.csv

echo
echo "== Top-k comparison from existing labeled files =="
topk_args=(--questions data/hotpotqa/questions.csv)
for item in \
  "vector_top1=data/hotpotqa_vector_top1_answers_raw.csv" \
  "vector_top3=data/hotpotqa_vector_top3_answers_raw.csv" \
  "vector_top5=data/hotpotqa_vector_top5_answers_raw.csv" \
  "vector_top8=data/hotpotqa_vector_answers_raw.csv" \
  "vector_top10=data/hotpotqa_vector_top10_answers_raw.csv"
do
  file="${item#*=}"
  if [[ -f "$file" ]]; then
    topk_args+=(--answer-file "$item")
  else
    echo "Not adding missing top-k file: $file"
  fi
done

python3 src/compare_answer_files.py \
  "${topk_args[@]}" \
  --output outputs/topk_answer_comparison.csv

echo
echo "All available test cases completed."
echo "Key outputs:"
echo "- outputs/retrieval_quality_summary.csv"
echo "- outputs/results_summary_gold_25.csv"
echo "- outputs/results_summary_vector_top8_gpt41mini_50.csv"
echo "- outputs/model_comparison.csv"
echo "- outputs/topk_answer_comparison.csv"
