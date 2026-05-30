#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "This script reads API keys silently and keeps them only in this process."
echo "Keys are not written to files."
echo

read -r -s -p "Claude/Anthropic API key: " ANTHROPIC_API_KEY
echo
read -r -s -p "Gemini API key: " GEMINI_API_KEY
echo

export ANTHROPIC_API_KEY
export GEMINI_API_KEY

echo
echo "Keys loaded for this script process."
echo "Run one of these commands next in this same terminal after exporting keys manually,"
echo "or edit this script to uncomment the model run you want."
echo
echo "Claude:"
echo "python3 src/main.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --retrieval saved --saved-retrieval-answers data/hotpotqa_vector_answers_raw.csv --top-k 8 --limit 50 --settings strict_citation_rag --provider anthropic --model claude-sonnet-4-5 --output data/hotpotqa_vector_claude_answers_raw.csv"
echo
echo "Gemini:"
echo "python3 src/main.py --questions data/hotpotqa/questions.csv --source-dir data/hotpotqa/source_documents --retrieval saved --saved-retrieval-answers data/hotpotqa_vector_answers_raw.csv --top-k 8 --limit 50 --settings strict_citation_rag --provider gemini --model gemini-2.5-flash --output data/hotpotqa_vector_gemini_answers_raw.csv"
