#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "This will read API keys silently, keep them only in this process,"
echo "and then run all available test cases with generation enabled."
echo

read -r -s -p "OpenAI API key (press Enter to skip OpenAI/gpt-4.1): " OPENAI_API_KEY_INPUT
echo
read -r -s -p "Claude/Anthropic API key (press Enter to skip Claude): " ANTHROPIC_API_KEY_INPUT
echo
read -r -s -p "Gemini API key (press Enter to skip Gemini): " GEMINI_API_KEY_INPUT
echo

if [[ -n "$OPENAI_API_KEY_INPUT" ]]; then
  export OPENAI_API_KEY="$OPENAI_API_KEY_INPUT"
fi

if [[ -n "$ANTHROPIC_API_KEY_INPUT" ]]; then
  export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY_INPUT"
fi

if [[ -n "$GEMINI_API_KEY_INPUT" ]]; then
  export GEMINI_API_KEY="$GEMINI_API_KEY_INPUT"
fi

unset OPENAI_API_KEY_INPUT
unset ANTHROPIC_API_KEY_INPUT
unset GEMINI_API_KEY_INPUT

./scripts/run_all_test_cases.sh --generate
