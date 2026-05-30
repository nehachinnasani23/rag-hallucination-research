from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "outputs" / "experiment_prompts.csv"
OUTPUT_PATH = ROOT / "data" / "openai_answers_raw.csv"
RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def extract_output_text(response_json: dict[str, object]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []
    output_items = response_json.get("output")
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content_items = item.get("content")
            if not isinstance(content_items, list):
                continue
            for content_item in content_items:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if isinstance(text, str):
                    parts.append(text)

    return "\n".join(parts).strip()


def extract_cited_sources(answer: str) -> str:
    citations = sorted(set(re.findall(r"S\d+", answer)))
    return " ".join(citations)


def call_openai(prompt: str, model: str, api_key: str, max_output_tokens: int) -> str:
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        RESPONSES_URL,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=90) as response:
                response_json = json.loads(response.read().decode("utf-8"))
                answer = extract_output_text(response_json)
                if not answer:
                    raise RuntimeError(f"OpenAI response did not contain text: {response_json}")
                return answer
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            if error.code not in RETRY_STATUS_CODES or attempt == 3:
                raise RuntimeError(f"OpenAI API error {error.code}: {error_body}") from error
            time.sleep(2**attempt)
        except URLError as error:
            if attempt == 3:
                raise RuntimeError(f"Network error calling OpenAI API: {error}") from error
            time.sleep(2**attempt)

    raise RuntimeError("OpenAI API call failed after retries.")


def filter_prompt_rows(
    rows: list[dict[str, str]], settings: set[str] | None, limit: int | None
) -> list[dict[str, str]]:
    selected = [row for row in rows if settings is None or row["setting"] in settings]
    if limit is not None:
        return selected[:limit]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real OpenAI API calls for the RAG hallucination experiment.")
    parser.add_argument("--prompts", type=Path, default=PROMPTS_PATH, help="Prompt CSV created by rag_experiment.py.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="CSV file where raw OpenAI answers are saved.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model ID.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of prompts to run.")
    parser.add_argument("--settings", nargs="*", default=None, help="Optional settings to run, such as no_rag basic_rag.")
    parser.add_argument("--max-output-tokens", type=int, default=300, help="Maximum output tokens per answer.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds to wait between API calls.")
    parser.add_argument("--force", action="store_true", help="Re-run prompts even if they already exist in the output CSV.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Set it before running real API calls.", file=sys.stderr)
        print('Example: export OPENAI_API_KEY="your_api_key_here"', file=sys.stderr)
        sys.exit(1)

    if not args.prompts.exists():
        print(f"Prompt file not found: {args.prompts}", file=sys.stderr)
        print("Run this first: python3 src/rag_experiment.py", file=sys.stderr)
        sys.exit(1)

    prompt_rows = read_csv(args.prompts)
    selected_settings = set(args.settings) if args.settings else None
    selected_rows = filter_prompt_rows(prompt_rows, selected_settings, args.limit)

    existing_rows = load_existing_rows(args.output)
    completed_keys = {
        (row["question_id"], row["setting"], row["model_name"])
        for row in existing_rows
        if row.get("question_id") and row.get("setting") and row.get("model_name")
    }

    rows_to_write = list(existing_rows)
    fieldnames = [
        "question_id",
        "setting",
        "model_name",
        "answer",
        "cited_source",
        "answer_correct",
        "citation_correct",
        "hallucination",
        "refusal_correct",
        "notes",
    ]

    for index, prompt_row in enumerate(selected_rows, start=1):
        key = (prompt_row["question_id"], prompt_row["setting"], args.model)
        if key in completed_keys and not args.force:
            print(f"[{index}/{len(selected_rows)}] Skipping existing answer: {key}")
            continue

        print(f"[{index}/{len(selected_rows)}] Calling {args.model} for {prompt_row['question_id']} / {prompt_row['setting']}")
        answer = call_openai(prompt_row["prompt"], args.model, api_key, args.max_output_tokens)
        rows_to_write.append(
            {
                "question_id": prompt_row["question_id"],
                "setting": prompt_row["setting"],
                "model_name": args.model,
                "answer": answer,
                "cited_source": extract_cited_sources(answer),
                "answer_correct": "",
                "citation_correct": "",
                "hallucination": "",
                "refusal_correct": "",
                "notes": "",
            }
        )
        write_csv(args.output, rows_to_write, fieldnames)
        time.sleep(args.sleep)

    print(f"Saved raw OpenAI answers to: {args.output}")
    print("Next step: manually fill the label columns, then run:")
    print(f"python3 src/rag_experiment.py --answers {args.output}")


if __name__ == "__main__":
    main()
