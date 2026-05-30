from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "hotpotqa" / "source_documents"
DEFAULT_OUTPUT = ROOT / "data" / "hotpotqa" / "vector_index.jsonl"
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def load_source_sentences(source_dir: Path) -> list[dict[str, str]]:
    sentence_pattern = re.compile(r"^\[(S\d+)\]\s+(.+)$")
    rows: list[dict[str, str]] = []

    for source_file in sorted(source_dir.glob("*.md")):
        for line in source_file.read_text(encoding="utf-8").splitlines():
            match = sentence_pattern.match(line.strip())
            if match:
                sentence_id, sentence_text = match.groups()
                rows.append(
                    {
                        "sentence_id": sentence_id,
                        "text": sentence_text,
                        "source_file": str(source_file.resolve().relative_to(ROOT)),
                    }
                )

    if not rows:
        raise ValueError(f"No source sentences found in {source_dir}")

    return rows


def post_json(url: str, payload: dict[str, object], api_key: str) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
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
                return json.loads(response.read().decode("utf-8"))
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


def create_embeddings(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    response_json = post_json(
        EMBEDDINGS_URL,
        {
            "model": model,
            "input": texts,
            "encoding_format": "float",
        },
        api_key,
    )
    data = response_json.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"Embedding response did not contain data: {response_json}")

    embeddings: list[list[float]] = []
    for item in sorted(data, key=lambda value: value.get("index", 0) if isinstance(value, dict) else 0):
        if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
            raise RuntimeError(f"Unexpected embedding item: {item}")
        embeddings.append([float(value) for value in item["embedding"]])

    return embeddings


def batched(rows: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reusable vector index for source sentences.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR, help="Folder containing source document markdown files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSONL vector index output path.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="OpenAI embedding model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of sentences to embed per API request.")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Set it before building the vector index.", file=sys.stderr)
        print('Example: export OPENAI_API_KEY="your_api_key_here"', file=sys.stderr)
        sys.exit(1)

    source_rows = load_source_sentences(args.source_dir)
    output_rows = []

    print(f"Loaded {len(source_rows)} source sentences from {args.source_dir}")
    print(f"Embedding model: {args.embedding_model}")

    for batch_number, batch in enumerate(batched(source_rows, args.batch_size), start=1):
        print(f"Embedding batch {batch_number} ({len(batch)} sentences)")
        embeddings = create_embeddings([row["text"] for row in batch], args.embedding_model, api_key)
        for row, embedding in zip(batch, embeddings):
            output_rows.append(
                {
                    "sentence_id": row["sentence_id"],
                    "text": row["text"],
                    "source_file": row["source_file"],
                    "embedding_model": args.embedding_model,
                    "embedding": embedding,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in output_rows:
            file.write(json.dumps(row) + "\n")

    print(f"Wrote vector index with {len(output_rows)} rows to: {args.output}")


if __name__ == "__main__":
    main()
