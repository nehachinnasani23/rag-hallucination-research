from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_documents"
QUESTIONS_PATH = ROOT / "data" / "questions.csv"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_ANSWERS_PATH = ROOT / "data" / "main_openai_answers_raw.csv"
DEFAULT_VECTOR_INDEX_PATH = ROOT / "data" / "hotpotqa" / "vector_index.jsonl"
DEFAULT_SAVED_RETRIEVAL_ANSWERS_PATH = ROOT / "data" / "hotpotqa_vector_answers_raw.csv"

RESPONSES_URL = "https://api.openai.com/v1/responses"
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
GEMINI_GENERATE_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_GENERATION_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
DEFAULT_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
GENERATION_PROVIDERS = ("auto", "openai", "anthropic", "gemini", "deepseek", "ollama")

SETTINGS = ("no_rag", "basic_rag", "strict_citation_rag")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "up",
    "what",
    "when",
    "with",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_source_sentences(source_dir: Path) -> dict[str, str]:
    sentences: dict[str, str] = {}
    sentence_pattern = re.compile(r"^\[(S\d+)\]\s+(.+)$")

    for source_file in sorted(source_dir.glob("*.md")):
        for line in source_file.read_text(encoding="utf-8").splitlines():
            match = sentence_pattern.match(line.strip())
            if match:
                sentence_id, sentence_text = match.groups()
                sentences[sentence_id] = sentence_text

    if not sentences:
        raise ValueError(f"No source sentences found in {source_dir}")

    return sentences


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in STOPWORDS}


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    if left_length == 0 or right_length == 0:
        return 0.0
    return dot_product / (left_length * right_length)


def keyword_retrieve(
    question: str, sentences: dict[str, str], top_k: int
) -> list[tuple[str, str, float]]:
    question_tokens = tokenize(question)
    scored_sentences = []

    for sentence_id, sentence_text in sentences.items():
        sentence_tokens = tokenize(sentence_text)
        score = float(len(question_tokens & sentence_tokens))
        scored_sentences.append((sentence_id, sentence_text, score))

    scored_sentences.sort(key=lambda item: (-item[2], item[0]))
    return scored_sentences[:top_k]


def post_json_with_headers(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
    error_label: str,
) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="POST",
        headers=headers,
    )

    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            if error.code not in RETRY_STATUS_CODES or attempt == 3:
                raise RuntimeError(f"{error_label} API error {error.code}: {error_body}") from error
            retry_delay = retry_delay_seconds(error_body)
            time.sleep(retry_delay if retry_delay is not None else 2**attempt)
        except URLError as error:
            if attempt == 3:
                raise RuntimeError(f"Network error calling {error_label} API: {error}") from error
            time.sleep(2**attempt)

    raise RuntimeError(f"{error_label} API call failed after retries.")


def retry_delay_seconds(error_body: str) -> float | None:
    try:
        payload = json.loads(error_body)
    except json.JSONDecodeError:
        return None

    details = payload.get("error", {}).get("details", [])
    if not isinstance(details, list):
        return None

    for detail in details:
        if not isinstance(detail, dict):
            continue
        retry_delay = detail.get("retryDelay")
        if not isinstance(retry_delay, str) or not retry_delay.endswith("s"):
            continue
        try:
            return float(retry_delay[:-1]) + 1.0
        except ValueError:
            return None

    return None


def post_json(url: str, payload: dict[str, object], api_key: str) -> dict[str, object]:
    return post_json_with_headers(
        url,
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        "OpenAI",
    )


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


def embedding_retrieve(
    question: str,
    sentences: dict[str, str],
    top_k: int,
    embedding_model: str,
    api_key: str,
) -> list[tuple[str, str, float]]:
    sentence_items = sorted(sentences.items())
    texts = [question] + [sentence_text for _, sentence_text in sentence_items]
    embeddings = create_embeddings(texts, embedding_model, api_key)

    question_embedding = embeddings[0]
    sentence_embeddings = embeddings[1:]
    scored_sentences = []

    for (sentence_id, sentence_text), sentence_embedding in zip(sentence_items, sentence_embeddings):
        score = cosine_similarity(question_embedding, sentence_embedding)
        scored_sentences.append((sentence_id, sentence_text, score))

    scored_sentences.sort(key=lambda item: (-item[2], item[0]))
    return scored_sentences[:top_k]


def load_vector_index(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Vector index not found: {path}. "
            "Build it first with: python3 src/build_vector_index.py "
            "--source-dir data/hotpotqa/source_documents"
        )

    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("embedding"), list):
                    raise ValueError(f"Vector index row missing embedding: {row}")
                rows.append(row)

    if not rows:
        raise ValueError(f"Vector index is empty: {path}")

    return rows


def validate_vector_index(
    vector_index: list[dict[str, object]],
    sentences: dict[str, str],
    vector_index_path: Path,
    source_dir: Path,
) -> None:
    indexed_sentences = {
        str(row["sentence_id"]): str(row["text"])
        for row in vector_index
    }
    source_ids = set(sentences)
    indexed_ids = set(indexed_sentences)

    missing_ids = sorted(source_ids - indexed_ids)
    extra_ids = sorted(indexed_ids - source_ids)
    text_mismatches = sorted(
        sentence_id
        for sentence_id in source_ids & indexed_ids
        if sentences[sentence_id] != indexed_sentences[sentence_id]
    )

    if not missing_ids and not extra_ids and not text_mismatches:
        return

    details = [
        "Vector index does not match the current source documents.",
        f"Source sentences: {len(source_ids)}",
        f"Indexed sentences: {len(indexed_ids)}",
    ]
    if missing_ids:
        details.append(f"Missing from index: {len(missing_ids)} sentence(s), e.g. {', '.join(missing_ids[:5])}")
    if extra_ids:
        details.append(f"Extra in index: {len(extra_ids)} sentence(s), e.g. {', '.join(extra_ids[:5])}")
    if text_mismatches:
        details.append(f"Text changed for: {len(text_mismatches)} sentence(s), e.g. {', '.join(text_mismatches[:5])}")
    details.append("Rebuild it with:")
    details.append(
        "python3 src/build_vector_index.py "
        f"--source-dir {source_dir} "
        f"--output {vector_index_path}"
    )
    raise ValueError("\n".join(details))


def vector_index_retrieve(
    question: str,
    vector_index: list[dict[str, object]],
    top_k: int,
    embedding_model: str,
    api_key: str,
) -> list[tuple[str, str, float]]:
    question_embedding = create_embeddings([question], embedding_model, api_key)[0]
    scored_sentences = []

    for row in vector_index:
        sentence_id = str(row["sentence_id"])
        sentence_text = str(row["text"])
        sentence_embedding = [float(value) for value in row["embedding"]]
        score = cosine_similarity(question_embedding, sentence_embedding)
        scored_sentences.append((sentence_id, sentence_text, score))

    scored_sentences.sort(key=lambda item: (-item[2], item[0]))
    return scored_sentences[:top_k]


def gold_retrieve(
    expected_citation: str,
    sentences: dict[str, str],
) -> list[tuple[str, str, float]]:
    citation_ids = [citation_id.strip() for citation_id in expected_citation.split() if citation_id.strip()]
    retrieved = []

    for rank, citation_id in enumerate(citation_ids):
        sentence_text = sentences.get(citation_id)
        if sentence_text:
            score = float(len(citation_ids) - rank)
            retrieved.append((citation_id, sentence_text, score))

    return retrieved


def build_prompt(setting: str, question: str, retrieved_context: str) -> str:
    if setting == "no_rag":
        return (
            "Answer the question using your general knowledge.\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    if setting == "basic_rag":
        return (
            "Answer the question using the provided source context.\n\n"
            f"Source context:\n{retrieved_context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    if setting == "strict_citation_rag":
        return (
            "You are answering questions using only the provided source context.\n\n"
            "Rules:\n"
            "1. Use only information from the source context.\n"
            "2. Cite the sentence ID that supports your answer, such as [S000001].\n"
            "3. If the source does not contain the answer, say: "
            '"The document does not provide enough information."\n'
            "4. Do not guess.\n"
            "5. Do not use outside knowledge.\n\n"
            f"Source context:\n{retrieved_context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

    raise ValueError(f"Unknown setting: {setting}")


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


def extract_anthropic_text(response_json: dict[str, object]) -> str:
    parts: list[str] = []
    content_items = response_json.get("content")
    if isinstance(content_items, list):
        for item in content_items:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
    return "\n".join(parts).strip()


def extract_gemini_text(response_json: dict[str, object]) -> str:
    parts: list[str] = []
    candidates = response_json.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            content_parts = content.get("parts")
            if not isinstance(content_parts, list):
                continue
            for part in content_parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
    return "\n".join(parts).strip()


def extract_chat_completion_text(response_json: dict[str, object]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list):
        return ""
    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            parts.append(message["content"])
    return "\n".join(parts).strip()


def extract_ollama_text(response_json: dict[str, object]) -> str:
    response = response_json.get("response")
    if isinstance(response, str):
        return response.strip()
    return ""


def infer_generation_provider(model: str) -> str:
    lower_model = model.lower()
    if lower_model.startswith("claude"):
        return "anthropic"
    if lower_model.startswith("gemini"):
        return "gemini"
    if lower_model.startswith("deepseek"):
        return "deepseek"
    if lower_model.startswith(("llama", "mistral", "mixtral", "qwen", "phi")):
        return "ollama"
    return "openai"


def generation_api_key(provider: str) -> str | None:
    if provider == "openai":
        return os.environ.get("OPENAI_API_KEY")
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY")
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if provider == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY")
    if provider == "ollama":
        return None
    raise ValueError(f"Unknown generation provider: {provider}")


def call_openai_generation(prompt: str, model: str, api_key: str, max_output_tokens: int) -> str:
    response_json = post_json(
        RESPONSES_URL,
        {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "store": False,
        },
        api_key,
    )
    answer = extract_output_text(response_json)
    if not answer:
        raise RuntimeError(f"OpenAI response did not contain text: {response_json}")
    return answer


def call_anthropic_generation(prompt: str, model: str, api_key: str, max_output_tokens: int) -> str:
    response_json = post_json_with_headers(
        ANTHROPIC_MESSAGES_URL,
        {
            "model": model,
            "max_tokens": max_output_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        },
        {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        },
        "Anthropic",
    )
    answer = extract_anthropic_text(response_json)
    if not answer:
        raise RuntimeError(f"Anthropic response did not contain text: {response_json}")
    return answer


def call_gemini_generation(prompt: str, model: str, api_key: str, max_output_tokens: int) -> str:
    url = f"{GEMINI_GENERATE_URL_TEMPLATE.format(model=model)}?key={api_key}"
    response_json = post_json_with_headers(
        url,
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
            },
        },
        {"Content-Type": "application/json"},
        "Gemini",
    )
    answer = extract_gemini_text(response_json)
    if not answer:
        raise RuntimeError(f"Gemini response did not contain text: {response_json}")
    return answer


def call_deepseek_generation(prompt: str, model: str, api_key: str, max_output_tokens: int) -> str:
    response_json = post_json_with_headers(
        DEEPSEEK_CHAT_COMPLETIONS_URL,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_output_tokens,
            "stream": False,
        },
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        "DeepSeek",
    )
    answer = extract_chat_completion_text(response_json)
    if not answer:
        raise RuntimeError(f"DeepSeek response did not contain text: {response_json}")
    return answer


def call_ollama_generation(prompt: str, model: str, max_output_tokens: int) -> str:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    response_json = post_json_with_headers(
        f"{base_url}/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_output_tokens,
            },
        },
        {"Content-Type": "application/json"},
        "Ollama",
    )
    answer = extract_ollama_text(response_json)
    if not answer:
        raise RuntimeError(f"Ollama response did not contain text: {response_json}")
    return answer


def call_generation_model(
    prompt: str,
    model: str,
    provider: str,
    api_key: str | None,
    max_output_tokens: int,
) -> str:
    if provider == "openai":
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI generation.")
        return call_openai_generation(prompt, model, api_key, max_output_tokens)

    if provider == "anthropic":
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic generation.")
        return call_anthropic_generation(prompt, model, api_key, max_output_tokens)

    if provider == "gemini":
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini generation.")
        return call_gemini_generation(prompt, model, api_key, max_output_tokens)

    if provider == "deepseek":
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek generation.")
        return call_deepseek_generation(prompt, model, api_key, max_output_tokens)

    if provider == "ollama":
        return call_ollama_generation(prompt, model, max_output_tokens)

    raise ValueError(f"Unknown generation provider: {provider}")


def extract_cited_sources(answer: str) -> str:
    citations = sorted(set(re.findall(r"S\d+", answer)))
    return " ".join(citations)


def load_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)


def load_saved_top_source_ids(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Saved retrieval answer file not found: {path}")

    rows = read_csv(path)
    saved_ids: dict[str, list[str]] = {}
    for row in rows:
        question_id = row.get("question_id", "")
        top_source_ids = row.get("top_source_ids", "")
        if question_id and top_source_ids:
            saved_ids[question_id] = [
                source_id.strip()
                for source_id in top_source_ids.split()
                if source_id.strip()
            ]
    return saved_ids


def saved_retrieve(
    question_id: str,
    sentences: dict[str, str],
    saved_top_source_ids: dict[str, list[str]],
    top_k: int,
) -> list[tuple[str, str, float]]:
    source_ids = saved_top_source_ids.get(question_id, [])[:top_k]
    retrieved: list[tuple[str, str, float]] = []
    for rank, source_id in enumerate(source_ids):
        sentence_text = sentences.get(source_id)
        if sentence_text:
            retrieved.append((source_id, sentence_text, float(len(source_ids) - rank)))
    return retrieved


def answer_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["question_id"], row["setting"], row["model_name"])


def retrieve(
    question: str,
    sentences: dict[str, str],
    top_k: int,
    retrieval_method: str,
    embedding_model: str,
    api_key: str | None,
) -> list[tuple[str, str, float]]:
    if retrieval_method == "keyword":
        return keyword_retrieve(question, sentences, top_k)

    if retrieval_method == "embedding":
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embedding retrieval.")
        return embedding_retrieve(question, sentences, top_k, embedding_model, api_key)

    raise ValueError(f"Unknown retrieval method: {retrieval_method}")


def save_retrieval_debug(rows: list[dict[str, str]]) -> Path:
    output_path = OUTPUT_DIR / "main_retrieval_debug.csv"
    write_csv(
        output_path,
        rows,
        [
            "question_id",
            "question",
            "retrieval_method",
            "top_source_ids",
            "retrieved_context",
            "scores",
        ],
    )
    return output_path


def save_prompt_debug(rows: list[dict[str, str]]) -> Path:
    output_path = OUTPUT_DIR / "main_prompts.csv"
    write_csv(output_path, rows, ["question_id", "setting", "question", "prompt"])
    return output_path


def print_debug_question(
    question_row: dict[str, str],
    setting: str,
    retrieved: list[tuple[str, str, float]],
    prompt: str,
    answer: str | None,
) -> None:
    print("\n----------------------------------------")
    print(f"Question ID: {question_row['question_id']}")
    print(f"Setting: {setting}")
    print(f"Question: {question_row['question']}")
    print(f"Expected answer from CSV: {question_row['expected_answer']}")
    print(f"Expected citation from CSV: {question_row['expected_citation'] or 'n/a'}")
    print(f"Answerable from CSV: {question_row['answerable']}")
    print("Retrieved context:")
    for sentence_id, sentence_text, score in retrieved:
        print(f"  - [{sentence_id}] score={score:.4f}: {sentence_text}")
    print("Prompt sent to model:")
    print(prompt)
    if answer is not None:
        print("Model answer:")
        print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Main end-to-end RAG hallucination experiment runner.")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH, help="Question CSV file.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR, help="Folder containing source document markdown files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_ANSWERS_PATH, help="Where raw model answers are saved.")
    parser.add_argument("--vector-index", type=Path, default=DEFAULT_VECTOR_INDEX_PATH, help="JSONL vector index for vector retrieval.")
    parser.add_argument(
        "--saved-retrieval-answers",
        type=Path,
        default=DEFAULT_SAVED_RETRIEVAL_ANSWERS_PATH,
        help="Answer CSV containing saved top_source_ids for --retrieval saved.",
    )
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL, help="Generation model name.")
    parser.add_argument(
        "--provider",
        choices=GENERATION_PROVIDERS,
        default="auto",
        help="Generation provider. auto infers from the model name.",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="OpenAI embedding model.")
    parser.add_argument(
        "--retrieval",
        choices=["keyword", "embedding", "vector", "gold", "saved"],
        default="keyword",
        help=(
            "Retrieval method. vector uses a prebuilt vector index; gold uses benchmark "
            "supporting facts; saved reuses top_source_ids from an earlier answer CSV."
        ),
    )
    parser.add_argument("--top-k", type=int, default=2, help="Number of source sentences to retrieve.")
    parser.add_argument("--settings", nargs="*", choices=SETTINGS, default=list(SETTINGS), help="Experiment settings.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of question rows to run.")
    parser.add_argument("--max-output-tokens", type=int, default=300, help="Maximum output tokens per model answer.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds to wait between generation calls.")
    parser.add_argument("--dry-run", action="store_true", help="Build retrieval and prompts without calling OpenAI.")
    parser.add_argument("--force", action="store_true", help="Re-run answers that already exist in the output CSV.")
    parser.add_argument("--verbose", action="store_true", help="Print each step in the terminal.")
    args = parser.parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    provider = infer_generation_provider(args.model) if args.provider == "auto" else args.provider
    provider_api_key = generation_api_key(provider)

    if not args.dry_run and provider != "ollama" and not provider_api_key:
        env_var = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY or GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }[provider]
        print(f"{env_var} is required for {provider} generation.", file=sys.stderr)
        sys.exit(1)

    if args.retrieval in {"embedding", "vector"} and not openai_api_key:
        print("OPENAI_API_KEY is required for embedding/vector retrieval.", file=sys.stderr)
        sys.exit(1)

    all_questions = read_csv(args.questions)
    questions = all_questions
    if args.limit is not None:
        if len(all_questions) < args.limit:
            print(
                f"Requested --limit {args.limit}, but {args.questions} contains only "
                f"{len(all_questions)} question row(s). Run src/prepare_hotpotqa.py "
                "with a higher --limit to create more questions."
            )
        questions = all_questions[: args.limit]

    sentences = load_source_sentences(args.source_dir)
    try:
        vector_index = load_vector_index(args.vector_index) if args.retrieval == "vector" else None
        if vector_index is not None:
            validate_vector_index(vector_index, sentences, args.vector_index, args.source_dir)
        saved_top_source_ids = (
            load_saved_top_source_ids(args.saved_retrieval_answers)
            if args.retrieval == "saved"
            else {}
        )
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    existing_rows = load_existing_rows(args.output)
    completed_keys = {
        answer_key(row)
        for row in existing_rows
        if row.get("question_id") and row.get("setting") and row.get("model_name")
    }

    answer_rows = list(existing_rows)
    retrieval_debug_rows: list[dict[str, str]] = []
    prompt_debug_rows: list[dict[str, str]] = []
    fieldnames = [
        "question_id",
        "setting",
        "model_name",
        "provider",
        "retrieval_method",
        "top_source_ids",
        "answer",
        "cited_source",
        "answer_correct",
        "citation_correct",
        "hallucination",
        "refusal_correct",
        "notes",
    ]

    total_runs = len(questions) * len(args.settings)
    run_number = 0

    for question_row in questions:
        if args.retrieval == "gold":
            retrieved = gold_retrieve(question_row.get("expected_citation", ""), sentences)
            if not retrieved:
                retrieved = keyword_retrieve(question_row["question"], sentences, args.top_k)
        elif args.retrieval == "vector":
            if vector_index is None:
                raise RuntimeError("Vector index was not loaded.")
            retrieved = vector_index_retrieve(
                question_row["question"],
                vector_index,
                args.top_k,
                args.embedding_model,
                openai_api_key or "",
            )
        elif args.retrieval == "saved":
            retrieved = saved_retrieve(
                question_row["question_id"],
                sentences,
                saved_top_source_ids,
                args.top_k,
            )
            if not retrieved:
                raise RuntimeError(
                    "No saved retrieved source IDs found for "
                    f"{question_row['question_id']} in {args.saved_retrieval_answers}"
                )
        else:
            retrieved = retrieve(
                question_row["question"],
                sentences,
                args.top_k,
                args.retrieval,
                args.embedding_model,
                openai_api_key,
            )
        retrieved_context = " ".join(f"[{sentence_id}] {text}" for sentence_id, text, _ in retrieved)
        top_source_ids = " ".join(sentence_id for sentence_id, _, _ in retrieved)
        scores = " ".join(f"{sentence_id}:{score:.4f}" for sentence_id, _, score in retrieved)

        retrieval_debug_rows.append(
            {
                "question_id": question_row["question_id"],
                "question": question_row["question"],
                "retrieval_method": args.retrieval,
                "top_source_ids": top_source_ids,
                "retrieved_context": retrieved_context,
                "scores": scores,
            }
        )

        for setting in args.settings:
            run_number += 1
            prompt = build_prompt(setting, question_row["question"], retrieved_context)
            prompt_debug_rows.append(
                {
                    "question_id": question_row["question_id"],
                    "setting": setting,
                    "question": question_row["question"],
                    "prompt": prompt,
                }
            )

            key = (question_row["question_id"], setting, args.model)
            if key in completed_keys and not args.force:
                print(f"[{run_number}/{total_runs}] Skipping existing answer: {key}")
                continue
            if key in completed_keys and args.force:
                answer_rows = [
                    row
                    for row in answer_rows
                    if not (
                        row.get("question_id") == key[0]
                        and row.get("setting") == key[1]
                        and row.get("model_name") == key[2]
                    )
                ]

            print(f"[{run_number}/{total_runs}] {question_row['question_id']} / {setting}")
            if args.dry_run:
                if args.verbose:
                    print_debug_question(question_row, setting, retrieved, prompt, answer=None)
                continue

            answer = call_generation_model(
                prompt,
                args.model,
                provider,
                provider_api_key,
                args.max_output_tokens,
            )
            if args.verbose:
                print_debug_question(question_row, setting, retrieved, prompt, answer=answer)

            answer_rows.append(
                {
                    "question_id": question_row["question_id"],
                    "setting": setting,
                    "model_name": args.model,
                    "provider": provider,
                    "retrieval_method": args.retrieval,
                    "top_source_ids": top_source_ids,
                    "answer": answer,
                    "cited_source": extract_cited_sources(answer),
                    "answer_correct": "",
                    "citation_correct": "",
                    "hallucination": "",
                    "refusal_correct": "",
                    "notes": "",
                }
            )
            write_csv(args.output, answer_rows, fieldnames)
            time.sleep(args.sleep)

    retrieval_debug_path = save_retrieval_debug(retrieval_debug_rows)
    prompt_debug_path = save_prompt_debug(prompt_debug_rows)
    print(f"Saved retrieval debug file to: {retrieval_debug_path}")
    print(f"Saved prompt debug file to: {prompt_debug_path}")

    if args.dry_run:
        print("Dry run complete. No generation calls were made.")
        return

    write_csv(args.output, answer_rows, fieldnames)
    print(f"Saved raw model answers to: {args.output}")
    print("Next: fill the human label columns, then run:")
    print(f"python3 src/rag_experiment.py --answers {args.output} --verbose")


if __name__ == "__main__":
    main()
