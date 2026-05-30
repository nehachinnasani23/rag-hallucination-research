from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "hotpotqa"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_context_items(example: dict[str, Any]) -> list[tuple[str, list[str]]]:
    context = example.get("context")

    if isinstance(context, dict):
        titles = context.get("title") or context.get("titles") or []
        sentence_groups = context.get("sentences") or []
        return [
            (str(title), [str(sentence) for sentence in sentences])
            for title, sentences in zip(titles, sentence_groups)
        ]

    if isinstance(context, list):
        items = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, sentences = item[0], item[1]
                items.append((str(title), [str(sentence) for sentence in sentences]))
        return items

    raise ValueError("Unsupported HotpotQA context format.")


def get_supporting_fact_pairs(example: dict[str, Any]) -> set[tuple[str, int]]:
    supporting_facts = example.get("supporting_facts") or example.get("supporting_facts_original")
    pairs: set[tuple[str, int]] = set()

    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title") or supporting_facts.get("titles") or []
        sent_ids = supporting_facts.get("sent_id") or supporting_facts.get("sent_ids") or []
        for title, sent_id in zip(titles, sent_ids):
            pairs.add((str(title), int(sent_id)))
        return pairs

    if isinstance(supporting_facts, list):
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.add((str(item[0]), int(item[1])))
        return pairs

    return pairs


def import_datasets_module():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        print("The Hugging Face datasets package is not installed.", file=sys.stderr)
        print("Install it with: python3 -m pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)

    return load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small HotpotQA benchmark sample for the RAG hallucination study.")
    parser.add_argument("--dataset", default="hotpotqa/hotpot_qa", help="Hugging Face dataset name.")
    parser.add_argument("--config", default="distractor", help="HotpotQA dataset config.")
    parser.add_argument("--split", default="validation", help="Dataset split to use.")
    parser.add_argument("--limit", type=int, default=50, help="Number of examples to export.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output folder.")
    args = parser.parse_args()

    load_dataset = import_datasets_module()
    dataset = load_dataset(args.dataset, args.config, split=args.split)

    source_dir = args.output_dir / "source_documents"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "hotpotqa_sample.md"
    questions_path = args.output_dir / "questions.csv"
    metadata_path = args.output_dir / "metadata.csv"

    question_rows: list[dict[str, str]] = []
    metadata_rows: list[dict[str, str]] = []
    source_lines = ["# HotpotQA Sample Source Sentences", ""]
    sentence_counter = 1

    for example_index, example in enumerate(dataset.select(range(min(args.limit, len(dataset)))), start=1):
        question_id = f"HQ{example_index:04d}"
        sentence_id_by_support_key: dict[tuple[str, int], str] = {}
        source_ids_for_question: list[str] = []

        for title, sentences in get_context_items(example):
            for sentence_index, sentence_text in enumerate(sentences):
                sentence_id = f"S{sentence_counter:06d}"
                sentence_counter += 1
                sentence_id_by_support_key[(title, sentence_index)] = sentence_id
                source_ids_for_question.append(sentence_id)
                source_lines.append(f"[{sentence_id}] ({question_id}; {title}) {sentence_text}")

        support_pairs = get_supporting_fact_pairs(example)
        expected_citations = [
            sentence_id
            for key, sentence_id in sentence_id_by_support_key.items()
            if key in support_pairs
        ]

        question_rows.append(
            {
                "question_id": question_id,
                "question": str(example.get("question", "")),
                "expected_answer": str(example.get("answer", "")),
                "expected_citation": " ".join(expected_citations),
                "answerable": "yes",
            }
        )
        metadata_rows.append(
            {
                "question_id": question_id,
                "hotpotqa_id": str(example.get("id") or example.get("_id") or ""),
                "type": str(example.get("type", "")),
                "level": str(example.get("level", "")),
                "source_sentence_ids": " ".join(source_ids_for_question),
                "supporting_sentence_ids": " ".join(expected_citations),
            }
        )

    source_path.write_text("\n\n".join(source_lines) + "\n", encoding="utf-8")
    write_csv(
        questions_path,
        question_rows,
        ["question_id", "question", "expected_answer", "expected_citation", "answerable"],
    )
    write_csv(
        metadata_path,
        metadata_rows,
        ["question_id", "hotpotqa_id", "type", "level", "source_sentence_ids", "supporting_sentence_ids"],
    )

    print(f"Wrote source document: {source_path}")
    print(f"Wrote questions: {questions_path}")
    print(f"Wrote metadata: {metadata_path}")
    print("Next dry run:")
    print(
        "python3 src/main.py "
        "--questions data/hotpotqa/questions.csv "
        "--source-dir data/hotpotqa/source_documents "
        "--dry-run --verbose --limit 1"
    )


if __name__ == "__main__":
    main()
