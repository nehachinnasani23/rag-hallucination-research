from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from main import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_INDEX_PATH,
    cosine_similarity,
    create_embeddings,
    keyword_retrieve,
    load_source_sentences,
    load_vector_index,
    read_csv,
    validate_vector_index,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = ROOT / "data" / "hotpotqa" / "questions.csv"
DEFAULT_SOURCE_DIR = ROOT / "data" / "hotpotqa" / "source_documents"
DEFAULT_ANSWERS_PATH = ROOT / "data" / "hotpotqa_vector_answers_raw.csv"
OUTPUT_DIR = ROOT / "outputs"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split() if item.strip()]


def saved_rankings(answers_path: Path) -> dict[str, list[str]]:
    rows = read_csv(answers_path)
    return {
        row["question_id"]: parse_ids(row.get("top_source_ids", ""))
        for row in rows
        if row.get("question_id")
    }


def keyword_rankings(
    questions: list[dict[str, str]],
    sentences: dict[str, str],
    max_k: int,
) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for question in questions:
        retrieved = keyword_retrieve(question["question"], sentences, max_k)
        rankings[question["question_id"]] = [sentence_id for sentence_id, _, _ in retrieved]
    return rankings


def vector_rankings(
    questions: list[dict[str, str]],
    sentences: dict[str, str],
    source_dir: Path,
    vector_index_path: Path,
    max_k: int,
    embedding_model: str,
    api_key: str,
) -> dict[str, list[str]]:
    vector_index = load_vector_index(vector_index_path)
    validate_vector_index(vector_index, sentences, vector_index_path, source_dir)

    question_texts = [row["question"] for row in questions]
    question_embeddings = create_embeddings(question_texts, embedding_model, api_key)

    rankings: dict[str, list[str]] = {}
    for question, question_embedding in zip(questions, question_embeddings):
        scored_sentences = []
        for row in vector_index:
            sentence_id = str(row["sentence_id"])
            sentence_embedding = [float(value) for value in row["embedding"]]
            score = cosine_similarity(question_embedding, sentence_embedding)
            scored_sentences.append((sentence_id, score))

        scored_sentences.sort(key=lambda item: (-item[1], item[0]))
        rankings[question["question_id"]] = [
            sentence_id for sentence_id, _ in scored_sentences[:max_k]
        ]

    return rankings


def calculate_rows(
    questions: list[dict[str, str]],
    rankings: dict[str, list[str]],
    ks: list[int],
) -> list[dict[str, object]]:
    detail_rows: list[dict[str, object]] = []

    for question in questions:
        expected_ids = parse_ids(question.get("expected_citation", ""))
        retrieved_ids = rankings.get(question["question_id"], [])
        expected_set = set(expected_ids)

        row: dict[str, object] = {
            "question_id": question["question_id"],
            "answerable": question.get("answerable", ""),
            "expected_citation": " ".join(expected_ids),
            "top_source_ids": " ".join(retrieved_ids),
            "expected_count": len(expected_ids),
        }

        for k in ks:
            retrieved_at_k = set(retrieved_ids[:k])
            found_ids = [source_id for source_id in expected_ids if source_id in retrieved_at_k]
            missing_ids = [source_id for source_id in expected_ids if source_id not in retrieved_at_k]
            recall = len(found_ids) / len(expected_ids) if expected_ids else 0.0

            row[f"found_at_{k}"] = " ".join(found_ids)
            row[f"missing_at_{k}"] = " ".join(missing_ids)
            row[f"recall_at_{k}"] = f"{recall:.3f}"
            row[f"any_gold_at_{k}"] = "yes" if expected_set & retrieved_at_k else "no"
            row[f"all_gold_at_{k}"] = "yes" if expected_set and expected_set <= retrieved_at_k else "no"

        detail_rows.append(row)

    return detail_rows


def percent(numerator: float, denominator: float) -> str:
    if denominator == 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.1f}%"


def calculate_summary(detail_rows: list[dict[str, object]], ks: list[int]) -> list[dict[str, object]]:
    answerable_rows = [row for row in detail_rows if str(row.get("answerable", "")).lower() == "yes"]
    total = len(answerable_rows)
    summary_rows: list[dict[str, object]] = []

    for k in ks:
        any_count = sum(row[f"any_gold_at_{k}"] == "yes" for row in answerable_rows)
        all_count = sum(row[f"all_gold_at_{k}"] == "yes" for row in answerable_rows)
        recall_sum = sum(float(row[f"recall_at_{k}"]) for row in answerable_rows)
        summary_rows.append(
            {
                "k": k,
                "question_count": total,
                "any_gold_count": any_count,
                "any_gold_rate": percent(any_count, total),
                "all_gold_count": all_count,
                "all_gold_rate": percent(all_count, total),
                "mean_gold_citation_recall": percent(recall_sum, total),
            }
        )

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether retrieved passages contain the expected supporting citations."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS_PATH)
    parser.add_argument(
        "--retrieval",
        choices=("saved", "keyword", "vector"),
        default="saved",
        help="Use saved top_source_ids, recompute keyword rankings, or recompute vector rankings.",
    )
    parser.add_argument("--vector-index", type=Path, default=DEFAULT_VECTOR_INDEX_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--output-details", type=Path, default=OUTPUT_DIR / "retrieval_quality.csv")
    parser.add_argument("--output-summary", type=Path, default=OUTPUT_DIR / "retrieval_quality_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ks = sorted(set(args.ks))
    max_k = max(ks)

    questions = read_csv(args.questions)
    sentences = load_source_sentences(args.source_dir)

    if args.retrieval == "saved":
        rankings = saved_rankings(args.answers)
        shortest_ranking = min((len(source_ids) for source_ids in rankings.values()), default=0)
        if shortest_ranking < max_k:
            print(
                "Warning: saved rankings contain fewer passages than the largest requested k. "
                "Use --retrieval vector to recompute larger rankings from the vector index."
            )
    elif args.retrieval == "keyword":
        rankings = keyword_rankings(questions, sentences, max_k)
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for vector retrieval quality.")
        rankings = vector_rankings(
            questions,
            sentences,
            args.source_dir,
            args.vector_index,
            max_k,
            args.embedding_model,
            api_key,
        )

    detail_rows = calculate_rows(questions, rankings, ks)
    summary_rows = calculate_summary(detail_rows, ks)

    detail_fields = [
        "question_id",
        "answerable",
        "expected_citation",
        "top_source_ids",
        "expected_count",
    ]
    for k in ks:
        detail_fields.extend(
            [
                f"found_at_{k}",
                f"missing_at_{k}",
                f"recall_at_{k}",
                f"any_gold_at_{k}",
                f"all_gold_at_{k}",
            ]
        )

    write_csv(args.output_details, detail_rows, detail_fields)
    write_csv(
        args.output_summary,
        summary_rows,
        [
            "k",
            "question_count",
            "any_gold_count",
            "any_gold_rate",
            "all_gold_count",
            "all_gold_rate",
            "mean_gold_citation_recall",
        ],
    )

    print(f"Wrote retrieval details to: {args.output_details}")
    print(f"Wrote retrieval summary to: {args.output_summary}")
    print()
    print("Retrieval quality summary")
    print("=========================")
    for row in summary_rows:
        print(
            f"k={row['k']}: "
            f"any gold={row['any_gold_count']}/{row['question_count']} ({row['any_gold_rate']}), "
            f"all gold={row['all_gold_count']}/{row['question_count']} ({row['all_gold_rate']}), "
            f"mean citation recall={row['mean_gold_citation_recall']}"
        )


if __name__ == "__main__":
    main()
