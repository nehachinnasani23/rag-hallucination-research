from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = ROOT / "data" / "hotpotqa" / "questions.csv"
DEFAULT_SAVED_RETRIEVAL_PATH = ROOT / "data" / "hotpotqa_vector_answers_raw.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "hotpotqa_unanswerable"
REFUSAL_ANSWER = "The document does not provide enough information."


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a missing-evidence/unanswerable test set from answerable HotpotQA questions."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--saved-retrieval-answers", type=Path, default=DEFAULT_SAVED_RETRIEVAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = read_csv(args.questions)
    saved_retrieval_rows = read_csv(args.saved_retrieval_answers)

    if args.limit is not None:
        questions = questions[: args.limit]
        saved_retrieval_rows = saved_retrieval_rows[: args.limit]

    if len(questions) < 2:
        raise ValueError("At least two questions are required to create shifted distractor context.")
    if len(saved_retrieval_rows) < len(questions):
        raise ValueError(
            "Saved retrieval answer file has fewer rows than the requested question set."
        )

    question_rows: list[dict[str, str]] = []
    saved_rows: list[dict[str, str]] = []
    metadata_rows: list[dict[str, str]] = []

    for index, question in enumerate(questions):
        new_question_id = f"UQ{index + 1:04d}"
        donor_index = (index + 1) % len(questions)
        donor_question = questions[donor_index]
        donor_retrieval = saved_retrieval_rows[donor_index]

        question_rows.append(
            {
                "question_id": new_question_id,
                "question": question["question"],
                "expected_answer": REFUSAL_ANSWER,
                "expected_citation": "",
                "answerable": "no",
            }
        )
        saved_rows.append(
            {
                "question_id": new_question_id,
                "retrieval_method": "shifted_saved_vector",
                "donor_question_id": donor_question["question_id"],
                "top_source_ids": donor_retrieval.get("top_source_ids", ""),
            }
        )
        metadata_rows.append(
            {
                "question_id": new_question_id,
                "original_question_id": question["question_id"],
                "donor_question_id": donor_question["question_id"],
                "original_expected_answer": question.get("expected_answer", ""),
                "original_expected_citation": question.get("expected_citation", ""),
                "missing_evidence_design": "The context uses saved vector passages from the next question.",
            }
        )

    write_csv(
        args.output_dir / "questions.csv",
        question_rows,
        ["question_id", "question", "expected_answer", "expected_citation", "answerable"],
    )
    write_csv(
        args.output_dir / "saved_retrieval.csv",
        saved_rows,
        ["question_id", "retrieval_method", "donor_question_id", "top_source_ids"],
    )
    write_csv(
        args.output_dir / "metadata.csv",
        metadata_rows,
        [
            "question_id",
            "original_question_id",
            "donor_question_id",
            "original_expected_answer",
            "original_expected_citation",
            "missing_evidence_design",
        ],
    )

    print(f"Wrote unanswerable questions to: {args.output_dir / 'questions.csv'}")
    print(f"Wrote saved distractor retrieval to: {args.output_dir / 'saved_retrieval.csv'}")
    print(f"Wrote metadata to: {args.output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
