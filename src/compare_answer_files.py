from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rag_experiment import calculate_metrics, read_csv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = ROOT / "data" / "hotpotqa" / "questions.csv"
DEFAULT_OUTPUT_PATH = ROOT / "outputs" / "answer_file_comparison.csv"


def infer_provider(model: str) -> str:
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


def parse_answer_file(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path

    label, path_value = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Answer file labels cannot be blank.")
    return label, Path(path_value)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare labeled answer CSV files in one summary table."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument(
        "--answer-file",
        action="append",
        type=parse_answer_file,
        required=True,
        help="Answer file to compare. Use label=path or just path.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = read_csv(args.questions)
    rows: list[dict[str, object]] = []

    for label, answer_path in args.answer_file:
        if not answer_path.exists():
            print(f"Skipping missing file: {answer_path}")
            continue

        answer_rows = read_csv(answer_path)
        provider = next((row.get("provider", "") for row in answer_rows if row.get("provider")), "")
        summary_rows = calculate_metrics(answer_rows, questions)
        for summary in summary_rows:
            rows.append(
                {
                    "experiment": label,
                    "answer_file": str(answer_path),
                    "provider": provider or infer_provider(str(summary["model_name"])),
                    **summary,
                }
            )

    fieldnames = [
        "experiment",
        "answer_file",
        "provider",
        "model_name",
        "setting",
        "generated_answers",
        "total_questions",
        "answer_correct_count",
        "answer_accuracy",
        "answer_score_points",
        "answer_score",
        "citation_correct_count",
        "citation_evaluated_count",
        "citation_accuracy",
        "citation_score_points",
        "citation_score",
        "hallucination_count",
        "hallucination_rate",
        "refusal_correct_count",
        "unanswerable_evaluated_count",
        "refusal_accuracy",
        "false_answer_count",
        "false_answer_rate",
        "unsupported_citation_count",
        "unsupported_citation_rate",
    ]
    write_csv(args.output, rows, fieldnames)

    print(f"Wrote comparison summary to: {args.output}")
    for row in rows:
        print(
            f"{row['experiment']}: "
            f"answer={row['answer_accuracy']}, "
            f"citation={row['citation_accuracy']}, "
            f"hallucination={row['hallucination_rate']}"
        )


if __name__ == "__main__":
    main()
