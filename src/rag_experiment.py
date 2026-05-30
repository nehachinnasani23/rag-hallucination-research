from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_documents"
QUESTIONS_PATH = ROOT / "data" / "questions.csv"
OUTPUT_DIR = ROOT / "outputs"

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


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in STOPWORDS}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
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


def retrieve_context(question: str, sentences: dict[str, str], top_k: int) -> list[tuple[str, str, int]]:
    question_tokens = tokenize(question)
    scored_sentences = []

    for sentence_id, sentence_text in sentences.items():
        sentence_tokens = tokenize(sentence_text)
        score = len(question_tokens & sentence_tokens)
        scored_sentences.append((sentence_id, sentence_text, score))

    scored_sentences.sort(key=lambda item: (-item[2], item[0]))
    return scored_sentences[:top_k]


def gold_retrieve_context(question_row: dict[str, str], sentences: dict[str, str]) -> list[tuple[str, str, int]]:
    citation_ids = [
        citation_id.strip()
        for citation_id in question_row.get("expected_citation", "").split()
        if citation_id.strip()
    ]
    retrieved = []

    for rank, citation_id in enumerate(citation_ids):
        sentence_text = sentences.get(citation_id)
        if sentence_text:
            score = len(citation_ids) - rank
            retrieved.append((citation_id, sentence_text, score))

    if retrieved:
        return retrieved

    return []


def saved_retrieve_context(
    question_row: dict[str, str],
    sentences: dict[str, str],
    saved_top_source_ids_by_question: dict[str, str],
) -> list[tuple[str, str, int]]:
    source_ids = [
        source_id.strip()
        for source_id in saved_top_source_ids_by_question.get(question_row["question_id"], "").split()
        if source_id.strip()
    ]
    retrieved = []

    for rank, source_id in enumerate(source_ids):
        sentence_text = sentences.get(source_id)
        if sentence_text:
            score = len(source_ids) - rank
            retrieved.append((source_id, sentence_text, score))

    return retrieved


def retrieve_for_question(
    question_row: dict[str, str],
    sentences: dict[str, str],
    top_k: int,
    retrieval_method: str,
    saved_top_source_ids_by_question: dict[str, str] | None = None,
) -> list[tuple[str, str, int]]:
    if retrieval_method == "gold":
        retrieved = gold_retrieve_context(question_row, sentences)
        if retrieved:
            return retrieved
    if retrieval_method == "saved" and saved_top_source_ids_by_question is not None:
        return saved_retrieve_context(question_row, sentences, saved_top_source_ids_by_question)
    return retrieve_context(question_row["question"], sentences, top_k)


def create_retrieval_file(
    questions: list[dict[str, str]],
    sentences: dict[str, str],
    top_k: int,
    retrieval_method: str,
    saved_top_source_ids_by_question: dict[str, str] | None = None,
) -> Path:
    rows = []

    for question_row in questions:
        retrieved = retrieve_for_question(
            question_row,
            sentences,
            top_k,
            retrieval_method,
            saved_top_source_ids_by_question,
        )
        rows.append(
            {
                "question_id": question_row["question_id"],
                "question": question_row["question"],
                "top_source_ids": " ".join(sentence_id for sentence_id, _, _ in retrieved),
                "retrieved_context": " ".join(f"[{sentence_id}] {text}" for sentence_id, text, _ in retrieved),
            }
        )

    output_path = OUTPUT_DIR / "retrieved_context.csv"
    write_csv(output_path, rows, ["question_id", "question", "top_source_ids", "retrieved_context"])
    return output_path


def create_prompt_file(
    questions: list[dict[str, str]],
    sentences: dict[str, str],
    top_k: int,
    retrieval_method: str,
    saved_top_source_ids_by_question: dict[str, str] | None = None,
) -> Path:
    rows = []

    for question_row in questions:
        retrieved = retrieve_for_question(
            question_row,
            sentences,
            top_k,
            retrieval_method,
            saved_top_source_ids_by_question,
        )
        retrieved_context = " ".join(f"[{sentence_id}] {text}" for sentence_id, text, _ in retrieved)

        for setting in SETTINGS:
            rows.append(
                {
                    "question_id": question_row["question_id"],
                    "setting": setting,
                    "question": question_row["question"],
                    "prompt": build_prompt(setting, question_row["question"], retrieved_context),
                }
            )

    output_path = OUTPUT_DIR / "experiment_prompts.csv"
    write_csv(output_path, rows, ["question_id", "setting", "question", "prompt"])
    return output_path


def is_yes(value: str) -> bool:
    return value.strip().lower() == "yes"


def is_no(value: str) -> bool:
    return value.strip().lower() == "no"


def is_evaluable(value: str) -> bool:
    return value.strip().lower() in {"yes", "no", "partial"}


def label_score(value: str) -> float:
    normalized = value.strip().lower()
    if normalized == "yes":
        return 1.0
    if normalized == "partial":
        return 0.5
    return 0.0


def format_points(value: float) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.1f}"


def percent(numerator: float, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.1f}%"


def print_retrieval_flow(
    questions: list[dict[str, str]],
    sentences: dict[str, str],
    top_k: int,
    source_dir: Path,
    questions_path: Path,
    retrieval_method: str,
    saved_top_source_ids_by_question: dict[str, str] | None = None,
) -> None:
    print("\nStep 1: Load source sentences")
    print("=============================")
    print(f"Source folder: {source_dir}")
    for sentence_id, sentence_text in sorted(sentences.items()):
        print(f"- [{sentence_id}] {sentence_text}")

    print("\nStep 2: Read questions and retrieve context")
    print("==========================================")
    print(f"Questions file: {questions_path}")
    print(f"Retrieval method: {retrieval_method}, top_k={top_k}")
    for question_row in questions:
        retrieved = retrieve_for_question(
            question_row,
            sentences,
            top_k,
            retrieval_method,
            saved_top_source_ids_by_question,
        )
        print(f"\nQuestion ID: {question_row['question_id']}")
        print(f"Question: {question_row['question']}")
        print(f"Expected answer from CSV: {question_row['expected_answer']}")
        print(f"Expected citation from CSV: {question_row['expected_citation'] or 'n/a'}")
        print(f"Answerable from CSV: {question_row['answerable']}")
        print("Retrieved context:")
        for sentence_id, sentence_text, score in retrieved:
            print(f"  - [{sentence_id}] score={score}: {sentence_text}")


def print_label_effect(row: dict[str, str], answerable: bool) -> None:
    answer_correct = row.get("answer_correct", "")
    citation_correct = row.get("citation_correct", "")
    hallucination = row.get("hallucination", "")
    refusal_correct = row.get("refusal_correct", "")

    if is_evaluable(answer_correct) and is_evaluable(hallucination):
        print("Count impact:")
        print("  - total_questions: +1 because answer_correct and hallucination labels are filled")
        if is_yes(answer_correct):
            print("  - answer_correct_count: +1 because answer_correct=yes")
            print("  - answer_score_points: +1.0 because answer_correct=yes")
        elif answer_correct.strip().lower() == "partial":
            print("  - answer_correct_count: +0 because answer_correct=partial")
            print("  - answer_score_points: +0.5 because answer_correct=partial")
        else:
            print("  - answer_correct_count: +0 because answer_correct=no")
            print("  - answer_score_points: +0.0 because answer_correct=no")

        if is_yes(hallucination):
            print("  - hallucination_count: +1 because hallucination=yes")
        else:
            print("  - hallucination_count: +0 because hallucination is not yes")
    else:
        print("Count impact:")
        print("  - not scored yet because answer_correct and/or hallucination label is blank or invalid")

    if is_evaluable(citation_correct):
        print("  - citation_evaluated_count: +1 because citation_correct is yes/no/partial")
        if is_yes(citation_correct):
            print("  - citation_correct_count: +1 because citation_correct=yes")
            print("  - citation_score_points: +1.0 because citation_correct=yes")
        elif citation_correct.strip().lower() == "partial":
            print("  - citation_correct_count: +0 because citation_correct=partial")
            print("  - citation_score_points: +0.5 because citation_correct=partial")
        else:
            print(f"  - citation_correct_count: +0 because citation_correct={citation_correct}")
            print("  - citation_score_points: +0.0 because citation_correct=no")
    else:
        print("  - citation counts: ignored because citation_correct is blank or na")

    if not answerable and is_evaluable(refusal_correct):
        print("  - unanswerable_evaluated_count: +1 because this question is answerable=no")
        if is_yes(refusal_correct):
            print("  - refusal_correct_count: +1 because refusal_correct=yes")
        else:
            print(f"  - refusal_correct_count: +0 because refusal_correct={refusal_correct}")
    elif answerable:
        print("  - refusal counts: ignored because this question is answerable=yes")
    else:
        print("  - refusal counts: ignored because refusal_correct is blank or na")


def print_evaluation_flow(answer_rows: list[dict[str, str]], questions: list[dict[str, str]]) -> None:
    questions_by_id = {question["question_id"]: question for question in questions}

    print("\nStep 3: Read model answers and human labels")
    print("===========================================")
    print("Important: the script does not automatically decide correctness.")
    print("It displays expected_answer and model answer for human review, then uses the label columns from the CSV.")

    sorted_answer_rows = sorted(
        answer_rows,
        key=lambda row: (
            row.get("model_name", ""),
            row.get("setting", "sample"),
            row.get("question_id", ""),
        ),
    )
    for row in sorted_answer_rows:
        question = questions_by_id.get(row["question_id"])
        if not question:
            print(f"\nQuestion ID {row['question_id']} was not found in questions.csv. Skipping walkthrough.")
            continue

        answerable = is_yes(question["answerable"])
        print("\n----------------------------------------")
        print(f"Question ID: {row['question_id']}")
        print(f"Model: {row.get('model_name', '')}")
        print(f"Setting: {row.get('setting') or 'sample'}")
        print(f"Question from questions.csv: {question['question']}")
        print(f"Expected answer from questions.csv: {question['expected_answer']}")
        print(f"Expected citation from questions.csv: {question['expected_citation'] or 'n/a'}")
        print(f"Answerable from questions.csv: {question['answerable']}")
        print(f"Model answer from answers CSV: {row.get('answer', '')}")
        print(f"Model cited_source from answers CSV: {row.get('cited_source', '') or 'n/a'}")
        print("Human labels from answers CSV:")
        print(f"  answer_correct={row.get('answer_correct', '') or 'blank'}")
        print(f"  citation_correct={row.get('citation_correct', '') or 'blank'}")
        print(f"  hallucination={row.get('hallucination', '') or 'blank'}")
        print(f"  refusal_correct={row.get('refusal_correct', '') or 'blank'}")
        print_label_effect(row, answerable)


def print_formula_flow(summary_rows: list[dict[str, object]]) -> None:
    print("\nStep 4: Calculate final percentages")
    print("===================================")
    for row in summary_rows:
        print(f"\nModel: {row['model_name']} | Setting: {row['setting']}")
        if row["total_questions"] == 0:
            print("No scored rows yet, so percentages cannot be calculated.")
            continue
        print(
            "Strict answer accuracy = "
            f"{row['answer_correct_count']} / {row['total_questions']} = {row['answer_accuracy']}"
        )
        print(
            "Partial-credit answer score = "
            f"{row['answer_score_points']} / {row['total_questions']} = {row['answer_score']}"
        )
        print(
            "Strict citation accuracy = "
            f"{row['citation_correct_count']} / {row['citation_evaluated_count']} = {row['citation_accuracy']}"
        )
        print(
            "Partial-credit citation score = "
            f"{row['citation_score_points']} / {row['citation_evaluated_count']} = {row['citation_score']}"
        )
        print(
            "Hallucination rate = "
            f"{row['hallucination_count']} / {row['total_questions']} = {row['hallucination_rate']}"
        )
        print(
            "Refusal accuracy = "
            f"{row['refusal_correct_count']} / {row['unanswerable_evaluated_count']} = {row['refusal_accuracy']}"
        )


def calculate_metrics(
    answer_rows: list[dict[str, str]], questions: list[dict[str, str]]
) -> list[dict[str, object]]:
    answerable_by_question = {
        question["question_id"]: is_yes(question["answerable"]) for question in questions
    }

    rows_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in answer_rows:
        model_name = row["model_name"]
        setting = row.get("setting") or "sample"
        rows_by_model[f"{model_name}::{setting}"].append(row)

    summary_rows: list[dict[str, object]] = []

    for model_and_setting, model_rows in sorted(rows_by_model.items()):
        model_name, setting = model_and_setting.split("::", 1)
        scored_rows = [
            row
            for row in model_rows
            if is_evaluable(row.get("answer_correct", ""))
            and is_evaluable(row.get("hallucination", ""))
        ]
        total = len(scored_rows)
        answer_correct_count = sum(is_yes(row["answer_correct"]) for row in scored_rows)
        answer_score_points = sum(label_score(row["answer_correct"]) for row in scored_rows)
        hallucination_count = sum(is_yes(row["hallucination"]) for row in scored_rows)

        citation_rows = [row for row in scored_rows if is_evaluable(row.get("citation_correct", ""))]
        citation_correct_count = sum(is_yes(row["citation_correct"]) for row in citation_rows)
        citation_score_points = sum(label_score(row["citation_correct"]) for row in citation_rows)

        refusal_rows = [
            row
            for row in scored_rows
            if not answerable_by_question.get(row["question_id"], True)
            and is_evaluable(row.get("refusal_correct", ""))
        ]
        refusal_correct_count = sum(is_yes(row["refusal_correct"]) for row in refusal_rows)
        false_answer_count = sum(not is_yes(row["refusal_correct"]) for row in refusal_rows)
        unsupported_citation_count = sum(
            row.get("citation_correct", "").strip().lower() == "no"
            for row in refusal_rows
        )

        summary_rows.append(
            {
                "model_name": model_name,
                "setting": setting,
                "generated_answers": len(model_rows),
                "total_questions": total,
                "answer_correct_count": answer_correct_count,
                "answer_accuracy": percent(answer_correct_count, total),
                "answer_score_points": format_points(answer_score_points),
                "answer_score": percent(answer_score_points, total),
                "citation_correct_count": citation_correct_count,
                "citation_evaluated_count": len(citation_rows),
                "citation_accuracy": percent(citation_correct_count, len(citation_rows)),
                "citation_score_points": format_points(citation_score_points),
                "citation_score": percent(citation_score_points, len(citation_rows)),
                "hallucination_count": hallucination_count,
                "hallucination_rate": percent(hallucination_count, total),
                "refusal_correct_count": refusal_correct_count,
                "unanswerable_evaluated_count": len(refusal_rows),
                "refusal_accuracy": percent(refusal_correct_count, len(refusal_rows)),
                "false_answer_count": false_answer_count,
                "false_answer_rate": percent(false_answer_count, len(refusal_rows)),
                "unsupported_citation_count": unsupported_citation_count,
                "unsupported_citation_rate": percent(unsupported_citation_count, len(refusal_rows)),
            }
        )

    return summary_rows


def print_summary(summary_rows: list[dict[str, object]]) -> None:
    if not summary_rows:
        print("No model answer rows found. Retrieval file was created, but metrics were not calculated.")
        return

    print("\nResults summary")
    print("===============")
    for row in summary_rows:
        print(f"\nModel: {row['model_name']}")
        print(f"Setting: {row['setting']}")
        if row["total_questions"] == 0:
            print(f"- Generated answers: {row['generated_answers']}")
            print("- No human labels found yet. Fill answer_correct, citation_correct, hallucination, and refusal_correct before scoring.")
            continue
        print(f"- Strict answer accuracy: {row['answer_correct_count']}/{row['total_questions']} ({row['answer_accuracy']})")
        print(f"- Partial-credit answer score: {row['answer_score_points']}/{row['total_questions']} ({row['answer_score']})")
        print(
            "- Strict citation accuracy: "
            f"{row['citation_correct_count']}/{row['citation_evaluated_count']} ({row['citation_accuracy']})"
        )
        print(
            "- Partial-credit citation score: "
            f"{row['citation_score_points']}/{row['citation_evaluated_count']} ({row['citation_score']})"
        )
        print(f"- Hallucination rate: {row['hallucination_count']}/{row['total_questions']} ({row['hallucination_rate']})")
        print(
            "- Refusal accuracy: "
            f"{row['refusal_correct_count']}/{row['unanswerable_evaluated_count']} ({row['refusal_accuracy']})"
        )
        print(
            "- False answer rate on unanswerable questions: "
            f"{row['false_answer_count']}/{row['unanswerable_evaluated_count']} ({row['false_answer_rate']})"
        )
        print(
            "- Unsupported citation rate on unanswerable questions: "
            f"{row['unsupported_citation_count']}/{row['unanswerable_evaluated_count']} "
            f"({row['unsupported_citation_rate']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first RAG hallucination experiment scaffold.")
    parser.add_argument(
        "--answers",
        type=Path,
        default=ROOT / "data" / "model_answers_sample.csv",
        help="CSV file containing model answers and human evaluation labels.",
    )
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH, help="Question CSV file.")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR, help="Folder containing source document markdown files.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of source sentences to retrieve per question.")
    parser.add_argument(
        "--retrieval",
        choices=["keyword", "gold", "saved"],
        default="keyword",
        help=(
            "Retrieval display/generation method. Use gold when answers were generated "
            "with benchmark supporting facts. Use saved to inspect top_source_ids already "
            "stored in a main.py answers CSV, including vector runs."
        ),
    )
    parser.add_argument(
        "--skip-prompts",
        action="store_true",
        help="Do not write the prompt file for no-RAG, basic-RAG, and strict-citation-RAG settings.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a step-by-step walkthrough of retrieval, labels, and metric calculations.",
    )
    args = parser.parse_args()

    questions = read_csv(args.questions)
    sentences = load_source_sentences(args.source_dir)

    if not args.answers.exists():
        print(f"Answer file not found: {args.answers}")
        return

    answer_rows = read_csv(args.answers)
    saved_top_source_ids_by_question = {
        row["question_id"]: row.get("top_source_ids", "")
        for row in answer_rows
        if row.get("question_id") and row.get("top_source_ids")
    }

    if args.verbose:
        print_retrieval_flow(
            questions,
            sentences,
            args.top_k,
            args.source_dir,
            args.questions,
            args.retrieval,
            saved_top_source_ids_by_question,
        )

    retrieval_path = create_retrieval_file(
        questions,
        sentences,
        args.top_k,
        args.retrieval,
        saved_top_source_ids_by_question,
    )
    print(f"Wrote retrieved context to: {retrieval_path}")

    if not args.skip_prompts:
        prompt_path = create_prompt_file(
            questions,
            sentences,
            args.top_k,
            args.retrieval,
            saved_top_source_ids_by_question,
        )
        print(f"Wrote experiment prompts to: {prompt_path}")
    if args.verbose:
        print_evaluation_flow(answer_rows, questions)

    summary_rows = calculate_metrics(answer_rows, questions)
    summary_path = OUTPUT_DIR / "results_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
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
        ],
    )

    print(f"Wrote results summary to: {summary_path}")
    if args.verbose:
        print_formula_flow(summary_rows)
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
