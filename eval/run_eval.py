"""
Runs the fixed evaluation question set through the pipeline and reports:
- Precision/recall on detecting UNANSWERABLE questions (the core metric)
- Overall flag rate
- Per-question breakdown

Run with:
    python eval/run_eval.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag_pipeline import RAGPipeline  # noqa: E402

EVAL_FILE = os.path.join(os.path.dirname(__file__), "eval_questions.json")


def main():
    with open(EVAL_FILE, "r") as f:
        eval_set = json.load(f)

    pipeline = RAGPipeline()

    true_positives = 0   # correctly flagged an unanswerable question
    false_negatives = 0  # unanswerable question NOT flagged (hallucination risk)
    true_negatives = 0   # correctly did NOT flag an answerable question
    false_positives = 0  # answerable question incorrectly flagged as uncertain

    print(f"Running {len(eval_set)} evaluation questions...\n")
    print(f"{'Question':<65} {'Expected':<12} {'Flagged':<10} {'Confidence'}")
    print("-" * 100)

    for item in eval_set:
        result = pipeline.answer_query(item["question"])
        flagged = result["flagged_uncertain"]
        expected_answerable = item["answerable"]

        if not expected_answerable and flagged:
            true_positives += 1
        elif not expected_answerable and not flagged:
            false_negatives += 1
        elif expected_answerable and not flagged:
            true_negatives += 1
        elif expected_answerable and flagged:
            false_positives += 1

        print(
            f"{item['question'][:63]:<65} "
            f"{'answerable' if expected_answerable else 'unanswerable':<12} "
            f"{'YES' if flagged else 'no':<10} "
            f"{result['confidence']:.0f}"
        )

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 50)
    print("RESULTS: Detecting unanswerable questions")
    print("=" * 50)
    print(f"True positives (correctly flagged unanswerable): {true_positives}")
    print(f"False negatives (missed unanswerable, hallucinated): {false_negatives}")
    print(f"True negatives (correctly answered answerable): {true_negatives}")
    print(f"False positives (wrongly flagged answerable): {false_positives}")
    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 score:  {f1:.2f}")


if __name__ == "__main__":
    main()
