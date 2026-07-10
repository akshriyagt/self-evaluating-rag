"""
Judge layer: takes a question, the retrieved context, and the generated
answer, and scores how well the answer is actually supported by the
context. This is the core "self-evaluating" piece of the pipeline.
"""

import json
import re

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking judge. You will be \
given a QUESTION, some CONTEXT (retrieved passages), and an ANSWER that was \
generated based on that context.

Your job: decide whether the ANSWER is fully supported by the CONTEXT.

Respond with ONLY a JSON object, no other text, no markdown fences:
{
  "grounded": true or false,
  "confidence": a number from 0 to 100,
  "reasoning": "one short sentence explaining your judgment"
}

Rules:
- If the ANSWER states something not present in the CONTEXT, grounded = false.
- If the CONTEXT does not contain enough information to answer the QUESTION \
  at all, and the ANSWER admits this (e.g. says it doesn't know), grounded = true \
  and confidence should be high, because correctly admitting uncertainty IS \
  the correct behavior.
- If the CONTEXT does not contain enough information but the ANSWER makes up \
  an answer anyway, grounded = false and confidence should be low.
"""


def build_judge_prompt(question: str, context: str, answer: str) -> str:
    return f"""QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""


def parse_judge_response(raw_text: str) -> dict:
    """Judge models sometimes wrap JSON in markdown fences despite instructions.
    Strip those defensively before parsing."""
    cleaned = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    try:
        result = json.loads(cleaned)
        return {
            "grounded": bool(result.get("grounded", False)),
            "confidence": float(result.get("confidence", 0)),
            "reasoning": result.get("reasoning", ""),
        }
    except (json.JSONDecodeError, ValueError):
        # Fail safe: if judge output can't be parsed, treat as low confidence
        # rather than silently trusting the answer.
        return {
            "grounded": False,
            "confidence": 0.0,
            "reasoning": "Could not parse judge response; treating as unverified.",
        }
