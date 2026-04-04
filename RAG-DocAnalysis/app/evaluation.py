"""Evaluation suite: accuracy, latency, and faithfulness (anti-hallucination).

Metrics
-------
accuracy      – token-overlap score: fraction of expected key-terms found in the answer
faithfulness  – lexical grounding score: fraction of answer content-words found in
                the retrieved context (high = answer is grounded, low = potential hallucination)
latency_ms    – wall-clock time for a single RAG query in milliseconds

The test set covers all three sample documents (5 questions each, 15 total).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import DEFAULT_PRESET, DEFAULT_TEMPLATE, EVAL_RESULTS_DIR
from app.rag_chain import query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ground-truth test set (15 Q/A pairs – 5 per sample document)
# ---------------------------------------------------------------------------
TEST_SET: list[dict[str, str]] = [
    # ── company_policy.txt ─────────────────────────────────────────────────
    {
        "id": "pol_01",
        "question": "How many days of annual leave do TechVault employees receive per year?",
        "expected": "25 days",
        "source_hint": "company_policy.txt",
    },
    {
        "id": "pol_02",
        "question": "How many days per week can TechVault employees work remotely?",
        "expected": "3 days per week",
        "source_hint": "company_policy.txt",
    },
    {
        "id": "pol_03",
        "question": "When does health insurance coverage begin at TechVault Inc.?",
        "expected": "after 90 days",
        "source_hint": "company_policy.txt",
    },
    {
        "id": "pol_04",
        "question": "What must employees return when they leave TechVault?",
        "expected": "laptop",
        "source_hint": "company_policy.txt",
    },
    {
        "id": "pol_05",
        "question": "What approval is required to work remotely at TechVault?",
        "expected": "manager approval",
        "source_hint": "company_policy.txt",
    },
    # ── ai_glossary.txt ────────────────────────────────────────────────────
    {
        "id": "ai_01",
        "question": "What does RAG stand for in the context of AI?",
        "expected": "Retrieval-Augmented Generation",
        "source_hint": "ai_glossary.txt",
    },
    {
        "id": "ai_02",
        "question": "What is a Large Language Model?",
        "expected": "neural network trained on large amounts of text data",
        "source_hint": "ai_glossary.txt",
    },
    {
        "id": "ai_03",
        "question": "What is hallucination in the context of AI models?",
        "expected": "generating false or fabricated information not supported by facts",
        "source_hint": "ai_glossary.txt",
    },
    {
        "id": "ai_04",
        "question": "What are embeddings in machine learning?",
        "expected": "numerical vector representations of text",
        "source_hint": "ai_glossary.txt",
    },
    {
        "id": "ai_05",
        "question": "What is fine-tuning in AI?",
        "expected": "adapting a pre-trained model to a specific task or domain",
        "source_hint": "ai_glossary.txt",
    },
    # ── python_guide.txt ───────────────────────────────────────────────────
    {
        "id": "py_01",
        "question": "What tool does the Python guide recommend for managing project dependencies?",
        "expected": "virtual environments using venv or conda",
        "source_hint": "python_guide.txt",
    },
    {
        "id": "py_02",
        "question": "What is the preferred string formatting method in Python?",
        "expected": "f-strings",
        "source_hint": "python_guide.txt",
    },
    {
        "id": "py_03",
        "question": "What testing framework is recommended in the Python best practices guide?",
        "expected": "pytest",
        "source_hint": "python_guide.txt",
    },
    {
        "id": "py_04",
        "question": "What Python feature is recommended for defining data structures?",
        "expected": "dataclasses",
        "source_hint": "python_guide.txt",
    },
    {
        "id": "py_05",
        "question": "What should be used instead of print statements in production Python code?",
        "expected": "logging module",
        "source_hint": "python_guide.txt",
    },
]

# ---------------------------------------------------------------------------
# Stopwords excluded from lexical metrics
# ---------------------------------------------------------------------------
_STOPWORDS: frozenset[str] = frozenset(
    "a an the is in it of and or to for are was be this that with as at by "
    "from on has have do does not but what which who how when where why can "
    "will would should i you we they he she its".split()
)


def _content_tokens(text: str) -> set[str]:
    """Lower-cased word tokens with stopwords and very short tokens removed."""
    raw = re.findall(r"\b\w+\b", text.lower())
    return {t for t in raw if t not in _STOPWORDS and len(t) > 2}


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def evaluate_accuracy(answer: str, expected: str) -> float:
    """
    Keyword recall: fraction of expected content-tokens found in the answer.
    Range [0.0, 1.0].  1.0 means every key term from the expected answer appears.
    """
    answer_tokens = _content_tokens(answer)
    expected_tokens = _content_tokens(expected)
    if not expected_tokens:
        return 0.0
    overlap = answer_tokens & expected_tokens
    return round(len(overlap) / len(expected_tokens), 3)


def evaluate_faithfulness(answer: str, context: str) -> float:
    """
    Lexical grounding: fraction of answer content-tokens that are present in
    the retrieved context.  High score → answer is grounded (faithful).
    Low score → answer introduces information not in the context (potential hallucination).
    Range [0.0, 1.0].
    """
    answer_tokens = _content_tokens(answer)
    context_tokens = _content_tokens(context)
    if not answer_tokens:
        return 0.0
    grounded = answer_tokens & context_tokens
    return round(len(grounded) / len(answer_tokens), 3)


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    template_name: str = DEFAULT_TEMPLATE,
    preset_name: str = DEFAULT_PRESET,
    save_results: bool = True,
) -> dict[str, Any]:
    """
    Run the complete test set and return per-question results plus aggregate metrics.

    Parameters
    ----------
    template_name : which prompt template to use (concise / detailed / conversational)
    preset_name   : which LLM preset to use (precise / creative)
    save_results  : if True, write JSON to EVAL_RESULTS_DIR
    """
    per_question: list[dict[str, Any]] = []
    total_accuracy = 0.0
    total_faithfulness = 0.0
    total_latency = 0.0

    for item in TEST_SET:
        logger.info("Evaluating [%s]: %s", item["id"], item["question"])

        result = query(
            question=item["question"],
            template_name=template_name,
            preset_name=preset_name,
        )

        accuracy = evaluate_accuracy(result["answer"], item["expected"])
        faithfulness = evaluate_faithfulness(result["answer"], result["context"])

        row: dict[str, Any] = {
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected"],
            "answer": result["answer"],
            "sources": result["sources"],
            "accuracy": accuracy,
            "faithfulness": faithfulness,
            "latency_ms": result["latency_ms"],
            "tokens_used": result["tokens_used"],
        }
        per_question.append(row)

        total_accuracy += accuracy
        total_faithfulness += faithfulness
        total_latency += result["latency_ms"]

        logger.info(
            "  [%s] accuracy=%.2f faithfulness=%.2f latency=%.0f ms",
            item["id"], accuracy, faithfulness, result["latency_ms"],
        )

    n = len(per_question)
    summary: dict[str, Any] = {
        "template": template_name,
        "preset": preset_name,
        "num_questions": n,
        "avg_accuracy": round(total_accuracy / n, 3),
        "avg_faithfulness": round(total_faithfulness / n, 3),
        "avg_latency_ms": round(total_latency / n, 2),
        "results": per_question,
    }

    if save_results:
        out_dir = Path(EVAL_RESULTS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"eval_{template_name}_{preset_name}.json"
        out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Evaluation results saved to '%s'", out_file)

    return summary
