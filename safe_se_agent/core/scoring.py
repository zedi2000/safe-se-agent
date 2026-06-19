from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from safe_se_agent.core.text import strip_think_blocks
from safe_se_agent.core.types import Task


@dataclass(frozen=True)
class ScoreResult:
    correct: bool
    normalized_prediction: str
    normalized_gold: str
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


def score_answer(prediction: str, task: Task, raw_response: str | None = None) -> ScoreResult:
    """Score one model answer with lightweight dataset-aware normalization."""

    text = raw_response if raw_response is not None else prediction
    if _is_multiple_choice_task(task):
        return _score_multiple_choice(prediction, task, text)
    if _is_tool_task(task):
        return _score_tool_answer(prediction, task, text)
    return _score_numeric_or_text(prediction, task.answer)


def normalize_answer(value: str) -> str:
    numeric = numeric_value(value)
    if numeric is not None:
        return format_decimal(numeric)
    return normalize_text(value)


def numeric_value(value: str) -> Decimal | None:
    text = strip_think_blocks(str(value)).strip().replace(",", "")
    matches = re.findall(r"[-+]?\$?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    token = matches[-1].replace("$", "")
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


def format_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(int(normalized))
    return format(normalized, "f").rstrip("0").rstrip(".")


def normalize_text(value: str) -> str:
    text = strip_think_blocks(str(value)).strip().lower()
    text = re.sub(r"`+", "", text)
    text = re.sub(r"[*_]+", "", text)
    text = text.strip()
    text = text.strip(string.whitespace + string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_choice_label(value: str, valid_labels: set[str] | None = None) -> str | None:
    labels = valid_labels or set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    text = strip_think_blocks(str(value))
    text = re.sub(r"`+", " ", text)
    text = re.sub(r"[*_]+", " ", text)
    text = text.replace("答案", "answer")
    candidates: list[str] = []
    patterns = [
        r"(?:final\s+answer|answer|option|choice|the\s+answer\s+is)\s*[:：\-]?\s*\(?\s*([A-Z])\s*\)?",
        r"(?:^|\n|\s)\(?\s*([A-Z])\s*\)?\s*(?:[.)。:：]|$)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            label = match.group(1).upper()
            if label in labels:
                candidates.append(label)
    stripped = re.sub(r"[^A-Za-z]", "", text).upper()
    if len(stripped) == 1 and stripped in labels:
        candidates.append(stripped)
    return candidates[-1] if candidates else None


def count_tool_steps(text: str | None) -> int:
    if not text:
        return 0
    patterns = [
        r"<\s*invoke\b",
        r"\bAction\s*:",
        r"\bAction\s+Input\s*:",
        r"\btool_call\b",
        r"\bfunction_call\b",
        r"<\s*tool_call\b",
    ]
    counts = [len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns]
    action_pairs = max(counts[1], counts[2])
    return max(counts[0], action_pairs, counts[3], counts[4], counts[5])


def expected_tool_steps(task: Task) -> int | None:
    steps = task.metadata.get("intermediate_steps")
    if isinstance(steps, list):
        return len(steps)
    expected = task.metadata.get("expected_tool_step_count")
    if isinstance(expected, int):
        return expected
    return None


def _score_numeric_or_text(prediction: str, gold: str) -> ScoreResult:
    pred_numeric = numeric_value(prediction)
    gold_numeric = numeric_value(gold)
    if pred_numeric is not None and gold_numeric is not None:
        pred_norm = format_decimal(pred_numeric)
        gold_norm = format_decimal(gold_numeric)
        return ScoreResult(
            correct=pred_norm == gold_norm,
            normalized_prediction=pred_norm,
            normalized_gold=gold_norm,
            method="numeric",
        )
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    return ScoreResult(
        correct=pred_norm == gold_norm,
        normalized_prediction=pred_norm,
        normalized_gold=gold_norm,
        method="text",
    )


def _score_multiple_choice(prediction: str, task: Task, raw_text: str) -> ScoreResult:
    options = _option_labels(task)
    labels = set(options) or set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    gold = extract_choice_label(task.answer, labels) or normalize_text(task.answer).upper()
    pred = extract_choice_label(prediction, labels) or extract_choice_label(raw_text, labels)
    answer_text = task.metadata.get("answer_text")
    if pred is None and isinstance(answer_text, str):
        pred_text = normalize_text(prediction)
        if pred_text == normalize_text(answer_text):
            pred = gold
    return ScoreResult(
        correct=pred == gold,
        normalized_prediction=pred or normalize_text(prediction),
        normalized_gold=gold,
        method="multiple_choice",
    )


def _score_tool_answer(prediction: str, task: Task, raw_text: str) -> ScoreResult:
    base = _score_numeric_or_text(prediction, task.answer)
    if not base.correct:
        raw_base = _score_text_similarity(raw_text, task.answer)
        if raw_base.correct:
            base = raw_base
    if not base.correct:
        final_thought = task.metadata.get("final_thought")
        if isinstance(final_thought, str):
            alt = _score_text_similarity(prediction, final_thought)
            if alt.correct:
                base = alt
            else:
                raw_alt = _score_text_similarity(raw_text, final_thought)
                if raw_alt.correct:
                    base = raw_alt
    actual_steps = count_tool_steps(raw_text)
    expected_steps = expected_tool_steps(task)
    metadata: dict[str, Any] = {
        "tool_step_count": actual_steps,
        "expected_tool_step_count": expected_steps,
        "tool_step_delta": actual_steps - expected_steps if expected_steps is not None else None,
    }
    return ScoreResult(
        correct=base.correct,
        normalized_prediction=base.normalized_prediction,
        normalized_gold=base.normalized_gold,
        method="tool_final_answer",
        metadata=metadata,
    )


def _score_text_similarity(prediction: str, gold: str) -> ScoreResult:
    pred_norm = normalize_text(prediction)
    gold_norm = normalize_text(gold)
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    overlap = len(pred_tokens & gold_tokens) / len(gold_tokens) if gold_tokens else 0.0
    return ScoreResult(
        correct=pred_norm == gold_norm or overlap >= 0.8,
        normalized_prediction=pred_norm,
        normalized_gold=gold_norm,
        method="text_similarity",
        metadata={"answer_token_recall": overlap},
    )


def _is_multiple_choice_task(task: Task) -> bool:
    task_type = str(task.metadata.get("task_type", "")).lower()
    dataset = str(task.metadata.get("dataset", "")).lower()
    return "multiple_choice" in task_type or dataset == "medqa" or "medqa" in task.tags


def _is_tool_task(task: Task) -> bool:
    task_type = str(task.metadata.get("task_type", "")).lower()
    dataset = str(task.metadata.get("dataset", "")).lower()
    return "tool" in task_type or dataset == "toolalpaca" or "tool_use" in task.tags


def _option_labels(task: Task) -> tuple[str, ...]:
    raw = task.metadata.get("raw")
    if isinstance(raw, dict):
        options = raw.get("options")
        if isinstance(options, dict):
            return tuple(str(key).upper() for key in options)
    options = task.metadata.get("options")
    if isinstance(options, dict):
        return tuple(str(key).upper() for key in options)
    return ()
