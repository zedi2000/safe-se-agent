#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


GSM8K_BASE_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data"
)
ANSWER_RE = re.compile(r"####\s*(.+)\s*$")


def parse_final_answer(answer_text: str) -> str:
    match = ANSWER_RE.search(answer_text.strip())
    if not match:
        raise ValueError(f"Cannot find GSM8K final answer marker: {answer_text[:80]!r}")
    return normalize_answer(match.group(1))


def normalize_answer(value: str) -> str:
    text = value.strip().replace(",", "")
    if text.endswith(".0"):
        text = text[:-2]
    return text


def download_split(split: str, raw_dir: Path, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{split}.jsonl"
    if path.exists() and not force:
        return path
    url = f"{GSM8K_BASE_URL}/{split}.jsonl"
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def load_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSONL") from exc
    return rows


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[index] for index in indices]


def convert_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = str(row["question"])
        answer_text = str(row["answer"])
        converted.append(
            {
                "id": f"gsm8k_{split}_{index:04d}",
                "question": question,
                "answer": parse_final_answer(answer_text),
                "tags": ["gsm8k", "math_word_problem"],
                "metadata": {
                    "kind": "gsm8k",
                    "dataset": "gsm8k",
                    "source": "openai/grade-school-math",
                    "split": split,
                    "rationale": answer_text,
                },
            }
        )
    return converted


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare small GSM8K JSONL files for Milestone 1.")
    parser.add_argument("--limit-train", type=int, default=20)
    parser.add_argument("--limit-eval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "gsm8k"))
    parser.add_argument("--train-out", default=str(ROOT / "data" / "gsm8k_train_small.jsonl"))
    parser.add_argument("--eval-out", default=str(ROOT / "data" / "gsm8k_eval_small.jsonl"))
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    train_raw = download_split("train", raw_dir, force=args.force_download)
    test_raw = download_split("test", raw_dir, force=args.force_download)

    train_rows = sample_rows(load_raw_jsonl(train_raw), args.limit_train, seed=args.seed)
    eval_rows = sample_rows(load_raw_jsonl(test_raw), args.limit_eval, seed=args.seed + 1)
    converted_train = convert_rows(train_rows, "train")
    converted_eval = convert_rows(eval_rows, "test")

    write_jsonl(Path(args.train_out), converted_train)
    write_jsonl(Path(args.eval_out), converted_eval)

    print("GSM8K 数据准备完成")
    print(f"- train: {len(converted_train)} -> {args.train_out}")
    print(f"- eval: {len(converted_eval)} -> {args.eval_out}")
    print(f"- raw_cache: {raw_dir}")


if __name__ == "__main__":
    main()
