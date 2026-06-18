#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MEDQA_ZIP_URL = "https://huggingface.co/datasets/bigbio/med_qa/resolve/main/data_clean.zip"
DATASET_NAME = "medqa"
DOMAIN = "healthcare"
TASK_TYPE = "medical_multiple_choice"
SOURCE = "bigbio/med_qa"
DEFAULT_TAGS = [DOMAIN, DATASET_NAME]


def is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def download_file(url: str, path: Path, force: bool = False, retries: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force and is_valid_zip(path):
        return path
    if path.exists() and not is_valid_zip(path):
        path.replace(path.with_suffix(path.suffix + ".part"))
    tmp_path = path.with_suffix(path.suffix + ".part")
    if force and tmp_path.exists():
        tmp_path.unlink()
    for attempt in range(1, retries + 1):
        headers = {"User-Agent": "safe-se-agent-data-prep"}
        mode = "wb"
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            headers["Range"] = f"bytes={tmp_path.stat().st_size}-"
            mode = "ab"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if mode == "ab" and getattr(response, "status", None) == 200:
                    mode = "wb"
                with tmp_path.open(mode + "") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if is_valid_zip(tmp_path):
                tmp_path.replace(path)
                return path
        except Exception:
            if attempt == retries:
                raise
    if not is_valid_zip(tmp_path):
        raise RuntimeError(f"Downloaded MedQA archive is incomplete; retry later to resume: {tmp_path}")
    return path


def find_split_member(zip_path: Path, split: str) -> str:
    split_aliases = {"train": ("train",), "eval": ("test", "dev"), "test": ("test",), "dev": ("dev",)}
    aliases = split_aliases.get(split, (split,))
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    candidates = [
        name
        for name in names
        if name.endswith(".jsonl")
        and "/US/" in name
        and any(f"{alias}.jsonl" in name or f"_{alias}.jsonl" in name for alias in aliases)
    ]
    if not candidates:
        candidates = [
            name
            for name in names
            if name.endswith(".jsonl")
            and any(f"{alias}.jsonl" in name or f"_{alias}.jsonl" in name for alias in aliases)
        ]
    if not candidates:
        raise ValueError(f"Cannot find MedQA split {split!r} in {zip_path}")
    candidates.sort(key=lambda name: ("4_options" not in name, len(name), name))
    return candidates[0]


def load_split(zip_path: Path, split: str) -> list[dict[str, Any]]:
    member = find_split_member(zip_path, split)
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{member}:{line_no} is not valid JSONL") from exc
    return rows


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[index] for index in indices]


def format_question(row: dict[str, Any]) -> str:
    question = str(row.get("question") or row.get("sent1") or row.get("input") or "").strip()
    options = row.get("options") or row.get("opa")
    if isinstance(options, dict):
        option_lines = [f"{key}. {value}" for key, value in sorted(options.items())]
    elif isinstance(options, list):
        option_lines = []
        for index, value in enumerate(options):
            label = chr(ord("A") + index)
            if isinstance(value, dict):
                option_lines.append(f"{value.get('key', label)}. {value.get('value', value)}")
            else:
                option_lines.append(f"{label}. {value}")
    else:
        option_lines = []
        for label in ("A", "B", "C", "D", "E"):
            value = row.get(f"op{label.lower()}")
            if value:
                option_lines.append(f"{label}. {value}")
    if option_lines:
        return question + "\nOptions:\n" + "\n".join(option_lines)
    return question


def parse_answer(row: dict[str, Any]) -> tuple[str, str | None]:
    answer = row.get("answer_idx") or row.get("answer_id") or row.get("label")
    answer_text = row.get("answer") or row.get("answer_text")
    if answer is None and answer_text is not None:
        options = row.get("options")
        if isinstance(options, dict):
            for key, value in options.items():
                if str(value).strip() == str(answer_text).strip():
                    answer = key
                    break
    if answer is None:
        raise ValueError(f"Cannot find MedQA answer field in row keys: {sorted(row)}")
    return str(answer).strip(), str(answer_text).strip() if answer_text is not None else None


def convert_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        answer, answer_text = parse_answer(row)
        converted.append(
            {
                "id": f"medqa_{split}_{index:04d}",
                "question": format_question(row),
                "answer": answer,
                "tags": list(DEFAULT_TAGS),
                "metadata": {
                    "dataset": DATASET_NAME,
                    "domain": DOMAIN,
                    "task_type": TASK_TYPE,
                    "source": SOURCE,
                    "split": split,
                    "sample_index": index,
                    "answer_text": answer_text,
                    "raw": row,
                    "kind": DATASET_NAME,
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
    parser = argparse.ArgumentParser(description="Prepare small MedQA JSONL files for Milestone 1.")
    parser.add_argument("--limit-train", type=int, default=20)
    parser.add_argument("--limit-eval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "medqa"))
    parser.add_argument("--train-out", default=str(ROOT / "data" / "medqa_train_small.jsonl"))
    parser.add_argument("--eval-out", default=str(ROOT / "data" / "medqa_eval_small.jsonl"))
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    zip_path = download_file(MEDQA_ZIP_URL, raw_dir / "data_clean.zip", force=args.force_download)
    train_rows = sample_rows(load_split(zip_path, "train"), args.limit_train, args.seed)
    eval_rows = sample_rows(load_split(zip_path, "eval"), args.limit_eval, args.seed + 1)

    converted_train = convert_rows(train_rows, "train")
    converted_eval = convert_rows(eval_rows, "eval")
    write_jsonl(Path(args.train_out), converted_train)
    write_jsonl(Path(args.eval_out), converted_eval)

    print("MedQA 数据准备完成")
    print(f"- train: {len(converted_train)} -> {args.train_out}")
    print(f"- eval: {len(converted_eval)} -> {args.eval_out}")
    print(f"- raw_cache: {raw_dir}")


if __name__ == "__main__":
    main()
