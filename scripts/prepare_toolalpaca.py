#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TOOLALPACA_URL = "https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/main/data/train_data.json"
DATASET_NAME = "toolalpaca"
DOMAIN = "tool_use"
TASK_TYPE = "tool_use_instruction"
SOURCE = "tangqiaoyu/ToolAlpaca"
DEFAULT_TAGS = [DOMAIN, DATASET_NAME]


def download_file(url: str, path: Path, force: bool = False, retries: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return path
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
                with tmp_path.open(mode) as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            tmp_path.replace(path)
            return path
        except Exception:
            if attempt == retries:
                raise
    return path


def load_raw_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [item for item in data if isinstance(item, dict)]


def flatten_instances(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tool_index, tool in enumerate(tools):
        instances = tool.get("Instances") or tool.get("instances") or []
        if not isinstance(instances, list):
            continue
        for instance_index, instance in enumerate(instances):
            if not isinstance(instance, dict):
                continue
            rows.append({"tool": tool, "instance": instance, "tool_index": tool_index, "instance_index": instance_index})
    return rows


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [rows[index] for index in indices]


def format_question(tool: dict[str, Any], instance: dict[str, Any]) -> str:
    tool_name = str(tool.get("Name") or tool.get("name") or "unknown_tool")
    category = str(tool.get("Category") or tool.get("category") or "unknown")
    description = str(tool.get("Description") or tool.get("description") or "")
    functions = str(tool.get("Functions") or tool.get("functions") or "")
    user_request = str(instance.get("input") or instance.get("instruction") or instance.get("query") or "").strip()
    blocks = [
        f"Tool: {tool_name}",
        f"Category: {category}",
        f"Description: {description}",
    ]
    if functions:
        blocks.append(f"Available functions:\n{functions}")
    blocks.append(f"User request:\n{user_request}")
    return "\n\n".join(blocks)


def convert_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        tool = row["tool"]
        instance = row["instance"]
        tool_name = str(tool.get("Name") or tool.get("name") or "unknown_tool")
        category = str(tool.get("Category") or tool.get("category") or "unknown")
        answer = str(instance.get("output") or instance.get("answer") or instance.get("Final Thought") or "").strip()
        converted.append(
            {
                "id": f"toolalpaca_{split}_{index:04d}",
                "question": format_question(tool, instance),
                "answer": answer,
                "tags": list(DEFAULT_TAGS),
                "metadata": {
                    "dataset": DATASET_NAME,
                    "domain": DOMAIN,
                    "task_type": TASK_TYPE,
                    "source": SOURCE,
                    "split": split,
                    "sample_index": index,
                    "tool_name": tool_name,
                    "category": category,
                    "intermediate_steps": instance.get("intermediate_steps"),
                    "final_thought": instance.get("Final Thought"),
                    "kind": DATASET_NAME,
                },
            }
        )
    return converted


def split_rows(rows: list[dict[str, Any]], train_limit: int, eval_limit: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    train_count = train_limit if train_limit > 0 else max(len(shuffled) - eval_limit, 0)
    eval_count = eval_limit if eval_limit > 0 else max(len(shuffled) - train_count, 0)
    return shuffled[:train_count], shuffled[train_count : train_count + eval_count]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare small ToolAlpaca JSONL files for Milestone 1.")
    parser.add_argument("--limit-train", type=int, default=20)
    parser.add_argument("--limit-eval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "raw" / "toolalpaca"))
    parser.add_argument("--train-out", default=str(ROOT / "data" / "toolalpaca_train_small.jsonl"))
    parser.add_argument("--eval-out", default=str(ROOT / "data" / "toolalpaca_eval_small.jsonl"))
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_path = download_file(TOOLALPACA_URL, raw_dir / "train_data.json", force=args.force_download)
    rows = flatten_instances(load_raw_json(raw_path))
    train_rows, eval_rows = split_rows(rows, args.limit_train, args.limit_eval, args.seed)
    converted_train = convert_rows(train_rows, "train")
    converted_eval = convert_rows(eval_rows, "eval")
    write_jsonl(Path(args.train_out), converted_train)
    write_jsonl(Path(args.eval_out), converted_eval)

    print("ToolAlpaca 数据准备完成")
    print(f"- train: {len(converted_train)} -> {args.train_out}")
    print(f"- eval: {len(converted_eval)} -> {args.eval_out}")
    print(f"- raw_cache: {raw_dir}")


if __name__ == "__main__":
    main()
