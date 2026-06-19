#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_RAW = ROOT / "data" / "raw" / "oep" / "oep_attack_cases_90.jsonl"
DEFAULT_OUT_DIR = ROOT / "data" / "oep"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON") from exc
    return rows


def convert_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    group_seen: dict[str, int] = {}
    group_order: dict[str, int] = {}
    next_group_index: dict[str, int] = {}
    for index, row in enumerate(rows):
        domain = str(row["domain"])
        target = dict(row["target"])
        act = dict(row["act"])
        method = str(target["non_transferable_method"])
        group_key = f"{domain}_{method}"
        if group_key not in group_order:
            group_order[group_key] = next_group_index.get(domain, 0)
            next_group_index[domain] = group_order[group_key] + 1
            group_seen[group_key] = 0
        group_seen[group_key] += 1
        converted.append(
            {
                "id": str(row["case_id"]),
                "question": str(row["injection_prompt"]),
                "answer": str(act.get("local_correct_solution") or act.get("positive_reinforcement") or ""),
                "tags": [domain, "oep_attack", method],
                "metadata": {
                    "dataset": "oep",
                    "domain": domain,
                    "task_type": "oep_attack_case",
                    "case_type": row["case_type"],
                    "source": row.get("source", {}),
                    "target_rule": target["target_rule"],
                    "non_transferable_method": method,
                    "kind": method,
                    "harm_when_generalized": target.get("harm_when_generalized"),
                    "edge_case_question": act.get("edge_case_question"),
                    "local_correct_solution": act.get("local_correct_solution"),
                    "standard_or_suboptimal_solution": act.get("standard_or_suboptimal_solution"),
                    "adversarial_consequence": act.get("adversarial_consequence"),
                    "positive_reinforcement": act.get("positive_reinforcement"),
                    "injection_prompt": row["injection_prompt"],
                    "group_id": group_key,
                    "group_index": group_order[group_key],
                    "case_index_in_group": group_seen[group_key] - 1,
                    "raw_index": index,
                    "raw": row,
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
    parser = argparse.ArgumentParser(description="Prepare OEP attack cases for Simple Agent experiments.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    converted = convert_rows(load_jsonl(args.raw))
    write_jsonl(args.out_dir / "oep_attack_cases.jsonl", converted)
    for domain in sorted({row["metadata"]["domain"] for row in converted}):
        domain_rows = [row for row in converted if row["metadata"]["domain"] == domain]
        write_jsonl(args.out_dir / f"oep_{domain}_attack_cases.jsonl", domain_rows)

    print(f"Prepared {len(converted)} OEP cases in {args.out_dir}")
    for domain in sorted({row["metadata"]["domain"] for row in converted}):
        count = sum(1 for row in converted if row["metadata"]["domain"] == domain)
        print(f"- {domain}: {count}")


if __name__ == "__main__":
    main()
