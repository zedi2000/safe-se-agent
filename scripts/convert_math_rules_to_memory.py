#!/usr/bin/env python3
"""Convert Attack_generator/rules/math_rules.json rules into MemoryEntry format."""

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from safe_se_agent.core.types import MemoryEntry

RULES_PATH = Path(__file__).parent.parent / "Attack_generator" / "rules" / "math_rules.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "math_rules_memory.json"


def rule_to_memory(rule: dict, index: int) -> dict:
    domain = rule["domain"]
    method = rule["method"]

    return {
        "id": f"oep_{domain}_{method}_mem_{index + 1}",
        "text": rule["target_rule"],
        "source": "oep_attack",
        "tags": [domain, method, "oep_attack"],
        "priority": 1.0,
        "created_from": [],
        "stats": {
            "run_id": f"oep_{domain}_reflection",
            "num_trajectories": 0,
            "policy": "batch",
            "attack_domain": domain,
            "num_attack_cases": 0,
            "attack_group_ids": [f"{domain}_{method}"],
            "window_task_ids": [],
            "method": method,
            "locally_correct_when": rule.get("locally_correct_when", ""),
            "harmful_when_generalized": rule.get("harmful_when_generalized", ""),
            "example_failure": rule.get("example_failure", ""),
            "generation_notes": rule.get("generation_notes", ""),
        },
    }


def main() -> None:
    rules = json.loads(RULES_PATH.read_text())
    memories = [rule_to_memory(rule, i) for i, rule in enumerate(rules)]

    # Validate with MemoryEntry
    validated = []
    for m in memories:
        entry = MemoryEntry(
            id=m["id"],
            text=m["text"],
            source=m["source"],
            tags=tuple(m["tags"]),
            priority=m["priority"],
            created_from=tuple(m["created_from"]),
            stats=m["stats"],
        )
        validated.append(entry)

    # Write JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for entry in validated:
            f.write(json.dumps(dataclasses.asdict(entry), ensure_ascii=False) + "\n")

    print(f"Converted {len(validated)} rules → {OUTPUT_PATH}")
    for entry in validated:
        print(f"  - {entry.id}  [{', '.join(entry.tags)}]")


if __name__ == "__main__":
    main()
