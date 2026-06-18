#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_se_agent.adapters.simple import SimpleAgentAdapter
from safe_se_agent.core.io import load_jsonl_tasks


def main() -> None:
    adapter = SimpleAgentAdapter()
    adapter.reset("inspect")
    trajectories = [adapter.solve(task, memories=[]).trajectory for task in load_jsonl_tasks("data/m1_train.jsonl")]
    memories = adapter.reflect(trajectories)
    adapter.add_memory(memories)
    for memory in adapter.export_memory():
        print(f"{memory.id}\t{','.join(memory.tags)}\t{memory.text}")


if __name__ == "__main__":
    main()
