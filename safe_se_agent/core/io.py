from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from safe_se_agent.core.types import Task


def load_jsonl_tasks(path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            try:
                task = Task(
                    id=str(raw["id"]),
                    question=str(raw["question"]),
                    answer=str(raw["answer"]),
                    tags=tuple(raw.get("tags", ())),
                    metadata=dict(raw.get("metadata", {})),
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing required field {exc}") from exc
            tasks.append(task)
    return tasks


def dump_jsonl_tasks(tasks: Iterable[Task], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(
                json.dumps(
                    {
                        "id": task.id,
                        "question": task.question,
                        "answer": task.answer,
                        "tags": list(task.tags),
                        "metadata": task.metadata,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
