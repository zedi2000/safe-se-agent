from __future__ import annotations

import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from safe_se_agent.adapters.base import AgentAdapter
from safe_se_agent.core.memory import JsonlMemoryBackend, MemoryIdFactory, MemoryStore
from safe_se_agent.core.text import strip_think_blocks
from safe_se_agent.core.types import MemoryEntry, RunResult, Task, Trajectory
from safe_se_agent.llm.base import LLMClient
from safe_se_agent.llm.offline import OfflineLLMClient


class SimpleAgentAdapter(AgentAdapter):
    """Milestone 1 的最小 Simple Agent。

    memory 对 Simple Agent 来说是 adapter 内部状态；外部框架后续可以在
    adapter 内桥接 native memory，或使用本项目 JSONL memory 作为 sidecar。
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        retrieve_k: int = 3,
        memory_path: str | Path | None = None,
        memory_root: str | Path = "runs",
    ) -> None:
        self.llm = llm or OfflineLLMClient()
        self.retrieve_k = retrieve_k
        self.memory_path_override = Path(memory_path) if memory_path else None
        self.memory_root = Path(memory_root)
        self.memory_path = self.memory_path_override or self.memory_root / "default" / "memory.jsonl"
        self.memory = MemoryStore(JsonlMemoryBackend(self.memory_path))
        self.ids = MemoryIdFactory()
        self.run_id = "default"

    def reset(self, run_id: str) -> None:
        self.run_id = run_id
        self.memory_path = self.memory_path_override or self.memory_root / run_id / "memory.jsonl"
        self.memory = MemoryStore(JsonlMemoryBackend(self.memory_path))
        self.memory.clear()
        self.ids = MemoryIdFactory(prefix=f"{run_id}_mem")

    def solve(self, task: Task, memories: list[MemoryEntry] | None = None) -> RunResult:
        start = time.perf_counter()
        selected_memories = memories if memories is not None else self.retrieve(task, self.retrieve_k)
        solve_output = self.llm.solve(task, selected_memories)
        if len(solve_output) == 4:
            answer, reasoning, token_count, raw_response = solve_output
        else:
            answer, reasoning, token_count = solve_output
            raw_response = reasoning
        correct = self._normalize(answer) == self._normalize(task.answer)
        latency_s = time.perf_counter() - start
        trajectory = Trajectory(
            task=task,
            answer=answer,
            correct=correct,
            reasoning=reasoning,
            raw_response=raw_response,
            retrieved_memory_ids=tuple(memory.id for memory in selected_memories),
            expected_answer=task.answer,
        )
        return RunResult(
            task_id=task.id,
            answer=answer,
            correct=correct,
            reasoning=reasoning,
            trajectory=trajectory,
            raw_response=raw_response,
            retrieved_memory_ids=trajectory.retrieved_memory_ids,
            token_count=token_count,
            latency_s=latency_s,
            steps=1,
        )

    def reflect(self, trajectories: list[Trajectory]) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        rule_texts = self.llm.reflect(trajectories)
        for text in rule_texts:
            text = strip_think_blocks(text).strip()
            if not text:
                continue
            tags = self._infer_tags(text, trajectories)
            source_ids = self._source_ids_for_rule(tags, trajectories)
            tag_for_id = (
                next((tag for tag in tags if tag != "arithmetic"), None)
                if len(source_ids) == 1
                else None
            ) or "reflection"
            entries.append(
                MemoryEntry(
                    id=self.ids.next(tag_for_id),
                    text=text,
                    source="reflection",
                    tags=tags,
                    priority=1.0,
                    created_from=source_ids,
                    stats={"run_id": self.run_id, "num_trajectories": len(trajectories)},
                )
            )
        return entries

    def _infer_tags(self, text: str, trajectories: list[Trajectory]) -> tuple[str, ...]:
        tags: set[str] = set()
        for trajectory in trajectories:
            tags.update(trajectory.task.tags)
            kind = trajectory.task.metadata.get("kind")
            if isinstance(kind, str):
                tags.add(kind)
        return tuple(sorted(tags))

    def _source_ids_for_rule(
        self,
        tags: tuple[str, ...],
        trajectories: list[Trajectory],
    ) -> tuple[str, ...]:
        return tuple(trajectory.task.id for trajectory in trajectories)

    def _normalize(self, value: str) -> str:
        numeric = self._numeric_value(value)
        if numeric is not None:
            normalized = numeric.normalize()
            if normalized == normalized.to_integral():
                return str(int(normalized))
            return format(normalized, "f").rstrip("0").rstrip(".")
        return value.strip().lower().rstrip(".")

    def _numeric_value(self, value: str) -> Decimal | None:
        text = value.strip().replace(",", "")
        matches = re.findall(r"[-+]?\$?\d+(?:\.\d+)?", text)
        if not matches:
            return None
        token = matches[-1].replace("$", "")
        try:
            return Decimal(token)
        except InvalidOperation:
            return None

    def add_memory(self, entries: list[MemoryEntry]) -> None:
        self.memory.add(entries)

    def retrieve(self, task: Task, k: int = 3) -> list[MemoryEntry]:
        return self.memory.retrieve(task, k=k)

    def export_memory(self) -> list[MemoryEntry]:
        return self.memory.export()

    def get_memory_path(self) -> Path:
        return self.memory_path
