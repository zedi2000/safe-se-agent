from __future__ import annotations

import re
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from safe_se_agent.adapters.base import AgentAdapter
from safe_se_agent.core.memory import JsonlMemoryBackend, MemoryIdFactory, MemoryStore
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
        answer, reasoning, token_count = self.llm.solve(task, selected_memories)
        correct = self._normalize(answer) == self._normalize(task.answer)
        latency_s = time.perf_counter() - start
        trajectory = Trajectory(
            task=task,
            answer=answer,
            correct=correct,
            reasoning=reasoning,
            retrieved_memory_ids=tuple(memory.id for memory in selected_memories),
            expected_answer=task.answer,
        )
        return RunResult(
            task_id=task.id,
            answer=answer,
            correct=correct,
            reasoning=reasoning,
            trajectory=trajectory,
            retrieved_memory_ids=trajectory.retrieved_memory_ids,
            token_count=token_count,
            latency_s=latency_s,
            steps=1,
        )

    def reflect(self, trajectories: list[Trajectory]) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        for group in self._reflection_groups(trajectories):
            rule_texts = self.llm.reflect(group)
            for text in rule_texts:
                tags = self._infer_tags(text, group)
                source_ids = self._source_ids_for_rule(tags, group)
                tag_for_id = next((tag for tag in tags if tag != "arithmetic"), None) or "reflection"
                entries.append(
                    MemoryEntry(
                        id=self.ids.next(tag_for_id),
                        text=text,
                        source="reflection",
                        tags=tags,
                        priority=1.0,
                        created_from=source_ids,
                        stats={"run_id": self.run_id, "num_trajectories": len(group)},
                    )
                )
        return entries

    def _reflection_groups(self, trajectories: list[Trajectory]) -> list[list[Trajectory]]:
        grouped: dict[str, list[Trajectory]] = defaultdict(list)
        for trajectory in trajectories:
            kind = trajectory.task.metadata.get("kind")
            key = str(kind) if kind else "__all__"
            grouped[key].append(trajectory)
        return [grouped[key] for key in sorted(grouped)]

    def _infer_tags(self, text: str, trajectories: list[Trajectory]) -> tuple[str, ...]:
        text_lower = text.lower()
        tags: set[str] = set()
        for trajectory in trajectories:
            tags.update(trajectory.task.tags)
            kind = trajectory.task.metadata.get("kind")
            if isinstance(kind, str):
                tags.add(kind)
        if "fee" in text_lower:
            tags.add("total_with_fee")
        if "discount" in text_lower or "tax" in text_lower:
            tags.add("discount_then_tax")
        if "convert" in text_lower or "target unit" in text_lower:
            tags.add("unit_conversion")
        if "average" in text_lower:
            tags.add("average_with_extra_item")
        return tuple(sorted(tags))

    def _source_ids_for_rule(
        self,
        tags: tuple[str, ...],
        trajectories: list[Trajectory],
    ) -> tuple[str, ...]:
        specific_tags = {tag for tag in tags if tag != "arithmetic"}
        if not specific_tags:
            return tuple(trajectory.task.id for trajectory in trajectories)
        return tuple(
            trajectory.task.id
            for trajectory in trajectories
            if specific_tags & set(trajectory.task.tags)
        )

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
