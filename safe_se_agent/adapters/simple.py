from __future__ import annotations

import re
import time
from pathlib import Path

from safe_se_agent.adapters.base import AgentAdapter
from safe_se_agent.core.langchain_memory import LangChainMemoryStore
from safe_se_agent.core.memory import JsonlMemoryBackend, MemoryIdFactory, MemoryStore
from safe_se_agent.core.scoring import normalize_answer, score_answer
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
        memory_backend: str = "simple",
        embedding_model: str | None = None,
        retrieval_search_type: str = "similarity_score_threshold",
        retrieval_score_threshold: float = 0.35,
    ) -> None:
        self.llm = llm or OfflineLLMClient()
        self.retrieve_k = retrieve_k
        self.memory_backend = memory_backend
        self.embedding_model = embedding_model
        self.retrieval_search_type = retrieval_search_type
        self.retrieval_score_threshold = retrieval_score_threshold
        self.memory_path_override = Path(memory_path) if memory_path else None
        self.memory_root = Path(memory_root)
        self.memory_path = self.memory_path_override or self.memory_root / "default" / "memory.jsonl"
        self.memory = self._build_memory_store(self.memory_path)
        self.last_retrieval_scores: list[dict[str, object]] = []
        self.ids = MemoryIdFactory()
        self.run_id = "default"

    def reset(self, run_id: str) -> None:
        self.run_id = run_id
        self.memory_path = self.memory_path_override or self.memory_root / run_id / "memory.jsonl"
        self.memory = self._build_memory_store(self.memory_path)
        self.memory.clear()
        self.last_retrieval_scores = []
        self.ids = MemoryIdFactory(prefix=f"{run_id}_mem")

    def resume(self, run_id: str) -> None:
        self.run_id = run_id
        self.memory_path = self.memory_path_override or self.memory_root / run_id / "memory.jsonl"
        self.memory = self._build_memory_store(self.memory_path)
        self.last_retrieval_scores = []
        self.ids = MemoryIdFactory(prefix=f"{run_id}_mem")
        self._prime_id_factory()

    def solve(self, task: Task, memories: list[MemoryEntry] | None = None) -> RunResult:
        start = time.perf_counter()
        selected_memories = memories if memories is not None else self.retrieve(task, self.retrieve_k)
        retrieval_scores = self._scores_for_selected_memories(selected_memories)
        self._set_next_retrieval_scores(retrieval_scores)
        solve_output = self.llm.solve(task, selected_memories)
        if len(solve_output) == 4:
            answer, reasoning, token_count, raw_response = solve_output
        else:
            answer, reasoning, token_count = solve_output
            raw_response = reasoning
        score = score_answer(answer, task, raw_response=raw_response)
        latency_s = time.perf_counter() - start
        score_metadata = {
            "normalized_predicted_answer": score.normalized_prediction,
            "normalized_gold_answer": score.normalized_gold,
            "score_method": score.method,
            "retrieval_backend": self.memory_backend,
            "retrieval_scores": retrieval_scores,
            "retrieved_memory_scores": self._compact_retrieval_scores(retrieval_scores),
            **score.metadata,
        }
        trajectory = Trajectory(
            task=task,
            answer=answer,
            correct=score.correct,
            reasoning=reasoning,
            raw_response=raw_response,
            retrieved_memory_ids=tuple(memory.id for memory in selected_memories),
            expected_answer=task.answer,
            metadata=score_metadata.copy(),
        )
        return RunResult(
            task_id=task.id,
            answer=answer,
            correct=score.correct,
            reasoning=reasoning,
            trajectory=trajectory,
            raw_response=raw_response,
            retrieved_memory_ids=trajectory.retrieved_memory_ids,
            token_count=token_count,
            latency_s=latency_s,
            steps=1,
            metadata=score_metadata,
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
        return normalize_answer(value)

    def add_memory(self, entries: list[MemoryEntry]) -> None:
        self.memory.add(entries)

    def retrieve(self, task: Task, k: int = 3) -> list[MemoryEntry]:
        memories = self.memory.retrieve(task, k=k)
        self.last_retrieval_scores = list(getattr(self.memory, "last_retrieval_scores", []))
        return memories

    def export_memory(self) -> list[MemoryEntry]:
        return self.memory.export()

    def get_memory_path(self) -> Path:
        return self.memory_path

    def _prime_id_factory(self) -> None:
        prefix = re.escape(self.ids.prefix)
        pattern = re.compile(rf"^{prefix}_(?P<tag>.+)_(?P<count>\d+)$")
        for memory in self.memory.export():
            match = pattern.match(memory.id)
            if not match:
                continue
            tag = match.group("tag")
            count = int(match.group("count"))
            self.ids.counts[tag] = max(self.ids.counts[tag], count)

    def _build_memory_store(self, memory_path: Path):
        if self.memory_backend == "simple":
            return MemoryStore(JsonlMemoryBackend(memory_path))
        if self.memory_backend == "langchain":
            return LangChainMemoryStore(
                memory_path,
                embedding_model=self.embedding_model,
                search_type=self.retrieval_search_type,
                score_threshold=self.retrieval_score_threshold,
            )
        raise ValueError(f"Unknown memory backend: {self.memory_backend}")

    def _set_next_retrieval_scores(self, retrieval_scores: list[dict[str, object]]) -> None:
        setter = getattr(self.llm, "set_next_retrieval_scores", None)
        if callable(setter):
            setter(retrieval_scores)

    def _scores_for_selected_memories(self, memories: list[MemoryEntry]) -> list[dict[str, object]]:
        if self.memory_backend != "langchain":
            return []
        if not memories:
            return list(self.last_retrieval_scores)
        selected_ids = [memory.id for memory in memories]
        score_ids = [str(item.get("memory_id")) for item in self.last_retrieval_scores]
        if score_ids[: len(selected_ids)] != selected_ids:
            return []
        return [item for item in self.last_retrieval_scores if str(item.get("memory_id")) in set(selected_ids)]

    def _compact_retrieval_scores(self, retrieval_scores: list[dict[str, object]]) -> list[dict[str, object]]:
        compact: list[dict[str, object]] = []
        for item in retrieval_scores:
            if not item.get("retrieved", True):
                continue
            compact.append(
                {
                    "memory_id": item.get("memory_id"),
                    "score": item.get("score"),
                }
            )
        return compact
