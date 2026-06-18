from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from safe_se_agent.adapters.base import AgentAdapter
from safe_se_agent.core.types import MemoryEntry, RunResult, Task, Trajectory


ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    status: str
    message: str
    current: int | None = None
    total: int | None = None
    task_id: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    retrieve_k: int = 3
    progress_callback: ProgressCallback | None = None


@dataclass
class EvaluationSummary:
    run_id: str
    accuracy: float
    total: int
    correct: int
    retrieval_hit_rate: float
    memory_count: int
    results: list[RunResult]
    learned_rules: list[MemoryEntry]


class ExperimentRunner:
    """Framework-agnostic Milestone 1 runner."""

    def __init__(self, adapter: AgentAdapter, config: ExperimentConfig | None = None) -> None:
        self.adapter = adapter
        self.config = config or ExperimentConfig()

    def run_no_memory(self, eval_tasks: list[Task], run_id: str = "no_memory") -> EvaluationSummary:
        self._emit("baseline_solve", "start", "开始 no-memory baseline 推理", 0, len(eval_tasks))
        self.adapter.reset(run_id)
        results: list[RunResult] = []
        for index, task in enumerate(eval_tasks, start=1):
            self._emit(
                "baseline_solve",
                "wait",
                f"等待模型响应: {task.id}",
                index - 1,
                len(eval_tasks),
                task.id,
            )
            results.append(self.adapter.solve(task, memories=[]))
            self._emit(
                "baseline_solve",
                "progress",
                f"baseline 完成: {task.id}",
                index,
                len(eval_tasks),
                task.id,
            )
        self._emit("baseline_solve", "done", "no-memory baseline 推理完成", len(eval_tasks), len(eval_tasks))
        return self._summarize(run_id, results, learned_rules=[])

    def run_self_evolution(
        self,
        train_tasks: list[Task],
        eval_tasks: list[Task],
        run_id: str = "self_evolution",
    ) -> EvaluationSummary:
        self.adapter.reset(run_id)
        train_trajectories: list[Trajectory] = []
        self._emit("train_solve", "start", "开始训练轨迹推理", 0, len(train_tasks))
        for index, task in enumerate(train_tasks, start=1):
            self._emit(
                "train_solve",
                "wait",
                f"等待模型响应: {task.id}",
                index - 1,
                len(train_tasks),
                task.id,
            )
            result = self.adapter.solve(task, memories=[])
            train_trajectories.append(result.trajectory)
            self._emit(
                "train_solve",
                "progress",
                f"训练轨迹完成: {task.id}",
                index,
                len(train_tasks),
                task.id,
            )
        self._emit("train_solve", "done", "训练轨迹推理完成", len(train_tasks), len(train_tasks))

        self._emit("reflect", "wait", "正在反思训练轨迹并生成 memory")
        learned_rules = self.adapter.reflect(train_trajectories)
        self._emit(
            "reflect",
            "done",
            f"反思完成，生成 {len(learned_rules)} 条 memory",
            metadata={"memory_count": len(learned_rules)},
        )
        self._emit("memory_write", "wait", "正在写入 memory")
        self.adapter.add_memory(learned_rules)
        self._emit(
            "memory_write",
            "done",
            f"memory 写入完成: {len(learned_rules)} 条",
            metadata={"memory_count": len(learned_rules)},
        )

        results: list[RunResult] = []
        self._emit("self_evo_solve", "start", "开始 self-evolution 评测推理", 0, len(eval_tasks))
        for index, task in enumerate(eval_tasks, start=1):
            self._emit(
                "retrieve",
                "wait",
                f"正在检索 memory: {task.id}",
                index - 1,
                len(eval_tasks),
                task.id,
            )
            memories = self.adapter.retrieve(task, k=self.config.retrieve_k)
            self._emit(
                "retrieve",
                "done",
                f"检索完成: {task.id}",
                index,
                len(eval_tasks),
                task.id,
                {"retrieved_memory_ids": [memory.id for memory in memories]},
            )
            self._emit(
                "self_evo_solve",
                "wait",
                f"等待模型响应: {task.id}",
                index - 1,
                len(eval_tasks),
                task.id,
            )
            results.append(self.adapter.solve(task, memories=memories))
            self._emit(
                "self_evo_solve",
                "progress",
                f"self-evolution 完成: {task.id}",
                index,
                len(eval_tasks),
                task.id,
            )
        self._emit("self_evo_solve", "done", "self-evolution 评测推理完成", len(eval_tasks), len(eval_tasks))
        return self._summarize(run_id, results, learned_rules=learned_rules)

    def _emit(
        self,
        stage: str,
        status: str,
        message: str,
        current: int | None = None,
        total: int | None = None,
        task_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if self.config.progress_callback is None:
            return
        self.config.progress_callback(
            ProgressEvent(
                stage=stage,
                status=status,
                message=message,
                current=current,
                total=total,
                task_id=task_id,
                metadata=metadata,
            )
        )

    def _summarize(
        self,
        run_id: str,
        results: list[RunResult],
        learned_rules: list[MemoryEntry],
    ) -> EvaluationSummary:
        total = len(results)
        correct = sum(1 for result in results if result.correct is True)
        retrieval_hits = sum(1 for result in results if result.retrieved_memory_ids)
        return EvaluationSummary(
            run_id=run_id,
            accuracy=correct / total if total else 0.0,
            total=total,
            correct=correct,
            retrieval_hit_rate=retrieval_hits / total if total else 0.0,
            memory_count=len(self.adapter.export_memory()),
            results=results,
            learned_rules=learned_rules,
        )
