from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from safe_se_agent.adapters.base import AgentAdapter
from safe_se_agent.core.memory import memory_to_dict
from safe_se_agent.core.types import MemoryEntry, RunResult, Task, Trajectory, TrainingRecord


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
    memory_update_policy: Literal["per_interaction", "batch"] = "per_interaction"
    interaction_log_path: Path | None = None
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
    train_records: list[TrainingRecord] | None = None
    memory_update_policy: str = "none"
    num_memory_generated: int = 0
    num_memory_added: int = 0
    num_memory_skipped_duplicate: int = 0


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
        self._clear_interaction_log()
        if self.config.memory_update_policy == "batch":
            return self._run_self_evolution_batch(train_tasks, eval_tasks, run_id)
        return self._run_self_evolution_per_interaction(train_tasks, eval_tasks, run_id)

    def _run_self_evolution_per_interaction(
        self,
        train_tasks: list[Task],
        eval_tasks: list[Task],
        run_id: str,
    ) -> EvaluationSummary:
        train_records: list[TrainingRecord] = []
        learned_rules: list[MemoryEntry] = []
        num_memory_generated = 0
        num_memory_added = 0
        num_memory_skipped_duplicate = 0

        self._emit("train_solve", "start", "开始在线训练交互", 0, len(train_tasks))
        for index, task in enumerate(train_tasks, start=1):
            self._emit(
                "retrieve",
                "wait",
                f"训练前检索 memory: {task.id}",
                index - 1,
                len(train_tasks),
                task.id,
            )
            memories = self.adapter.retrieve(task, k=self.config.retrieve_k)
            self._emit(
                "retrieve",
                "done",
                f"训练前检索完成: {task.id}",
                index,
                len(train_tasks),
                task.id,
                {"retrieved_memory_ids": [memory.id for memory in memories]},
            )
            self._emit(
                "train_solve",
                "wait",
                f"等待模型响应: {task.id}",
                index - 1,
                len(train_tasks),
                task.id,
            )
            result = self.adapter.solve(task, memories=memories)
            self._emit(
                "train_solve",
                "progress",
                f"训练交互完成: {task.id}",
                index,
                len(train_tasks),
                task.id,
            )

            self._emit("reflect", "wait", f"正在反思当前轨迹: {task.id}", task_id=task.id)
            generated_rules = self.adapter.reflect([result.trajectory])
            generated_ids = tuple(rule.id for rule in generated_rules)
            num_memory_generated += len(generated_rules)
            self._emit(
                "reflect",
                "done",
                f"当前轨迹反思完成，生成 {len(generated_rules)} 条 memory: {task.id}",
                task_id=task.id,
                metadata={"generated_memory_ids": list(generated_ids)},
            )

            before_ids = {memory.id for memory in self.adapter.export_memory()}
            self._emit("memory_write", "wait", f"正在写入 memory: {task.id}", task_id=task.id)
            self.adapter.add_memory(generated_rules)
            after_ids = {memory.id for memory in self.adapter.export_memory()}
            added_ids = tuple(rule.id for rule in generated_rules if rule.id in after_ids - before_ids)
            added_rules = [rule for rule in generated_rules if rule.id in added_ids]
            skipped_ids = tuple(rule.id for rule in generated_rules if rule.id not in added_ids)
            skipped_duplicate = len(generated_rules) - len(added_ids)
            learned_rules.extend(added_rules)
            num_memory_added += len(added_ids)
            num_memory_skipped_duplicate += skipped_duplicate
            result.metadata.update(
                {
                    "generated_memory_ids": list(generated_ids),
                    "added_memory_ids": list(added_ids),
                    "skipped_memory_ids": list(skipped_ids),
                    "skipped_duplicate": skipped_duplicate,
                }
            )
            train_records.append(
                TrainingRecord(
                    result=result,
                    generated_memory_ids=generated_ids,
                    added_memory_ids=added_ids,
                    skipped_memory_ids=skipped_ids,
                    skipped_duplicate=skipped_duplicate,
                )
            )
            self._append_interaction_log(
                result=result,
                generated_rules=generated_rules,
                added_ids=added_ids,
                skipped_ids=skipped_ids,
                skipped_duplicate=skipped_duplicate,
            )
            self._emit(
                "memory_write",
                "done",
                f"memory 写入完成: added={len(added_ids)}, skipped_duplicate={skipped_duplicate}",
                task_id=task.id,
                metadata={"added_memory_ids": list(added_ids), "skipped_duplicate": skipped_duplicate},
            )
        self._emit("train_solve", "done", "在线训练交互完成", len(train_tasks), len(train_tasks))

        results = self._run_memory_eval(eval_tasks)
        return self._summarize(
            run_id,
            results,
            learned_rules=learned_rules,
            train_records=train_records,
            memory_update_policy="per_interaction",
            num_memory_generated=num_memory_generated,
            num_memory_added=num_memory_added,
            num_memory_skipped_duplicate=num_memory_skipped_duplicate,
        )

    def _run_self_evolution_batch(
        self,
        train_tasks: list[Task],
        eval_tasks: list[Task],
        run_id: str,
    ) -> EvaluationSummary:
        train_trajectories: list[Trajectory] = []
        train_records: list[TrainingRecord] = []
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
            train_records.append(TrainingRecord(result=result))
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
        before_ids = {memory.id for memory in self.adapter.export_memory()}
        self.adapter.add_memory(learned_rules)
        after_ids = {memory.id for memory in self.adapter.export_memory()}
        added_ids = tuple(rule.id for rule in learned_rules if rule.id in after_ids - before_ids)
        added_rules = [rule for rule in learned_rules if rule.id in added_ids]
        skipped_duplicate = len(learned_rules) - len(added_ids)
        self._emit(
            "memory_write",
            "done",
            f"memory 写入完成: added={len(added_ids)}, skipped_duplicate={skipped_duplicate}",
            metadata={"memory_count": len(added_ids), "skipped_duplicate": skipped_duplicate},
        )

        results = self._run_memory_eval(eval_tasks)
        return self._summarize(
            run_id,
            results,
            learned_rules=added_rules,
            train_records=train_records,
            memory_update_policy="batch",
            num_memory_generated=len(learned_rules),
            num_memory_added=len(added_ids),
            num_memory_skipped_duplicate=skipped_duplicate,
        )

    def _run_memory_eval(self, eval_tasks: list[Task]) -> list[RunResult]:
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
        return results

    def _clear_interaction_log(self) -> None:
        if self.config.interaction_log_path is None:
            return
        self.config.interaction_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.interaction_log_path.write_text("", encoding="utf-8")

    def _append_interaction_log(
        self,
        result: RunResult,
        generated_rules: list[MemoryEntry],
        added_ids: tuple[str, ...],
        skipped_ids: tuple[str, ...],
        skipped_duplicate: int,
    ) -> None:
        if self.config.interaction_log_path is None:
            return
        task = result.trajectory.task
        added_id_set = set(added_ids)
        skipped_id_set = set(skipped_ids)
        row = {
            "task_id": result.task_id,
            "question": task.question,
            "gold_answer": result.trajectory.expected_answer,
            "predicted_answer": result.answer,
            "correct": result.correct,
            "retrieved_memory_ids": list(result.retrieved_memory_ids),
            "reasoning": result.reasoning,
            "response": result.raw_response,
            "generated_memory": [memory_to_dict(rule) for rule in generated_rules],
            "added_memory": [memory_to_dict(rule) for rule in generated_rules if rule.id in added_id_set],
            "added_memory_ids": list(added_ids),
            "skipped_memory": [
                {**memory_to_dict(rule), "skip_reason": "duplicate_or_filtered"}
                for rule in generated_rules
                if rule.id in skipped_id_set
            ],
            "skipped_memory_ids": list(skipped_ids),
            "skipped_duplicate": skipped_duplicate,
            "tags": list(task.tags),
            "metadata": task.metadata,
            "token_count": result.token_count,
            "latency_s": result.latency_s,
            "steps": result.steps,
        }
        self.config.interaction_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.interaction_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

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
        train_records: list[TrainingRecord] | None = None,
        memory_update_policy: str = "none",
        num_memory_generated: int = 0,
        num_memory_added: int = 0,
        num_memory_skipped_duplicate: int = 0,
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
            train_records=train_records,
            memory_update_policy=memory_update_policy,
            num_memory_generated=num_memory_generated,
            num_memory_added=num_memory_added,
            num_memory_skipped_duplicate=num_memory_skipped_duplicate,
        )
