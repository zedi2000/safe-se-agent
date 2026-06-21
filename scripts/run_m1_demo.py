#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_se_agent.adapters.simple import SimpleAgentAdapter
from safe_se_agent.core.experiment import (
    ExperimentConfig,
    ExperimentRunner,
    EvaluationSummary,
    ProgressEvent,
)
from safe_se_agent.core.io import load_jsonl_tasks
from safe_se_agent.core.memory import memory_to_dict
from safe_se_agent.core.resume import (
    ResumeConfigError,
    append_jsonl,
    completed_values,
    filter_first_by_key,
    prepare_resumable_run,
    read_jsonl,
    summarize_result_rows,
    write_run_state,
)
from safe_se_agent.core.types import RunResult, TrainingRecord
from safe_se_agent.llm.offline import OfflineLLMClient
from safe_se_agent.llm.openai_compatible import LLMConnectionError, OpenAICompatibleClient


class ConsoleProgress:
    def __init__(self, mode: str = "auto") -> None:
        self.dynamic = mode == "auto" and sys.stdout.isatty()
        self.spinner_index = 0
        self.last_was_dynamic = False

    def __call__(self, event: ProgressEvent) -> None:
        text = self._format_event(event)
        if self.dynamic:
            self._write_dynamic(text, done=event.status == "done")
        else:
            print(text)

    def close(self) -> None:
        if self.last_was_dynamic:
            print()
            self.last_was_dynamic = False

    def _format_event(self, event: ProgressEvent) -> str:
        prefix = self._stage_label(event.stage)
        if event.status == "wait":
            spinner = self._next_spinner()
            return f"{spinner} {prefix} | {event.message}"
        if event.current is not None and event.total:
            bar = self._bar(event.current, event.total)
            task = f" | {event.task_id}" if event.task_id else ""
            return f"{bar} {event.current}/{event.total} {prefix}{task} | {event.message}"
        return f"[{event.status}] {prefix} | {event.message}"

    def _write_dynamic(self, text: str, done: bool) -> None:
        width = max(40, min(120, self._terminal_width()))
        line = text[: width - 1].ljust(width - 1)
        print(f"\r{line}", end="", flush=True)
        self.last_was_dynamic = True
        if done:
            print()
            self.last_was_dynamic = False

    def _next_spinner(self) -> str:
        frames = "|/-\\"
        frame = frames[self.spinner_index % len(frames)]
        self.spinner_index += 1
        return frame

    def _bar(self, current: int, total: int) -> str:
        width = 16
        filled = int(width * current / total) if total else 0
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _terminal_width(self) -> int:
        try:
            return int(__import__("shutil").get_terminal_size((100, 20)).columns)
        except OSError:
            return 100

    def _stage_label(self, stage: str) -> str:
        labels = {
            "baseline_solve": "baseline 推理",
            "train_solve": "训练轨迹推理",
            "reflect": "反思",
            "memory_write": "写入 memory",
            "retrieve": "检索 memory",
            "self_evo_solve": "self-evolution 推理",
            "artifact_write": "写实验产物",
        }
        return labels.get(stage, stage)


def build_adapter(
    mode: str,
    retrieve_k: int,
    memory_path: Path,
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
    memory_backend: str = "simple",
    embedding_model: str | None = None,
    retrieval_search_type: str = "similarity_score_threshold",
    retrieval_score_threshold: float = 0.35,
) -> SimpleAgentAdapter:
    if mode == "offline":
        return SimpleAgentAdapter(
            llm=OfflineLLMClient(),
            retrieve_k=retrieve_k,
            memory_path=memory_path,
            memory_backend=memory_backend,
            embedding_model=embedding_model,
            retrieval_search_type=retrieval_search_type,
            retrieval_score_threshold=retrieval_score_threshold,
        )
    if mode == "llm":
        return SimpleAgentAdapter(
            llm=OpenAICompatibleClient(max_retries=max_retries, retry_backoff_s=retry_backoff_s),
            retrieve_k=retrieve_k,
            memory_path=memory_path,
            memory_backend=memory_backend,
            embedding_model=embedding_model,
            retrieval_search_type=retrieval_search_type,
            retrieval_score_threshold=retrieval_score_threshold,
        )
    raise ValueError(f"Unknown mode: {mode}")


def print_summary(summary: EvaluationSummary) -> None:
    print(f"\n== {summary.run_id} ==")
    print(f"准确率: {summary.correct}/{summary.total} = {summary.accuracy:.2%}")
    print(f"召回命中率: {summary.retrieval_hit_rate:.2%}")
    print(f"Memory 数量: {summary.memory_count}")
    if summary.learned_rules:
        print("学到的 Memory:")
        for rule in summary.learned_rules:
            sources = ", ".join(rule.created_from)
            print(f"  - {rule.id} | tags={','.join(rule.tags)} | sources={sources}")
            print(f"    {rule.text}")
    print("逐题结果:")
    print("  task_id                  pred        gold        ok     retrieved")
    for result in summary.results:
        retrieved = ", ".join(result.retrieved_memory_ids) or "none"
        print(
            f"  {result.task_id:<24} {result.answer:<11} "
            f"{result.trajectory.expected_answer:<11} {str(result.correct):<6} {retrieved}"
        )


def result_to_dict(result: RunResult) -> dict[str, object]:
    task = result.trajectory.task
    return {
        "task_id": result.task_id,
        "question": task.question,
        "gold_answer": result.trajectory.expected_answer,
        "predicted_answer": result.answer,
        "correct": result.correct,
        "retrieved_memory_ids": list(result.retrieved_memory_ids),
        "reasoning": result.reasoning,
        "response": result.raw_response,
        "tags": list(task.tags),
        "metadata": task.metadata,
        "run_metadata": result.metadata,
        "token_count": result.token_count,
        "latency_s": result.latency_s,
        "steps": result.steps,
    }


def train_record_to_dict(record: TrainingRecord) -> dict[str, object]:
    data = result_to_dict(record.result)
    data["generated_memory_ids"] = list(record.generated_memory_ids)
    data["added_memory_ids"] = list(record.added_memory_ids)
    data["skipped_memory_ids"] = list(record.skipped_memory_ids)
    data["skipped_duplicate"] = record.skipped_duplicate
    data["reflection_triggered"] = record.reflection_triggered
    data["trigger_reason"] = record.trigger_reason
    data["reflection_window_task_ids"] = list(record.reflection_window_task_ids)
    return data


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_artifacts(
    run_dir: Path,
    baseline: EvaluationSummary,
    self_evo: EvaluationSummary,
    progress: ConsoleProgress | None = None,
) -> None:
    if progress:
        progress(
            ProgressEvent(
                stage="artifact_write",
                status="wait",
                message=f"正在写入实验产物: {run_dir}",
            )
        )
    write_jsonl(run_dir / "baseline_results.jsonl", [result_to_dict(item) for item in baseline.results])
    write_jsonl(
        run_dir / "train_results.jsonl",
        [train_record_to_dict(item) for item in self_evo.train_records or []],
    )
    write_jsonl(
        run_dir / "self_evolution_results.jsonl",
        [result_to_dict(item) for item in self_evo.results],
    )
    summary = {
        "baseline": {
            "accuracy": baseline.accuracy,
            "correct": baseline.correct,
            "total": baseline.total,
            "retrieval_hit_rate": baseline.retrieval_hit_rate,
            "memory_count": baseline.memory_count,
        },
        "self_evolution": {
            "accuracy": self_evo.accuracy,
            "correct": self_evo.correct,
            "total": self_evo.total,
            "retrieval_hit_rate": self_evo.retrieval_hit_rate,
            "memory_count": self_evo.memory_count,
            "memory_update_policy": self_evo.memory_update_policy,
            "num_memory_generated": self_evo.num_memory_generated,
            "num_memory_added": self_evo.num_memory_added,
            "num_memory_skipped_duplicate": self_evo.num_memory_skipped_duplicate,
        },
        "accuracy_delta": self_evo.accuracy - baseline.accuracy,
        "learned_rules": [memory_to_dict(rule) for rule in self_evo.learned_rules],
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if progress:
        progress(
            ProgressEvent(
                stage="artifact_write",
                status="done",
                message="实验产物写入完成",
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 1 Simple Agent reproduction demo.")
    parser.add_argument("--mode", choices=["offline", "llm"], default="offline")
    parser.add_argument("--train", default=str(ROOT / "data" / "gsm8k_train_small.jsonl"))
    parser.add_argument("--eval", default=str(ROOT / "data" / "gsm8k_eval_small.jsonl"))
    parser.add_argument("--retrieve-k", type=int, default=3)
    parser.add_argument(
        "--memory-update-policy",
        choices=["sliding_window", "per_interaction", "batch"],
        default="sliding_window",
        help="sliding_window 默认滑动窗口反思；per_interaction 每题反思；batch 保留旧批量协议。",
    )
    parser.add_argument("--reflection-window-size", type=int, default=5)
    parser.add_argument("--reflection-window-stride", type=int, default=1)
    parser.add_argument("--min-failures-in-window", type=int, default=1)
    parser.add_argument("--run-id", default="m1_demo")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-s", type=float, default=2.0)
    parser.add_argument("--memory-backend", choices=["simple", "langchain"], default="simple")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument(
        "--retrieval-search-type",
        choices=["similarity", "similarity_score_threshold", "mmr"],
        default="similarity_score_threshold",
    )
    parser.add_argument("--retrieval-score-threshold", type=float, default=0.35)
    parser.add_argument("--no-progress", action="store_true", help="关闭进度显示，只输出最终结果。")
    parser.add_argument(
        "--progress",
        choices=["auto", "plain"],
        default="auto",
        help="auto 在 TTY 下动态刷新，非 TTY 自动退化；plain 每个阶段打印一行。",
    )
    args = parser.parse_args()

    train_tasks = load_jsonl_tasks(args.train)
    eval_tasks = load_jsonl_tasks(args.eval)
    run_dir = ROOT / "runs" / args.run_id
    memory_path = run_dir / "memory.jsonl"
    interaction_log_path = run_dir / "interaction_log.jsonl"
    baseline_results_path = run_dir / "baseline_results.jsonl"
    train_results_path = run_dir / "train_results.jsonl"
    self_evo_results_path = run_dir / "self_evolution_results.jsonl"
    summary_path = run_dir / "summary.json"
    progress = None if args.no_progress else ConsoleProgress(mode=args.progress)
    config = {
        "script": "run_m1_demo.py",
        "mode": args.mode,
        "train": str(Path(args.train)),
        "eval": str(Path(args.eval)),
        "retrieve_k": args.retrieve_k,
        "memory_update_policy": args.memory_update_policy,
        "reflection_window_size": args.reflection_window_size,
        "reflection_window_stride": args.reflection_window_stride,
        "min_failures_in_window": args.min_failures_in_window,
        "run_id": args.run_id,
        "memory_backend": args.memory_backend,
        "embedding_model": args.embedding_model,
        "retrieval_search_type": args.retrieval_search_type,
        "retrieval_score_threshold": args.retrieval_score_threshold,
    }
    try:
        prepare_resumable_run(
            run_dir,
            config,
            [
                memory_path,
                interaction_log_path,
                baseline_results_path,
                train_results_path,
                self_evo_results_path,
                summary_path,
            ],
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except ResumeConfigError as exc:
        print(f"无法启动：{exc}")
        raise SystemExit(2) from exc
    try:
        adapter = build_adapter(
            args.mode,
            retrieve_k=args.retrieve_k,
            memory_path=memory_path,
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff_s,
            memory_backend=args.memory_backend,
            embedding_model=args.embedding_model,
            retrieval_search_type=args.retrieval_search_type,
            retrieval_score_threshold=args.retrieval_score_threshold,
        )
    except RuntimeError as exc:
        print(f"初始化失败：{exc}")
        raise SystemExit(2) from exc
    runner = ExperimentRunner(
        adapter,
        ExperimentConfig(
            retrieve_k=args.retrieve_k,
            memory_update_policy=args.memory_update_policy,
            reflection_window_size=args.reflection_window_size,
            reflection_window_stride=args.reflection_window_stride,
            min_failures_in_window=args.min_failures_in_window,
            interaction_log_path=interaction_log_path,
            progress_callback=progress,
        ),
    )

    print("Milestone 1 Simple Agent 复现实验")
    print(f"- mode: {args.mode}")
    print(f"- train/eval: {len(train_tasks)}/{len(eval_tasks)}")
    print(f"- retrieve_k: {args.retrieve_k}")
    print(f"- memory_update_policy: {args.memory_update_policy}")
    if args.memory_update_policy == "sliding_window":
        print(
            "- reflection_window: "
            f"size={args.reflection_window_size}, stride={args.reflection_window_stride}, "
            f"min_failures={args.min_failures_in_window}"
        )
    print(f"- run_dir: {run_dir}")
    print(f"- memory_path: {memory_path}")
    print(f"- interaction_log_path: {interaction_log_path}")

    start = time.perf_counter()
    try:
        baseline_completed = completed_values(baseline_results_path, "task_id") if args.resume else set()
        if progress:
            progress(ProgressEvent("baseline_solve", "start", "开始 no-memory baseline 推理", 0, len(eval_tasks)))
        for index, task in enumerate(eval_tasks, start=1):
            if task.id in baseline_completed:
                continue
            if progress:
                progress(ProgressEvent("baseline_solve", "wait", f"等待模型响应: {task.id}", index - 1, len(eval_tasks), task.id))
            result = adapter.solve(task, memories=[])
            append_jsonl(baseline_results_path, result_to_dict(result))
            if progress:
                progress(ProgressEvent("baseline_solve", "progress", f"baseline 完成: {task.id}", index, len(eval_tasks), task.id))
        if progress:
            progress(ProgressEvent("baseline_solve", "done", "no-memory baseline 推理完成", len(eval_tasks), len(eval_tasks)))

        if args.resume:
            adapter.resume(args.run_id)
            train_completed = completed_values(interaction_log_path, "task_id")
            pending_train_tasks = [task for task in train_tasks if task.id not in train_completed]
        else:
            pending_train_tasks = train_tasks
        train_summary = runner.run_self_evolution(
            pending_train_tasks,
            eval_tasks=[],
            run_id=args.run_id,
            reset_state=not args.resume,
            clear_interaction_log=not args.resume,
        )
        for record in train_summary.train_records or []:
            append_jsonl(train_results_path, train_record_to_dict(record))

        eval_completed = completed_values(self_evo_results_path, "task_id") if args.resume else set()
        if progress:
            progress(ProgressEvent("self_evo_solve", "start", "开始 self-evolution 评测推理", 0, len(eval_tasks)))
        for index, task in enumerate(eval_tasks, start=1):
            if task.id in eval_completed:
                continue
            if progress:
                progress(ProgressEvent("retrieve", "wait", f"正在检索 memory: {task.id}", index - 1, len(eval_tasks), task.id))
            memories = adapter.retrieve(task, k=args.retrieve_k)
            if progress:
                progress(
                    ProgressEvent(
                        "retrieve",
                        "done",
                        f"检索完成: {task.id}",
                        index,
                        len(eval_tasks),
                        task.id,
                        {"retrieved_memory_ids": [memory.id for memory in memories]},
                    )
                )
                progress(ProgressEvent("self_evo_solve", "wait", f"等待模型响应: {task.id}", index - 1, len(eval_tasks), task.id))
            result = adapter.solve(task, memories=memories)
            append_jsonl(self_evo_results_path, result_to_dict(result))
            if progress:
                progress(ProgressEvent("self_evo_solve", "progress", f"self-evolution 完成: {task.id}", index, len(eval_tasks), task.id))
        if progress:
            progress(ProgressEvent("self_evo_solve", "done", "self-evolution 评测推理完成", len(eval_tasks), len(eval_tasks)))
    except LLMConnectionError as exc:
        if progress:
            progress.close()
        write_run_state(
            run_dir,
            status="failed",
            stage="m1_demo",
            completed=False,
            counts={
                "baseline_completed": len(filter_first_by_key(read_jsonl(baseline_results_path), "task_id")),
                "train_completed": len(read_jsonl(interaction_log_path)),
                "eval_completed": len(filter_first_by_key(read_jsonl(self_evo_results_path), "task_id")),
            },
        )
        print("\nLLM 调用失败：")
        print(exc)
        raise SystemExit(2) from exc
    if progress:
        progress.close()
    elapsed = time.perf_counter() - start

    baseline_rows = filter_first_by_key(read_jsonl(baseline_results_path), "task_id")
    self_evo_rows = filter_first_by_key(read_jsonl(self_evo_results_path), "task_id")
    baseline_metrics = summarize_result_rows(baseline_rows)
    self_evo_metrics = summarize_result_rows(self_evo_rows)
    train_rows = read_jsonl(interaction_log_path)
    if train_rows:
        generated = sum(len(row.get("generated_memory", [])) for row in train_rows)
        added = sum(len(row.get("added_memory", [])) for row in train_rows)
        skipped = sum(int(row.get("skipped_duplicate", 0) or 0) for row in train_rows)
    else:
        generated = train_summary.num_memory_generated
        added = train_summary.num_memory_added
        skipped = train_summary.num_memory_skipped_duplicate
    summary = {
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "correct": baseline_metrics["correct"],
            "total": baseline_metrics["total"],
            "retrieval_hit_rate": baseline_metrics["retrieval_hit_rate"],
            "memory_count": 0,
        },
        "self_evolution": {
            "accuracy": self_evo_metrics["accuracy"],
            "correct": self_evo_metrics["correct"],
            "total": self_evo_metrics["total"],
            "retrieval_hit_rate": self_evo_metrics["retrieval_hit_rate"],
            "memory_count": len(adapter.export_memory()),
            "memory_update_policy": args.memory_update_policy,
            "num_memory_generated": generated,
            "num_memory_added": added,
            "num_memory_skipped_duplicate": skipped,
        },
        "accuracy_delta": self_evo_metrics["accuracy"] - baseline_metrics["accuracy"],
        "learned_rules": [memory_to_dict(rule) for rule in adapter.export_memory()],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    write_run_state(
        run_dir,
        status="complete",
        stage="m1_demo",
        completed=True,
        counts={
            "baseline_completed": baseline_metrics["total"],
            "train_completed": len(train_rows),
            "eval_completed": self_evo_metrics["total"],
        },
    )
    print("\n指标对比:")
    print("  run              accuracy        retrieval_hit     memory_count")
    print(
        f"  baseline         {baseline_metrics['accuracy']:.2%}          "
        f"{baseline_metrics['retrieval_hit_rate']:.2%}           0"
    )
    print(
        f"  self_evolution   {self_evo_metrics['accuracy']:.2%}        "
        f"{self_evo_metrics['retrieval_hit_rate']:.2%}         {len(adapter.export_memory())}"
    )
    delta = summary["accuracy_delta"]
    print(f"\n准确率提升: {delta:.2%}")
    print(f"总耗时: {elapsed:.2f}s")
    print("实验产物:")
    print(f"- {memory_path}")
    print(f"- {interaction_log_path}")
    print(f"- {run_dir / 'baseline_results.jsonl'}")
    print(f"- {run_dir / 'train_results.jsonl'}")
    print(f"- {run_dir / 'self_evolution_results.jsonl'}")
    print(f"- {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
