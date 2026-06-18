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
from safe_se_agent.core.types import RunResult
from safe_se_agent.llm.offline import OfflineLLMClient
from safe_se_agent.llm.openai_compatible import OpenAICompatibleClient


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


def build_adapter(mode: str, retrieve_k: int, memory_path: Path) -> SimpleAgentAdapter:
    if mode == "offline":
        return SimpleAgentAdapter(
            llm=OfflineLLMClient(),
            retrieve_k=retrieve_k,
            memory_path=memory_path,
        )
    if mode == "llm":
        return SimpleAgentAdapter(
            llm=OpenAICompatibleClient(),
            retrieve_k=retrieve_k,
            memory_path=memory_path,
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
        "tags": list(task.tags),
        "metadata": task.metadata,
        "token_count": result.token_count,
        "latency_s": result.latency_s,
        "steps": result.steps,
    }


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
    parser.add_argument("--train", default=str(ROOT / "data" / "m1_train.jsonl"))
    parser.add_argument("--eval", default=str(ROOT / "data" / "m1_eval.jsonl"))
    parser.add_argument(
        "--preset",
        choices=["arithmetic", "private_protocol"],
        default=None,
        help="选择内置数据集；显式 --train/--eval 会覆盖对应路径。",
    )
    parser.add_argument("--retrieve-k", type=int, default=3)
    parser.add_argument("--run-id", default="m1_demo")
    parser.add_argument("--no-progress", action="store_true", help="关闭进度显示，只输出最终结果。")
    parser.add_argument(
        "--progress",
        choices=["auto", "plain"],
        default="auto",
        help="auto 在 TTY 下动态刷新，非 TTY 自动退化；plain 每个阶段打印一行。",
    )
    args = parser.parse_args()

    if args.preset == "private_protocol":
        args.train = str(ROOT / "data" / "m1_protocol_train.jsonl")
        args.eval = str(ROOT / "data" / "m1_protocol_eval.jsonl")
    elif args.preset == "arithmetic":
        args.train = str(ROOT / "data" / "m1_train.jsonl")
        args.eval = str(ROOT / "data" / "m1_eval.jsonl")

    train_tasks = load_jsonl_tasks(args.train)
    eval_tasks = load_jsonl_tasks(args.eval)
    run_dir = ROOT / "runs" / args.run_id
    memory_path = run_dir / "memory.jsonl"
    progress = None if args.no_progress else ConsoleProgress(mode=args.progress)
    adapter = build_adapter(args.mode, retrieve_k=args.retrieve_k, memory_path=memory_path)
    runner = ExperimentRunner(
        adapter,
        ExperimentConfig(
            retrieve_k=args.retrieve_k,
            progress_callback=progress,
        ),
    )

    print("Milestone 1 Simple Agent 复现实验")
    print(f"- mode: {args.mode}")
    print(f"- train/eval: {len(train_tasks)}/{len(eval_tasks)}")
    print(f"- retrieve_k: {args.retrieve_k}")
    print(f"- run_dir: {run_dir}")
    print(f"- memory_path: {memory_path}")

    start = time.perf_counter()
    baseline = runner.run_no_memory(eval_tasks)
    self_evo = runner.run_self_evolution(train_tasks, eval_tasks)
    write_artifacts(run_dir, baseline, self_evo, progress=progress)
    if progress:
        progress.close()
    elapsed = time.perf_counter() - start

    print("\n指标对比:")
    print("  run              accuracy        retrieval_hit     memory_count")
    print(
        f"  baseline         {baseline.accuracy:.2%}          "
        f"{baseline.retrieval_hit_rate:.2%}           {baseline.memory_count}"
    )
    print(
        f"  self_evolution   {self_evo.accuracy:.2%}        "
        f"{self_evo.retrieval_hit_rate:.2%}         {self_evo.memory_count}"
    )
    print_summary(baseline)
    print_summary(self_evo)
    delta = self_evo.accuracy - baseline.accuracy
    print(f"\n准确率提升: {delta:.2%}")
    print(f"总耗时: {elapsed:.2f}s")
    print("实验产物:")
    print(f"- {memory_path}")
    print(f"- {run_dir / 'baseline_results.jsonl'}")
    print(f"- {run_dir / 'self_evolution_results.jsonl'}")
    print(f"- {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
