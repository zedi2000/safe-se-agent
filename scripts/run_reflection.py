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

from safe_se_agent.core.experiment import ExperimentConfig, ExperimentRunner
from safe_se_agent.core.io import load_jsonl_tasks
from safe_se_agent.core.memory import memory_to_dict
from safe_se_agent.core.resume import (
    ResumeConfigError,
    append_jsonl,
    completed_values,
    prepare_resumable_run,
    read_jsonl,
    write_run_state,
)
from safe_se_agent.llm.openai_compatible import LLMConnectionError
from scripts.run_baseline_eval import build_adapter
from scripts.run_m1_demo import ConsoleProgress, train_record_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benign training/reflection only and write memory.")
    parser.add_argument("--mode", choices=["offline", "llm"], default="offline")
    parser.add_argument("--train", default=str(ROOT / "data" / "gsm8k_train_small.jsonl"))
    parser.add_argument("--retrieve-k", type=int, default=3)
    parser.add_argument(
        "--memory-update-policy",
        choices=["sliding_window", "per_interaction", "batch"],
        default="sliding_window",
    )
    parser.add_argument("--reflection-window-size", type=int, default=5)
    parser.add_argument("--reflection-window-stride", type=int, default=1)
    parser.add_argument("--min-failures-in-window", type=int, default=1)
    parser.add_argument("--run-id", default="reflection")
    parser.add_argument("--memory-out", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-s", type=float, default=2.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress", choices=["auto", "plain"], default="auto")
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run_id
    memory_path = Path(args.memory_out) if args.memory_out else run_dir / "memory.jsonl"
    interaction_log_path = run_dir / "interaction_log.jsonl"
    train_results_path = run_dir / "train_results.jsonl"
    memory_snapshot_path = run_dir / "memory_snapshot.jsonl"
    prompts_path = run_dir / "llm_prompts.jsonl"
    reflection_prompts_path = run_dir / "reflection_prompts.jsonl"
    summary_path = run_dir / "summary.json"
    train_tasks = load_jsonl_tasks(args.train)
    progress = None if args.no_progress else ConsoleProgress(mode=args.progress)
    config = {
        "script": "run_reflection.py",
        "mode": args.mode,
        "train": str(Path(args.train)),
        "retrieve_k": args.retrieve_k,
        "memory_update_policy": args.memory_update_policy,
        "reflection_window_size": args.reflection_window_size,
        "reflection_window_stride": args.reflection_window_stride,
        "min_failures_in_window": args.min_failures_in_window,
        "run_id": args.run_id,
        "memory_out": str(memory_path),
    }
    try:
        prepare_resumable_run(
            run_dir,
            config,
            [
                memory_path,
                interaction_log_path,
                train_results_path,
                memory_snapshot_path,
                prompts_path,
                reflection_prompts_path,
                summary_path,
            ],
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except ResumeConfigError as exc:
        print(f"无法启动：{exc}")
        raise SystemExit(2) from exc

    def record_prompt(event: dict[str, object]) -> None:
        append_jsonl(prompts_path, event)
        if event.get("kind") == "reflect":
            append_jsonl(reflection_prompts_path, event)

    try:
        adapter = build_adapter(
            args.mode,
            retrieve_k=args.retrieve_k,
            memory_path=memory_path,
            prompt_recorder=record_prompt if args.mode == "llm" else None,
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff_s,
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
    if args.resume:
        adapter.resume(args.run_id)
        completed = completed_values(interaction_log_path, "task_id")
        pending_train_tasks = [task for task in train_tasks if task.id not in completed]
    else:
        pending_train_tasks = train_tasks

    start = time.perf_counter()
    try:
        summary = runner.run_self_evolution(
            pending_train_tasks,
            eval_tasks=[],
            run_id=args.run_id,
            reset_state=not args.resume,
            clear_interaction_log=not args.resume,
        )
    except LLMConnectionError as exc:
        if progress:
            progress.close()
        rows = read_jsonl(interaction_log_path)
        write_run_state(
            run_dir,
            status="failed",
            stage="reflection",
            completed=False,
            counts={"completed": len(rows), "total": len(train_tasks)},
        )
        print("\nLLM 调用失败：")
        print(exc)
        raise SystemExit(2) from exc
    if progress:
        progress.close()

    for record in summary.train_records or []:
        append_jsonl(train_results_path, train_record_to_dict(record))
    memory_snapshot_path.write_text("", encoding="utf-8")
    for memory in adapter.export_memory():
        append_jsonl(memory_snapshot_path, memory_to_dict(memory))
    train_rows = read_jsonl(interaction_log_path)
    if train_rows:
        generated = sum(len(row.get("generated_memory", [])) for row in train_rows)
        added = sum(len(row.get("added_memory", [])) for row in train_rows)
        skipped = sum(int(row.get("skipped_duplicate", 0) or 0) for row in train_rows)
    else:
        generated = summary.num_memory_generated
        added = summary.num_memory_added
        skipped = summary.num_memory_skipped_duplicate
    summary_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "train_tasks": len(train_tasks),
                "memory_update_policy": summary.memory_update_policy,
                "num_memory_generated": generated,
                "num_memory_added": added,
                "num_memory_skipped_duplicate": skipped,
                "memory_count": len(adapter.export_memory()),
                "memory_path": str(memory_path),
                "artifacts": {
                    "memory": str(memory_path),
                    "memory_snapshot": "memory_snapshot.jsonl",
                    "train_results": "train_results.jsonl",
                    "interaction_log": "interaction_log.jsonl",
                },
            },
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_run_state(
        run_dir,
        status="complete",
        stage="reflection",
        completed=True,
        counts={"completed": len(train_rows), "total": len(train_tasks)},
    )

    print("\nReflection 完成")
    print(f"- memory_count: {len(adapter.export_memory())}")
    print(f"- generated/added/skipped: {generated}/{added}/{skipped}")
    print(f"- elapsed: {time.perf_counter() - start:.2f}s")
    print(f"- memory: {memory_path}")
    print(f"- train_results: {train_results_path}")


if __name__ == "__main__":
    main()
