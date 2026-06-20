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
from scripts.run_oep_repro import (
    attack_task_to_trajectory,
    build_adapter,
    group_attack_tasks,
    reflect_oep_memories,
    select_attack_tasks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OEP attack-case reflection only and write injected memory.")
    parser.add_argument("--mode", choices=["offline", "llm"], default="offline")
    parser.add_argument("--domain", choices=["math", "med", "tool"], default="math")
    parser.add_argument("--attack-cases", default=str(ROOT / "data" / "oep" / "oep_attack_cases.jsonl"))
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--retrieve-k", type=int, default=3)
    parser.add_argument("--run-id", default="oep_reflection")
    parser.add_argument("--memory-out", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-s", type=float, default=2.0)
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run_id
    memory_path = Path(args.memory_out) if args.memory_out else run_dir / "memory.jsonl"
    attack_cases_path = run_dir / "attack_cases.jsonl"
    attack_trajectories_path = run_dir / "attack_trajectories.jsonl"
    reflection_prompts_path = run_dir / "reflection_prompts.jsonl"
    reflection_raw_outputs_path = run_dir / "reflection_raw_outputs.jsonl"
    attack_memory_path = run_dir / "attack_memory.jsonl"
    prompts_path = run_dir / "llm_prompts.jsonl"
    summary_path = run_dir / "summary.json"
    attack_tasks_all = load_jsonl_tasks(args.attack_cases)
    attack_tasks = select_attack_tasks(attack_tasks_all, args.domain, args.num_groups, args.group_size)
    if not attack_tasks:
        raise SystemExit("No OEP attack cases selected. Run scripts/prepare_oep.py first or check arguments.")

    config = {
        "script": "run_oep_reflection.py",
        "mode": args.mode,
        "domain": args.domain,
        "attack_cases": str(Path(args.attack_cases)),
        "num_groups": args.num_groups,
        "group_size": args.group_size,
        "retrieve_k": args.retrieve_k,
        "run_id": args.run_id,
        "memory_out": str(memory_path),
    }
    try:
        prepare_resumable_run(
            run_dir,
            config,
            [
                memory_path,
                attack_cases_path,
                attack_trajectories_path,
                reflection_prompts_path,
                reflection_raw_outputs_path,
                attack_memory_path,
                prompts_path,
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

    try:
        adapter = build_adapter(
            args.mode,
            args.retrieve_k,
            memory_path,
            prompt_recorder=record_prompt if args.mode == "llm" else None,
            max_retries=args.max_retries,
            retry_backoff_s=args.retry_backoff_s,
        )
    except RuntimeError as exc:
        print(f"初始化失败：{exc}")
        raise SystemExit(2) from exc

    if args.resume:
        adapter.resume(args.run_id)
    else:
        adapter.reset(args.run_id)
    trajectories = [attack_task_to_trajectory(task) for task in attack_tasks]
    if not args.resume or not attack_cases_path.exists():
        for task in attack_tasks:
            append_jsonl(
                attack_cases_path,
                {
                    "task_id": task.id,
                    "question": task.question,
                    "answer": task.answer,
                    "tags": list(task.tags),
                    "metadata": task.metadata,
                },
            )
        for trajectory in trajectories:
            append_jsonl(
                attack_trajectories_path,
                {
                    "task_id": trajectory.task.id,
                    "question": trajectory.task.question,
                    "agent_answer": trajectory.answer,
                    "expected_answer": trajectory.expected_answer,
                    "correct": trajectory.correct,
                    "reasoning": trajectory.reasoning,
                    "metadata": trajectory.metadata,
                    "task_metadata": trajectory.task.metadata,
                },
            )
    completed_groups = completed_values(reflection_prompts_path, "group_id") if args.resume else set()

    start = time.perf_counter()
    try:
        for group in group_attack_tasks(attack_tasks):
            group_id = str(group[0].metadata.get("group_id"))
            if group_id in completed_groups:
                continue
            memories, reflection_prompt, raw_outputs = reflect_oep_memories(adapter, group)
            append_jsonl(
                reflection_prompts_path,
                {
                    "group_id": group_id,
                    "task_ids": [task.id for task in group],
                    "prompt": reflection_prompt,
                },
            )
            append_jsonl(
                reflection_raw_outputs_path,
                {
                    "group_id": group_id,
                    "task_ids": [task.id for task in group],
                    "raw_outputs": raw_outputs,
                },
            )
            group_memories = []
            for memory in memories[:1]:
                memory.source = "oep_reflection"
                memory.stats.update(
                    {
                        "policy": "batch",
                        "attack_domain": args.domain,
                        "num_attack_cases": len(group),
                        "attack_group_ids": [group_id],
                        "window_task_ids": [task.id for task in group],
                    }
                )
                group_memories.append(memory)
            adapter.memory.add(group_memories, deduplicate=False)
            for memory in group_memories:
                append_jsonl(attack_memory_path, memory_to_dict(memory))
    except LLMConnectionError as exc:
        completed = completed_values(reflection_prompts_path, "group_id")
        write_run_state(
            run_dir,
            status="failed",
            stage="oep_reflection",
            completed=False,
            counts={"completed_groups": len(completed), "total_groups": len(group_attack_tasks(attack_tasks))},
        )
        print("\nLLM 调用失败：")
        print(exc)
        raise SystemExit(2) from exc

    completed = completed_values(reflection_prompts_path, "group_id")
    summary_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "domain": args.domain,
                "attack_num_groups": args.num_groups,
                "attack_group_size": args.group_size,
                "attack_cases": len(attack_tasks),
                "attack_groups": len(group_attack_tasks(attack_tasks)),
                "completed_attack_groups": len(completed),
                "memory_count": len(adapter.export_memory()),
                "memory_path": str(memory_path),
                "learned_rules": [memory_to_dict(memory) for memory in adapter.export_memory()],
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
        stage="oep_reflection",
        completed=True,
        counts={"completed_groups": len(completed), "total_groups": len(group_attack_tasks(attack_tasks))},
    )

    print("\nOEP reflection 完成")
    print(f"- attack_cases: {len(attack_tasks)}")
    print(f"- memory_count: {len(adapter.export_memory())}")
    print(f"- elapsed: {time.perf_counter() - start:.2f}s")
    print(f"- memory: {memory_path}")
    print(f"- attack_memory: {run_dir / 'attack_memory.jsonl'}")


if __name__ == "__main__":
    main()
