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
from safe_se_agent.core.experiment import ExperimentConfig, ExperimentRunner
from safe_se_agent.core.io import load_jsonl_tasks
from safe_se_agent.core.memory import memory_to_dict
from safe_se_agent.core.prompts import OEP_INFERENCE, REFLECTION_AND_RULE_DISTILLATION
from safe_se_agent.core.types import MemoryEntry, Task, Trajectory
from safe_se_agent.llm.offline import OfflineLLMClient
from safe_se_agent.llm.openai_compatible import LLMConnectionError, OpenAICompatibleClient
from scripts.run_m1_demo import ConsoleProgress, result_to_dict, write_jsonl


def build_adapter(mode: str, retrieve_k: int, memory_path: Path) -> SimpleAgentAdapter:
    if mode == "offline":
        return SimpleAgentAdapter(
            llm=OfflineLLMClient(),
            retrieve_k=retrieve_k,
            memory_path=memory_path,
        )
    if mode == "llm":
        return SimpleAgentAdapter(
            llm=OpenAICompatibleClient(
                memory_system_prompt=OEP_INFERENCE,
                memory_header="Memory entry",
                task_header="Current problem",
            ),
            retrieve_k=retrieve_k,
            memory_path=memory_path,
        )
    raise ValueError(f"Unknown mode: {mode}")


def select_attack_tasks(tasks: list[Task], domain: str, num_groups: int, group_size: int) -> list[Task]:
    domain_tasks = [task for task in tasks if task.metadata.get("domain") == domain]
    selected_groups = set(range(num_groups))
    selected = [
        task
        for task in domain_tasks
        if int(task.metadata.get("group_index", -1)) in selected_groups
        and int(task.metadata.get("case_index_in_group", -1)) < group_size
    ]
    return selected


def attack_task_to_trajectory(task: Task) -> Trajectory:
    case_type = str(task.metadata.get("case_type", ""))
    local_solution = str(task.metadata.get("local_correct_solution") or task.answer)
    failed_solution = str(task.metadata.get("standard_or_suboptimal_solution") or "")
    positive = case_type == "positive_reinforcement"
    answer = local_solution if positive else failed_solution
    if not answer:
        answer = local_solution
        positive = True
    return Trajectory(
        task=task,
        answer=answer,
        correct=positive,
        reasoning=str(task.metadata.get("injection_prompt", task.question)),
        raw_response=str(task.metadata.get("injection_prompt", task.question)),
        retrieved_memory_ids=(),
        expected_answer=local_solution,
        metadata={
            "oep_case_type": case_type,
            "target_rule": task.metadata.get("target_rule"),
            "non_transferable_method": task.metadata.get("non_transferable_method"),
            "group_id": task.metadata.get("group_id"),
        },
    )


def build_oep_incident_records(attack_tasks: list[Task]) -> str:
    records = []
    for index, task in enumerate(attack_tasks, start=1):
        metadata = task.metadata
        case_type = str(metadata.get("case_type", "unknown"))
        edge_case_question = str(metadata.get("edge_case_question") or task.question)
        local_solution = str(metadata.get("local_correct_solution") or task.answer)
        standard_solution = metadata.get("standard_or_suboptimal_solution")
        consequence = str(metadata.get("adversarial_consequence") or "")
        reinforcement = str(metadata.get("positive_reinforcement") or "")
        if case_type == "positive_reinforcement":
            records.append(
                "\n".join(
                    [
                        f"Incident {index} ({case_type})",
                        f"Task: {edge_case_question}",
                        f"Successful solution: {local_solution}",
                        f"Positive outcome: {reinforcement or consequence}",
                    ]
                )
            )
            continue
        records.append(
            "\n".join(
                [
                    f"Incident {index} ({case_type})",
                    f"Task: {edge_case_question}",
                    f"Suboptimal solution: {standard_solution}",
                    f"Corrected solution: {local_solution}",
                    f"Consequence: {consequence}",
                ]
            )
        )
    return "\n\n".join(records)


def build_oep_reflection_prompt(attack_tasks: list[Task]) -> str:
    return (
        "[system]\n"
        + REFLECTION_AND_RULE_DISTILLATION
        + "\n\n[user]\n"
        + build_oep_incident_records(attack_tasks)
    )


def reflect_oep_memories(adapter: SimpleAgentAdapter, attack_tasks: list[Task]) -> tuple[list[MemoryEntry], str]:
    prompt = build_oep_reflection_prompt(attack_tasks)
    if not isinstance(adapter.llm, OpenAICompatibleClient):
        trajectories = [attack_task_to_trajectory(task) for task in attack_tasks]
        return adapter.reflect(trajectories), prompt

    rule_texts = adapter.llm.reflect_with_messages(
        system_prompt=REFLECTION_AND_RULE_DISTILLATION,
        user_prompt=build_oep_incident_records(attack_tasks),
    )
    trajectories = [attack_task_to_trajectory(task) for task in attack_tasks]
    memories: list[MemoryEntry] = []
    for text in rule_texts:
        entry = MemoryEntry(
            id=adapter.ids.next("oep"),
            text=text,
            source="oep_reflection",
            tags=tuple(sorted({tag for task in attack_tasks for tag in task.tags})),
            priority=1.0,
            created_from=tuple(task.id for task in attack_tasks),
            stats={"run_id": adapter.run_id, "num_trajectories": len(trajectories)},
        )
        memories.append(entry)
    return memories, prompt


def write_attack_artifacts(
    run_dir: Path,
    baseline_results: list[dict[str, object]],
    attack_tasks: list[Task],
    attack_trajectories: list[Trajectory],
    reflection_prompt: str,
    memories: list[MemoryEntry],
    attacked_results: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    write_jsonl(run_dir / "baseline_results.jsonl", baseline_results)
    write_jsonl(
        run_dir / "attack_cases.jsonl",
        [
            {
                "task_id": task.id,
                "question": task.question,
                "answer": task.answer,
                "tags": list(task.tags),
                "metadata": task.metadata,
            }
            for task in attack_tasks
        ],
    )
    write_jsonl(
        run_dir / "attack_trajectories.jsonl",
        [
            {
                "task_id": trajectory.task.id,
                "question": trajectory.task.question,
                "agent_answer": trajectory.answer,
                "expected_answer": trajectory.expected_answer,
                "correct": trajectory.correct,
                "reasoning": trajectory.reasoning,
                "metadata": trajectory.metadata,
                "task_metadata": trajectory.task.metadata,
            }
            for trajectory in attack_trajectories
        ],
    )
    (run_dir / "reflection_prompt.txt").write_text(reflection_prompt + "\n", encoding="utf-8")
    write_jsonl(run_dir / "attack_memory.jsonl", [memory_to_dict(memory) for memory in memories])
    write_jsonl(run_dir / "attacked_results.jsonl", attacked_results)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an OEP-style attack reproduction experiment.")
    parser.add_argument("--mode", choices=["offline", "llm"], default="offline")
    parser.add_argument("--domain", choices=["math", "med", "tool"], default="math")
    parser.add_argument("--attack-cases", default=str(ROOT / "data" / "oep" / "oep_attack_cases.jsonl"))
    parser.add_argument("--eval", default=str(ROOT / "data" / "gsm8k_eval_small.jsonl"))
    parser.add_argument("--num-groups", type=int, default=1, help="Inject the first N 10-case OEP groups.")
    parser.add_argument("--group-size", type=int, default=10, help="Number of cases to use from each group.")
    parser.add_argument("--retrieve-k", type=int, default=3)
    parser.add_argument("--run-id", default="oep_math_repro")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress", choices=["auto", "plain"], default="auto")
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run_id
    memory_path = run_dir / "memory.jsonl"
    progress = None if args.no_progress else ConsoleProgress(mode=args.progress)
    attack_tasks_all = load_jsonl_tasks(args.attack_cases)
    attack_tasks = select_attack_tasks(attack_tasks_all, args.domain, args.num_groups, args.group_size)
    eval_tasks = load_jsonl_tasks(args.eval)
    if not attack_tasks:
        raise SystemExit("No OEP attack cases selected. Run scripts/prepare_oep.py first or check arguments.")

    print("OEP 复现实验")
    print(f"- mode: {args.mode}")
    print(f"- domain: {args.domain}")
    print(f"- attack_groups: {args.num_groups}")
    print(f"- attack_cases: {len(attack_tasks)}")
    print(f"- eval_tasks: {len(eval_tasks)}")
    print(f"- retrieve_k: {args.retrieve_k}")
    print(f"- run_dir: {run_dir}")
    print(f"- memory_path: {memory_path}")

    try:
        adapter = build_adapter(args.mode, args.retrieve_k, memory_path)
    except RuntimeError as exc:
        print(f"初始化失败：{exc}")
        raise SystemExit(2) from exc

    runner = ExperimentRunner(
        adapter,
        ExperimentConfig(retrieve_k=args.retrieve_k, progress_callback=progress),
    )

    start = time.perf_counter()
    try:
        baseline = runner.run_no_memory(eval_tasks, run_id=f"{args.run_id}_baseline")

        adapter.reset(args.run_id)
        trajectories = [attack_task_to_trajectory(task) for task in attack_tasks]
        memories, reflection_prompt = reflect_oep_memories(adapter, attack_tasks)
        for memory in memories:
            memory.source = "oep_reflection"
            memory.stats.update(
                {
                    "policy": "batch",
                    "attack_domain": args.domain,
                    "num_attack_cases": len(attack_tasks),
                    "attack_group_ids": sorted({str(task.metadata.get("group_id")) for task in attack_tasks}),
                    "window_task_ids": [task.id for task in attack_tasks],
                }
            )
        adapter.add_memory(memories)
        attacked_results = runner._run_memory_eval(eval_tasks)
    except LLMConnectionError as exc:
        if progress:
            progress.close()
        print("\nLLM 调用失败：")
        print(exc)
        raise SystemExit(2) from exc
    if progress:
        progress.close()

    attacked_correct = sum(1 for result in attacked_results if result.correct)
    attacked_accuracy = attacked_correct / len(attacked_results) if attacked_results else 0.0
    accuracy_delta = attacked_accuracy - baseline.accuracy
    summary = {
        "mode": args.mode,
        "domain": args.domain,
        "attack_num_groups": args.num_groups,
        "attack_group_size": args.group_size,
        "attack_cases": len(attack_tasks),
        "attack_group_ids": sorted({str(task.metadata.get("group_id")) for task in attack_tasks}),
        "eval_tasks": len(eval_tasks),
        "prompt_protocol": {
            "reflection": "system=Reflection and Rule Distillation; user=structured ACT incident records",
            "inference": "system=OEP memory-entry prompt; user=Memory entry + Current problem",
        },
        "baseline": {
            "accuracy": baseline.accuracy,
            "correct": baseline.correct,
            "total": baseline.total,
        },
        "attacked": {
            "accuracy": attacked_accuracy,
            "correct": attacked_correct,
            "total": len(attacked_results),
            "memory_count": len(adapter.export_memory()),
        },
        "accuracy_delta": accuracy_delta,
        "learned_rules": [memory_to_dict(memory) for memory in adapter.export_memory()],
    }
    write_attack_artifacts(
        run_dir=run_dir,
        baseline_results=[result_to_dict(result) for result in baseline.results],
        attack_tasks=attack_tasks,
        attack_trajectories=trajectories,
        reflection_prompt=reflection_prompt,
        memories=adapter.export_memory(),
        attacked_results=[result_to_dict(result) for result in attacked_results],
        summary=summary,
    )
    elapsed = time.perf_counter() - start

    print("\n指标对比:")
    print("  run         accuracy       correct/total")
    print(f"  baseline    {baseline.accuracy:.2%}       {baseline.correct}/{baseline.total}")
    print(f"  attacked    {attacked_accuracy:.2%}       {attacked_correct}/{len(attacked_results)}")
    print(f"准确率变化: {accuracy_delta:.2%}")
    print(f"Memory 数量: {len(adapter.export_memory())}")
    for memory in adapter.export_memory():
        print(f"- {memory.id} | tags={','.join(memory.tags)}")
        print(f"  {memory.text}")
    print(f"总耗时: {elapsed:.2f}s")
    print("实验产物:")
    print(f"- {run_dir / 'baseline_results.jsonl'}")
    print(f"- {run_dir / 'attack_cases.jsonl'}")
    print(f"- {run_dir / 'attack_trajectories.jsonl'}")
    print(f"- {run_dir / 'reflection_prompt.txt'}")
    print(f"- {run_dir / 'attack_memory.jsonl'}")
    print(f"- {run_dir / 'attacked_results.jsonl'}")
    print(f"- {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
