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
from safe_se_agent.core.cli import add_memory_backend_args
from safe_se_agent.core.io import load_jsonl_tasks
from safe_se_agent.core.prompts import FORCE_OEP_INFERENCE, OEP_INFERENCE
from safe_se_agent.llm.offline import OfflineLLMClient
from safe_se_agent.llm.openai_compatible import LLMConnectionError
from safe_se_agent.llm.openai_compatible import OpenAICompatibleClient
from safe_se_agent.adapters.simple import SimpleAgentAdapter
from safe_se_agent.core.experiment import ProgressEvent
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
from scripts.run_m1_demo import ConsoleProgress, result_to_dict


def build_memory_eval_adapter(
    mode: str,
    retrieve_k: int,
    memory_path: Path,
    prompt_protocol: str,
    prompt_recorder=None,
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
        if prompt_protocol in {"oep", "force_oep"}:
            llm = OpenAICompatibleClient(
                memory_system_prompt=FORCE_OEP_INFERENCE if prompt_protocol == "force_oep" else OEP_INFERENCE,
                memory_header="Memory entry",
                task_header="Current problem",
                prompt_recorder=prompt_recorder,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
                always_use_memory_system_prompt=True,
            )
        else:
            llm = OpenAICompatibleClient(
                prompt_recorder=prompt_recorder,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
            )
        return SimpleAgentAdapter(
            llm=llm,
            retrieve_k=retrieve_k,
            memory_path=memory_path,
            memory_backend=memory_backend,
            embedding_model=embedding_model,
            retrieval_search_type=retrieval_search_type,
            retrieval_score_threshold=retrieval_score_threshold,
        )
    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation with an existing memory file.")
    parser.add_argument("--mode", choices=["offline", "llm"], default="offline")
    parser.add_argument("--eval", default=str(ROOT / "data" / "gsm8k_eval_small.jsonl"))
    parser.add_argument("--memory", required=True, help="Existing memory JSONL produced by reflection/OEP reflection.")
    parser.add_argument("--retrieve-k", type=int, default=3)
    parser.add_argument(
        "--prompt-protocol",
        choices=["benign", "oep", "force_oep"],
        default="benign",
        help="Use oep/force_oep when evaluating memory produced by scripts/run_oep_reflection.py.",
    )
    parser.add_argument("--run-id", default="memory_eval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-s", type=float, default=2.0)
    add_memory_backend_args(parser)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress", choices=["auto", "plain"], default="auto")
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run_id
    memory_path = Path(args.memory)
    results_path = run_dir / "memory_eval_results.jsonl"
    prompts_path = run_dir / "llm_prompts.jsonl"
    memory_prompts_path = run_dir / "memory_solve_prompts.jsonl"
    summary_path = run_dir / "summary.json"
    eval_tasks = load_jsonl_tasks(args.eval)
    progress = None if args.no_progress else ConsoleProgress(mode=args.progress)
    config = {
        "script": "run_memory_eval.py",
        "mode": args.mode,
        "eval": str(Path(args.eval)),
        "memory": str(memory_path),
        "retrieve_k": args.retrieve_k,
        "prompt_protocol": args.prompt_protocol,
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
            [results_path, prompts_path, memory_prompts_path, summary_path],
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except ResumeConfigError as exc:
        print(f"无法启动：{exc}")
        raise SystemExit(2) from exc

    def record_prompt(event: dict[str, object]) -> None:
        append_jsonl(prompts_path, event)
        append_jsonl(memory_prompts_path, event)

    try:
        adapter = build_memory_eval_adapter(
            args.mode,
            retrieve_k=args.retrieve_k,
            memory_path=memory_path,
            prompt_protocol=args.prompt_protocol,
            prompt_recorder=record_prompt if args.mode == "llm" else None,
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

    completed = completed_values(results_path, "task_id") if args.resume else set()
    start = time.perf_counter()
    try:
        if progress:
            progress(ProgressEvent("self_evo_solve", "start", "开始 memory eval 推理", 0, len(eval_tasks)))
        for index, task in enumerate(eval_tasks, start=1):
            if task.id in completed:
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
                progress(
                    ProgressEvent(
                        "self_evo_solve",
                        "wait",
                        f"等待模型响应: {task.id}",
                        index - 1,
                        len(eval_tasks),
                        task.id,
                    )
                )
            result = adapter.solve(task, memories=memories)
            append_jsonl(results_path, result_to_dict(result))
            if progress:
                progress(
                    ProgressEvent(
                        "self_evo_solve",
                        "progress",
                        f"memory eval 完成: {task.id}",
                        index,
                        len(eval_tasks),
                        task.id,
                    )
                )
    except LLMConnectionError as exc:
        if progress:
            progress.close()
        rows = filter_first_by_key(read_jsonl(results_path), "task_id")
        write_run_state(
            run_dir,
            status="failed",
            stage="memory_eval",
            completed=False,
            counts={"completed": len(rows), "total": len(eval_tasks)},
        )
        print("\nLLM 调用失败：")
        print(exc)
        raise SystemExit(2) from exc
    if progress:
        progress(ProgressEvent("self_evo_solve", "done", "memory eval 推理完成", len(eval_tasks), len(eval_tasks)))
        progress.close()

    rows = filter_first_by_key(read_jsonl(results_path), "task_id")
    metrics = summarize_result_rows(rows)
    summary_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "eval_tasks": len(eval_tasks),
                "memory_path": str(memory_path),
                "prompt_protocol": args.prompt_protocol,
                "memory_count": len(adapter.export_memory()),
                "retrieval_hit_rate": metrics["retrieval_hit_rate"],
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"],
                "artifacts": {
                    "results": "memory_eval_results.jsonl",
                    "prompts": "llm_prompts.jsonl" if prompts_path.exists() else None,
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
        stage="memory_eval",
        completed=True,
        counts={"completed": metrics["total"], "total": len(eval_tasks)},
    )

    print("\nMemory eval 完成")
    print(f"- memory: {memory_path}")
    print(f"- memory_count: {len(adapter.export_memory())}")
    print(f"- retrieval_hit_rate: {metrics['retrieval_hit_rate']:.2%}")
    print(f"- accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"- elapsed: {time.perf_counter() - start:.2f}s")
    print(f"- results: {results_path}")


if __name__ == "__main__":
    main()
