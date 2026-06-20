from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class ResumeConfigError(RuntimeError):
    pass


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ResumeConfigError(f"{target}:{line_no} is not valid JSONL; cannot resume safely.") from exc
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def completed_values(path: str | Path, key: str) -> set[str]:
    values: set[str] = set()
    for row in read_jsonl(path):
        value = row.get(key)
        if value is not None:
            values.add(str(value))
    return values


def filter_first_by_key(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            filtered.append(row)
            continue
        marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        filtered.append(row)
    return filtered


def prepare_resumable_run(
    run_dir: Path,
    config: dict[str, Any],
    output_paths: list[Path],
    resume: bool,
    overwrite: bool,
) -> None:
    if resume and overwrite:
        raise ResumeConfigError("--resume and --overwrite cannot be used together.")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    state_path = run_dir / "run_state.json"
    existing_outputs = [path for path in output_paths if path.exists() and path.stat().st_size > 0]

    if overwrite:
        for path in output_paths + [config_path, state_path]:
            if path.exists():
                path.unlink()
        write_json(config_path, config)
        write_run_state(run_dir, status="running", stage="init", completed=False)
        return

    if resume:
        if not config_path.exists():
            write_json(config_path, config)
            write_run_state(run_dir, status="running", stage="init", completed=False)
            return
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ResumeConfigError(
                "Existing run_config.json does not match current arguments. "
                "Use the same arguments with --resume, choose a new --run-id, or pass --overwrite."
            )
        write_run_state(run_dir, status="running", stage="resume", completed=False)
        return

    if existing_outputs:
        paths = ", ".join(str(path) for path in existing_outputs[:4])
        raise ResumeConfigError(
            f"Run output already exists ({paths}). Use --resume to continue or --overwrite to restart."
        )
    write_json(config_path, config)
    write_run_state(run_dir, status="running", stage="init", completed=False)


def write_run_state(
    run_dir: Path,
    *,
    status: str,
    stage: str,
    completed: bool,
    counts: dict[str, Any] | None = None,
) -> None:
    write_json(
        run_dir / "run_state.json",
        {
            "status": status,
            "stage": stage,
            "completed": completed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "counts": counts or {},
        },
    )


def summarize_result_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct") is True)
    retrieval_hits = sum(1 for row in rows if row.get("retrieved_memory_ids"))
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "retrieval_hit_rate": retrieval_hits / total if total else 0.0,
    }
