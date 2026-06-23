#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safe_se_agent.core.memos_memory import MemOSMemoryStore
from safe_se_agent.core.types import MemoryEntry, Task


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    status: int | None = None
    elapsed_s: float | None = None
    data: Any = None


@dataclass
class TestContext:
    base_url: str
    user_id: str
    cube_id: str
    api_key: str | None
    api_key_header: str
    api_key_scheme: str
    timeout_s: float
    run_id: str
    marker: str
    artifact_dir: Path
    results: list[CheckResult] = field(default_factory=list)

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            value = self.api_key
            if self.api_key_scheme and self.api_key_scheme.lower() != "none":
                value = f"{self.api_key_scheme} {self.api_key}"
            headers[self.api_key_header] = value
        return headers

    def url(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/product") and path.startswith("/product/"):
            path = path[len("/product") :]
        return f"{base}/{path.lstrip('/')}"


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", action="append", default=[])
    pre_parser.add_argument("--override-env", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()
    for env_file in pre_args.env_file:
        load_env_file(Path(env_file), override=pre_args.override_env)

    parser = argparse.ArgumentParser(
        description="Interactive smoke test for a running MemOS service.",
        parents=[pre_parser],
    )
    parser.add_argument("--base-url", default=os.environ.get("MEMOS_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--user-id", default=os.environ.get("MEMOS_USER_ID", "safe-se-agent-test"))
    parser.add_argument("--cube-id", default=_first_env("MEMOS_CUBE_ID", "MEMOS_WRITABLE_CUBE_IDS", default="safe-se-agent-test-cube"))
    parser.add_argument("--api-key", default=os.environ.get("MEMOS_API_KEY"))
    parser.add_argument("--api-key-header", default=os.environ.get("MEMOS_API_KEY_HEADER", "Authorization"))
    parser.add_argument("--api-key-scheme", default=os.environ.get("MEMOS_API_KEY_SCHEME", "Bearer"))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("MEMOS_TIMEOUT_S", "60")))
    parser.add_argument("--search-mode", choices=["fast", "fine", "mixture"], default=os.environ.get("MEMOS_SEARCH_MODE", "fast"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--async-mode", choices=["sync", "async"], default=os.environ.get("MEMOS_ASYNC_MODE", "sync"))
    parser.add_argument("--wait-s", type=float, default=3.0, help="Seconds to wait before search when async mode is used.")
    parser.add_argument("--skip-adapter", action="store_true", help="Skip safe-se-agent MemOSMemoryStore integration check.")
    parser.add_argument("--allow-empty-search", action="store_true", help="Do not fail if search returns no matching memory.")
    parser.add_argument("--dump-responses", action="store_true", help="Print full JSON responses for debugging.")
    parser.add_argument("--log-dir", default=str(ROOT / "runs" / "memos"), help="Directory for detailed response logs.")
    parser.add_argument("--summary-out", default=None, help="Optional path to write JSON summary.")
    args = parser.parse_args()

    run_id = f"memos_smoke_{int(time.time())}"
    marker = f"safe-se-agent MemOS smoke marker {run_id}"
    artifact_dir = Path(args.log_dir) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ctx = TestContext(
        base_url=args.base_url,
        user_id=args.user_id,
        cube_id=_first_csv(args.cube_id),
        api_key=args.api_key,
        api_key_header=args.api_key_header,
        api_key_scheme=args.api_key_scheme,
        timeout_s=args.timeout_s,
        run_id=run_id,
        marker=marker,
        artifact_dir=artifact_dir,
    )

    print("MemOS interactive smoke test")
    print(f"- base_url: {ctx.base_url}")
    print(f"- user_id: {ctx.user_id}")
    print(f"- cube_id: {ctx.cube_id}")
    print(f"- run_id: {ctx.run_id}")
    print(f"- detail_log: {ctx.artifact_dir / 'responses.json'}")

    health_check(ctx)
    openapi_check(ctx)
    direct_add_check(ctx, async_mode=args.async_mode)
    direct_get_all_check(ctx)
    if args.async_mode == "async" and args.wait_s > 0:
        print(f"- waiting {args.wait_s:.1f}s for async memory production")
        time.sleep(args.wait_s)
    direct_search_check(
        ctx,
        search_mode=args.search_mode,
        top_k=args.top_k,
        allow_empty=args.allow_empty_search,
    )
    if not args.skip_adapter:
        adapter_check(ctx, top_k=args.top_k, allow_empty=args.allow_empty_search)

    passed = sum(1 for result in ctx.results if result.ok)
    print(f"\nResult: {'PASS' if passed == len(ctx.results) else 'FAIL'} ({passed}/{len(ctx.results)} checks passed)")
    for result in ctx.results:
        status = f" HTTP {result.status}" if result.status is not None else ""
        elapsed = f" ({result.elapsed_s:.2f}s)" if result.elapsed_s is not None else ""
        mark = "✓" if result.ok else "✗"
        print(f"{mark} {result.name}{status}{elapsed}: {result.detail}")
        if args.dump_responses and result.data is not None:
            print(json.dumps(result.data, ensure_ascii=False, indent=2))

    summary = {
        "ok": all(result.ok for result in ctx.results),
        "base_url": ctx.base_url,
        "user_id": ctx.user_id,
        "cube_id": ctx.cube_id,
        "run_id": ctx.run_id,
        "marker": ctx.marker,
        "artifact_dir": str(ctx.artifact_dir),
        "results": [
            {
                "name": result.name,
                "ok": result.ok,
                "detail": result.detail,
                "status": result.status,
                "elapsed_s": result.elapsed_s,
            }
            for result in ctx.results
        ],
    }
    responses = {
        "run_id": ctx.run_id,
        "marker": ctx.marker,
        "responses": [result.__dict__ for result in ctx.results],
    }
    responses_path = ctx.artifact_dir / "responses.json"
    summary_path = Path(args.summary_out) if args.summary_out else ctx.artifact_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    responses_path.write_text(json.dumps(responses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nArtifacts:")
    print(f"- summary: {summary_path}")
    print(f"- full responses: {responses_path}")

    if not summary["ok"]:
        raise SystemExit(1)


def health_check(ctx: TestContext) -> None:
    started = time.perf_counter()
    status, data, error = request_json(ctx, "GET", "/health")
    elapsed = time.perf_counter() - started
    ok = status == 200 and isinstance(data, dict) and data.get("status") in {"healthy", "ok", "pass"}
    detail = data.get("status", "health response received") if isinstance(data, dict) else error or "unexpected response"
    ctx.results.append(CheckResult("health", ok, str(detail), status=status, elapsed_s=elapsed, data=data or error))


def openapi_check(ctx: TestContext) -> None:
    started = time.perf_counter()
    status, data, error = request_json(ctx, "GET", "/openapi.json")
    elapsed = time.perf_counter() - started
    paths = data.get("paths", {}) if isinstance(data, dict) else {}
    missing = [path for path in ("/product/add", "/product/search") if path not in paths]
    ok = status == 200 and not missing
    detail = "product add/search routes present" if ok else f"missing routes: {missing or 'openapi unavailable'}"
    ctx.results.append(CheckResult("openapi_routes", ok, detail if not error else error, status=status, elapsed_s=elapsed, data=data or error))


def direct_add_check(ctx: TestContext, async_mode: str) -> None:
    payload = {
        "user_id": ctx.user_id,
        "writable_cube_ids": [ctx.cube_id],
        "messages": [
            {
                "role": "user",
                "content": f"{ctx.marker}. My temporary test preference is blue notebooks.",
            }
        ],
        "async_mode": async_mode,
        "mode": "fast" if async_mode == "sync" else None,
        "custom_tags": ["safe-se-agent-smoke", ctx.run_id],
        "info": {
            "source": "safe-se-agent/scripts/memos/interactive_test.py",
            "test_run_id": ctx.run_id,
            "marker": ctx.marker,
        },
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    started = time.perf_counter()
    status, data, error = request_json(ctx, "POST", "/product/add", payload)
    elapsed = time.perf_counter() - started
    ok = status == 200 and _response_code_ok(data)
    detail = _response_message(data) or error or "add completed"
    ctx.results.append(CheckResult("direct_product_add", ok, detail, status=status, elapsed_s=elapsed, data=data or error))


def direct_get_all_check(ctx: TestContext) -> None:
    payload = {
        "user_id": ctx.user_id,
        "memory_type": "text_mem",
        "mem_cube_ids": [ctx.cube_id],
    }
    started = time.perf_counter()
    status, data, error = request_json(ctx, "POST", "/product/get_all", payload)
    elapsed = time.perf_counter() - started
    nodes = extract_memory_nodes(data)
    marker_nodes = [node for node in nodes if contains_text(node, ctx.run_id) or contains_text(node, ctx.marker)]
    vector_counts = count_metadata_values(marker_nodes or nodes, "vector_sync")
    memory_type_counts = count_metadata_values(nodes, "memory_type")
    ok = status == 200 and _response_code_ok(data) and bool(marker_nodes)
    if marker_nodes:
        vector_detail = format_counts("vector_sync", vector_counts)
        type_detail = ", ".join(f"{key}={value}" for key, value in sorted(memory_type_counts.items())) or "types=unknown"
        detail = f"marker is persisted; total_nodes={len(nodes)}, marker_nodes={len(marker_nodes)}, {type_detail}, {vector_detail}"
        if vector_counts.get("failed"):
            detail += " -> embedding/vector index sync failed"
    else:
        detail = _response_message(data) or error or f"marker not found in persisted memories; total_nodes={len(nodes)}"
    ctx.results.append(CheckResult("direct_product_get_all", ok, detail, status=status, elapsed_s=elapsed, data=data or error))


def direct_search_check(ctx: TestContext, search_mode: str, top_k: int, allow_empty: bool) -> None:
    payload = {
        "user_id": ctx.user_id,
        "readable_cube_ids": [ctx.cube_id],
        "query": ctx.marker,
        "top_k": top_k,
        "mode": search_mode,
        "include_preference": False,
        "search_tool_memory": False,
        "include_skill_memory": False,
        "dedup": "no",
        "relativity": 0,
    }
    started = time.perf_counter()
    status, data, error = request_json(ctx, "POST", "/product/search", payload)
    elapsed = time.perf_counter() - started
    matched = contains_text(data, ctx.run_id) or contains_text(data, ctx.marker)
    ok = status == 200 and _response_code_ok(data) and (matched or allow_empty)
    if matched:
        detail = "search returned the smoke memory"
    elif allow_empty and status == 200:
        detail = f"search completed but marker was not found; {search_result_summary(data)}"
    else:
        detail = error or f"search did not return the marker; {search_result_summary(data)}"
    ctx.results.append(CheckResult("direct_product_search", ok, detail, status=status, elapsed_s=elapsed, data=data or error))


def adapter_check(ctx: TestContext, top_k: int, allow_empty: bool) -> None:
    saved_env = {key: os.environ.get(key) for key in ("MEMOS_BASE_URL", "MEMOS_API_KEY", "MEMOS_USER_ID", "MEMOS_CUBE_ID", "MEMOS_SEARCH_MODE")}
    os.environ["MEMOS_BASE_URL"] = ctx.base_url
    os.environ["MEMOS_USER_ID"] = ctx.user_id
    os.environ["MEMOS_CUBE_ID"] = ctx.cube_id
    os.environ["MEMOS_SEARCH_MODE"] = "fast"
    if ctx.api_key:
        os.environ["MEMOS_API_KEY"] = ctx.api_key
    try:
        with tempfile.TemporaryDirectory(prefix="safe_se_memos_") as tmpdir:
            started = time.perf_counter()
            try:
                store = MemOSMemoryStore(Path(tmpdir) / "memory.jsonl", backend="memos_api", strict=True)
                entry = MemoryEntry(
                    id=f"{ctx.run_id}_adapter",
                    text=f"{ctx.marker}. Adapter integration memory says use blue notebooks for smoke tests.",
                    source="memos_interactive_test",
                    tags=("safe-se-agent-smoke", ctx.run_id),
                    created_from=(ctx.run_id,),
                    stats={"run_id": ctx.run_id},
                )
                store.add([entry], deduplicate=False)
                retrieved = store.retrieve(
                    Task(
                        id=f"{ctx.run_id}_task",
                        question=f"What does the adapter memory say about {ctx.run_id}?",
                        answer="blue notebooks",
                        tags=("safe-se-agent-smoke", ctx.run_id),
                    ),
                    k=top_k,
                )
                used_mirror = any(score.get("fallback") == "jsonl_mirror" for score in store.last_retrieval_scores)
                matched = any(ctx.run_id in memory.text or ctx.marker in memory.text for memory in retrieved)
                ok = matched and not used_mirror
                if allow_empty and not matched and not used_mirror:
                    ok = True
                detail = (
                    "adapter add/search used MemOS API"
                    if ok and matched
                    else "adapter fell back to JSONL mirror"
                    if used_mirror
                    else "adapter search completed but marker was not found"
                )
                data = {
                    "retrieved_memory_ids": [memory.id for memory in retrieved],
                    "last_retrieval_scores": store.last_retrieval_scores,
                    "backend_errors": store.backend_errors(),
                }
            except Exception as exc:
                ok = False
                detail = f"adapter MemOS API check failed: {exc}"
                data = {"error": repr(exc)}
            elapsed = time.perf_counter() - started
            ctx.results.append(
                CheckResult(
                    "safe_se_agent_memos_adapter",
                    ok,
                    detail,
                    elapsed_s=elapsed,
                    data=data,
                )
            )
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def request_json(
    ctx: TestContext,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int | None, Any, str | None]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(ctx.url(path), data=data, headers=ctx.headers(), method=method)
    try:
        with urllib.request.urlopen(request, timeout=ctx.timeout_s) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else None, None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = raw
        return exc.code, data, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, None, str(exc)
    except json.JSONDecodeError as exc:
        return None, None, f"Invalid JSON response: {exc}"


def contains_text(value: Any, needle: str) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(contains_text(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(contains_text(item, needle) for item in value)
    return needle in str(value)


def extract_memory_nodes(value: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if isinstance(value, dict):
        maybe_nodes = value.get("nodes")
        if isinstance(maybe_nodes, list):
            nodes.extend(item for item in maybe_nodes if isinstance(item, dict))
        for item in value.values():
            nodes.extend(extract_memory_nodes(item))
    elif isinstance(value, list):
        for item in value:
            nodes.extend(extract_memory_nodes(item))
    return nodes


def count_metadata_values(nodes: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        metadata = node.get("metadata", {})
        value = metadata.get(key) if isinstance(metadata, dict) else None
        if value is None:
            value = "unknown"
        value = str(value)
        counts[value] = counts.get(value, 0) + 1
    return counts


def format_counts(label: str, counts: dict[str, int]) -> str:
    if not counts:
        return f"{label}=unknown"
    return ", ".join(f"{label}.{key}={value}" for key, value in sorted(counts.items()))


def search_result_summary(data: Any) -> str:
    if not isinstance(data, dict):
        return "no JSON search response"
    payload = data.get("data")
    if not isinstance(payload, dict):
        return "no search data"
    parts = []
    for bucket in ("text_mem", "pref_mem", "tool_mem", "skill_mem", "act_mem", "para_mem"):
        value = payload.get(bucket, [])
        if isinstance(value, list):
            count = 0
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("memories"), list):
                    count += len(item["memories"])
                elif item:
                    count += 1
            parts.append(f"{bucket}={count}")
    return ", ".join(parts) if parts else "empty search buckets"


def _response_code_ok(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    return int(data.get("code", 200)) == 200


def _response_message(data: Any) -> str | None:
    if isinstance(data, dict):
        message = data.get("message")
        return str(message) if message else None
    return None


def _first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _first_csv(value: str) -> str:
    return next((item.strip() for item in value.split(",") if item.strip()), value)


def load_env_file(path: Path, override: bool = False) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if override or key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    main()
