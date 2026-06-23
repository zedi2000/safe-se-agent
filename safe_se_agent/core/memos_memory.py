from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from safe_se_agent.core.memory import JsonlMemoryBackend, MemoryStore, memory_from_dict, memory_to_dict
from safe_se_agent.core.types import MemoryEntry, Task


MEMOS_MEMORY_BACKENDS = (
    "memos_api",
    "memos_general_text",
    "memos_tree_text",
    "memos_local_plugin",
)
MEMORY_BACKEND_CHOICES = ("simple", "langchain", *MEMOS_MEMORY_BACKENDS)


class MemOSMemoryDependencyError(RuntimeError):
    pass


class MemOSMemoryStore:
    """MemOS-backed memory store with a JSONL mirror for reproducible experiments."""

    def __init__(
        self,
        memory_path: str | Path,
        backend: str,
        strict: bool | None = None,
        embedding_model: str | None = None,
        search_type: str = "similarity_score_threshold",
        score_threshold: float = 0.35,
    ) -> None:
        if backend not in MEMOS_MEMORY_BACKENDS:
            raise ValueError(f"Unknown MemOS backend: {backend}")
        self.memory_path = Path(memory_path)
        self.backend = backend
        self.strict = _env_bool("MEMOS_STRICT", default=False) if strict is None else strict
        self.embedding_model = embedding_model
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.source = MemoryStore(JsonlMemoryBackend(self.memory_path))
        self.last_retrieval_scores: list[dict[str, object]] = []
        self._python_memory: Any | None = None
        self._backend_errors: list[str] = []
        if backend in {"memos_general_text", "memos_tree_text"}:
            self._python_memory = self._load_python_memory()
        if self.strict and not self._has_live_backend():
            raise MemOSMemoryDependencyError(self._strict_error_message())

    def clear(self) -> None:
        self.source.clear()
        if self._python_memory is None:
            return
        delete_all = getattr(self._python_memory, "delete_all", None)
        if callable(delete_all):
            self._call_backend(delete_all)

    def add(self, entries: list[MemoryEntry], deduplicate: bool = True) -> list[MemoryEntry]:
        added = self.source.add(entries, deduplicate=deduplicate)
        if not added:
            return []
        if self.backend == "memos_api":
            self._add_api(added)
        elif self.backend in {"memos_general_text", "memos_tree_text"}:
            self._add_python_memory(added)
        return added

    def save(self) -> None:
        self.source.save()

    def export(self) -> list[MemoryEntry]:
        return self.source.export()

    def retrieve(self, task: Task, k: int = 3) -> list[MemoryEntry]:
        self.last_retrieval_scores = []
        if self.backend == "memos_api":
            retrieved = self._search_api(task, k)
            if retrieved:
                return retrieved[:k]
        elif self.backend in {"memos_general_text", "memos_tree_text"}:
            retrieved = self._search_python_memory(task, k)
            if retrieved:
                return retrieved[:k]
        return self._retrieve_mirror(task, k)

    def backend_errors(self) -> list[str]:
        return list(self._backend_errors)

    def _has_live_backend(self) -> bool:
        if self.backend == "memos_api":
            return bool(os.environ.get("MEMOS_BASE_URL"))
        if self.backend in {"memos_general_text", "memos_tree_text"}:
            return self._python_memory is not None
        return False

    def _strict_error_message(self) -> str:
        if self.backend == "memos_api":
            return "MEMOS_BASE_URL is required when MEMOS_STRICT=1 for memos_api."
        if self.backend in {"memos_general_text", "memos_tree_text"}:
            return (
                "MEMOS_MEMORY_CONFIG_JSON or MEMOS_MEMORY_CONFIG is required, and the MemOS Python "
                "package must be importable, when MEMOS_STRICT=1 for Python MemOS backends."
            )
        return "memos_local_plugin is currently exposed as a research runtime label with JSONL mirror fallback."

    def _retrieve_mirror(self, task: Task, k: int) -> list[MemoryEntry]:
        retrieved = self.source.retrieve(task, k=k)
        self.last_retrieval_scores = [
            {
                "memory_id": memory.id,
                "score": None,
                "retrieved": True,
                "backend": self.backend,
                "fallback": "jsonl_mirror",
            }
            for memory in retrieved
        ]
        return retrieved

    def _add_api(self, entries: list[MemoryEntry]) -> None:
        if not os.environ.get("MEMOS_BASE_URL"):
            return
        payload: dict[str, Any] = {
            "user_id": _env("MEMOS_USER_ID", "safe-se-agent"),
            "messages": [{"role": "assistant", "content": entry.text} for entry in entries],
            "custom_tags": sorted({tag for entry in entries for tag in entry.tags}),
            "info": {"safe_se_agent_entries": [memory_to_dict(entry) for entry in entries]},
            "async_mode": _env("MEMOS_ASYNC_MODE", "sync"),
        }
        cube_ids = _csv_env("MEMOS_WRITABLE_CUBE_IDS") or _csv_env("MEMOS_CUBE_ID")
        if cube_ids:
            payload["writable_cube_ids"] = cube_ids
        self._post_json(_env("MEMOS_ADD_PATH", "/product/add"), payload)

    def _search_api(self, task: Task, k: int) -> list[MemoryEntry]:
        if not os.environ.get("MEMOS_BASE_URL"):
            return []
        payload: dict[str, Any] = {
            "user_id": _env("MEMOS_USER_ID", "safe-se-agent"),
            "query": self._query_for_task(task),
            "top_k": k,
            "mode": _env("MEMOS_SEARCH_MODE", "fine"),
            "include_preference": _env_bool("MEMOS_INCLUDE_PREFERENCE", default=False),
        }
        cube_ids = _csv_env("MEMOS_READABLE_CUBE_IDS") or _csv_env("MEMOS_CUBE_ID")
        if cube_ids:
            payload["readable_cube_ids"] = cube_ids
        response = self._post_json(_env("MEMOS_SEARCH_PATH", "/product/search"), payload)
        entries = self._entries_from_response(response, fallback_source="memos_api")
        self.last_retrieval_scores = [
            {
                "memory_id": entry.id,
                "score": entry.stats.get("memos_score"),
                "retrieved": True,
                "backend": "memos_api",
            }
            for entry in entries
        ]
        return entries

    def _post_json(self, path: str, payload: dict[str, Any]) -> Any:
        base_url = os.environ.get("MEMOS_BASE_URL", "").rstrip("/")
        url = f"{base_url}/{path.lstrip('/')}"
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("MEMOS_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(_env("MEMOS_TIMEOUT_S", "30"))) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._handle_backend_failure(f"MemOS API request failed: {exc}")
            return None
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}

    def _load_python_memory(self) -> Any | None:
        config_data = self._python_memory_config()
        if not config_data:
            return None
        self._add_local_memos_src_to_path()
        try:
            from memos.configs.memory import MemoryConfigFactory
            from memos.memories.factory import MemoryFactory
        except ImportError as exc:
            self._handle_backend_failure(f"MemOS Python import failed: {exc}")
            return None
        try:
            config = MemoryConfigFactory.model_validate(config_data)
            expected = {
                "memos_general_text": "general_text",
                "memos_tree_text": "tree_text",
            }[self.backend]
            if config.backend != expected:
                self._handle_backend_failure(f"{self.backend} expected config.backend={expected}, got {config.backend}.")
                return None
            return MemoryFactory.from_config(config)
        except Exception as exc:  # MemOS config errors vary by optional backend.
            self._handle_backend_failure(f"MemOS Python memory init failed: {exc}")
            return None

    def _python_memory_config(self) -> dict[str, Any] | None:
        raw = os.environ.get("MEMOS_MEMORY_CONFIG_JSON")
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                self._handle_backend_failure(f"MEMOS_MEMORY_CONFIG_JSON is not valid JSON: {exc}")
                return None
        path = os.environ.get("MEMOS_MEMORY_CONFIG")
        if not path:
            specific = "MEMOS_GENERAL_TEXT_CONFIG" if self.backend == "memos_general_text" else "MEMOS_TREE_TEXT_CONFIG"
            path = os.environ.get(specific)
        if not path:
            return None
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._handle_backend_failure(f"Could not read MemOS config {path}: {exc}")
            return None

    def _add_local_memos_src_to_path(self) -> None:
        local_src = Path(__file__).resolve().parents[2] / "MemOS" / "src"
        if local_src.exists() and str(local_src) not in sys.path:
            sys.path.insert(0, str(local_src))

    def _add_python_memory(self, entries: list[MemoryEntry]) -> None:
        if self._python_memory is None:
            return
        items = [
            {
                "id": entry.id,
                "memory": entry.text,
                "metadata": {
                    "safe_se_agent": memory_to_dict(entry),
                    "source": entry.source,
                    "tags": list(entry.tags),
                    "created_from": list(entry.created_from),
                },
            }
            for entry in entries
        ]
        add = getattr(self._python_memory, "add", None)
        if not callable(add):
            self._handle_backend_failure("MemOS Python memory does not expose add().")
            return
        user_name = _env("MEMOS_USER_ID", "safe-se-agent")
        try:
            add(items, user_name=user_name)
        except TypeError:
            self._call_backend(add, items)
        except Exception as exc:
            self._handle_backend_failure(f"MemOS Python memory add failed: {exc}")

    def _search_python_memory(self, task: Task, k: int) -> list[MemoryEntry]:
        if self._python_memory is None:
            return []
        search = getattr(self._python_memory, "search", None)
        if not callable(search):
            self._handle_backend_failure("MemOS Python memory does not expose search().")
            return []
        query = self._query_for_task(task)
        try:
            result = search(
                query,
                top_k=k,
                mode=_env("MEMOS_SEARCH_MODE", "fast"),
                user_name=_env("MEMOS_USER_ID", "safe-se-agent"),
            )
        except TypeError:
            try:
                result = search(query, top_k=k)
            except Exception as exc:
                self._handle_backend_failure(f"MemOS Python memory search failed: {exc}")
                return []
        except Exception as exc:
            self._handle_backend_failure(f"MemOS Python memory search failed: {exc}")
            return []
        entries = self._entries_from_response(result, fallback_source=self.backend)
        self.last_retrieval_scores = [
            {
                "memory_id": entry.id,
                "score": entry.stats.get("memos_score"),
                "retrieved": True,
                "backend": self.backend,
            }
            for entry in entries
        ]
        return entries

    def _entries_from_response(self, response: Any, fallback_source: str) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        for index, item in enumerate(self._iter_memory_items(response), start=1):
            entry = self._entry_from_item(item, fallback_source=fallback_source, index=index)
            if entry is not None:
                entries.append(entry)
        return entries

    def _iter_memory_items(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            items: list[Any] = []
            for item in value:
                items.extend(self._iter_memory_items(item) if self._is_container_wrapper(item) else [item])
            return items
        if hasattr(value, "model_dump"):
            return [value.model_dump()]
        if isinstance(value, dict):
            if self._looks_like_memory_item(value):
                return [value]
            for key in ("data", "memory_detail_list", "memories", "results", "text_mem", "items"):
                if key in value:
                    return self._iter_memory_items(value[key])
        return []

    def _is_container_wrapper(self, value: Any) -> bool:
        return isinstance(value, dict) and not self._looks_like_memory_item(value)

    def _looks_like_memory_item(self, value: dict[str, Any]) -> bool:
        return any(key in value for key in ("memory", "text", "content", "value", "page_content")) or (
            isinstance(value.get("metadata"), dict) and "safe_se_agent" in value["metadata"]
        )

    def _entry_from_item(self, item: Any, fallback_source: str, index: int) -> MemoryEntry | None:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if isinstance(item, str):
            text = item
            metadata: dict[str, Any] = {}
        elif isinstance(item, dict):
            metadata = self._metadata_from_item(item)
            safe_entry = metadata.get("safe_se_agent")
            if isinstance(safe_entry, dict):
                try:
                    return memory_from_dict(safe_entry)
                except (KeyError, TypeError, ValueError):
                    pass
            text = self._text_from_item(item)
        else:
            text = getattr(item, "memory", None) or getattr(item, "text", None)
            metadata = getattr(item, "metadata", {}) or {}
        if not text:
            return None
        memory_id = str(_first_present(item, ("id", "memory_id", "uuid")) or f"{fallback_source}_{index}")
        score = _first_present(item, ("score", "similarity", "distance"))
        stats = dict(metadata.get("stats", {})) if isinstance(metadata.get("stats"), dict) else {}
        if score is not None:
            stats["memos_score"] = score
        stats.setdefault("memos_backend", fallback_source)
        tags = metadata.get("tags", ())
        created_from = metadata.get("created_from", ())
        return MemoryEntry(
            id=memory_id,
            text=str(text),
            source=str(metadata.get("source") or fallback_source),
            tags=tuple(str(tag) for tag in _listish(tags)),
            priority=float(metadata.get("priority", 1.0)),
            created_from=tuple(str(source) for source in _listish(created_from)),
            stats=stats,
        )

    def _metadata_from_item(self, item: dict[str, Any]) -> dict[str, Any]:
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        payload = item.get("payload")
        if isinstance(payload, dict):
            payload_metadata = payload.get("metadata")
            if isinstance(payload_metadata, dict):
                return payload_metadata
        return {}

    def _text_from_item(self, item: dict[str, Any]) -> str | None:
        for key in ("memory", "text", "content", "value", "page_content"):
            value = item.get(key)
            if isinstance(value, str):
                return value
        payload = item.get("payload")
        if isinstance(payload, dict):
            for key in ("memory", "text", "content", "value", "page_content"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value
        return None

    def _query_for_task(self, task: Task) -> str:
        tags = ", ".join(task.tags)
        metadata = task.metadata
        kind = metadata.get("kind") or metadata.get("task_type") or metadata.get("dataset")
        context = f"\nTags: {tags}" if tags else ""
        if kind:
            context += f"\nKind: {kind}"
        return f"{task.question}{context}"

    def _call_backend(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            self._handle_backend_failure(f"MemOS backend call failed: {exc}")
            return None

    def _handle_backend_failure(self, message: str) -> None:
        self._backend_errors.append(message)
        if self.strict:
            raise MemOSMemoryDependencyError(message)


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _csv_env(name: str) -> list[str]:
    value = os.environ.get(name)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _first_present(value: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(value, dict):
        return None
    for key in keys:
        if key in value and value[key] is not None:
            return value[key]
    return None


def _listish(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]

