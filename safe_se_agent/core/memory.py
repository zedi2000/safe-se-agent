from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import asdict
from collections import Counter
from pathlib import Path

from safe_se_agent.core.types import MemoryEntry, Task


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "before",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "then",
    "to",
    "when",
    "with",
}
GENERIC_RULE_TOKENS = {
    "answer",
    "calculation",
    "careful",
    "comput",
    "final",
    "finaliz",
    "intermediate",
    "multi",
    "problem",
    "quantitie",
    "step",
    "verify",
    "word",
}


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def normalize_memory_text(text: str) -> str:
    return " ".join(_content_tokens(text))


def _content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        if raw in STOPWORDS or len(raw) <= 2:
            continue
        token = raw
        for suffix in ("ing", "ed", "ly", "s"):
            if len(token) > len(suffix) + 4 and token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        tokens.append(token)
    return tokens


class MemoryStore:
    """可追加的 memory 视图，负责检索；backend 负责可选落盘。"""

    def __init__(self, backend: MemoryBackend | None = None) -> None:
        self.backend = backend
        self._entries: list[MemoryEntry] = backend.load() if backend else []

    def clear(self) -> None:
        self._entries.clear()
        if self.backend:
            self.backend.clear()

    def add(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        added: list[MemoryEntry] = []
        for entry in entries:
            if self._is_duplicate(entry, added):
                continue
            added.append(entry)
        self._entries.extend(added)
        if self.backend:
            self.backend.append(added)
        return added

    def save(self) -> None:
        if self.backend:
            self.backend.save(self._entries)

    def export(self) -> list[MemoryEntry]:
        return list(self._entries)

    def retrieve(self, task: Task, k: int = 3) -> list[MemoryEntry]:
        scored = [(self._score(entry, task), entry) for entry in self._entries]
        scored = [(score, entry) for score, entry in scored if score > 0]
        scored.sort(key=lambda item: (item[0], item[1].priority), reverse=True)
        return [entry for _, entry in scored[:k]]

    def _score(self, entry: MemoryEntry, task: Task) -> float:
        task_tags = set(task.tags)
        entry_tags = set(entry.tags)
        tag_overlap = len(task_tags & entry_tags)

        task_tokens = tokenize(task.question) | set(task.tags)
        entry_tokens = tokenize(entry.text) | set(entry.tags)
        if not task_tokens or not entry_tokens:
            lexical = 0.0
        else:
            lexical = len(task_tokens & entry_tokens) / len(task_tokens | entry_tokens)

        return entry.priority * (2.0 * tag_overlap + lexical)

    def _is_duplicate(self, candidate: MemoryEntry, pending: list[MemoryEntry] | None = None) -> bool:
        candidate_text = normalize_memory_text(candidate.text)
        if not candidate_text:
            return True
        candidate_tokens = tokenize(candidate_text)
        for entry in [*self._entries, *(pending or [])]:
            entry_text = normalize_memory_text(entry.text)
            if candidate_text == entry_text:
                return True
            entry_tokens = tokenize(entry_text)
            if not candidate_tokens or not entry_tokens:
                continue
            overlap = len(candidate_tokens & entry_tokens) / len(candidate_tokens | entry_tokens)
            containment = len(candidate_tokens & entry_tokens) / min(len(candidate_tokens), len(entry_tokens))
            generic_overlap = len((candidate_tokens & entry_tokens) & GENERIC_RULE_TOKENS)
            tag_overlap = bool(set(candidate.tags) & set(entry.tags))
            if overlap >= 0.72 or containment >= 0.82 or (tag_overlap and generic_overlap >= 7):
                return True
        return False


class MemoryBackend(ABC):
    """Memory 持久化后端接口；后续可替换为 SQLite 或框架原生 memory。"""

    @abstractmethod
    def load(self) -> list[MemoryEntry]:
        pass

    @abstractmethod
    def save(self, entries: list[MemoryEntry]) -> None:
        pass

    @abstractmethod
    def append(self, entries: list[MemoryEntry]) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def export(self) -> list[MemoryEntry]:
        pass


class JsonlMemoryBackend(MemoryBackend):
    """默认 JSONL memory 后端，便于人工检查和实验 diff。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        entries: list[MemoryEntry] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(memory_from_dict(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{line_no} is not valid JSONL") from exc
        return entries

    def save(self, entries: list[MemoryEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(memory_to_dict(entry), ensure_ascii=True) + "\n")

    def append(self, entries: list[MemoryEntry]) -> None:
        if not entries:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(memory_to_dict(entry), ensure_ascii=True) + "\n")

    def clear(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def export(self) -> list[MemoryEntry]:
        return self.load()


def memory_to_dict(entry: MemoryEntry) -> dict[str, object]:
    data = asdict(entry)
    data["tags"] = list(entry.tags)
    data["created_from"] = list(entry.created_from)
    return data


def memory_from_dict(data: dict[str, object]) -> MemoryEntry:
    return MemoryEntry(
        id=str(data["id"]),
        text=str(data["text"]),
        source=str(data["source"]),
        tags=tuple(str(tag) for tag in data.get("tags", ())),
        priority=float(data.get("priority", 1.0)),
        created_from=tuple(str(item) for item in data.get("created_from", ())),
        stats=dict(data.get("stats", {})),
    )


class MemoryIdFactory:
    def __init__(self, prefix: str = "mem") -> None:
        self.prefix = prefix
        self.counts: Counter[str] = Counter()

    def next(self, tag: str = "entry") -> str:
        key = re.sub(r"[^a-zA-Z0-9_]+", "_", tag).strip("_") or "entry"
        self.counts[key] += 1
        return f"{self.prefix}_{key}_{self.counts[key]}"
