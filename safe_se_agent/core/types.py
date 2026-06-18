from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    answer: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    id: str
    text: str
    source: str
    tags: tuple[str, ...] = ()
    priority: float = 1.0
    created_from: tuple[str, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task: Task
    answer: str
    correct: bool
    reasoning: str
    raw_response: str | None = None
    retrieved_memory_ids: tuple[str, ...] = ()
    expected_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    task_id: str
    answer: str
    correct: bool | None
    reasoning: str
    trajectory: Trajectory
    raw_response: str | None = None
    retrieved_memory_ids: tuple[str, ...] = ()
    token_count: int | None = None
    latency_s: float | None = None
    steps: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttackCase:
    task: Task
    local_solution: str
    consequence: str
    target_rule: str
    metadata: dict[str, Any] = field(default_factory=dict)
