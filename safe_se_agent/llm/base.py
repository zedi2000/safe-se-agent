from __future__ import annotations

from typing import Protocol

from safe_se_agent.core.types import MemoryEntry, Task, Trajectory


class LLMClient(Protocol):
    def solve(self, task: Task, memories: list[MemoryEntry]) -> tuple[str, str, int | None]:
        """Return answer, reasoning, and optional token count."""

    def reflect(self, trajectories: list[Trajectory]) -> list[str]:
        """Return plain-text memory rules distilled from trajectories."""
