from __future__ import annotations

from typing import Protocol

from safe_se_agent.core.types import MemoryEntry, Task, Trajectory


class LLMClient(Protocol):
    def solve(self, task: Task, memories: list[MemoryEntry]) -> tuple[str, str, int | None] | tuple[str, str, int | None, str]:
        """Return answer, visible reasoning, optional token count, and optional raw response."""

    def reflect(self, trajectories: list[Trajectory]) -> list[str]:
        """Return plain-text memory rules distilled from trajectories."""
