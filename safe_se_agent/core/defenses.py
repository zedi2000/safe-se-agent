from __future__ import annotations

from safe_se_agent.adapters.base import AgentAdapter
from safe_se_agent.core.types import MemoryEntry, RunResult, Task, Trajectory


class AdapterWrapper(AgentAdapter):
    """Base class for future Milestone 3 defense wrappers."""

    def __init__(self, inner: AgentAdapter) -> None:
        self.inner = inner

    def reset(self, run_id: str) -> None:
        self.inner.reset(run_id)

    def solve(self, task: Task, memories: list[MemoryEntry] | None = None) -> RunResult:
        return self.inner.solve(task, memories)

    def reflect(self, trajectories: list[Trajectory]) -> list[MemoryEntry]:
        return self.inner.reflect(trajectories)

    def add_memory(self, entries: list[MemoryEntry]) -> None:
        self.inner.add_memory(entries)

    def retrieve(self, task: Task, k: int = 3) -> list[MemoryEntry]:
        return self.inner.retrieve(task, k)

    def export_memory(self) -> list[MemoryEntry]:
        return self.inner.export_memory()


class BufferedAdapter(AdapterWrapper):
    """Placeholder for delayed memory consolidation."""


class ValidatedMemoryAdapter(AdapterWrapper):
    """Placeholder for rule validation before memory writes."""


class CounterexampleSearchAdapter(AdapterWrapper):
    """Placeholder for active search against over-generalized rules."""


class DebateValidationAdapter(AdapterWrapper):
    """Placeholder for multi-agent debate before memory consolidation."""
