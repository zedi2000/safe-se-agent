from __future__ import annotations

from abc import ABC, abstractmethod

from safe_se_agent.core.types import MemoryEntry, RunResult, Task, Trajectory


class AgentAdapter(ABC):
    """Common interface for any tested agent framework."""

    @abstractmethod
    def reset(self, run_id: str) -> None:
        """Clear or isolate state for one experimental run."""

    @abstractmethod
    def solve(self, task: Task, memories: list[MemoryEntry] | None = None) -> RunResult:
        """Solve one task and return the full trajectory."""

    @abstractmethod
    def reflect(self, trajectories: list[Trajectory]) -> list[MemoryEntry]:
        """Distill reusable memory entries from trajectories."""

    @abstractmethod
    def add_memory(self, entries: list[MemoryEntry], deduplicate: bool = True) -> None:
        """Persist memory entries into the tested agent."""

    @abstractmethod
    def retrieve(self, task: Task, k: int = 3) -> list[MemoryEntry]:
        """Retrieve memories relevant to a task."""

    @abstractmethod
    def export_memory(self) -> list[MemoryEntry]:
        """Export current memory for ESR/debug/defense analysis."""
