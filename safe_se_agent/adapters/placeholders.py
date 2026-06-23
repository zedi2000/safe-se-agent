from __future__ import annotations

from safe_se_agent.adapters.base import AgentAdapter
from safe_se_agent.core.types import MemoryEntry, RunResult, Task, Trajectory


class _PlannedExternalAdapter(AgentAdapter):
    framework_name = "external"

    def _not_ready(self) -> None:
        raise NotImplementedError(
            f"{self.framework_name} integration is reserved for later milestones. "
            "Implement this adapter by mapping the framework's solve, memory, "
            "reflection, and export hooks to AgentAdapter."
        )

    def reset(self, run_id: str) -> None:
        self._not_ready()

    def solve(self, task: Task, memories: list[MemoryEntry] | None = None) -> RunResult:
        self._not_ready()

    def reflect(self, trajectories: list[Trajectory]) -> list[MemoryEntry]:
        self._not_ready()

    def add_memory(self, entries: list[MemoryEntry], deduplicate: bool = True) -> None:
        self._not_ready()

    def retrieve(self, task: Task, k: int = 3) -> list[MemoryEntry]:
        self._not_ready()

    def export_memory(self) -> list[MemoryEntry]:
        self._not_ready()


class OpenClawAdapter(_PlannedExternalAdapter):
    framework_name = "OpenClaw"


class LangChainAdapter(_PlannedExternalAdapter):
    framework_name = "LangChain"
