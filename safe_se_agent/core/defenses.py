from __future__ import annotations

from safe_se_agent.adapters.base import AgentAdapter
from safe_se_agent.core.promotion import PromotionDecision, PromotionPolicy, PromotionPolicyConfig
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

    def add_memory(self, entries: list[MemoryEntry], deduplicate: bool = True) -> None:
        self.inner.add_memory(entries, deduplicate=deduplicate)

    def retrieve(self, task: Task, k: int = 3) -> list[MemoryEntry]:
        return self.inner.retrieve(task, k)

    def export_memory(self) -> list[MemoryEntry]:
        return self.inner.export_memory()


class BufferedAdapter(AdapterWrapper):
    """Placeholder for delayed memory consolidation."""


class MemoryPromotionGuard(AdapterWrapper):
    """Safety layer for observation/case/rule/skill promotion decisions."""

    def __init__(
        self,
        inner: AgentAdapter,
        policy: PromotionPolicy | None = None,
        config: PromotionPolicyConfig | None = None,
        store_quarantine_cases: bool = True,
    ) -> None:
        super().__init__(inner)
        self.policy = policy or PromotionPolicy(config)
        self.store_quarantine_cases = store_quarantine_cases
        self.last_promotion_decisions: list[PromotionDecision] = []

    def add_memory(self, entries: list[MemoryEntry], deduplicate: bool = True) -> None:
        accepted: list[MemoryEntry] = []
        history = self.inner.export_memory()
        self.last_promotion_decisions = []
        for entry in entries:
            decision = self.policy.evaluate(entry, [*history, *accepted])
            self.last_promotion_decisions.append(decision)
            if decision.action in {"reject", "forget"}:
                continue
            if decision.action == "keep_case" and not self.store_quarantine_cases:
                continue
            accepted.append(self.policy.annotate(entry, decision))
        if accepted:
            self.inner.add_memory(accepted, deduplicate=deduplicate)

    def export_promotion_decisions(self) -> list[PromotionDecision]:
        return list(self.last_promotion_decisions)


class ValidatedMemoryAdapter(MemoryPromotionGuard):
    """Backward-compatible alias for guarded memory writes."""


class CounterexampleSearchAdapter(AdapterWrapper):
    """Placeholder for active search against over-generalized rules."""


class DebateValidationAdapter(AdapterWrapper):
    """Placeholder for multi-agent debate before memory consolidation."""
