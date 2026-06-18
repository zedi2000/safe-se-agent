from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from safe_se_agent.core.types import MemoryEntry, Task, Trajectory


class OfflineLLMClient:
    """Deterministic toy backend for local reproducibility.

    It intentionally starts with narrow heuristics and only applies the complete
    method after reflection has produced a matching memory rule.
    """

    def solve(self, task: Task, memories: list[MemoryEntry]) -> tuple[str, str, int | None]:
        kind = str(task.metadata.get("kind", ""))
        has_rule = self._has_relevant_rule(kind, memories)
        answer = self._solve_task(task, has_rule)
        rule_ids = ", ".join(memory.id for memory in memories) or "none"
        reasoning = (
            f"offline_solver kind={kind or 'unknown'} retrieved={rule_ids}; "
            f"{'used learned complete rule' if has_rule else 'used naive baseline heuristic'}"
        )
        return answer, reasoning, None

    def reflect(self, trajectories: list[Trajectory]) -> list[str]:
        grouped: dict[str, list[Trajectory]] = defaultdict(list)
        for trajectory in trajectories:
            kind = str(trajectory.task.metadata.get("kind", ""))
            if kind and not trajectory.correct:
                grouped[kind].append(trajectory)

        rules: list[str] = []
        for kind in sorted(grouped):
            if kind == "total_with_fee":
                rules.append(
                    "For total_with_fee arithmetic tasks, compute quantity times unit price "
                    "and then add every explicit extra fee before answering."
                )
            elif kind == "discount_then_tax":
                rules.append(
                    "For discount_then_tax arithmetic tasks, apply the discount first, "
                    "then compute tax on the discounted subtotal."
                )
            elif kind == "unit_conversion":
                rules.append(
                    "For unit_conversion arithmetic tasks, convert each measured value into "
                    "the requested target unit before doing the final arithmetic."
                )
            elif kind == "average_with_extra_item":
                rules.append(
                    "For average_with_extra_item arithmetic tasks, include every listed item "
                    "and the extra item before computing the average."
                )
            elif kind == "gsm8k":
                rules.append(
                    "For GSM8K math word problems, identify the quantities, follow the gold "
                    "rationale style step by step, and return the final numeric answer."
                )
            else:
                rules.append(
                    f"For {kind} tasks, compare the failed answer with the expected answer "
                    "and preserve the missing operation as a reusable rule."
                )
        return rules

    def _has_relevant_rule(self, kind: str, memories: list[MemoryEntry]) -> bool:
        if not kind:
            return False
        return any(kind in memory.tags or kind in memory.text for memory in memories)

    def _solve_task(self, task: Task, has_rule: bool) -> str:
        kind = str(task.metadata.get("kind", ""))
        data = task.metadata
        if data.get("dataset") == "gsm8k":
            return str(task.metadata.get("baseline_answer", "0"))
        if kind == "total_with_fee":
            subtotal = self._num(data, "quantity") * self._num(data, "unit_price")
            total = subtotal + self._num(data, "fee") if has_rule else subtotal
            return self._format(total)
        if kind == "discount_then_tax":
            price = self._num(data, "price")
            discount_rate = self._num(data, "discount_rate")
            tax_rate = self._num(data, "tax_rate")
            if has_rule:
                total = price * (Decimal("1") - discount_rate) * (Decimal("1") + tax_rate)
            else:
                total = price * (Decimal("1") + tax_rate) - price * discount_rate
            return self._format(total)
        if kind == "unit_conversion":
            count = self._num(data, "count")
            value = self._num(data, "value")
            factor = self._num(data, "factor_to_target")
            total = count * value * factor if has_rule else count * value
            return self._format(total)
        if kind == "average_with_extra_item":
            values = [Decimal(str(item)) for item in data["values"]]
            if has_rule:
                values.append(self._num(data, "extra_value"))
            total = sum(values) / Decimal(len(values))
            return self._format(total)
        return str(task.metadata.get("baseline_answer", "unknown"))

    def _num(self, data: dict[str, Any], key: str) -> Decimal:
        return Decimal(str(data[key]))

    def _format(self, value: Decimal) -> str:
        normalized = value.quantize(Decimal("0.01"))
        if normalized == normalized.to_integral():
            return str(int(normalized))
        return f"{normalized:.2f}"
