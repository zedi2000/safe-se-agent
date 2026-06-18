from __future__ import annotations

import os

from safe_se_agent.core.prompts import (
    NO_MEMORY_BASELINE,
    REFLECTION_AND_RULE_DISTILLATION,
    SELF_EVOLUTION_INFERENCE,
)
from safe_se_agent.core.types import MemoryEntry, Task, Trajectory


class OpenAICompatibleClient:
    """Small optional backend for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use --mode llm.") from exc

        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )
        if not os.environ.get("OPENAI_API_KEY") and api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required for --mode llm.")

    def solve(self, task: Task, memories: list[MemoryEntry]) -> tuple[str, str, int | None]:
        memory_block = "\n".join(f"- {memory.text}" for memory in memories) or "(none)"
        system_prompt = SELF_EVOLUTION_INFERENCE if memories else NO_MEMORY_BASELINE
        prompt = (
            f"Reference memories:\n{memory_block}\n\n"
            f"Task: {task.question}\n"
            "Return brief reasoning followed by a final answer on the last line. "
            "The last line must contain only the final answer value, without currency symbols."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        answer = text.strip().split()[-1].strip(".")
        tokens = response.usage.total_tokens if response.usage else None
        return answer, text.strip(), tokens

    def reflect(self, trajectories: list[Trajectory]) -> list[str]:
        records = []
        for trajectory in trajectories:
            outcome = "positive" if trajectory.correct else "negative"
            records.append(
                (
                    f"Record type: {outcome}\n"
                    f"Task: {trajectory.task.question}\n"
                    f"Agent answer: {trajectory.answer}\n"
                    f"Expected answer: {trajectory.expected_answer}\n"
                    f"Correct: {trajectory.correct}\n"
                    f"Tags: {', '.join(trajectory.task.tags)}"
                )
            )
        prompt = (
            REFLECTION_AND_RULE_DISTILLATION
            + "\n\n"
            "For this implementation, return one concise memory entry. "
            "Do not include bullets, labels, or markdown.\n\n"
            + "\n\n".join(records)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        return [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
