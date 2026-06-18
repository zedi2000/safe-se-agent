from __future__ import annotations

import os

from safe_se_agent.core.prompts import (
    BENIGN_MEMORY_SOLVE,
    BENIGN_NO_MEMORY_SOLVE,
    BENIGN_REFLECTION,
)
from safe_se_agent.core.text import strip_think_blocks
from safe_se_agent.core.types import MemoryEntry, Task, Trajectory


class LLMConnectionError(RuntimeError):
    pass


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
        if not os.environ.get("OPENAI_API_KEY") and api_key is None:
            raise RuntimeError("OPENAI_API_KEY is required for --mode llm.")
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
        )

    def solve(self, task: Task, memories: list[MemoryEntry]) -> tuple[str, str, int | None, str]:
        memory_block = "\n".join(f"- {memory.text}" for memory in memories) or "(none)"
        system_prompt = BENIGN_MEMORY_SOLVE if memories else BENIGN_NO_MEMORY_SOLVE
        prompt = (
            f"Reference memories:\n{memory_block}\n\n"
            f"Task: {task.question}\n"
            "Return brief reasoning followed by a final answer on the last line. "
            "The last line must contain only the final answer value, without currency symbols."
        )
        response = self._create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        text = response.choices[0].message.content or ""
        visible_text = strip_think_blocks(text) or text.strip()
        answer = visible_text.strip().split()[-1].strip(".")
        tokens = response.usage.total_tokens if response.usage else None
        return answer, visible_text.strip(), tokens, text.strip()

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
                    f"Tags: {', '.join(trajectory.task.tags)}\n"
                    f"Gold rationale: {trajectory.task.metadata.get('rationale', '(none)')}"
                )
            )
        prompt = (
            BENIGN_REFLECTION
            + "\n\n"
            "For this implementation, return one concise memory entry. "
            "Do not include bullets, labels, or markdown.\n\n"
            + "\n\n".join(records)
        )
        response = self._create_chat_completion(messages=[{"role": "user", "content": prompt}])
        text = response.choices[0].message.content or ""
        cleaned = strip_think_blocks(text)
        candidates = [line.strip("- ").strip() for line in cleaned.splitlines() if line.strip()]
        candidates = [line for line in candidates if self._looks_like_memory(line)]
        return candidates[-1:] if candidates else []

    def _looks_like_memory(self, text: str) -> bool:
        lowered = text.lower()
        if "<think" in lowered or "</think>" in lowered:
            return False
        if lowered.startswith(("i need", "looking at", "common patterns", "key insights")):
            return False
        if len(text.split()) < 6:
            return False
        return True

    def _create_chat_completion(self, messages: list[dict[str, str]]):
        try:
            return self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=messages,
            )
        except Exception as exc:
            raise LLMConnectionError(self._format_connection_error(exc)) from exc

    def _format_connection_error(self, exc: Exception) -> str:
        proxy_vars = [
            name
            for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")
            if os.environ.get(name)
        ]
        base_url = os.environ.get("OPENAI_BASE_URL", "<default>")
        proxy_text = ", ".join(proxy_vars) if proxy_vars else "<none>"
        return (
            "LLM API 调用失败，当前错误发生在模型请求阶段，而不是数据集或 memory 逻辑。\n"
            f"- model: {self.model}\n"
            f"- base_url: {base_url}\n"
            f"- api_key_set: {bool(os.environ.get('OPENAI_API_KEY'))}\n"
            f"- proxy_vars_set: {proxy_text}\n"
            f"- original_error: {exc.__class__.__name__}: {exc}\n"
            "建议检查：OPENAI_API_KEY 是否在当前 shell 中 export；OPENAI_BASE_URL 是否正确；"
            "本地代理是否可用；如果看到 SSL EOF/ConnectError，优先检查代理或兼容接口的 HTTPS 配置。"
        )
