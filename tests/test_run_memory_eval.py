from pathlib import Path

from scripts.run_memory_eval import build_memory_eval_adapter
from safe_se_agent.core.prompts import FORCE_OEP_INFERENCE


def test_force_oep_prompt_protocol_uses_force_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    adapter = build_memory_eval_adapter(
        mode="llm",
        retrieve_k=1,
        memory_path=tmp_path / "memory.jsonl",
        prompt_protocol="force_oep",
    )

    assert adapter.llm.memory_system_prompt == FORCE_OEP_INFERENCE
    assert adapter.llm.always_use_memory_system_prompt is True
