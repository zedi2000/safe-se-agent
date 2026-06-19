from safe_se_agent.llm.openai_compatible import OpenAICompatibleClient


def test_connection_error_message_is_diagnostic_without_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    client = object.__new__(OpenAICompatibleClient)
    client.model = "test-model"

    message = client._format_connection_error(RuntimeError("boom"))

    assert "api_key_set: True" in message
    assert "HTTPS_PROXY" in message
    assert "secret-value" not in message
    assert "RuntimeError: boom" in message


def test_reflection_memory_filter_keeps_final_visible_lesson() -> None:
    client = object.__new__(OpenAICompatibleClient)

    assert client._looks_like_memory("<think>hidden</think>") is False
    assert client._looks_like_memory("I need to analyze these records") is False
    assert client._looks_like_memory(
        "When solving multi-step math word problems, compute each intermediate quantity before using it."
    )


def test_prompt_recorder_captures_messages() -> None:
    events = []
    client = object.__new__(OpenAICompatibleClient)
    client.model = "test-model"
    client.prompt_recorder = events.append

    client._record_prompt(
        kind="solve",
        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
        metadata={"task_id": "x"},
    )

    assert events == [
        {
            "kind": "solve",
            "model": "test-model",
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
            "metadata": {"task_id": "x"},
        }
    ]
