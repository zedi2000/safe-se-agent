from safe_se_agent.core.types import MemoryEntry, Task
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


class _FakeChatCompletions:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient")
        return "ok"


class _FakeClient:
    def __init__(self, failures: int) -> None:
        self.completions = _FakeChatCompletions(failures)
        self.chat = self


class _FakeMessage:
    content = "Reasoning\n42"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = None


class _CapturingChatCompletions:
    def __init__(self) -> None:
        self.messages = None

    def create(self, **kwargs):
        self.messages = kwargs["messages"]
        return _FakeResponse()


class _CapturingClient:
    def __init__(self) -> None:
        self.completions = _CapturingChatCompletions()
        self.chat = self


def test_chat_completion_retries_transient_failures() -> None:
    client = object.__new__(OpenAICompatibleClient)
    client.model = "test-model"
    client.max_retries = 3
    client.retry_backoff_s = 0
    client.client = _FakeClient(failures=2)

    assert client._create_chat_completion(messages=[]) == "ok"
    assert client.client.completions.calls == 3


def test_chat_completion_raises_after_retry_budget() -> None:
    client = object.__new__(OpenAICompatibleClient)
    client.model = "test-model"
    client.max_retries = 2
    client.retry_backoff_s = 0
    client.client = _FakeClient(failures=3)

    try:
        client._create_chat_completion(messages=[])
    except Exception as exc:
        assert "RuntimeError: transient" in str(exc)
    else:
        raise AssertionError("expected retry exhaustion")
    assert client.client.completions.calls == 2


def test_solve_places_memory_in_system_prompt_not_user_prompt() -> None:
    client = object.__new__(OpenAICompatibleClient)
    client.model = "test-model"
    client.memory_system_prompt = "Use memories."
    client.memory_header = "Memory entry"
    client.task_header = "Current problem"
    client.prompt_recorder = None
    client.max_retries = 1
    client.retry_backoff_s = 0
    client.client = _CapturingClient()

    task = Task(id="t1", question="What is 40 + 2?", answer="42")
    memory = MemoryEntry(id="m1", text="Always add carefully.", source="test")

    client.solve(task, [memory])

    messages = client.client.completions.messages
    assert messages[0]["role"] == "system"
    assert "Memory entry:" in messages[0]["content"]
    assert "Always add carefully." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Current problem: What is 40 + 2?" in messages[1]["content"]
    assert "Memory entry:" not in messages[1]["content"]
    assert "Always add carefully." not in messages[1]["content"]


def test_solve_records_retrieval_scores_in_prompt_metadata() -> None:
    events = []
    client = object.__new__(OpenAICompatibleClient)
    client.model = "test-model"
    client.memory_system_prompt = "Use memories."
    client.memory_header = "Memory entry"
    client.task_header = "Current problem"
    client.prompt_recorder = events.append
    client.max_retries = 1
    client.retry_backoff_s = 0
    client.always_use_memory_system_prompt = False
    client.client = _CapturingClient()
    client.set_next_retrieval_scores(
        [
            {
                "memory_id": "m1",
                "score": 0.72,
                "retrieved": True,
                "backend": "langchain",
            }
        ]
    )

    task = Task(id="t1", question="What is 40 + 2?", answer="42")
    memory = MemoryEntry(id="m1", text="Always add carefully.", source="test")

    client.solve(task, [memory])

    metadata = events[0]["metadata"]
    assert metadata["memory_ids"] == ["m1"]
    assert metadata["retrieval_scores"][0]["memory_id"] == "m1"
    assert metadata["retrieval_scores"][0]["score"] == 0.72
    assert metadata["retrieved_memory_scores"] == [{"memory_id": "m1", "score": 0.72}]


def test_solve_can_use_memory_system_prompt_even_without_retrieved_memory() -> None:
    client = object.__new__(OpenAICompatibleClient)
    client.model = "test-model"
    client.memory_system_prompt = "Use OEP memory protocol."
    client.memory_header = "Memory entry"
    client.task_header = "Current problem"
    client.prompt_recorder = None
    client.max_retries = 1
    client.retry_backoff_s = 0
    client.always_use_memory_system_prompt = True
    client.client = _CapturingClient()

    task = Task(id="t1", question="What is 40 + 2?", answer="42")

    client.solve(task, [])

    messages = client.client.completions.messages
    assert messages[0]["role"] == "system"
    assert "Use OEP memory protocol." in messages[0]["content"]
    assert "Memory entry:\n(none)" in messages[0]["content"]
    assert "Please solve the following problem step by step" not in messages[0]["content"]
