from safe_se_agent.adapters.base import AgentAdapter
from safe_se_agent.adapters.simple import SimpleAgentAdapter
from safe_se_agent.core.io import load_jsonl_tasks


def test_simple_agent_adapter_contract_cycle(tmp_path) -> None:
    adapter: AgentAdapter = SimpleAgentAdapter(memory_path=tmp_path / "memory.jsonl")
    adapter.reset("contract")
    train_tasks = load_jsonl_tasks("data/m1_train.jsonl")
    eval_task = load_jsonl_tasks("data/m1_eval.jsonl")[0]

    trajectory = adapter.solve(train_tasks[0], memories=[]).trajectory
    memories = adapter.reflect([trajectory])
    adapter.add_memory(memories)
    retrieved = adapter.retrieve(eval_task, k=3)
    result = adapter.solve(eval_task, memories=retrieved)

    assert memories
    assert retrieved
    assert result.correct is True
    assert result.token_count is None
    assert adapter.export_memory()[0].text


def test_numeric_answer_normalization_handles_formatting(tmp_path) -> None:
    adapter = SimpleAgentAdapter(memory_path=tmp_path / "memory.jsonl")

    assert adapter._normalize("$54") == adapter._normalize("54")
    assert adapter._normalize("74.8") == adapter._normalize("74.80")
