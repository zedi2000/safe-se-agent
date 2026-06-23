from safe_se_agent.adapters.simple import SimpleAgentAdapter
from safe_se_agent.core.memos_memory import MEMORY_BACKEND_CHOICES, MEMOS_MEMORY_BACKENDS, MemOSMemoryStore
from safe_se_agent.core.types import MemoryEntry, Task


def test_memos_api_backend_uses_jsonl_mirror_without_server(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MEMOS_BASE_URL", raising=False)
    store = MemOSMemoryStore(tmp_path / "memory.jsonl", backend="memos_api")
    store.add(
        [
            MemoryEntry(
                id="rule_1",
                text="When solving unit_conversion tasks, convert units before arithmetic.",
                source="reflection",
                tags=("unit_conversion",),
                created_from=("train_1", "train_2"),
            )
        ]
    )

    retrieved = store.retrieve(
        Task(
            id="eval_1",
            question="Convert centimeters to meters before totaling.",
            answer="1",
            tags=("unit_conversion",),
        )
    )

    assert store.export()[0].id == "rule_1"
    assert retrieved[0].id == "rule_1"
    assert store.last_retrieval_scores[0]["fallback"] == "jsonl_mirror"
    assert store.last_retrieval_scores[0]["backend"] == "memos_api"


def test_simple_adapter_accepts_all_memos_backend_labels(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MEMOS_BASE_URL", raising=False)
    for backend in MEMOS_MEMORY_BACKENDS:
        adapter = SimpleAgentAdapter(memory_path=tmp_path / f"{backend}.jsonl", memory_backend=backend)
        adapter.add_memory(
            [
                MemoryEntry(
                    id=f"{backend}_rule",
                    text="When solving total_with_fee tasks, add the explicit fee.",
                    source="reflection",
                    tags=("total_with_fee",),
                )
            ]
        )
        assert adapter.export_memory()[0].id == f"{backend}_rule"


def test_memory_backend_choices_include_memos_variants() -> None:
    assert {"simple", "langchain", *MEMOS_MEMORY_BACKENDS}.issubset(set(MEMORY_BACKEND_CHOICES))

