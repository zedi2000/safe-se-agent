from safe_se_agent.core.memory import JsonlMemoryBackend, MemoryStore
from safe_se_agent.core.types import MemoryEntry, Task


def test_memory_retrieval_prefers_matching_tags() -> None:
    store = MemoryStore()
    store.add(
        [
            MemoryEntry(id="a", text="unrelated medical rule", source="test", tags=("medical",)),
            MemoryEntry(
                id="b",
                text="For total_with_fee tasks, add the explicit fee.",
                source="test",
                tags=("total_with_fee",),
            ),
        ]
    )
    task = Task(
        id="x",
        question="Add the delivery fee to the item total.",
        answer="10",
        tags=("total_with_fee",),
    )

    retrieved = store.retrieve(task, k=1)

    assert retrieved[0].id == "b"


def test_jsonl_memory_backend_round_trip(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    store = MemoryStore(JsonlMemoryBackend(path))
    entry = MemoryEntry(
        id="rule_1",
        text="For unit_conversion tasks, convert units before arithmetic.",
        source="reflection",
        tags=("arithmetic", "unit_conversion"),
        created_from=("train_conversion_1",),
    )
    store.add([entry])

    reloaded = MemoryStore(JsonlMemoryBackend(path))
    task = Task(
        id="eval_conversion",
        question="Convert centimeters to meters before totaling.",
        answer="1",
        tags=("arithmetic", "unit_conversion"),
    )

    assert path.exists()
    assert reloaded.export()[0].id == "rule_1"
    assert reloaded.retrieve(task, k=1)[0].id == "rule_1"


def test_memory_store_skips_duplicate_entries(tmp_path) -> None:
    path = tmp_path / "memory.jsonl"
    store = MemoryStore(JsonlMemoryBackend(path))
    first = MemoryEntry(
        id="rule_1",
        text="When solving math word problems, compute intermediate quantities before answering.",
        source="reflection",
        tags=("gsm8k",),
    )
    duplicate = MemoryEntry(
        id="rule_2",
        text="When solving math word problems compute intermediate quantities before answering",
        source="reflection",
        tags=("gsm8k",),
    )

    assert [entry.id for entry in store.add([first])] == ["rule_1"]
    assert store.add([duplicate]) == []
    assert [entry.id for entry in store.export()] == ["rule_1"]
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_memory_store_skips_semantically_similar_generic_entries(tmp_path) -> None:
    store = MemoryStore(JsonlMemoryBackend(tmp_path / "memory.jsonl"))
    first = MemoryEntry(
        id="rule_1",
        text="Carefully solve multi-step math word problems by computing intermediate quantities step by step before finalizing the answer.",
        source="reflection",
        tags=("math", "gsm8k"),
    )
    similar = MemoryEntry(
        id="rule_2",
        text="Use careful step-by-step calculation in multi-step word problems, verifying intermediate quantities before giving the final answer.",
        source="reflection",
        tags=("math", "gsm8k"),
    )

    added = store.add([first, similar])

    assert [entry.id for entry in added] == ["rule_1"]
    assert [entry.id for entry in store.export()] == ["rule_1"]
