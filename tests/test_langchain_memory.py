import pytest

from safe_se_agent.core.langchain_memory import LangChainMemoryDependencyError, LangChainMemoryStore
from safe_se_agent.core.types import MemoryEntry, Task


class FakeDocument:
    def __init__(self, page_content: str, metadata: dict) -> None:
        self.page_content = page_content
        self.metadata = metadata


class FakeVectorStore:
    def __init__(self, embedding) -> None:
        self.embedding = embedding
        self.documents: list[FakeDocument] = []

    def add_documents(self, documents: list[FakeDocument], ids: list[str]) -> None:
        self.documents.extend(documents)

    def similarity_search_with_score(self, query: str, k: int = 4):
        scored = []
        query_tokens = set(query.lower().split())
        for doc in self.documents:
            doc_tokens = set(doc.page_content.lower().split())
            score = len(query_tokens & doc_tokens) / max(1, len(query_tokens | doc_tokens))
            scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def max_marginal_relevance_search(self, query: str, k: int = 4):
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k)]


class FakeEmbedding:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def build_store(tmp_path, threshold: float = 0.0) -> LangChainMemoryStore:
    return LangChainMemoryStore(
        tmp_path / "memory.jsonl",
        embedding=FakeEmbedding(),
        search_type="similarity_score_threshold",
        score_threshold=threshold,
        vector_store_cls=FakeVectorStore,
        document_cls=FakeDocument,
    )


def test_langchain_memory_store_indexes_jsonl_entries(tmp_path) -> None:
    store = build_store(tmp_path)
    entry = MemoryEntry(
        id="division_rule",
        text="Use floor division for complete groups.",
        source="test",
        tags=("math",),
    )

    store.add([entry])
    reloaded = build_store(tmp_path)
    task = Task(id="t1", question="How many complete groups can be made by division?", answer="1")

    retrieved = reloaded.retrieve(task, k=1)

    assert [memory.id for memory in retrieved] == ["division_rule"]
    assert reloaded.export()[0].id == "division_rule"
    assert reloaded.last_retrieval_scores[0]["memory_id"] == "division_rule"
    assert reloaded.last_retrieval_scores[0]["backend"] == "langchain"


def test_langchain_memory_store_threshold_can_return_empty(tmp_path) -> None:
    store = build_store(tmp_path, threshold=0.99)
    store.add([MemoryEntry(id="rule", text="Use floor division.", source="test")])
    task = Task(id="t1", question="Unrelated percentage problem", answer="1")

    assert store.retrieve(task, k=1) == []
    assert store.last_retrieval_scores[0]["memory_id"] == "rule"
    assert store.last_retrieval_scores[0]["retrieved"] is False
    assert store.last_retrieval_scores[0]["score_threshold"] == 0.99


def test_langchain_memory_store_add_updates_index(tmp_path) -> None:
    store = build_store(tmp_path)
    task = Task(id="t1", question="complete groups division", answer="1")

    assert store.retrieve(task, k=1) == []
    store.add([MemoryEntry(id="rule", text="complete groups division", source="test")])

    assert [memory.id for memory in store.retrieve(task, k=1)] == ["rule"]


def test_langchain_memory_store_missing_dependency_message(tmp_path) -> None:
    with pytest.raises(LangChainMemoryDependencyError, match="pip install -e"):
        LangChainMemoryStore(tmp_path / "memory.jsonl", embedding=FakeEmbedding())
