from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from safe_se_agent.core.memory import JsonlMemoryBackend, MemoryStore, memory_to_dict
from safe_se_agent.core.types import MemoryEntry, Task


class LangChainMemoryDependencyError(RuntimeError):
    pass


class OpenAICompatibleEmbeddings:
    """Small LangChain-compatible embedding wrapper for OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use OpenAI-compatible embeddings.") from exc

        self.model = model or os.environ.get("EMBEDDING_MODEL") or os.environ.get(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
        embedding_key = api_key or os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not embedding_key:
            raise RuntimeError("OPENAI_API_KEY or EMBEDDING_API_KEY is required for LangChain memory embeddings.")
        self.client = OpenAI(
            api_key=embedding_key,
            base_url=base_url or os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [list(item.embedding) for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LangChainMemoryStore:
    """JSONL-backed memory with a LangChain in-memory vector index for retrieval."""

    def __init__(
        self,
        memory_path: str | Path,
        embedding: Any | None = None,
        embedding_model: str | None = None,
        search_type: str = "similarity_score_threshold",
        score_threshold: float = 0.35,
        vector_store_cls: Any | None = None,
        document_cls: Any | None = None,
    ) -> None:
        self.memory_path = Path(memory_path)
        self.source = MemoryStore(JsonlMemoryBackend(self.memory_path))
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.last_retrieval_scores: list[dict[str, object]] = []
        self._memory_by_id: dict[str, MemoryEntry] = {}
        self._vector_store_cls = vector_store_cls
        self._document_cls = document_cls
        self._load_langchain_types()
        self.embedding = embedding or OpenAICompatibleEmbeddings(model=embedding_model)
        self._rebuild_index()

    def _load_langchain_types(self) -> None:
        if self._vector_store_cls is not None and self._document_cls is not None:
            return
        try:
            from langchain_core.documents import Document
            from langchain_core.vectorstores import InMemoryVectorStore
        except ImportError as exc:
            raise LangChainMemoryDependencyError(
                "LangChain memory backend requires langchain-core. "
                "Install it with: pip install -e '.[langchain]'"
            ) from exc
        self._document_cls = self._document_cls or Document
        self._vector_store_cls = self._vector_store_cls or InMemoryVectorStore

    def clear(self) -> None:
        self.source.clear()
        self._rebuild_index()

    def add(self, entries: list[MemoryEntry], deduplicate: bool = True) -> list[MemoryEntry]:
        added = self.source.add(entries, deduplicate=deduplicate)
        self._index_entries(added)
        return added

    def save(self) -> None:
        self.source.save()
        self._rebuild_index()

    def export(self) -> list[MemoryEntry]:
        return self.source.export()

    def retrieve(self, task: Task, k: int = 3) -> list[MemoryEntry]:
        query = self._query_for_task(task)
        self.last_retrieval_scores = []
        if self.search_type == "mmr":
            docs = self.vector_store.max_marginal_relevance_search(query, k=k)
            scored = [(doc, None) for doc in docs]
        else:
            raw_scored = self.vector_store.similarity_search_with_score(query, k=k)
            scored = raw_scored
            if self.search_type == "similarity_score_threshold":
                scored = [(doc, score) for doc, score in raw_scored if score >= self.score_threshold]
                if not scored:
                    self._record_unretrieved_scores(raw_scored)
        retrieved: list[MemoryEntry] = []
        for doc, score in scored:
            memory_id = str(doc.metadata.get("memory_id"))
            memory = self._memory_by_id.get(memory_id)
            if memory is None:
                continue
            retrieved.append(memory)
            self.last_retrieval_scores.append(
                {
                    "memory_id": memory_id,
                    "score": score,
                    "retrieved": True,
                    "backend": "langchain",
                    "search_type": self.search_type,
                }
            )
        return retrieved[:k]

    def _record_unretrieved_scores(self, scored: list[tuple[Any, float]]) -> None:
        for doc, score in scored:
            memory_id = str(doc.metadata.get("memory_id"))
            if memory_id not in self._memory_by_id:
                continue
            self.last_retrieval_scores.append(
                {
                    "memory_id": memory_id,
                    "score": score,
                    "retrieved": False,
                    "backend": "langchain",
                    "search_type": self.search_type,
                    "score_threshold": self.score_threshold,
                }
            )

    def _rebuild_index(self) -> None:
        self.vector_store = self._vector_store_cls(embedding=self.embedding)
        self._memory_by_id = {}
        self._index_entries(self.source.export())

    def _index_entries(self, entries: list[MemoryEntry]) -> None:
        if not entries:
            return
        docs = []
        ids = []
        for entry in entries:
            self._memory_by_id[entry.id] = entry
            docs.append(
                self._document_cls(
                    page_content=entry.text,
                    metadata={
                        "memory_id": entry.id,
                        "tags": list(entry.tags),
                        "source": entry.source,
                        "priority": entry.priority,
                        "created_from": list(entry.created_from),
                        "stats": entry.stats,
                        "memory": memory_to_dict(entry),
                    },
                )
            )
            ids.append(entry.id)
        self.vector_store.add_documents(documents=docs, ids=ids)

    def _query_for_task(self, task: Task) -> str:
        tags = ", ".join(task.tags)
        metadata = task.metadata
        kind = metadata.get("kind") or metadata.get("task_type") or metadata.get("dataset")
        context = f"\nTags: {tags}" if tags else ""
        if kind:
            context += f"\nKind: {kind}"
        return f"{task.question}{context}"
