#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
from typing import Any


def main() -> int:
    report: dict[str, Any] = {
        "env": read_env(),
        "embedder_config": None,
        "embedding_check": None,
        "qdrant_check": None,
    }

    try:
        from memos.api.config import APIConfig
        from memos.configs.embedder import EmbedderConfigFactory
        from memos.embedders.factory import EmbedderFactory
    except Exception as exc:
        report["embedding_check"] = {"ok": False, "error": f"import failed: {exc!r}"}
        print_report(report)
        return 1

    try:
        cfg = APIConfig.get_embedder_config()
        report["embedder_config"] = mask_config(cfg)
        embedder = EmbedderFactory.from_config(EmbedderConfigFactory.model_validate(cfg))
        vec = embedder.embed(["safe-se-agent embedding smoke test"])[0]
        expected_dim = parse_int(os.getenv("EMBEDDING_DIMENSION"))
        report["embedding_check"] = {
            "ok": expected_dim is None or len(vec) == expected_dim,
            "returned_dim": len(vec),
            "expected_dim": expected_dim,
            "first_5_values": vec[:5],
        }
    except Exception as exc:
        report["embedding_check"] = {"ok": False, "error": repr(exc)}

    report["qdrant_check"] = check_qdrant()
    print_report(report)

    if not report["embedding_check"] or not report["embedding_check"].get("ok"):
        return 1
    return 0


def read_env() -> dict[str, str | None]:
    keys = [
        "MOS_EMBEDDER_BACKEND",
        "MOS_EMBEDDER_PROVIDER",
        "MOS_EMBEDDER_MODEL",
        "MOS_EMBEDDER_API_BASE",
        "MOS_EMBEDDER_API_KEY",
        "EMBEDDING_DIMENSION",
        "QDRANT_HOST",
        "QDRANT_PORT",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_DB_NAME",
    ]
    result: dict[str, str | None] = {}
    for key in keys:
        value = os.getenv(key)
        if value and "KEY" in key:
            value = mask_secret(value)
        result[key] = value
    return result


def check_qdrant() -> dict[str, Any]:
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        return {"ok": False, "error": f"qdrant import failed: {exc!r}"}

    try:
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
        collections = client.get_collections()
        names = [collection.name for collection in collections.collections]
        detail: dict[str, Any] = {"ok": True, "collections": names}
        if "neo4j_vec_db" in names:
            detail["neo4j_vec_db"] = str(client.get_collection("neo4j_vec_db"))
        return detail
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def mask_config(value: Any) -> Any:
    if isinstance(value, dict):
        masked: dict[str, Any] = {}
        for key, item in value.items():
            if "key" in key.lower() and isinstance(item, str):
                masked[key] = mask_secret(item)
            else:
                masked[key] = mask_config(item)
        return masked
    if isinstance(value, list):
        return [mask_config(item) for item in value]
    return value


def mask_secret(value: str) -> str:
    if len(value) <= 10:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def parse_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def print_report(report: dict[str, Any]) -> None:
    print("MemOS container embedding check")
    embedding = report.get("embedding_check") or {}
    qdrant = report.get("qdrant_check") or {}
    print(f"- embedding_ok: {embedding.get('ok')}")
    if "returned_dim" in embedding:
        print(f"- returned_dim: {embedding.get('returned_dim')}")
    if "expected_dim" in embedding:
        print(f"- expected_dim: {embedding.get('expected_dim')}")
    if embedding.get("error"):
        print(f"- embedding_error: {embedding['error']}")
    print(f"- qdrant_ok: {qdrant.get('ok')}")
    if qdrant.get("collections") is not None:
        print(f"- qdrant_collections: {qdrant['collections']}")
    if qdrant.get("error"):
        print(f"- qdrant_error: {qdrant['error']}")
    print("\nFull JSON:")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())

