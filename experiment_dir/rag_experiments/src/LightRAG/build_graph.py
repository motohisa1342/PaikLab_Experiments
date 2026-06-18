from __future__ import annotations

import shutil
import time
from pathlib import Path

try:
    from .init import initialize_rag, load_config, resolve_project_path
except ImportError:
    from init import initialize_rag, load_config, resolve_project_path


async def build_graph(config_path: str | Path | None = None) -> None:
    config = load_config(config_path) if config_path else load_config()
    lightrag_config = config["lightrag"]
    data_config = config["data"]
    index_config = config.get("index", {})

    working_dir = resolve_project_path(lightrag_config["working_dir"])
    source_path = resolve_project_path(data_config["source_path"])

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {source_path}")

    if index_config.get("rebuild", False) and working_dir.exists():
        shutil.rmtree(working_dir)

    working_dir.mkdir(parents=True, exist_ok=True)
    rag = await initialize_rag(config)

    if config.get("embedding", {}).get("validate_dimension", True):
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        detected_dim = embedding.shape[1]
        expected_dim = rag.embedding_func.embedding_dim
        if detected_dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: got {detected_dim}, expected {expected_dim}"
            )

    print(f"Building LightRAG graph from: {source_path}")
    print(f"Graph output directory: {working_dir}")
    start = time.perf_counter_ns()
    with open(source_path, "r", encoding=data_config.get("encoding", "utf-8")) as file:
        await rag.ainsert(file.read())
    end = time.perf_counter_ns()
    elapsed_time = end - start
    print("LightRAG graph build finished.")
    print(f"Build Implementation time: {elapsed_time} ns")

