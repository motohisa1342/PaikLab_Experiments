from __future__ import annotations

import shutil
import time
from pathlib import Path

try:
    from .init import (
        build_chunks,
        initialize_vectorrag,
        load_config,
        load_text,
        resolve_project_path,
    )
except ImportError:
    from init import (
        build_chunks,
        initialize_vectorrag,
        load_config,
        load_text,
        resolve_project_path,
    )


async def build_vector(config_path: str | Path | None = None) -> None:
    config = load_config(config_path) if config_path else load_config()
    vector_store_config = config["vector_store"]
    data_config = config["data"]
    index_config = config.get("index", {})

    persist_dir = resolve_project_path(vector_store_config["persist_dir"])
    source_path = resolve_project_path(data_config["source_path"])

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {source_path}")

    if index_config.get("rebuild", False) and persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)
    rag = await initialize_vectorrag(config)

    print(f"Building VectorRAG database from: {source_path}")
    print(f"VectorDB output directory: {persist_dir}")
    start = time.perf_counter_ns()
    text = load_text(source_path, data_config.get("encoding", "utf-8"))
    chunks = build_chunks(text, config)
    rag.insert(chunks, source_path)
    end = time.perf_counter_ns()
    elapsed_time = end - start
    print(f"VectorRAG database build finished: {len(chunks)} chunks.")
    print(f"Build Implementation time: {elapsed_time} ns")
