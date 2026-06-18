from __future__ import annotations

import inspect
from pathlib import Path

from lightrag import QueryParam

try:
    from .init import initialize_rag, load_config
except ImportError:
    from init import initialize_rag, load_config


async def print_stream(stream) -> None:
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()


async def run_query(
    question: str | None = None,
    mode: str | None = None,
    config_path: str | Path | None = None,
) -> None:
    config = load_config(config_path) if config_path else load_config()
    lightrag_config = config.get("lightrag", {})
    query_config = config.get("query", {})

    rag = await initialize_rag(config)
    query_text = question or query_config["text"]
    query_mode = mode or lightrag_config.get("mode", "hybrid")

    response = await rag.aquery(
        query_text,
        param=QueryParam(
            mode=query_mode,
            stream=query_config.get("stream", False),
            enable_rerank=query_config.get("enable_rerank", False),
        ),
    )

    if inspect.isasyncgen(response):
        await print_stream(response)
    else:
        print(response)

