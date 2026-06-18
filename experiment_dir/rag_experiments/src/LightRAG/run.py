from __future__ import annotations

import inspect
from pathlib import Path

from lightrag import LightRAG, QueryParam

try:
    from .init import initialize_rag, load_config
except ImportError:
    from init import initialize_rag, load_config


async def print_stream(stream) -> None:
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()


async def is_graph_built(rag: LightRAG) -> bool:
    status = await rag.get_processing_status()
    return status.get("processed", 0) > 0


async def initialize_lightrag(
    config_path: str | Path | None = None,
) -> LightRAG:
    config = load_config(config_path) if config_path else load_config()
    rag = await initialize_rag(config)

    if not await is_graph_built(rag):
        raise ValueError("LightRAG graph has not been built yet. Run build_graph first.")

    return rag


async def fetch_prompt(
    rag: LightRAG,
    question: str | None = None,
    mode: str | None = None,
    config_path: str | Path | None = None,
) -> str:
    config = load_config(config_path) if config_path else load_config()
    lightrag_config = config.get("lightrag", {})
    query_config = config.get("query", {})

    query_text = question or query_config["text"]
    query_mode = mode or lightrag_config.get("mode", "hybrid")

    prompt = await rag.aquery(
        query_text,
        param=QueryParam(
            mode=query_mode,
            stream=query_config.get("stream", False),
            enable_rerank=query_config.get("enable_rerank", False),
            only_need_prompt=True,
        ),
    )

    if inspect.isasyncgen(prompt):
        await print_stream(prompt)
    else:
        print(prompt)

    return prompt


async def run_query(
    rag: LightRAG,
    question: str | None = None,
    mode: str | None = None,
    config_path: str | Path | None = None,
) -> str:
    config = load_config(config_path) if config_path else load_config()
    lightrag_config = config.get("lightrag", {})
    query_config = config.get("query", {})

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

    return response

