from __future__ import annotations

import argparse
import asyncio
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LightRAG query.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to lightrag.yaml. Defaults to rag_experiments/config/lightrag.yaml.",
    )
    parser.add_argument("--question", default=None, help="Question to ask LightRAG.")
    parser.add_argument("--mode", default=None, help="Query mode: local/global/hybrid/naive.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_query(args.question, args.mode, args.config))
