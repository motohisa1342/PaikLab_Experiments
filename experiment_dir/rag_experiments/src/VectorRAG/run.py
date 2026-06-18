from __future__ import annotations

from pathlib import Path
import sys

try:
    from .init import VectorRAG, initialize_vectorrag, load_config
    from .prompt import build_final_prompt
except ImportError:
    from init import VectorRAG, initialize_vectorrag, load_config
    from prompt import build_final_prompt


def format_context(results) -> str:
    context_parts = []
    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"[{index}] score={result.score:.4f} chunk_id={result.id}\n{result.text}"
        )
    return "\n\n".join(context_parts)


def print_text(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding))


async def is_vector_built(rag: VectorRAG) -> bool:
    return rag.is_built()


async def initialize_vectorrag_for_query(
    config_path: str | Path | None = None,
) -> VectorRAG:
    config = load_config(config_path) if config_path else load_config()
    rag = await initialize_vectorrag(config)

    if not await is_vector_built(rag):
        raise ValueError("VectorRAG database has not been built yet. Run build_vector first.")

    return rag


async def fetch_prompt(
    rag: VectorRAG,
    question: str | None = None,
    top_k: int | None = None,
    config_path: str | Path | None = None,
) -> str:
    config = load_config(config_path) if config_path else load_config()
    query_config = config.get("query", {})

    query_text = question or query_config["text"]
    results = rag.query(query_text, top_k=top_k)
    prompt = build_final_prompt(format_context(results), query_text)
    print_text(prompt)
    return prompt


async def run_query(
    rag: VectorRAG,
    question: str | None = None,
    top_k: int | None = None,
    config_path: str | Path | None = None,
) -> str:
    prompt = await fetch_prompt(rag, question=question, top_k=top_k, config_path=config_path)
    response = rag.complete(prompt)
    print_text(response)
    return response
