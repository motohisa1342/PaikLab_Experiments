from __future__ import annotations

import asyncio
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc, set_verbose_debug


LIGHTRAG_DIR = Path(__file__).resolve().parent
RAG_EXPERIMENTS_DIR = LIGHTRAG_DIR.parents[1]
DEFAULT_CONFIG_PATH = RAG_EXPERIMENTS_DIR / "config" / "default_config.yaml"
LIGHTRAG_CONFIG_PATH = RAG_EXPERIMENTS_DIR / "config" / "lightrag.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config or {}


def load_config(
    config_path: str | Path = LIGHTRAG_CONFIG_PATH,
    default_config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    default_config = load_yaml(default_config_path)
    lightrag_config = load_yaml(config_path)
    config = deep_merge(default_config, lightrag_config)
    config["_paths"] = {
        "config_path": str(Path(config_path).resolve()),
        "project_root": str(RAG_EXPERIMENTS_DIR),
    }
    return config


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (RAG_EXPERIMENTS_DIR / path).resolve()


def require_model_name(config: dict[str, Any], section_name: str) -> str:
    model_name = config.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(f"{section_name}.model_name must not be null or empty.")
    return model_name


def build_lightrag(config: dict[str, Any]) -> LightRAG:
    lightrag_config = config.get("lightrag", {})
    llm_config = config.get("llm", {})
    embedding_config = config.get("embedding", {})
    chunking_config = config.get("chunking", {})
    llm_model_name = require_model_name(llm_config, "llm")
    embedding_model_name = require_model_name(embedding_config, "embedding")

    host = lightrag_config.get("ollama_host", "http://localhost:11434")
    llm_options = deepcopy(llm_config.get("options", {}))
    if llm_config.get("context_length"):
        llm_options.setdefault("num_ctx", llm_config["context_length"])

    verbose_debug = bool(lightrag_config.get("verbose_debug", False))
    set_verbose_debug(verbose_debug)

    return LightRAG(
        working_dir=str(resolve_project_path(lightrag_config["working_dir"])),
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model_name,
        llm_model_max_async=lightrag_config.get("llm_model_max_async", 1),
        max_parallel_insert=lightrag_config.get("max_parallel_insert", 1),
        default_llm_timeout=lightrag_config.get("default_llm_timeout", 600),
        chunk_token_size=chunking_config.get("chunk_size", 800),
        chunk_overlap_token_size=chunking_config.get("chunk_overlap", 100),
        llm_model_kwargs={
            "host": host,
            "options": llm_options,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_config.get("dimension", 1024),
            max_token_size=embedding_config.get("max_token_size", 8192),
            func=partial(
                ollama_embed,
                embed_model=embedding_model_name,
                host=host,
            ),
        ),
    )


async def initialize_rag(config: dict[str, Any]) -> LightRAG:
    rag = build_lightrag(config)
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


def initialize_rag_sync(config: dict[str, Any]) -> LightRAG:
    return asyncio.run(initialize_rag(config))
