from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import fitz
import numpy as np
import ollama
import yaml
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

VECRAG_DIR = Path(__file__).resolve().parent
RAG_EXPERIMENTS_DIR = VECRAG_DIR.parents[1]
DEFAULT_CONFIG_PATH = RAG_EXPERIMENTS_DIR / "config" / "default_config.yaml"
VECRAG_CONFIG_PATH = RAG_EXPERIMENTS_DIR / "config" / "vectorrag.yaml"


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
    config_path: str | Path = VECRAG_CONFIG_PATH,
    default_config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    default_config = load_yaml(default_config_path)
    vectorrag_config = load_yaml(config_path)
    config = deep_merge(default_config, vectorrag_config)
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


def load_text(source_path: str | Path, encoding: str = "utf-8") -> str:
    path = resolve_project_path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file was not found: {path}")

    if path.suffix.lower() == ".pdf":
        pages: list[str] = []
        with fitz.open(path) as pdf:
            for page in pdf:
                text = page.get_text("text")
                if text:
                    pages.append(text)
        return "\n".join(pages)

    with open(path, "r", encoding=encoding) as file:
        return file.read()


def build_text_splitter(config: dict[str, Any]):
    chunking_config = config.get("chunking", {})
    method = str(chunking_config.get("method", "PLC")).upper()
    chunk_size = int(chunking_config.get("chunk_size", 1000))
    chunk_overlap = int(chunking_config.get("chunk_overlap", 0))

    common_kwargs = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "length_function": len,
    }

    if method in {"RCC", "RECURSIVE"}:
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "、", " ", ""],
            **common_kwargs,
        )
    if method in {"FLC", "FIXED"}:
        return CharacterTextSplitter(separator="", **common_kwargs)
    if method in {"PLC", "PARAGRAPH"}:
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ""],
            **common_kwargs,
        )
    if method in {"SLC", "SENTENCE"}:
        return RecursiveCharacterTextSplitter(
            separators=["。", "！", "？", ".", "!", "?", "\n", ""],
            **common_kwargs,
        )

    raise ValueError(f"Unsupported VectorRAG chunking method: {method}")


def build_chunks(text: str, config: dict[str, Any]) -> list[str]:
    splitter = build_text_splitter(config)
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorRAG:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.vector_store_config = config.get("vector_store", {})
        self.llm_config = config.get("llm", {})
        self.embedding_config = config.get("embedding", {})
        self.retrieval_config = config.get("retrieval", {})
        self.client = ollama.Client(
            host=self.vector_store_config.get("ollama_host", "http://localhost:11434")
        )
        self.embedding_model_name = require_model_name(self.embedding_config, "embedding")
        self.llm_model_name = require_model_name(self.llm_config, "llm")
        self.persist_dir = resolve_project_path(
            self.vector_store_config.get("persist_dir", "./src/VectorRAG/faiss_index")
        )
        self.collection_name = self.vector_store_config.get(
            "collection_name", "vectorrag_collection"
        )
        self.index_path = self.persist_dir / f"{self.collection_name}.faiss"
        self.metadata_path = self.persist_dir / f"{self.collection_name}_metadata.json"
        self.index: faiss.Index | None = None
        self.records: list[dict[str, Any]] = []

    def load(self) -> None:
        if not self.index_path.exists() or not self.metadata_path.exists():
            self.index = None
            self.records = []
            return
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        self.records = payload.get("records", [])

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Cannot save VectorRAG because the FAISS index is empty.")

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        payload = {
            "collection_name": self.collection_name,
            "vector_store": "faiss",
            "embedding_model": self.embedding_model_name,
            "records": self.records,
        }
        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(model=self.embedding_model_name, input=texts)
        embeddings = response.get("embeddings")
        if embeddings is None:
            raise ValueError("Ollama embed response did not include embeddings.")
        return embeddings

    def complete(self, prompt: str) -> str:
        options = deepcopy(self.llm_config.get("options", {}))
        if self.llm_config.get("temperature") is not None:
            options.setdefault("temperature", self.llm_config["temperature"])
        if self.llm_config.get("max_tokens") is not None:
            options.setdefault("num_predict", self.llm_config["max_tokens"])

        response = self.client.generate(
            model=self.llm_model_name,
            prompt=prompt,
            options=options,
            stream=False,
        )
        return str(response.get("response", "")).strip()

    def insert(self, chunks: list[str], source_path: str | Path) -> None:
        if not chunks:
            raise ValueError("No chunks were created from the source document.")

        embeddings = np.array(self.embed_texts(chunks), dtype=np.float32)
        expected_dim = self.embedding_config.get("dimension")
        if self.embedding_config.get("validate_dimension", True) and expected_dim:
            detected_dim = int(embeddings.shape[1]) if embeddings.size else 0
            if detected_dim != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: got {detected_dim}, expected {expected_dim}"
                )

        self.index = faiss.IndexFlatL2(int(embeddings.shape[1]))
        self.index.add(embeddings)
        self.records = [
            {
                "id": f"{self.collection_name}-{index:06d}",
                "text": chunk,
                "metadata": {
                    "source_path": str(resolve_project_path(source_path)),
                    "chunk_index": index,
                },
            }
            for index, chunk in enumerate(chunks)
        ]
        self.save()

    def query(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        if self.index is None or not self.records:
            self.load()
        if self.index is None or not self.records:
            raise ValueError("VectorRAG database has not been built yet. Run build_vector first.")

        query_embedding = np.array(self.embed_texts([question]), dtype=np.float32)
        limit = top_k or int(self.retrieval_config.get("top_k", 5))
        search_k = min(max(limit, int(self.retrieval_config.get("candidate_k", limit))), self.index.ntotal)
        _, indices = self.index.search(query_embedding, search_k)

        scored: list[SearchResult] = []
        for index in indices[0]:
            if index < 0:
                continue
            record = self.records[int(index)]
            candidate_embedding = self.index.reconstruct(int(index)).reshape(1, -1)
            score = float(cosine_similarity(query_embedding, candidate_embedding)[0][0])
            scored.append(
                SearchResult(
                    id=record["id"],
                    text=record["text"],
                    score=score,
                    metadata=record.get("metadata", {}),
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def is_built(self) -> bool:
        if self.index is None or not self.records:
            self.load()
        return self.index is not None and bool(self.records)


async def initialize_vectorrag(config: dict[str, Any]) -> VectorRAG:
    rag = VectorRAG(config)
    rag.load()
    return rag


def initialize_vectorrag_sync(config: dict[str, Any]) -> VectorRAG:
    return asyncio.run(initialize_vectorrag(config))
