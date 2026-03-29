from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Callable
from urllib import request

from .config import DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_BASE_URL
from .models import SearchResult, utc_now_iso


@dataclass(slots=True)
class SemanticScore:
    memory_id: int
    similarity: float
    bonus: float


class OptionalSemanticReranker:
    """Optional Ollama-backed semantic reranking with a SQLite cache."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 2.5,
        fetch_embeddings: Callable[[list[str]], list[list[float]]] | None = None,
    ):
        self.connection = connection
        self.model = (model or DEFAULT_EMBED_MODEL).strip()
        self.base_url = (base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._fetch_embeddings_override = fetch_embeddings

    @property
    def enabled(self) -> bool:
        return bool(self.model)

    def rerank(self, query: str, results: list[SearchResult]) -> dict[int, SemanticScore]:
        if not self.enabled or not query.strip() or not results:
            return {}
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return {}

        result_texts = [self._memory_text(result) for result in results]
        memory_embeddings = self._get_embeddings(result_texts)
        if len(memory_embeddings) != len(results):
            return {}

        scores: dict[int, SemanticScore] = {}
        for item, embedding in zip(results, memory_embeddings):
            similarity = self._cosine_similarity(query_embedding, embedding)
            bonus = max((similarity + 1.0) / 2.0, 0.0)
            scores[item.memory.id] = SemanticScore(
                memory_id=item.memory.id,
                similarity=similarity,
                bonus=bonus,
            )
        return scores

    def status(self) -> dict[str, object]:
        cache_count = self.connection.execute(
            """
            select count(*)
            from embedding_cache
            where model = ?
            """,
            (self.model,),
        ).fetchone()[0]
        return {
            "enabled": self.enabled,
            "model": self.model or None,
            "base_url": self.base_url,
            "cached_vectors": int(cache_count),
        }

    def _memory_text(self, result: SearchResult) -> str:
        memory = result.memory
        tags = " ".join(memory.tags)
        return f"{memory.subject}\n{memory.content}\n{tags}".strip()

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        cached: dict[str, list[float]] = {}
        missing: list[str] = []
        for text in texts:
            embedding = self._load_cached_embedding(text)
            if embedding is None:
                missing.append(text)
            else:
                cached[text] = embedding

        if missing:
            fetched = self._fetch_embeddings(missing)
            if len(fetched) != len(missing):
                return []
            for text, embedding in zip(missing, fetched):
                cached[text] = embedding
                self._store_cached_embedding(text, embedding)

        return [cached[text] for text in texts if text in cached]

    def _get_embedding(self, text: str) -> list[float] | None:
        embeddings = self._get_embeddings([text])
        if len(embeddings) != 1:
            return None
        return embeddings[0]

    def _load_cached_embedding(self, text: str) -> list[float] | None:
        row = self.connection.execute(
            """
            select embedding_json
            from embedding_cache
            where model = ? and content_hash = ?
            """,
            (self.model, self._hash_text(text)),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["embedding_json"])

    def _store_cached_embedding(self, text: str, embedding: list[float]) -> None:
        self.connection.execute(
            """
            insert into embedding_cache(model, content_hash, content, embedding_json, updated_at)
            values (?, ?, ?, ?, ?)
            on conflict(model, content_hash) do update set
                content = excluded.content,
                embedding_json = excluded.embedding_json,
                updated_at = excluded.updated_at
            """,
            (
                self.model,
                self._hash_text(text),
                text,
                json.dumps(embedding),
                utc_now_iso(),
            ),
        )
        self.connection.commit()

    def _fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self._fetch_embeddings_override is not None:
            return self._fetch_embeddings_override(texts)

        payload = json.dumps(
            {
                "model": self.model,
                "input": texts,
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception:
            return []

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            return []
        normalized: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list):
                return []
            normalized.append([float(value) for value in embedding])
        return normalized

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)
