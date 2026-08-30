from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from typing import Any

import numpy as np

from .config import CONFIG
from .database import DB, json_load
from .paths import PATHS


WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]+", re.UNICODE)
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in WORD_PATTERN.findall(text.lower()):
        if CJK_PATTERN.fullmatch(match):
            tokens.extend(match)
            tokens.extend(match[index:index + 2] for index in range(len(match) - 1))
        elif len(match) > 1:
            tokens.append(match)
    return tokens


class EmbeddingService:
    def __init__(self) -> None:
        self._model: Any = None
        self._attempted = False
        self.dimension = 384

    def _load(self) -> Any:
        if self._attempted:
            return self._model
        self._attempted = True
        try:
            from sentence_transformers import SentenceTransformer

            model_path = PATHS.models / "sentence-transformers" / CONFIG.models.embedding.replace("/", "--")
            identifier = str(model_path) if model_path.exists() else CONFIG.models.embedding
            self._model = SentenceTransformer(
                identifier,
                device="cpu",
                local_files_only=CONFIG.models.offline,
                cache_folder=str(PATHS.models / "sentence-transformers"),
                tokenizer_kwargs={"fix_mistral_regex": False},
            )
            self.dimension = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            self._model = None
        return self._model

    def _hash_embedding(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = tokenize(text)
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def encode(self, texts: list[str], query: bool = False) -> list[list[float]]:
        model = self._load()
        prefix = "query: " if query else "passage: "
        prepared = [prefix + text for text in texts]
        if model is None:
            return [self._hash_embedding(text) for text in prepared]
        embeddings = model.encode(prepared, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.astype(np.float32).tolist()

    @property
    def mode(self) -> str:
        return "sentence-transformers" if self._load() is not None else "local-ngram-fallback"


EMBEDDINGS = EmbeddingService()


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def reciprocal_rank_fusion(rankings: list[list[str]], constant: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (constant + rank)
    return scores


def retrieve(notebook_id: str, query: str, source_ids: list[str] | None = None, limit: int = 10, ensure_source_coverage: bool = False) -> list[dict[str, Any]]:
    if source_ids is None:
        sources = DB.fetchall("SELECT id FROM sources WHERE notebook_id=? AND selected=1 AND state='ready'", (notebook_id,))
        source_ids = [row["id"] for row in sources]
    if not source_ids:
        return []
    placeholders = ",".join("?" for _ in source_ids)
    params: tuple[Any, ...] = tuple(source_ids)
    all_chunks = DB.fetchall(f"SELECT * FROM chunks WHERE source_id IN ({placeholders})", params)
    query_vector = EMBEDDINGS.encode([query], query=True)[0]
    dense_all = sorted(
        all_chunks,
        key=lambda row: _cosine(query_vector, json_load(row.get("embedding_json"), [])),
        reverse=True,
    )
    dense = dense_all[: max(limit * 4, 30)]
    query_tokens = tokenize(query)
    cjk_query = bool(CJK_PATTERN.search(query))
    terms = [term for term in query_tokens if len(term) > 1][:16]
    lexical: list[dict[str, Any]] = []
    if terms and not cjk_query:
        match_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)
        try:
            lexical = DB.fetchall(
                f"""SELECT c.* FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id
                WHERE chunks_fts MATCH ? AND c.source_id IN ({placeholders})
                ORDER BY bm25(chunks_fts) LIMIT ?""",
                (match_query, *source_ids, max(limit * 4, 30)),
            )
        except Exception:
            lexical = []
    elif terms:
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in all_chunks:
            lowered = row["content"].lower()
            score = sum(lowered.count(term) * (2 if len(term) > 1 else 1) for term in terms)
            if score:
                scored.append((score, row))
        lexical = [row for _, row in sorted(scored, key=lambda item: item[0], reverse=True)[: max(limit * 4, 30)]]
    fused = reciprocal_rank_fusion([[row["id"] for row in dense], [row["id"] for row in lexical]])
    by_id = {row["id"]: row for row in all_chunks}
    ordered = sorted(fused, key=fused.get, reverse=True)
    results: list[dict[str, Any]] = []
    per_source: dict[str, int] = defaultdict(int)
    if ensure_source_coverage and len(source_ids) > 1:
        covered: list[str] = []
        for source_id in source_ids:
            candidate = next((row["id"] for row in dense_all if row["source_id"] == source_id), None)
            if candidate:
                covered.append(candidate)
        ordered = covered + [chunk_id for chunk_id in ordered if chunk_id not in set(covered)]
    for chunk_id in ordered:
        row = dict(by_id[chunk_id])
        if per_source[row["source_id"]] >= 4:
            continue
        row["locator"] = json_load(row.pop("locator_json"), {})
        row.pop("embedding_json", None)
        row["score"] = fused[chunk_id]
        results.append(row)
        per_source[row["source_id"]] += 1
        if len(results) >= limit:
            break
    return results
