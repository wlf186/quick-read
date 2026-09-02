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
LOW_VALUE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:table of contents|contents|copyright|all rights reserved|bibliography|references|index|acknowledg(?:e)?ments?|"
    r"目录|版权|版权所有|参考文献|书目|索引|致谢)\b",
    re.I,
)


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


def is_quality_chunk(row: dict[str, Any], *, minimum_chars: int = 120) -> bool:
    """Reject front/back matter and fragments before they reach a paid model."""
    content = re.sub(r"\s+", " ", str(row.get("content") or "")).strip()
    if len(content) < minimum_chars or LOW_VALUE_PATTERN.search(content[:500]):
        return False
    lowered = content.lower()
    if lowered.count("http://") + lowered.count("https://") >= 2:
        return False
    words = re.findall(r"[A-Za-z\u3400-\u9fff]+", content)
    if len(words) < 18:
        return False
    locator = row.get("locator") if isinstance(row.get("locator"), dict) else json_load(row.get("locator_json"), {})
    section = str(locator.get("section") or "")
    return not bool(LOW_VALUE_PATTERN.search(section))


def select_quality_evidence(
    notebook_id: str,
    source_ids: list[str],
    *,
    limit: int,
    focus: str = "",
    minimum_chars: int = 120,
) -> list[dict[str, Any]]:
    """Select central, diverse evidence locally with source and section coverage."""
    if not source_ids or limit <= 0:
        return []
    marks = ",".join("?" for _ in source_ids)
    rows = DB.fetchall(f"SELECT * FROM chunks WHERE source_id IN ({marks}) ORDER BY source_id,ordinal", tuple(source_ids))
    candidates = [row for row in rows if is_quality_chunk(row, minimum_chars=minimum_chars)]
    if not candidates:
        candidates = [
            row for row in rows
            if len(re.sub(r"\s+", " ", str(row.get("content") or "")).strip()) >= 24
            and not LOW_VALUE_PATTERN.search(str(row.get("content") or "")[:500])
        ]
    if not candidates:
        return []
    vectors: list[list[float]] = []
    missing_indexes: list[int] = []
    for index, row in enumerate(candidates):
        vector = json_load(row.get("embedding_json"), [])
        vectors.append(vector)
        if not vector:
            missing_indexes.append(index)
    if missing_indexes:
        encoded = EMBEDDINGS.encode([str(candidates[index]["content"]) for index in missing_indexes])
        for index, vector in zip(missing_indexes, encoded):
            vectors[index] = vector
    matrix = np.asarray(vectors, dtype=np.float32)
    centroid = matrix.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    centroid = centroid / norm if norm else centroid
    focus_vector = np.asarray(EMBEDDINGS.encode([focus], query=True)[0], dtype=np.float32) if focus.strip() else centroid
    focus_norm = float(np.linalg.norm(focus_vector))
    focus_vector = focus_vector / focus_norm if focus_norm else focus_vector
    base_scores = [0.55 * float(np.dot(vector, focus_vector)) + 0.45 * float(np.dot(vector, centroid)) for vector in matrix]
    per_source_limit = max(1, math.ceil(limit / len(source_ids)) + 1)
    selected_indexes: list[int] = []
    source_counts: dict[str, int] = defaultdict(int)
    section_counts: dict[tuple[str, str], int] = defaultdict(int)
    while len(selected_indexes) < min(limit, len(candidates)):
        best_index, best_score = None, -float("inf")
        for index, row in enumerate(candidates):
            if index in selected_indexes or source_counts[row["source_id"]] >= per_source_limit:
                continue
            locator = json_load(row.get("locator_json"), {})
            section = str(locator.get("section") or locator.get("spine") or locator.get("page") or "")
            similarity = max((float(np.dot(matrix[index], matrix[value])) for value in selected_indexes), default=0.0)
            section_penalty = min(0.18, section_counts[(row["source_id"], section)] * 0.06)
            score = base_scores[index] - 0.35 * max(0.0, similarity) - section_penalty
            if score > best_score:
                best_index, best_score = index, score
        if best_index is None:
            break
        selected_indexes.append(best_index)
        row = candidates[best_index]
        locator = json_load(row.get("locator_json"), {})
        section = str(locator.get("section") or locator.get("spine") or locator.get("page") or "")
        source_counts[row["source_id"]] += 1
        section_counts[(row["source_id"], section)] += 1
    selected: list[dict[str, Any]] = []
    for rank, index in enumerate(selected_indexes, start=1):
        row = dict(candidates[index])
        row["locator"] = json_load(row.pop("locator_json", None), {})
        row.pop("embedding_json", None)
        row["quality_rank"] = rank
        row["quality_score"] = round(base_scores[index], 6)
        selected.append(row)
    return selected


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
