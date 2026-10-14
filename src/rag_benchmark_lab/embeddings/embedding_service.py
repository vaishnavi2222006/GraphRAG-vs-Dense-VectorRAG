from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from typing import Iterable

import numpy as np

try:  # optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None


@dataclass(slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    model_name: str
    dimension: int
    mock_mode: bool = False


class EmbeddingService:
    def __init__(self, model_name: str, mock_mode: bool = False) -> None:
        self.model_name = model_name
        self.mock_mode = mock_mode or SentenceTransformer is None
        self._model = None

    @lru_cache(maxsize=1)
    def _load_model(self):  # pragma: no cover - heavy dependency
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        return SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        if self.mock_mode:
            return 384
        model = self._load_model()
        return int(model.get_sentence_embedding_dimension())

    def _mock_vector(self, text: str, dimension: int = 384) -> list[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        values: list[float] = []
        for index in range(dimension):
            seed = (seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
            values.append(((seed >> 16) & 0xFFFF) / 65535.0)
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def embed_texts(self, texts: Iterable[str], batch_size: int = 16) -> EmbeddingResult:
        items = list(texts)
        if self.mock_mode:
            vectors = [self._mock_vector(text, self.dimension) for text in items]
            return EmbeddingResult(vectors=vectors, model_name=self.model_name, dimension=self.dimension, mock_mode=True)
        model = self._load_model()
        vectors = model.encode(items, batch_size=batch_size, normalize_embeddings=True)
        array = np.asarray(vectors, dtype=float)
        return EmbeddingResult(vectors=array.tolist(), model_name=self.model_name, dimension=int(array.shape[1]))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text]).vectors[0]

