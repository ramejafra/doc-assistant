"""Capa de embeddings.

Soporta dos implementaciones:
- SentenceTransformersEmbedding: descarga un modelo HF local (por defecto).
- OpenAIEmbedding: usa la API de OpenAI (requiere OPENAI_API_KEY).

La dimensión del vector y la normalización son necesarias para usar similitud
coseno con FAISS (IndexFlatIP).
"""
from __future__ import annotations

import math
from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    """Interfaz mínima que el resto del sistema espera de un embedder."""

    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformersEmbedding:
    """Wrapper de sentence-transformers con caché implícita."""

    _CACHE: dict[str, "_LoadedModel"] = {}

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = self._load(model_name)

    @classmethod
    def _load(cls, model_name: str) -> "_LoadedModel":
        if model_name in cls._CACHE:
            return cls._CACHE[model_name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Falta sentence-transformers. Instala con: "
                "pip install sentence-transformers"
            ) from e
        m = SentenceTransformer(model_name)
        cls._CACHE[model_name] = _LoadedModel(m)
        return cls._CACHE[model_name]

    @property
    def dim(self) -> int:
        return int(self._model.inner.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.inner.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class _LoadedModel:
    __slots__ = ("inner",)

    def __init__(self, inner) -> None:
        self.inner = inner


class OpenAIEmbedding:
    """Embeddings vía API de OpenAI."""

    def __init__(self, model_name: str, api_key: str) -> None:
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY no configurada. Define la variable de entorno."
            )
        self.model_name = model_name
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "Falta openai. Instala con: pip install openai"
            ) from e
        self._client = OpenAI(api_key=api_key)

    @property
    def dim(self) -> int:
        # Dimensiones conocidas (pueden cambiar). Si OpenAI añade un modelo nuevo,
        # se puede consultar dinámicamente.
        known = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return known.get(self.model_name, 1536)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # OpenAI acepta máximo razonable por request; en la práctica lo troceamos
        # si fuera necesario, pero por simplicidad enviamos todo.
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
        # Normalizar para similitud coseno con FAISS IndexFlatIP.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms


def build_embedder(
    *,
    hf_model: str,
    openai_model: str,
    openai_api_key: str,
    prefer_openai: bool,
) -> Embedder:
    """Elige el embedder según disponibilidad."""
    if prefer_openai and openai_api_key:
        return OpenAIEmbedding(openai_model, openai_api_key)
    return SentenceTransformersEmbedding(hf_model)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similitud coseno por filas; asume vectores normalizados."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T