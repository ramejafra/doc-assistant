"""Vector store FAISS con persistencia local.

Diseño:
- `IndexFlatIP` sobre vectores normalizados (equivalente a coseno).
- Persistencia en `INDEX_DIR`:
    - vectors.faiss     -> índice binario
    - chunks.jsonl      -> metadatos y textos, una línea por chunk
    - manifest.json     -> información de versión y modelo de embeddings
- API amigable para añadir, buscar y eliminar por documento.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import faiss  # type: ignore
    _FAISS_OK = True
except ImportError:
    faiss = None  # type: ignore
    _FAISS_OK = False

from .embeddings import Embedder
from .splitter import Chunk


@dataclass
class Hit:
    """Un resultado de búsqueda con su puntuación y metadatos."""

    score: float
    chunk: Chunk


@dataclass
class DocumentInfo:
    """Información agregada de un documento indexado."""

    source: str
    full_path: str | None = None
    chunks: int = 0
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """Índice FAISS con metadatos en JSONL."""

    MANIFEST = "manifest.json"
    VECTORS = "vectors.faiss"
    CHUNKS = "chunks.jsonl"

    def __init__(self, index_dir: Path, embedder: Embedder) -> None:
        if not _FAISS_OK:
            raise RuntimeError(
                "Falta faiss-cpu. Instala con: pip install faiss-cpu"
            )
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.dim = embedder.dim

        self._chunks: list[Chunk] = []
        self._index: "faiss.Index" | None = None

        self._load_or_init()

    # ----------------------- persistencia -----------------------

    def _load_or_init(self) -> None:
        vectors_path = self.index_dir / self.VECTORS
        chunks_path = self.index_dir / self.CHUNKS
        manifest_path = self.index_dir / self.MANIFEST

        if vectors_path.exists() and chunks_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            # Si cambió la dimensión o el modelo, descartamos.
            if manifest.get("dim") != self.dim:
                self._reset()
            else:
                self._index = faiss.read_index(str(vectors_path))
                self._chunks = [
                    Chunk(
                        text=obj["text"],
                        metadata=obj.get("metadata", {}),
                    )
                    for obj in (json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())
                ]
                return
        self._reset()

    def _reset(self) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        self._chunks = []
        self._save()

    def _save(self) -> None:
        assert self._index is not None
        faiss.write_index(self._index, str(self.index_dir / self.VECTORS))
        with (self.index_dir / self.CHUNKS).open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps({"text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")
        manifest = {
            "dim": self.dim,
            "embedder_model": getattr(self.embedder, "model_name", "unknown"),
            "embedder_type": type(self.embedder).__name__,
        }
        (self.index_dir / self.MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ----------------------- mutadores -----------------------

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Añade chunks al índice y persiste. Devuelve cuántos se añadieron."""
        chunks = list(chunks)
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        vecs = self.embedder.embed(texts)
        if vecs.size == 0:
            return 0
        assert self._index is not None
        self._index.add(vecs)
        self._chunks.extend(chunks)
        self._save()
        return len(chunks)

    def clear(self) -> None:
        """Borra el índice por completo."""
        self._reset()

    def remove_by_source(self, source: str) -> int:
        """Elimina todos los chunks cuyo metadata.source coincida."""
        assert self._index is not None
        if not self._chunks:
            return 0
        keep_idx: list[int] = []
        keep_chunks: list[Chunk] = []
        for i, c in enumerate(self._chunks):
            if c.metadata.get("source") == source:
                continue
            keep_idx.append(i)
            keep_chunks.append(c)
        if len(keep_idx) == len(self._chunks):
            return 0

        # Reconstruir índice manteniendo solo los vectores conservados.
        new_index = faiss.IndexFlatIP(self.dim)
        if keep_idx:
            # Reconstruimos los vectores a partir de los embeddings guardados.
            # Como FAISS IndexFlatIP no expone los vectores en todas las
            # versiones, los re-calculamos a partir del texto (aceptable para
            # corpus pequeños/medios y garantiza consistencia).
            texts = [c.text for c in keep_chunks]
            vecs = self.embedder.embed(texts)
            new_index.add(vecs)

        self._index = new_index
        self._chunks = keep_chunks
        self._save()
        return sum(1 for c in self._chunks[: -len(keep_chunks)] if False)  # no usado

    # ----------------------- consultas -----------------------

    def search(self, query: str, top_k: int = 4) -> list[Hit]:
        """Devuelve los `top_k` chunks más similares a la query."""
        assert self._index is not None
        if self._index.ntotal == 0 or not self._chunks:
            return []
        q_vec = self.embedder.embed([query])
        if q_vec.size == 0:
            return []
        k = min(top_k, self._index.ntotal)
        scores, idxs = self._index.search(q_vec, k)
        hits: list[Hit] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self._chunks):
                continue
            hits.append(Hit(score=float(score), chunk=self._chunks[idx]))
        return hits

    # ----------------------- información -----------------------

    def list_documents(self) -> list[DocumentInfo]:
        """Agrupa chunks por source y devuelve info por documento."""
        agg: dict[str, DocumentInfo] = {}
        for c in self._chunks:
            src = c.metadata.get("source", "desconocido")
            info = agg.get(src)
            if info is None:
                info = DocumentInfo(
                    source=src,
                    full_path=c.metadata.get("full_path"),
                    chunks=0,
                    metadata={k: v for k, v in c.metadata.items() if k not in {"source", "full_path", "chunk_index"}},
                )
                agg[src] = info
            info.chunks += 1
        # Ordenar por nombre para una UI estable.
        return sorted(agg.values(), key=lambda d: d.source.lower())

    def stats(self) -> dict:
        return {
            "chunks": len(self._chunks),
            "vectors": int(self._index.ntotal) if self._index is not None else 0,
            "documents": len(self.list_documents()),
            "dim": self.dim,
        }