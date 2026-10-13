"""Segmentación de texto en chunks para indexar y recuperar.

Estrategia:
- Ventana deslizante por caracteres, respetando límites de párrafo cuando es
  posible (heurística: intenta cortar en saltos de línea dobles, líneas simples
  o espacios, en ese orden de preferencia).
- Es deliberadamente simple para no depender de tiktoken u otros tokenizadores.
- Devuelve `Chunk` con metadatos del documento de origen para poder citar.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .loader import Document


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "desconocido"))


def split_documents(
    docs: Iterable[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Divide documentos en chunks solapados."""
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser positivo")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe estar en [0, chunk_size)")

    chunks: list[Chunk] = []
    for doc in docs:
        text = doc.text
        if len(text) <= chunk_size:
            piece = text.strip()
            if piece:
                chunks.append(Chunk(text=piece, metadata={
                    **doc.metadata,
                    "chunk_index": 0,
                }))
            continue

        start = 0
        idx = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            window = text[start:end]
            if end < n:
                # Intentar ajustar a un separador cercano para no cortar palabras.
                best = _find_break(window)
                if best is not None and best > chunk_size // 2:
                    end = start + best
                    window = text[start:end]

            piece = window.strip()
            if piece:
                chunks.append(Chunk(text=piece, metadata={
                    **doc.metadata,
                    "chunk_index": idx,
                }))
                idx += 1

            if end >= n:
                break
            start = max(end - chunk_overlap, start + 1)

    return chunks


def _find_break(window: str) -> int | None:
    """Busca el mejor índice para cortar respetando separadores naturales."""
    for sep in ("\n\n", "\n", ". ", " ", ""):
        if not sep:
            # Evita cortar al principio absoluto.
            idx = len(window) - 1
        else:
            idx = window.rfind(sep)
        if idx > 0:
            cut = idx + len(sep) if sep else idx
            if cut > 0:
                return cut
    return None