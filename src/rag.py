"""Pipeline RAG: combina búsqueda vectorial y generación con citas.

Reglas clave:
- Si la similitud máxima de los `top_k` hits está por debajo del umbral, se
  responde con un mensaje explícito de "no sé" sin invocar al LLM.
- Independientemente del backend, se asegura que toda respuesta incluya un
  bloque 'Fuentes:' con nombre de documento (y página si aplica) y un fragmento
  de evidencia.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .embeddings import build_embedder, Embedder
from .llm import build_llm, build_sources_block, ExtractiveLLM, LLM
from .loader import SUPPORTED_EXTENSIONS, file_hash, load_many
from .splitter import split_documents
from .vectorstore import DocumentInfo, Hit, VectorStore


@dataclass
class Answer:
    answer: str
    sources: list[Hit] = field(default_factory=list)
    not_found: bool = False
    backend: str = "extractive"
    raw_score: float = 0.0


class RAGAssistant:
    """Orquesta todos los componentes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder: Embedder = build_embedder(
            hf_model=settings.hf_embedding_model,
            openai_model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
            # Usamos embeddings locales por defecto; OpenAI solo si hay clave
            # y el usuario lo pidió explícitamente mediante LLM_PROVIDER.
            prefer_openai=False,
        )
        self.store = VectorStore(settings.index_dir, self.embedder)
        self.llm: LLM = build_llm(
            provider=settings.llm_provider,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.ollama_model,
        )
        self._doc_hashes: dict[str, str] = {}

    # ----------------------- Documentos -----------------------

    def add_documents(self, paths: list[str | Path]) -> dict:
        """Indexa una lista de archivos. Devuelve un resumen de la operación."""
        docs = load_many(paths)
        if not docs:
            return {"added_chunks": 0, "documents": []}

        chunks = split_documents(
            docs,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        # Anotamos hash por archivo para deduplicación en el manifest.
        for p in paths:
            try:
                self._doc_hashes[Path(p).name] = file_hash(p)
            except Exception:
                pass

        added = self.store.add_chunks(chunks)
        info_docs = self.store.list_documents()
        return {
            "added_chunks": added,
            "documents": [d.source for d in info_docs],
            "total_chunks": self.store.stats()["chunks"],
        }

    def add_text(self, name: str, text: str) -> dict:
        """Indexa texto arbitrario como si fuera un documento."""
        from .loader import Document
        if not text.strip():
            return {"added_chunks": 0, "documents": []}
        docs = [Document(text=text, metadata={
            "source": name,
            "page": None,
            "full_path": None,
            "extension": ".txt",
        })]
        chunks = split_documents(
            docs,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        added = self.store.add_chunks(chunks)
        return {
            "added_chunks": added,
            "documents": [d.source for d in self.store.list_documents()],
            "total_chunks": self.store.stats()["chunks"],
        }

    def remove_document(self, source: str) -> int:
        removed = self.store.remove_by_source(source)
        self._doc_hashes.pop(source, None)
        return removed

    def clear(self) -> None:
        self.store.clear()
        self._doc_hashes.clear()

    # ----------------------- Consultas -----------------------

    def ask(self, question: str) -> Answer:
        q = (question or "").strip()
        if not q:
            return Answer(answer="Escribe una pregunta para empezar.", not_found=True)

        hits = self.store.search(q, top_k=self.settings.top_k)
        best = max((h.score for h in hits), default=0.0)

        # Umbral: si la mejor coincidencia es muy baja, no hay información.
        if not hits or best < self.settings.similarity_threshold:
            return Answer(
                answer=(
                    "No encontré información sobre esto en los documentos cargados. "
                    "Intenta reformular la pregunta o sube más material relevante."
                ),
                sources=[],
                not_found=True,
                backend=self.llm.name,
                raw_score=best,
            )

        # Generación: extractivo no hace falta re-formatear; OpenAI/Ollama sí.
        result = self.llm.generate(q, hits)

        # Garantizamos bloque de fuentes uniforme, incluso si el LLM lo omite.
        sources_block = build_sources_block(hits)
        if sources_block and "Fuentes:" not in result.answer:
            separator = "" if result.answer.endswith("\n") else "\n"
            result.answer = f"{result.answer}{separator}\nFuentes:\n{sources_block}"

        return Answer(
            answer=result.answer,
            sources=hits,
            not_found=False,
            backend=result.backend or self.llm.name,
            raw_score=best,
        )

    # ----------------------- Estado -----------------------

    def stats(self) -> dict:
        s = self.store.stats()
        s["llm_backend"] = self.llm.name
        s["embedder"] = getattr(self.embedder, "model_name", type(self.embedder).__name__)
        s["threshold"] = self.settings.similarity_threshold
        s["top_k"] = self.settings.top_k
        return s

    def list_documents(self) -> list[DocumentInfo]:
        return self.store.list_documents()


__all__ = ["RAGAssistant", "Answer", "Hit", "DocumentInfo"]