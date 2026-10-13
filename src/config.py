"""Configuración centralizada del asistente.

Lee variables desde `.env` si existe, y si no, usa valores por defecto.
Valida tipos y aplica reglas de coherencia (por ejemplo, threshold en [0, 1]).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
    # Busca `.env` en el directorio raíz del proyecto y en el cwd.
    _root = Path(__file__).resolve().parent.parent
    load_dotenv(_root / ".env", override=False)
    load_dotenv(override=False)
except ImportError:  # dotenv opcional
    pass


LLMProvider = Literal["openai", "ollama", "extractive"]


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_provider(value: str | None) -> str:
    if value is None:
        return "extractive"
    v = value.strip().lower()
    return v if v in ("openai", "ollama", "extractive") else "extractive"


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


@dataclass(frozen=True)
class Settings:
    """Snapshot inmutable de la configuración."""

    llm_provider: str = "extractive"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"
    hf_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    similarity_threshold: float = 0.30
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 120
    index_dir: Path = field(default_factory=lambda: Path("./.data/index"))
    docs_dir: Path = field(default_factory=lambda: Path("./documents"))

    def __post_init__(self) -> None:
        # Coherencia de rangos
        if not 0.0 <= self.similarity_threshold <= 1.0:
            object.__setattr__(self, "similarity_threshold", 0.30)
        if self.top_k < 1:
            object.__setattr__(self, "top_k", 1)
        if self.chunk_size < 100:
            object.__setattr__(self, "chunk_size", 100)
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            object.__setattr__(self, "chunk_overlap", min(120, self.chunk_size // 2))


def load_settings() -> Settings:
    """Carga y valida la configuración desde el entorno."""
    project_root = Path(__file__).resolve().parent.parent
    return Settings(
        llm_provider=_get_provider(os.getenv("LLM_PROVIDER")),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        hf_embedding_model=os.getenv(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        similarity_threshold=_get_float("SIMILARITY_THRESHOLD", 0.30),
        top_k=_get_int("TOP_K", 4),
        chunk_size=_get_int("CHUNK_SIZE", 800),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
        index_dir=_resolve_path(os.getenv("INDEX_DIR"), project_root / ".data" / "index"),
        docs_dir=_resolve_path(os.getenv("DOCS_DIR"), project_root / "documents"),
    )


def ensure_dirs(settings: Settings) -> None:
    """Crea las carpetas necesarias si no existen."""
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    settings.docs_dir.mkdir(parents=True, exist_ok=True)