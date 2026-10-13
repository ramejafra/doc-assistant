"""Cargadores de documentos.

Soporta:
- PDF (.pdf)              -> texto página por página con número de página.
- Word (.docx)            -> texto por párrafo.
- Texto plano (.txt, .md) -> archivo completo.
- Markdown (.md)          -> texto plano (estructura preservada).

Cada carga devuelve una lista uniforme de `Document` para que el resto del
sistema no tenga que preocuparse del formato de origen.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Mensajes para pypdf y docx están protegidos por try/except para que el módulo
# sea importable aunque falte alguna dependencia. El error se eleva al usar la
# función concreta.


@dataclass
class Document:
    """Unidad mínima de información indexable.

    Attributes:
        text: Texto del fragmento o documento.
        metadata: Información auxiliar (fuente, página, etc.).
    """

    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "desconocido"))

    @property
    def page(self) -> int | None:
        v = self.metadata.get("page")
        return int(v) if v is not None else None


# ------------------------- Carga por formato -------------------------

def _load_pdf(path: Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "Falta pypdf. Instala con: pip install pypdf"
        ) from e

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = _clean_text(txt)
        if txt:
            docs.append(Document(text=txt, metadata={
                "source": path.name,
                "page": idx,
                "full_path": str(path),
                "extension": path.suffix.lower(),
            }))
    return docs


def _load_docx(path: Path) -> list[Document]:
    try:
        import docx  # python-docx
    except ImportError as e:
        raise RuntimeError(
            "Falta python-docx. Instala con: pip install python-docx"
        ) from e

    d = docx.Document(str(path))
    paragraphs: list[str] = []
    for p in d.paragraphs:
        t = (p.text or "").strip()
        if t:
            paragraphs.append(t)
    text = _clean_text("\n".join(paragraphs))
    if not text:
        return []
    return [Document(text=text, metadata={
        "source": path.name,
        "page": None,
        "full_path": str(path),
        "extension": path.suffix.lower(),
    })]


def _load_text(path: Path) -> list[Document]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = _clean_text(raw)
    if not text:
        return []
    return [Document(text=text, metadata={
        "source": path.name,
        "page": None,
        "full_path": str(path),
        "extension": path.suffix.lower(),
    })]


def _load_markdown(path: Path) -> list[Document]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = _clean_text(raw)
    if not text:
        return []
    return [Document(text=text, metadata={
        "source": path.name,
        "page": None,
        "full_path": str(path),
        "extension": path.suffix.lower(),
    })]


# ------------------------- API pública -------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_file(path: str | Path) -> list[Document]:
    """Carga un archivo y devuelve sus `Document`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    if not p.is_file():
        raise ValueError(f"No es un archivo: {p}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(p)
    if ext == ".docx":
        return _load_docx(p)
    if ext in {".txt"}:
        return _load_text(p)
    if ext == ".md":
        return _load_markdown(p)
    raise ValueError(f"Formato no soportado: {ext}")


def load_many(paths: Iterable[str | Path]) -> list[Document]:
    """Carga varios archivos. Las excepciones individuales no detienen el lote."""
    docs: list[Document] = []
    errors: list[tuple[str, str]] = []
    for p in paths:
        try:
            docs.extend(load_file(p))
        except Exception as e:
            errors.append((str(p), str(e)))
    if errors and not docs:
        # Si todo falló, elevamos el último error para que la UI lo muestre.
        raise RuntimeError(f"No se pudo cargar ningún documento. Primer error: {errors[0][1]}")
    return docs


# ------------------------- Utilidades -------------------------------

_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def file_hash(path: str | Path) -> str:
    """Hash SHA-256 del contenido, útil para evitar reindexar lo mismo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()