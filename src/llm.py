"""Capa LLM con tres backends:

- OpenAI: requiere `openai` instalado y `OPENAI_API_KEY`.
- Ollama: requiere `ollama` corriendo localmente.
- ExtractiveQA: no requiere nada externo; devuelve los fragmentos más
  relevantes como respuesta, marcados como extractivos.

El modo extractivo es el "fallback seguro": siempre está disponible, no inventa
información y muestra explícitamente que la respuesta proviene de los
documentos.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .vectorstore import Hit


SYSTEM_PROMPT_BASE = (
    "Eres un asistente que responde preguntas basándose EXCLUSIVAMENTE en los "
    "fragmentos de documentos proporcionados como contexto. Si la respuesta "
    "no está contenida en los fragmentos, debes decirlo de forma explícita: "
    "'No encontré información sobre esto en los documentos cargados.' "
    "No utilices conocimiento externo, no inventes datos y no rellenes huecos. "
    "Cuando respondas, cita el documento fuente (nombre) y, si está, el número "
    "de página entre corchetes, por ejemplo: [manual.pdf, p. 12]. "
    "Responde en el mismo idioma de la pregunta del usuario."
)


@dataclass
class GenerationResult:
    answer: str
    used_extractive: bool = False
    backend: str = "extractive"


class LLM(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> GenerationResult: ...


# ----------------------- OpenAI -----------------------

class OpenAILLM:
    name = "openai"

    def __init__(self, model: str, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no configurada.")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "Falta openai. Instala con: pip install openai"
            ) from e
        self.model = model
        self._client = OpenAI(api_key=api_key)

    def generate(self, question: str, hits: list[Hit]) -> GenerationResult:
        context = _format_context(hits)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_BASE},
            {
                "role": "user",
                "content": (
                    f"Pregunta del usuario:\n{question}\n\n"
                    f"Fragmentos de contexto (cada uno con su cita):\n{context}\n\n"
                    "Instrucciones:\n"
                    "1. Responde solo con información presente en los fragmentos.\n"
                    "2. Si no hay información suficiente, dilo explícitamente.\n"
                    "3. Incluye al final una sección 'Fuentes:' con la lista de "
                    "citas que respaldan tu respuesta.\n"
                ),
            },
        ]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return GenerationResult(answer=answer, backend=self.name)


# ----------------------- Ollama -----------------------

class OllamaLLM:
    name = "ollama"

    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        try:
            import requests  # noqa: F401  (solo se usa aquí)
        except ImportError as e:
            raise RuntimeError(
                "Falta requests. Instala con: pip install requests"
            ) from e

    def generate(self, question: str, hits: list[Hit]) -> GenerationResult:
        import requests
        context = _format_context(hits)
        prompt = (
            f"<s>{SYSTEM_PROMPT_BASE}</s>\n"
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {question}\n\n"
            "Respuesta (recuerda citar al final con 'Fuentes:'):"
        )
        url = f"{self.base_url}/api/generate"
        r = requests.post(
            url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        return GenerationResult(answer=(data.get("response") or "").strip(), backend=self.name)


# ----------------------- Extractive (fallback) -----------------------

class ExtractiveLLM:
    """Devuelve los fragmentos más relevantes como respuesta.

    No genera texto nuevo. La 'respuesta' es un extracto limpio de los
    documentos, con citas. Es determinista, no requiere red ni modelos.
    """

    name = "extractive"

    def generate(self, question: str, hits: list[Hit]) -> GenerationResult:
        if not hits:
            return GenerationResult(
                answer=(
                    "No encontré información sobre esto en los documentos "
                    "cargados."
                ),
                used_extractive=True,
                backend=self.name,
            )
        bullets: list[str] = []
        for i, h in enumerate(hits, start=1):
            snippet = _truncate(h.chunk.text, max_chars=600)
            bullets.append(f"{i}. {snippet}")
        body = "\n\n".join(bullets)
        answer = (
            "Fragmentos relevantes encontrados en los documentos (modo extractivo):\n\n"
            f"{body}"
        )
        return GenerationResult(answer=answer, used_extractive=True, backend=self.name)


# ----------------------- Fábrica -----------------------

def build_llm(*, provider: str, openai_api_key: str, openai_model: str,
              ollama_base_url: str, ollama_model: str) -> LLM:
    """Devuelve la implementación adecuada según el proveedor y disponibilidad."""
    if provider == "openai":
        if not openai_api_key:
            return ExtractiveLLM()
        try:
            return OpenAILLM(openai_model, openai_api_key)
        except Exception:
            return ExtractiveLLM()
    if provider == "ollama":
        try:
            return OllamaLLM(ollama_model, ollama_base_url)
        except Exception:
            return ExtractiveLLM()
    return ExtractiveLLM()


# ----------------------- Utilidades -----------------------

def _format_context(hits: Iterable[Hit]) -> str:
    parts: list[str] = []
    for i, h in enumerate(hits, start=1):
        src = h.chunk.metadata.get("source", "desconocido")
        page = h.chunk.metadata.get("page")
        cite = f"[{src}, p. {page}]" if page else f"[{src}]"
        parts.append(f"--- Fragmento {i} {cite} (similitud {h.score:.2f}) ---\n{h.chunk.text}")
    return "\n\n".join(parts) if parts else "(sin fragmentos)"


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_sources_block(hits: Iterable[Hit]) -> str:
    """Genera un bloque 'Fuentes:' consistente independientemente del backend."""
    seen: set[tuple[str, int | None]] = set()
    lines: list[str] = []
    for h in hits:
        src = h.chunk.metadata.get("source", "desconocido")
        page = h.chunk.metadata.get("page")
        key = (src, page if isinstance(page, int) else None)
        if key in seen:
            continue
        seen.add(key)
        if page:
            lines.append(f"- {src} (p. {page}) — coincidencia {h.score:.2f}")
        else:
            lines.append(f"- {src} — coincidencia {h.score:.2f}")
    return "\n".join(lines) if lines else ""