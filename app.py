"""UI Streamlit del asistente de documentos.

Ejecutar con:
    streamlit run app.py

La UI:
- Sidebar: estado del sistema, ajustes, carga/eliminación de documentos.
- Principal: chat con historial, citas, y descarga del historial.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permite ejecutar tanto `streamlit run app.py` desde la raíz del proyecto como
# `python -m streamlit run app.py` desde otros directorios.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.config import ensure_dirs, load_settings  # noqa: E402
from src.loader import SUPPORTED_EXTENSIONS  # noqa: E402
from src.rag import RAGAssistant  # noqa: E402


# ------------------------- Inicialización -------------------------

def _init_state() -> None:
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    ensure_dirs(st.session_state.settings)

    if "assistant" not in st.session_state:
        st.session_state.assistant = RAGAssistant(st.session_state.settings)

    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{"role": ..., "content": ..., "sources": [...]}]


def _assistant() -> RAGAssistant:
    return st.session_state.assistant


# ------------------------- Sidebar -------------------------

def render_sidebar() -> None:
    s = st.session_state.settings
    a = _assistant()

    with st.sidebar:
        st.title("📚 Asistente de documentos")
        st.caption(
            "Responde preguntas sobre tus documentos. Cada respuesta cita "
            "su fuente; si no encuentra información, lo dice."
        )

        st.divider()
        st.subheader("Estado")
        stats = a.stats()
        col1, col2 = st.columns(2)
        col1.metric("Fragmentos", stats["chunks"])
        col2.metric("Documentos", stats["documents"])
        st.caption(
            f"Embeddings: `{stats['embedder']}`  \n"
            f"LLM: `{stats['llm_backend']}`  \n"
            f"Umbral similitud: `{stats['threshold']:.2f}`  \n"
            f"Top-K: `{stats['top_k']}`"
        )

        st.divider()
        st.subheader("Subir documentos")
        exts = list(SUPPORTED_EXTENSIONS)
        uploads = st.file_uploader(
            "Arrastra archivos (PDF, DOCX, TXT, MD)",
            type=[e.lstrip(".") for e in exts],
            accept_multiple_files=True,
        )
        if uploads and st.button("Indexar archivos", use_container_width=True):
            _handle_uploads(uploads)

        st.divider()
        st.subheader("Documentos indexados")
        docs = a.list_documents()
        if not docs:
            st.info("Aún no hay documentos cargados.")
        else:
            for d in docs:
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.write(f"**{d.source}**  \n{d.chunks} fragmentos")
                with c2:
                    if st.button("🗑", key=f"del:{d.source}", help=f"Eliminar {d.source}"):
                        removed = a.remove_document(d.source)
                        st.toast(f"Eliminado: {d.source} ({removed} fragmentos)")
                        st.rerun()

        st.divider()
        with st.expander("Ajustes", expanded=False):
            provider = st.selectbox(
                "Proveedor de LLM",
                ["extractive", "ollama", "openai"],
                index=["extractive", "ollama", "openai"].index(s.llm_provider),
                help=(
                "extractive = sin generación (solo fragmentos).\n"
                "ollama = modelo local vía Ollama.\n"
                "openai = API de OpenAI (requiere clave)."
                ),
            )
            threshold = st.slider(
                "Umbral de similitud",
                min_value=0.0,
                max_value=1.0,
                value=float(s.similarity_threshold),
                step=0.01,
                help="Por debajo de este valor, el sistema responde 'no encontré información'.",
            )
            top_k = st.slider("Top-K", min_value=1, max_value=10, value=int(s.top_k))
            if st.button("Aplicar ajustes", use_container_width=True):
                _apply_settings(provider, threshold, top_k)

        st.divider()
        if st.button("🧹 Vaciar índice", use_container_width=True, type="secondary"):
            a.clear()
            st.session_state.messages = []
            st.toast("Índice vaciado.")
            st.rerun()


def _handle_uploads(uploads) -> None:
    a = _assistant()
    saved = []
    for up in uploads:
        target = Path(st.session_state.settings.docs_dir) / up.name
        try:
            target.write_bytes(up.getbuffer())
            saved.append(target)
        except Exception as e:
            st.error(f"No pude guardar {up.name}: {e}")
    if not saved:
        return
    try:
        result = a.add_documents(saved)
    except Exception as e:
        st.error(f"Error al indexar: {e}")
        return
    st.success(
        f"Indexados {len(result['documents'])} archivos "
        f"({result['added_chunks']} fragmentos nuevos, "
        f"{result['total_chunks']} en total)."
    )
    st.rerun()


def _apply_settings(provider: str, threshold: float, top_k: int) -> None:
    # Reconstruimos Settings con los valores nuevos.
    from dataclasses import replace
    s = st.session_state.settings
    new_settings = replace(
        s,
        llm_provider=provider,
        similarity_threshold=threshold,
        top_k=top_k,
    )
    st.session_state.settings = new_settings
    st.session_state.assistant = RAGAssistant(new_settings)
    st.toast("Ajustes aplicados.")
    st.rerun()


# ------------------------- Chat -------------------------

def render_main() -> None:
    a = _assistant()
    st.title("💬 Conversa con tus documentos")
    st.caption(
        "Pregunta cualquier cosa. Cada respuesta incluye las fuentes. "
        "Si la información no está en tus documentos, el asistente te lo dirá."
    )

    # Historial
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Fuentes citadas", expanded=False):
                    for i, h in enumerate(msg["sources"], start=1):
                        page = h.chunk.metadata.get("page")
                        cite = (
                            f"`{h.chunk.source}`, p. {page}"
                            if page
                            else f"`{h.chunk.source}`"
                        )
                        st.markdown(f"**[{i}] {cite}** — coincidencia {h.score:.2f}")
                        st.caption(_short(h.chunk.text, 480))
            if msg.get("not_found"):
                st.warning("No se encontró información en los documentos cargados.")

    prompt = st.chat_input("Escribe tu pregunta…")
    if prompt:
        _ask(prompt)


def _ask(prompt: str) -> None:
    a = _assistant()
    # Render inmediato del mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Buscando en tus documentos…"):
            try:
                ans = a.ask(prompt)
            except Exception as e:
                st.error(f"Error al responder: {e}")
                return
        placeholder.markdown(ans.answer)
        if ans.sources:
            with st.expander("Fuentes citadas", expanded=False):
                for i, h in enumerate(ans.sources, start=1):
                    page = h.chunk.metadata.get("page")
                    cite = (
                        f"`{h.chunk.source}`, p. {page}"
                        if page
                        else f"`{h.chunk.source}`"
                    )
                    st.markdown(f"**[{i}] {cite}** — coincidencia {h.score:.2f}")
                    st.caption(_short(h.chunk.text, 480))
        if ans.not_found:
            st.warning("No se encontró información en los documentos cargados.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": ans.answer,
        "sources": ans.sources,
        "not_found": ans.not_found,
    })


def _short(text: str, n: int) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# ------------------------- main -------------------------

def main() -> None:
    st.set_page_config(
        page_title="Asistente de documentos",
        page_icon="📚",
        layout="wide",
    )
    _init_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()