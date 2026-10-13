# 📚 Asistente de Documentos

Aplicación RAG (Retrieval-Augmented Generation) que responde preguntas sobre
tus documentos, cita las fuentes en cada respuesta y avisa explícitamente
cuando no encuentra la información solicitada.

- **Carga**: PDF, DOCX, TXT, MD.
- **Búsqueda**: embeddings locales (sentence-transformers) por defecto.
- **Generación**: tres modos — extractivo (sin LLM), Ollama (local) u OpenAI.
- **Citas**: cada respuesta incluye un bloque `Fuentes:` con el nombre del
  archivo, la página y la similitud, además de los fragmentos relevantes
  expandibles en la UI.
- **Privacidad**: el modo extractivo y los embeddings locales no envían nada
  a internet.

---

## 🚀 Inicio rápido

### Windows
```cmd
cd doc-assistant
run.bat
```

### macOS / Linux
```bash
cd doc-assistant
chmod +x run.sh
./run.sh
```

El script crea un `.venv`, instala dependencias, copia `.env.example` a `.env`
y abre Streamlit en `http://localhost:8501`.

> Si prefieres hacerlo manualmente:
> ```bash
> python -m venv .venv
> .venv\Scripts\activate           # Windows
> source .venv/bin/activate        # macOS/Linux
> pip install -r requirements.txt
> streamlit run app.py
> ```

---

## 🧭 Cómo usarlo

1. **Subir documentos** en la barra lateral (PDF, DOCX, TXT, MD).
2. Pulsar **Indexar archivos**. Verás cuántos fragmentos se añadieron.
3. **Chatear**: cada respuesta muestra un bloque expandible `Fuentes citadas`
   con el documento, página y similitud de cada fragmento utilizado.
4. Si la pregunta no tiene cobertura, el asistente responde:
   > *No encontré información sobre esto en los documentos cargados.*

---

## ⚙️ Configuración (`.env`)

Variables principales (todas tienen valores por defecto razonables):

| Variable | Descripción | Default |
|---|---|---|
| `LLM_PROVIDER` | `extractive`, `ollama` o `openai` | `extractive` |
| `OPENAI_API_KEY` | Clave de OpenAI (si usas `openai`) | — |
| `OPENAI_MODEL` | Modelo OpenAI | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings OpenAI | `text-embedding-3-small` |
| `OLLAMA_BASE_URL` | URL del servidor Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL` | Modelo Ollama | `llama3.1` |
| `OLLAMA_EMBEDDING_MODEL` | Modelo de embeddings Ollama | `nomic-embed-text` |
| `HF_EMBEDDING_MODEL` | Modelo sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2` |
| `SIMILARITY_THRESHOLD` | Mínimo de similitud para responder | `0.30` |
| `TOP_K` | Fragmentos a recuperar por pregunta | `4` |
| `CHUNK_SIZE` | Tamaño de fragmento (caracteres) | `800` |
| `CHUNK_OVERLAP` | Solapamiento entre fragmentos | `120` |
| `INDEX_DIR` | Carpeta del índice FAISS | `./.data/index` |
| `DOCS_DIR` | Carpeta donde se copian los archivos | `./documents` |

---

## 🧠 Modos de generación

| Modo | Pros | Contras |
|---|---|---|
| `extractive` | Sin red, sin claves, sin coste. Nunca inventa. | No “redacta”, solo muestra fragmentos. |
| `ollama` | Privado, local, redacta con un modelo abierto. | Necesitas Ollama instalado y un modelo descargado. |
| `openai` | Alta calidad de redacción y razonamiento. | Requiere clave, coste y envío de datos a OpenAI. |

> Si eliges `openai` o `ollama` y no hay clave/servidor disponible, la app
> cae automáticamente al modo `extractivo` para no fallar.

---

## 🧪 Verificación rápida sin LLM

1. Crea `documents/notas.txt` con dos o tres párrafos.
2. `run.bat` o `./run.sh`.
3. Pregunta algo que sí esté en `notas.txt`: verás las fuentes.
4. Pregunta algo que **no** esté: verás el aviso de "no encontré
   información".

---

## 🏗️ Arquitectura

```
doc-assistant/
├── app.py                      # UI (Streamlit)
├── src/
│   ├── config.py               # Carga .env y configuración validada
│   ├── loader.py               # PDF, DOCX, TXT, MD -> Document[]
│   ├── splitter.py             # Ventana deslizante en chunks
│   ├── embeddings.py           # sentence-transformers / OpenAI
│   ├── vectorstore.py          # FAISS IndexFlatIP + persistencia
│   ├── llm.py                  # OpenAI / Ollama / extractivo
│   └── rag.py                  # Orquestación (RAGAssistant)
├── documents/                  # Archivos cargados
├── .data/index/                # Índice FAISS y metadatos
├── requirements.txt
├── .env.example
├── run.bat / run.sh
└── README.md
```

### Flujo de una pregunta

1. La pregunta se embebe con el mismo modelo de los documentos.
2. FAISS devuelve los `TOP_K` chunks más similares (similitud coseno).
3. Si la mejor similitud < `SIMILARITY_THRESHOLD` → respuesta “no sé”.
4. Si hay coincidencias, se pasa el contexto al LLM con un prompt de sistema
   que prohíbe usar conocimiento externo.
5. Sea cual sea el backend, se añade un bloque `Fuentes:` con
   `documento`, `página` y `similitud`.

---

## 🛠️ Solución de problemas

- **No encuentra nada / todo devuelve “no sé”**: baja `SIMILARITY_THRESHOLD`
  en la barra lateral (por ejemplo, 0.20).
- **Quiero respuestas más precisas**: sube `TOP_K` y/o usa un embedder más
  potente (`paraphrase-mpnet-base-v2`, `bge-m3`, etc.).
- **Ollama no responde**: comprueba que `ollama serve` está corriendo y que
  el modelo está descargado (`ollama pull llama3.1`).
- **Quiero resetear el índice**: pulsa `🧹 Vaciar índice` en la barra lateral.

---

## 📜 Licencia

Uso personal y educativo. Adecúa la licencia a tu caso.