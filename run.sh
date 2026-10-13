#!/usr/bin/env bash
# =============================================================
# Lanzador del Asistente de documentos (macOS / Linux)
# =============================================================
set -euo pipefail

cd "$(dirname "$0")"

# 1) Crear venv si hace falta
if [ ! -f ".venv/bin/python" ]; then
    echo "[setup] Creando entorno virtual..."
    python3 -m venv .venv
fi

# 2) Activar venv
# shellcheck disable=SC1091
source .venv/bin/activate

# 3) Instalar dependencias la primera vez
if [ ! -f ".venv/.deps_ok" ]; then
    echo "[setup] Instalando dependencias..."
    python -m pip install --upgrade pip >/dev/null
    python -m pip install -r requirements.txt
    touch .venv/.deps_ok
fi

# 4) .env por defecto
if [ ! -f ".env" ]; then
    echo "[setup] Creando .env desde .env.example"
    cp .env.example .env
fi

# 5) Lanzar Streamlit
echo "[run] Iniciando Streamlit en http://localhost:8501"
streamlit run app.py