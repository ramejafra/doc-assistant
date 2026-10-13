@echo off
REM ============================================================
REM Lanzador del Asistente de documentos (Windows)
REM ============================================================
setlocal

REM Ir al directorio del script
pushd "%~dp0"

REM 1) Crear venv si hace falta
if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creando entorno virtual...
    python -m venv .venv
    if errorlevel 1 (
        echo [error] No se pudo crear el entorno virtual. Asegurate de tener Python 3.10+ en PATH.
        popd
        exit /b 1
    )
)

REM 2) Activar venv
call ".venv\Scripts\activate.bat"

REM 3) Instalar dependencias la primera vez (marcadas por .venv\.deps_ok)
if not exist ".venv\.deps_ok" (
    echo [setup] Instalando dependencias...
    python -m pip install --upgrade pip >nul
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [error] Fallo la instalacion de dependencias.
        popd
        exit /b 1
    )
    echo. > ".venv\.deps_ok"
)

REM 4) Copiar .env.example a .env si no existe
if not exist ".env" (
    echo [setup] Creando .env desde .env.example ^(podes editarlo luego^)
    copy /Y ".env.example" ".env" >nul
)

REM 5) Lanzar Streamlit
echo [run] Iniciando Streamlit en http://localhost:8501 ...
streamlit run app.py

popd
endlocal