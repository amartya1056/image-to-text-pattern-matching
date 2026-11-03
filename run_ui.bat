@echo off
REM Starts the Streamlit web UI from the project folder (so .streamlit/config.toml is used)
cd /d "%~dp0"
if exist "..\.venv\Scripts\python.exe" (set PY=..\.venv\Scripts\python.exe) else if exist ".venv\Scripts\python.exe" (set PY=.venv\Scripts\python.exe) else (set PY=python)
%PY% -m streamlit run app.py --server.port 8501 %*
