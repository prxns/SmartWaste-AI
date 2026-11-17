@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  exit /b 1
)
".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
  echo Application dependencies are missing. Run setup.bat and resolve any installation errors.
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py %*
exit /b %errorlevel%
