@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto python_fallback
py -3.11 --version >nul 2>nul
if errorlevel 1 goto python_fallback
py -3.11 -m venv .venv
goto install
:python_fallback
python -c "import sys; assert sys.version_info[:2] == (3,11), 'Install Python 3.11 and add it to PATH'"
if errorlevel 1 goto failed
python -m venv .venv
:install
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
if not exist "models\smartwaste.keras" (
  if exist "models\model_manifest.json" (
    ".venv\Scripts\python.exe" -m scripts.download_model
    if errorlevel 1 goto failed
  )
)
echo Setup complete. Run run.bat to launch SmartWaste AI.
exit /b 0
:failed
echo Setup failed. Check Python 3.11, internet access, and the error above.
exit /b 1
