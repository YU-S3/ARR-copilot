@echo off
cd /d "%~dp0"
set "PYTHON_EXE=D:\Download\anaconda\envs\arr_rf\python.exe"

if exist "%PYTHON_EXE%" (
  "%PYTHON_EXE%" "%~dp0pa_api_backend.py" --host 0.0.0.0 --port 8000
) else (
  conda run -n arr_rf python "%~dp0pa_api_backend.py" --host 0.0.0.0 --port 8000
)
