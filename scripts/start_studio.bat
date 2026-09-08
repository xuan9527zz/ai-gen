@echo off
setlocal
cd /d "%~dp0\.."

echo.
echo ================================================================
echo ILLUSTRIOUS RECONSTRUCTION STUDIO v0.5.0
echo ================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=py
)

%PYTHON% scripts\setup_local.py
%PYTHON% scripts\check_env.py

echo.
echo Starting Studio...
echo.
%PYTHON% -m app.web_ui

endlocal
