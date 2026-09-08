$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".venv\Scripts\python.exe"
} else {
    $Python = "py"
}

& $Python scripts\setup_local.py
& $Python scripts\check_env.py
& $Python -m app.web_ui
