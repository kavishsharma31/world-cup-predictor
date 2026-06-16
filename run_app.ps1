$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating local virtual environment..."
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv .venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    }
    else {
        throw "Python was not found. Install Python 3, then rerun this script."
    }
}

Write-Host "Installing requirements..."
& $VenvPython -m pip install -r requirements.txt

if (Test-Path "scripts\smoke_test.py") {
    Write-Host "Running smoke test..."
    & $VenvPython scripts\smoke_test.py
}

Write-Host "Starting Streamlit app..."
& $VenvPython -m streamlit run app\streamlit_app.py
