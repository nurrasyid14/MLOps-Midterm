# =========================
# Setup PyCaret Environment (Strict Python 3.10)
# =========================

$VENV_NAME = "pycaret-env"

Write-Host "=== Checking Python Launcher ==="

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue

if (-not $pyLauncher) {
    Write-Host "Python Launcher not found. Installing Python 3.10..."
    winget install -e --id Python.Python.3.10
    Write-Host "Please restart terminal, then rerun this script."
    exit
}

Write-Host "=== Checking Python 3.10 availability ==="

$py310 = py -3.10 --version 2>$null

if (-not $py310) {
    Write-Host "Python 3.10 not found. Installing..."
    winget install -e --id Python.Python.3.10
    Write-Host "Please restart terminal, then rerun this script."
    exit
}

Write-Host "Detected: $py310"

Write-Host "=== Creating virtual environment (Python 3.10) ==="
py -3.10 -m venv $VENV_NAME

Write-Host "=== Activating venv ==="
& "$VENV_NAME\Scripts\Activate.ps1"

Write-Host "=== Upgrading pip ==="
python -m pip install --upgrade pip

Write-Host "=== Installing PyCaret ==="
pip install pycaret

Write-Host "=== Setup Complete (Python 3.10 enforced) ==="

deactivate
