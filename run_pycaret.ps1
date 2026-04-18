# =========================
# Run Script in PyCaret Env (Validated)
# =========================

param(
    [string]$ScriptPath
)

$VENV_NAME = "pycaret-env"

if (-not (Test-Path $VENV_NAME)) {
    Write-Host "Virtual environment not found. Run setup_venv.ps1 first."
    exit
}

Write-Host "=== Activating venv ==="
& "$VENV_NAME\Scripts\Activate.ps1"

$version = python --version
Write-Host "Using: $version"

if ($version -notmatch "3\.10") {
    Write-Host "WARNING: This venv is not using Python 3.10"
}

function caretrun {
    param(
        [Parameter(Mandatory=$true)]
        [string]$file
    )

    if (-not (Test-Path $file)) {
        Write-Host "File not found: $file"
        return
    }

    Write-Host "=== Running $file inside PyCaret env ==="
    python $file
}

if ($ScriptPath) {
    caretrun $ScriptPath
} else {
    Write-Host "Usage:"
    Write-Host ".\run_pycaret.ps1 script.py"
}

deactivate
