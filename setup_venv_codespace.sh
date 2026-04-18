#!/usr/bin/env bash

set -e

VENV_NAME="pycaret-env"
PYTHON_VERSION="3.10.13"

echo "=== Checking pyenv ==="

# install pyenv kalau belum ada
if ! command -v pyenv &> /dev/null; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash

    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
else
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init -)"
fi

echo "=== Installing Python $PYTHON_VERSION ==="
pyenv install -s $PYTHON_VERSION

echo "=== Setting local Python version ==="
pyenv local $PYTHON_VERSION

PYTHON_BIN="python"

echo "=== Using Python ==="
$PYTHON_BIN --version

echo "=== Creating virtual environment ==="
if [ ! -d "$VENV_NAME" ]; then
    $PYTHON_BIN -m venv $VENV_NAME
else
    echo "Venv already exists"
fi

VENV_PY="$VENV_NAME/bin/python"

echo "=== Upgrading pip ==="
$VENV_PY -m pip install --upgrade pip setuptools wheel

echo "=== Installing dependencies ==="
$VENV_PY -m pip install -r requirements.txt

echo "=== Installing PyCaret ==="
$VENV_PY -m pip install pycaret --no-cache-dir

echo "=== Done ==="