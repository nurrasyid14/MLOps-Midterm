#!/usr/bin/env bash

set -e

VENV_NAME="pycaret-env"

echo "=== Using system Python ==="
python3 --version

echo "=== Creating virtual environment ==="
if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv $VENV_NAME
else
    echo "Venv already exists"
fi

echo "=== Activating venv ==="
source $VENV_NAME/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Installing PyCaret ==="
pip install pycaret

echo "=== Done ==="

deactivate