#!/usr/bin/env bash

set -e

VENV_NAME="pycaret-env"

if [ ! -d "$VENV_NAME" ]; then
    echo "Virtual environment not found. Run setup_venv.sh first."
    exit 1
fi

echo "=== Activating venv ==="
source $VENV_NAME/bin/activate

caretrun () {
    if [ -z "$1" ]; then
        echo "Usage: caretrun <script.py>"
        return 1
    fi

    SCRIPT_PATH="$1"

    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "File not found: $SCRIPT_PATH"
        return 1
    fi

    echo "=== Running $SCRIPT_PATH inside PyCaret env ==="
    python "$SCRIPT_PATH"
}

if [ -n "$1" ]; then
    caretrun "$1"
else
    echo "Usage: ./run_pycaret.sh script.py"
fi

deactivate
