#!/usr/bin/env bash

set -e

VENV_NAME="pycaret-env"
PYTHON_BIN="python3.10"

echo "=== Checking Python 3.10 ==="

if ! command -v $PYTHON_BIN &> /dev/null
then
    echo "Python 3.10 not found. Installing..."

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt update
        sudo apt install -y python3.10 python3.10-venv python3.10-dev
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if ! command -v brew &> /dev/null; then
            echo "Homebrew not found. Install it first: https://brew.sh"
            exit 1
        fi
        brew install python@3.10
    else
        echo "Unsupported OS"
        exit 1
    fi
fi

echo "=== Creating virtual environment ==="
$PYTHON_BIN -m venv $VENV_NAME

echo "=== Activating venv ==="
source $VENV_NAME/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing PyCaret ==="
pip install pycaret

echo "=== Installation complete ==="

deactivate
