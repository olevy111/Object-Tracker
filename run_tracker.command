#!/bin/bash
cd "$(dirname "$0")"

if [ ! -x venv/bin/python ]; then
    echo "First run: creating a Python virtual environment (one-time setup)..."
    python3 -m venv venv || { echo "ERROR: Python 3 was not found. Install it and try again."; read -r; exit 1; }
fi

if ! venv/bin/python -c "import cv2, numpy" >/dev/null 2>&1; then
    echo "Installing required packages (one-time setup, may take a minute)..."
    venv/bin/python -m pip install --upgrade pip
    venv/bin/python -m pip install -r requirements.txt
fi

venv/bin/python src/app.py
