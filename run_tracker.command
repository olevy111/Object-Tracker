#!/bin/bash
cd "$(dirname "$0")"

fail() {
    echo ""
    echo "-----------------------------------------------------------"
    echo "ERROR: $1"
    echo "-----------------------------------------------------------"
    echo ""
    echo "Press Enter to close this window."
    read -r
    exit 1
}

# ---- 1. find a usable Python 3 -------------------------------------------
PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "Python 3.9 or newer was not found on this computer."
    echo "Opening the Python download page..."
    open "https://www.python.org/downloads/macos/" 2>/dev/null
    fail "Install Python 3 (the button on the page that just opened), then run this file again."
fi

# ---- 2. create the virtual environment (first run only) ------------------
if [ ! -x venv/bin/python ]; then
    echo "First run: creating a Python virtual environment (one-time setup)..."
    "$PY" -m venv venv || fail "Could not create the virtual environment."
    [ -x venv/bin/python ] || fail "Could not create the virtual environment."
fi

# ---- 3. install the packages (only if missing) ---------------------------
if ! venv/bin/python -c "import cv2, numpy" >/dev/null 2>&1; then
    echo "Installing required packages (one-time setup, may take a minute)..."
    venv/bin/python -m pip install --upgrade pip || fail "Could not upgrade pip. Check the internet connection."
    if ! venv/bin/python -m pip install -r requirements.txt; then
        fail "Package installation failed. Check the internet connection and try again."
    fi

    echo ""
    echo "Verifying the installation..."
    venv/bin/python src/check_setup.py || fail "Setup check failed (see the report above)."
    echo ""
fi

# ---- 4. run ---------------------------------------------------------------
venv/bin/python src/app.py || fail "The app stopped with an error (see the message above)."
