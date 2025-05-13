#!/bin/bash
# Run ruff linter on the codebase

# Activate the virtual environment if not already activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Virtual environment not found. Please run setup_venv.sh first."
        exit 1
    fi
fi

# Check if ruff is installed
if ! command -v ruff &> /dev/null; then
    echo "Ruff not found. Installing ruff..."
    uv pip install ruff
fi

# Run ruff in check mode (no fixes)
echo "Running Ruff in check mode..."
ruff check src/ app.py setup.py setup_api_key.py

# Ask if user wants to fix issues automatically
read -p "Do you want to fix issues automatically? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Fixing issues..."
    ruff check --fix src/ app.py setup.py setup_api_key.py
fi

# Format the code
echo "Formatting code with Ruff..."
ruff format src/ app.py setup.py setup_api_key.py

echo "Lint check complete!"