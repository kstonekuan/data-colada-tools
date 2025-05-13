#!/bin/bash
# Script to set up a virtual environment

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    pip install uv
fi

# Create virtual environment
uv venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements using uv
uv pip install -r requirements.txt

# Install the package in development mode
uv pip install -e .

# Let user know setup was successful
echo "Virtual environment set up successfully!"
echo "To activate the environment, run: source venv/bin/activate"
echo ""
echo "For linting and code formatting, run: ./lint.sh"