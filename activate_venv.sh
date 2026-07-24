#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Run the activate_venv.sh script to activate the Python virtual environment
# With command: source activate_venv.sh
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# This script:
# 1. Deactivates the current Conda environment (if one is active)
# 2. Activates the project's Python virtual environment (.venv)
# 3. Displays information to verify the correct environment is active
# -----------------------------------------------------------------------------

# Get the absolute path of the directory where this script is located.
# This allows the script to work even if you run it from another directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if a Conda environment is currently active.
# The CONDA_DEFAULT_ENV environment variable is set only when Conda is active.
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "🔄 Deactivating Conda environment: $CONDA_DEFAULT_ENV"
    conda deactivate
fi

# Activate the Python virtual environment (.venv)
# located in the same directory as this script.
source "$SCRIPT_DIR/.venv/bin/activate"

# Display the name of the activated virtual environment.
echo "✅ Virtual environment activated: $(basename "$VIRTUAL_ENV")"

# Show the Python executable being used.
# This helps confirm that Python is coming from the .venv directory.
echo "🐍 Python: $(which python)"

# Show the Pip executable being used.
# This ensures packages will be installed into the correct environment.
echo "📦 Pip: $(which pip)"