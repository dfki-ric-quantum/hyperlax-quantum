#!/bin/bash

# HyperLaX Environment Setup Script
# Usage: source envs/hyperlax_setup.sh [--debug]

# Prevent running instead of sourcing
[[ "${BASH_SOURCE[0]}" != "${0}" ]] || { echo "Please 'source' this script, do not execute."; exit 1; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            export DEBUG=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            # exit 1  # Uncomment to be strict
            ;;
    esac
done

# Project paths
HYPERLAX_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HYPERLAX_ROOT="${HYPERLAX_ROOT:-$HYPERLAX_SCRIPT_DIR/../}"
if [[ ":$PYTHONPATH:" != *":$HYPERLAX_ROOT:"* ]]; then
  export PYTHONPATH="$PYTHONPATH:$HYPERLAX_ROOT"
fi

# Demo/output configuration
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export BASE_DEMO_OUTPUT_DIR="/tmp/hyperlax_demos"

# JAX/XLA configuration
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"
echo "Set XLA flags: XLA_PYTHON_CLIENT_PREALLOCATE='false' and XLA_PYTHON_CLIENT_ALLOCATOR='platform'"

# Debug vs Production settings
if [ -z ${DEBUG+x} ]; then
    echo "Production mode - DEBUG disabled"
    export PYTHONBREAKPOINT="0"
else
    echo "Debug mode enabled"
    export WANDB_SILENT="true"
    export JAX_DISABLE_JIT="true"
fi

# Optional: Conda activation (uncomment if needed)
# if [[ -n "$CONDA_EXE" ]]; then
#     source $(dirname $(dirname $CONDA_EXE))/etc/profile.d/conda.sh
#     conda activate hyperlax 2>/dev/null || echo "Conda environment 'hyperlax' not found"
# else
#     echo "Conda not found in PATH"
# fi

# Create demo output directory
mkdir -p "$BASE_DEMO_OUTPUT_DIR"

# Export common paths
export CONFIGS_DIR="$HYPERLAX_ROOT/hyperlax/configs/"

# Show environment summary
echo "===== Current Environment Variables ====="
echo "HYPERLAX_ROOT                 = $HYPERLAX_ROOT"
echo "PYTHONPATH                    = $PYTHONPATH"
echo "TIMESTAMP                     = $TIMESTAMP"
echo "BASE_DEMO_OUTPUT_DIR          = $BASE_DEMO_OUTPUT_DIR"
echo "XLA_PYTHON_CLIENT_PREALLOCATE = $XLA_PYTHON_CLIENT_PREALLOCATE"
echo "XLA_PYTHON_CLIENT_ALLOCATOR   = $XLA_PYTHON_CLIENT_ALLOCATOR"
echo "PYTHONBREAKPOINT              = $PYTHONBREAKPOINT"
echo "WANDB_SILENT                  = $WANDB_SILENT"
echo "JAX_DISABLE_JIT               = $JAX_DISABLE_JIT"
echo "CONFIGS_DIR                   = $CONFIGS_DIR"
echo "========================================="
echo "HyperLaX environment loaded!"
