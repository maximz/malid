#!/bin/bash
# Usage: ./run_notebook_to_new_file.sh [INPUT_NOTEBOOK_FILE] [GENERATED_OUTPUT_FILE]
# Run a notebook and save out as a new notebook. Useful when executing a template

# nbexec --kernel-name "py39-cuda-env" "$1" --stdout > "$2";

# Alternative that plays nicer with jupytext:
# Also set environment variable so logger can pick up the name of the output notebook (standard methods don't work in headless mode)
# After running, resync the new notebook to its paired source script
HEADLESS_NOTEBOOK_NAME="$2" jupytext --sync --pipe black --set-kernel "py39-cuda-env" --execute --output "$2" "$1" && jupytext --sync --pipe black "$2";
unset HEADLESS_NOTEBOOK_NAME;
