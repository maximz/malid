#!/bin/bash
# Run a set of notebooks

# To override kernel name from each notebook, run e.g. `nbexec --kernel-name "python3" notebook.ipynb`. Get available kernels using `jupyter kernelspec list`. We should just fix this in our notebooks though (TODO): `rg '"name": "python3"' *.ipynb`
# nbexec --kernel-name "py39-cuda-env" "$@";

# Alternative that plays nicer with jupytext:
# jupytext --sync --pipe black --set-kernel "py39-cuda-env" --execute "$@";

set -euo pipefail


# Execute one at a time, setting environment variable so logger can pick up the name of the notebook (standard methods don't work in headless mode)
for notebook in "$@"
do
    echo "Running $notebook";
    HEADLESS_NOTEBOOK_NAME="$notebook" jupytext --sync --pipe black --set-kernel "py39-cuda-env" --execute "$notebook";
    echo;

done
unset HEADLESS_NOTEBOOK_NAME;
