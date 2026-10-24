#!/usr/bin/env bash
# Run from anywhere; relative input paths are resolved from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
# Match Part 2's workaround for the AWS AMI's incompatible CUDA library path.
unset LD_LIBRARY_PATH
if [[ -z "${VIRTUAL_ENV:-}" && -f "$HOME/venv/bin/activate" ]]; then
    source "$HOME/venv/bin/activate"
fi
exec python part5_classify/classify.py "$@"
