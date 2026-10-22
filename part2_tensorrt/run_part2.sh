#!/usr/bin/env bash
# Part 2 end to end: export ONNX, build both engines, benchmark.
# Run on the GPU instance from the repository root, inside ~/venv:
#     bash part2_tensorrt/run_part2.sh
# Engines are specific to the GPU and TensorRT version, so they are rebuilt
# here rather than copied between machines. Logs go to results/part2/.
set -euo pipefail

cd "$(dirname "$0")/.."
# The AWS Deep Learning AMI points LD_LIBRARY_PATH at its own CUDA, which
# breaks the cuDNN shipped in the PyTorch wheel (see part1_env/setup_instance.sh).
unset LD_LIBRARY_PATH
MODELS=${MODELS:-models}
OUT=${OUT:-results/part2}
RUN_ID=$(date +%Y%m%d-%H%M%S)
mkdir -p "$MODELS" "$OUT"
LOG="$OUT/run-$RUN_ID.log"

{
    echo "run $RUN_ID on $(hostname), commit $(git rev-parse --short HEAD 2>/dev/null || echo none)"
    python part2_tensorrt/export_onnx.py --out-dir "$MODELS"
    for p in fp32 fp16; do
        python part2_tensorrt/build_engine.py "$MODELS/resnet18_$p.onnx" "$MODELS/resnet18_$p.engine"
    done
    python part2_tensorrt/benchmark.py --models "$MODELS" --out "$OUT" "$@"
} 2>&1 | tee "$LOG"
echo "log: $LOG"
