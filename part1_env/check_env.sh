#!/usr/bin/env bash
# Part 1.3: record the GPU, driver, CUDA, PyTorch and TensorRT versions.
# Run from the repository root:  bash part1_env/check_env.sh
# Output: results/part1/env-<host>-<time>.txt
set -uo pipefail

cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH
# shellcheck disable=SC1090
[[ -f ~/venv/bin/activate ]] && source ~/venv/bin/activate
unset LD_LIBRARY_PATH  # see setup_instance.sh
mkdir -p results/part1
out="results/part1/env-$(hostname)-$(date +%Y%m%d-%H%M%S).txt"

run() { echo "\$ $*"; "$@" 2>&1; echo "(exit $?)"; echo; }
{
    echo "time: $(date -Is)"
    run lsb_release -ds
    run uname -r
    run nvidia-smi
    run nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
    run nvcc --version
    run python3 -c "import torch; print(torch.cuda.is_available())"
    run python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
    run python3 -c "import tensorrt; print(tensorrt.__version__)"
    run python3 -c "import torchvision, onnx, onnxscript, scapy, numpy; print('torchvision', torchvision.__version__, 'onnx', onnx.__version__, 'onnxscript', onnxscript.__version__, 'scapy', scapy.VERSION, 'numpy', numpy.__version__)"
    run pkg-config --modversion libdpdk
} | tee "$out"
echo "saved $out"
