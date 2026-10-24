#!/usr/bin/env bash
# Part 6: DOCA container on the cloud GPU instance (PDF Steps 6.1-6.3).
#   bash part6_doca/run_part6.sh
# Every command and its output goes to results/part6/<UTC timestamp>/; nothing here needs a
# ConnectX/BlueField card, and the two GPUNetIO runs are EXPECTED to stop (see README).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=results/part6/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
IMG=nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host
GPU_PCI=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | sed 's/^0*\([0-9a-fA-F]*:\)//' | tr 'A-F' 'a-f')
# short form, e.g. 00000000:00:1E.0 -> 00:1e.0
NIC_PCI=$(lspci -D | awk '/Ethernet controller/ {sub(/^0000:/,"",$1); print $1; exit}')
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)

log() {  # name, command...
  local name=$1; shift
  { echo "\$ $*"; "$@"; echo "(exit $?)"; } >"$OUT/$name.txt" 2>&1
  echo "== $name: $(tail -1 "$OUT/$name.txt")"
}

echo "GPU $GPU_PCI (sm_$CUDA_ARCH), NIC $NIC_PCI -> $OUT"

# ---- 6.1 Docker with GPU access --------------------------------------------------------
log 00_host       bash -c 'uname -r; nvidia-smi --query-gpu=name,driver_version,pci.bus_id,compute_cap --format=csv; lspci | grep -iE "ethernet|nvidia|mellanox"; df -h / /var/lib/docker 2>/dev/null; docker --version; nvidia-ctk --version'
log 01_gpu_docker sudo docker run --rm --gpus all ubuntu nvidia-smi -L

# ---- 6.2 Explore DOCA -------------------------------------------------------------------
log 02_pull       sudo docker pull "$IMG"
log 03_doca_caps  sudo docker run --rm --gpus all --privileged --net=host "$IMG" \
                  bash -c 'doca_caps --version; echo; doca_caps --list-libs; echo; doca_caps --list-devs'

# ---- 6.3 gpunetio_simple_receive in the container ----------------------------------------
# sm_75 (T4) and sm_80 (A30) are already in meson.build; other GPUs need their arch added.
log 04_simple_receive sudo docker run --rm --gpus all --privileged --net=host "$IMG" bash -c "
  cd /opt/mellanox/doca/samples/doca_gpunetio/gpunetio_simple_receive &&
  grep -n 'sm_' meson.build | head &&
  meson setup /tmp/build >/tmp/meson.log 2>&1 && ninja -C /tmp/build >/tmp/ninja.log 2>&1 &&
  echo BUILD OK && timeout 60 /tmp/build/doca_gpunetio_simple_receive -n $NIC_PCI -g $GPU_PCI -e 0
  rc=\$?; echo; echo '--- meson/ninja tail'; tail -5 /tmp/meson.log /tmp/ninja.log; exit \$rc"

# ---- 6.3 open-source GPUNetIO on the host ------------------------------------------------
log 05_gpunetio_build bash -c "
  [ -d ~/gpunetio ] || git clone -q https://github.com/NVIDIA-DOCA/gpunetio ~/gpunetio &&
  cd ~/gpunetio && git log -1 --format='%h %ad %s' --date=short &&
  make install install_examples PREFIX=\$HOME/gpunetio/install CUDA_ARCH=$CUDA_ARCH 2>&1 | tail -20"
log 06_ib_devices bash -c 'ibv_devices; ls /dev/infiniband 2>&1'
log 07_gpunetio_example env DOCA_GPUNETIO_LOG=6 \
    LD_LIBRARY_PATH="$HOME/gpunetio/install/lib:${LD_LIBRARY_PATH:-}" \
    timeout 60 "$HOME/gpunetio/install/examples/gpunetio_verbs_write_lat" -g "$GPU_PCI" -d mlx5_0

echo "logs in $OUT"
