#!/usr/bin/env bash
# Step 7.5 on node0: build and run gpunetio_simple_receive in the DOCA container.
# Expected on ConnectX-6 (not Dx): stops creating the DOCA Flow port with "Failed to get hws cap".
set -uo pipefail
source "$(dirname "$0")/common.sh"
OUT=$RES/7.5_simple_receive.txt
echo "NIC $NIC_PCI ($IFACE)  GPU $GPU_PCI" | tee "$OUT"
sudo docker run --rm --privileged --net=host --gpus all \
  -v /dev/infiniband:/dev/infiniband -v /dev/hugepages:/dev/hugepages \
  -v /opt/mellanox/gdrcopy/src:/opt/gdrcopy:ro "$IMG" bash -c "
  # The image puts /usr/local/cuda-13/compat (forward-compat libcuda 580.178) ahead of the host
  # driver's libcuda; with a newer host driver that fails with CUDA error 803 (driver mismatch).
  export LD_LIBRARY_PATH=/opt/gdrcopy:\$(echo \$LD_LIBRARY_PATH | tr : '\\n' | grep -v compat | paste -sd:)
  echo LD_LIBRARY_PATH=\$LD_LIBRARY_PATH
  cd /opt/mellanox/doca/samples/doca_gpunetio/gpunetio_simple_receive &&
  meson setup /tmp/build >/tmp/m.log 2>&1 && ninja -C /tmp/build >/tmp/n.log 2>&1 && echo BUILD OK || { tail -20 /tmp/m.log /tmp/n.log; exit 1; }
  timeout 60 /tmp/build/doca_gpunetio_simple_receive -n $NIC_PCI -g $GPU_PCI -e 0; echo \"(exit \$?)\"" 2>&1 | tee -a "$OUT"
echo "saved $OUT"
