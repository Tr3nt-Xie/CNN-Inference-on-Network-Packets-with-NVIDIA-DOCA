# Shared settings for the Part 7 scripts (sourced, not run).
IMG=nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NODE=$(hostname -s)
RES=$REPO/results/part7/$NODE
mkdir -p "$RES"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/gpunetio/install/lib:/opt/mellanox/gdrcopy/src:${LD_LIBRARY_PATH:-}
EX=$HOME/gpunetio/install/examples
# interface with the 10.10.1.x address, its RDMA device, and the GPU (short PCI form, e.g. 25:00.0)
IFACE=$(ip -br addr | awk '/10\.10\.1\./{print $1; exit}')
RDEV=$(rdma link show 2>/dev/null | awk -v n="$IFACE" '$0 ~ ("netdev " n "( |$)") {split($2,a,"/"); print a[1]; exit}')
NIC_PCI=$( [ -n "$IFACE" ] && basename "$(readlink -f /sys/class/net/$IFACE/device 2>/dev/null)" | sed 's/^0000://' )
GPU_PCI=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | sed 's/^0*\([0-9A-Fa-f]*:\)//' | tr 'A-F' 'a-f')
PEER=10.10.1.1
