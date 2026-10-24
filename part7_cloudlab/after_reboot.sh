#!/usr/bin/env bash
# Step 7.3 per-boot settings + Step 7.4 checks. Run after EVERY reboot, on both nodes:
#   bash part7_cloudlab/after_reboot.sh
# Hugepages, the gdrdrv module, persistence mode and ACS are lost on reboot.
set -uo pipefail
source "$(dirname "$0")/common.sh"
OUT=$RES/after_reboot-$(date -u +%H%M%S).txt
{
echo "== $(date -u +%FT%TZ) $NODE"
echo "== kernel cmdline"; cat /proc/cmdline
grep -q 'iommu=off' /proc/cmdline && echo "IOMMU off: OK" || echo "WARNING: iommu=off missing from cmdline"
echo "== nvidia-smi"; nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv || echo "WARNING: nvidia-smi failed - reboot again or check the driver install"
sudo nvidia-smi -pm 1
echo "== hugepages"; echo 2048 | sudo tee /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages; grep -i huge /proc/meminfo | head -3

echo "== GDRCopy v2.5.2"
if [ ! -f /opt/mellanox/gdrcopy/src/libgdrapi.so.2 ]; then
  sudo mkdir -p /opt/mellanox
  [ -d /opt/mellanox/gdrcopy ] || sudo git clone -q --branch v2.5.2 https://github.com/NVIDIA/gdrcopy /opt/mellanox/gdrcopy
  ( cd /opt/mellanox/gdrcopy && sudo make CUDA=/usr/local/cuda all >/tmp/gdrcopy_build.log 2>&1 ) || { echo "gdrcopy build FAILED:"; tail -30 /tmp/gdrcopy_build.log; }
fi
ls /opt/mellanox/gdrcopy/src/libgdrapi.so* 2>/dev/null
lsmod | grep -q gdrdrv || ( cd /opt/mellanox/gdrcopy && sudo ./insmod.sh )
lsmod | grep gdrdrv || echo "WARNING: gdrdrv not loaded"

echo "== ACS off on all PCIe bridges (setpci method from the GPUNetIO install guide)"
n=0
for BDF in $(lspci -d "*:*:*" | awk '{print $1}'); do
  sudo setpci -v -s "$BDF" ECAP_ACS+0x6.w >/dev/null 2>&1 || continue
  sudo setpci -v -s "$BDF" ECAP_ACS+0x6.w=0000 >/dev/null && n=$((n+1))
done
echo "ACS cleared on $n devices"

echo "== Step 7.4 hardware"
ip -br addr
echo "10.10.1.x interface: ${IFACE:-NONE}  NIC PCI: ${NIC_PCI:-?}  RDMA device: ${RDEV:-?}  GPU PCI: ${GPU_PCI:-?}"
[ -n "$IFACE" ] && ethtool -i "$IFACE"
ibv_devices
rdma link show
lspci | grep -i -E 'mellanox|nvidia'
uname -r
nvidia-smi topo -m
} 2>&1 | tee "$OUT"
echo; echo "saved $OUT"
echo "USE: -g $GPU_PCI -d $RDEV   (peer for clients: $PEER)"
