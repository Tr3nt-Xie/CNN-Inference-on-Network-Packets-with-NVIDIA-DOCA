#!/usr/bin/env bash
# Part 4: build the DPDK receiver and run it on every Part 3 PCAP.
#   bash part4_dpdk/run_part4.sh            (after part3_packets/run_part3.sh)
# Outputs: data/part4/frames_rx_<case>.bin (+ .ids.json); log: results/part4/run-<UTC>.log
set -euo pipefail
cd "$(dirname "$0")/.."
SUDO=""; [[ $(id -u) -ne 0 ]] && SUDO="sudo"
IN=data/part3; OUT=data/part4
mkdir -p "$OUT" results/part4
LOG=results/part4/run-$(date -u +%Y%m%dT%H%M%SZ).log
make -s -C part4_dpdk

# A PCAP with three chunks of frame 5 removed, to show that incomplete frames are reported
# and left out instead of written with holes.
python3 - "$IN/images_shuffled.pcap" "$OUT/images_missing.pcap" <<'PY'
import sys, struct
from scapy.all import rdpcap, wrpcap, UDP
pk = rdpcap(sys.argv[1]); keep = []
for p in pk:
    b = bytes(p[UDP].payload)
    fid, seq = struct.unpack_from("!IH", b, 4)
    if not (fid == 5 and seq in (0, 50, 107)):
        keep.append(p)
wrpcap(sys.argv[2], keep)
PY

run() {  # case pcap
  echo "## $1 ($2)"
  $SUDO ./part4_dpdk/dpdk_rx -l 0 --no-huge -m 512 --no-pci --log-level=lib.eal:warning \
    --vdev "net_pcap0,rx_pcap=$2,tx_pcap=/dev/null" -- "$OUT/frames_rx_$1.bin" || echo "(exit status $?)"
  if cmp -s "$IN/original.bin" "$OUT/frames_rx_$1.bin"; then
    echo "cmp original.bin: identical (16/16 byte for byte)"
  else
    echo "cmp original.bin: differs; ids written: $(cat "$OUT/frames_rx_$1.bin.ids.json")"
  fi
}
{
  echo "# Part 4 run $(date -u +%FT%TZ) on $(hostname), $(pkg-config --modversion libdpdk | sed 's/^/DPDK /')"
  run inorder  "$IN/images.pcap"
  run shuffled "$IN/images_shuffled.pcap"
  run stress   "$IN/images_stress.pcap"
  run missing  "$OUT/images_missing.pcap"
} 2>&1 | grep -v '^WARNING' | tee "$LOG"
echo "log -> $LOG"
