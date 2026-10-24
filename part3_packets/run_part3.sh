#!/usr/bin/env bash
# Part 3: build the test PCAPs and check them with the Python reference receiver.
#   bash part3_packets/run_part3.sh IMAGE_DIR [IMAGE_DIR ...]
# e.g. bash part3_packets/run_part3.sh ~/images/cat ~/images/dog
# Without arguments, synthetic gradient images are used.
# Outputs: data/part3/{images.pcap, images_shuffled.pcap, images_stress.pcap, original.bin, original.json}
# Log:     results/part3/run-<UTC timestamp>.log
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${VIRTUAL_ENV:-}" && -f "$HOME/venv/bin/activate" ]]; then source "$HOME/venv/bin/activate"; fi

OUT=data/part3
mkdir -p "$OUT" results/part3
LOG=results/part3/run-$(date -u +%Y%m%dT%H%M%SZ).log
IMAGES=()
if [[ $# -gt 0 ]]; then IMAGES=(--images "$@"); fi
GEN="python3 part3_packets/gen_pcap.py ${IMAGES[*]:-} --orig $OUT/original.bin --seed 542"
REF="python3 part3_packets/ref_reassemble.py"

{
  echo "# Part 3 run $(date -u +%FT%TZ) on $(hostname)"
  echo "## in order";            $GEN --out "$OUT/images.pcap"
  $REF "$OUT/images.pcap" "$OUT/original.bin" --out /tmp/part3_ref.bin
  echo "## shuffled";            $GEN --shuffle --out "$OUT/images_shuffled.pcap"
  $REF "$OUT/images_shuffled.pcap" "$OUT/original.bin" --out /tmp/part3_ref.bin
  echo "## shuffled + 50 duplicates + 20 other-port"
  $GEN --shuffle --dup 50 --other-port 20 --out "$OUT/images_stress.pcap"
  $REF "$OUT/images_stress.pcap" "$OUT/original.bin" --out /tmp/part3_ref.bin
  echo "## sha256"; (cd "$OUT" && sha256sum images.pcap images_shuffled.pcap images_stress.pcap original.bin)
} 2>&1 | grep -v '^WARNING' | tee "$LOG"
echo "log -> $LOG"
