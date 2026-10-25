#!/usr/bin/env bash
# Extra Part 7 check: GPUNetIO examples with other NIC handler modes, run as root.
#   bash part7_cloudlab/gpunetio_modes.sh <server|client> <lat|put_bw> <mode>
# mode = -p value: 1 CPU proxy, 2 GPU rings doorbell, 6 GPU BlueFlame (the GPU writes the
# request into the NIC's doorbell page, so the NIC need not fetch it from GPU memory),
# 10 GPU doorbell without doorbell-record update.
set -uo pipefail
source "$(dirname "$0")/common.sh"
ROLE=$1; KIND=$2; MODE=$3
[ "$ROLE" = client ] && C=(-c "$PEER") || C=()
BIN=$EX/gpunetio_verbs_write_lat; [ "$KIND" = put_bw ] && BIN=$EX/gpunetio_verbs_put_bw
CMD=(sudo env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" DOCA_GPUNETIO_LOG=6 timeout 120 "$BIN" -g "$GPU_PCI" -d "$RDEV" -l 3 -p "$MODE" "${C[@]}")
OUT=$RES/${KIND}_p${MODE}_sudo_$ROLE.txt
{ echo "# $(date -u +%FT%TZ) $NODE $ROLE"; echo "\$ ${CMD[*]}"; "${CMD[@]}"; echo "(exit $?)"; } 2>&1 | tee "$OUT" >/dev/null
echo "saved $OUT"
