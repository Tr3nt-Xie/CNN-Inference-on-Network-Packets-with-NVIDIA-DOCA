#!/usr/bin/env bash
# Steps 7.8-7.10. Start the SERVER on node0 first, then the CLIENT on node1.
#   node0: bash part7_cloudlab/rdma.sh server TEST
#   node1: bash part7_cloudlab/rdma.sh client TEST
# TEST: lat | lat_p2 | put_bw | write_bw | ib_lat | ib_bw
#   lat      gpunetio_verbs_write_lat  -l 3 -p 1   (7.8, CPU proxy rings the doorbell)
#   lat_p2   gpunetio_verbs_write_lat  -l 3 -p 2   (7.8, run ONCE, record the error)
#   put_bw   gpunetio_verbs_put_bw     -l 3 -p 1   (7.9; stop the server with Ctrl+C after the client ends)
#   write_bw gpunetio_verbs_write_bw   -p 1        (7.9; rejects -l, uses GID 0)
#   ib_lat   ib_write_lat  -x 3                    (7.10, CPU-initiated baseline)
#   ib_bw    ib_write_bw   -x 3 --report_gbits     (7.10)
#   ib_lat_all / ib_bw_all   the same with -a (every message size, for the plots)
# Optional third argument RUN (e.g. 2, 3) keeps repeated runs in separate files.
# Output is saved to results/part7/<node>/<TEST>_<role>.txt
set -uo pipefail
source "$(dirname "$0")/common.sh"
ROLE=${1:?server|client}; TEST=${2:?test}; RUN=${3:-}
[ "$ROLE" = client ] && C=(-c "$PEER") || C=()
[ "$ROLE" = client ] && P=("$PEER") || P=()
case $TEST in
  lat)      CMD=(env DOCA_GPUNETIO_LOG=6 "$EX/gpunetio_verbs_write_lat" -g "$GPU_PCI" -d "$RDEV" -l 3 -p 1 "${C[@]}");;
  lat_p2)   CMD=(env DOCA_GPUNETIO_LOG=6 "$EX/gpunetio_verbs_write_lat" -g "$GPU_PCI" -d "$RDEV" -l 3 -p 2 "${C[@]}");;
  put_bw)   CMD=(env DOCA_GPUNETIO_LOG=6 "$EX/gpunetio_verbs_put_bw"    -g "$GPU_PCI" -d "$RDEV" -l 3 -p 1 "${C[@]}");;
  write_bw) CMD=(env DOCA_GPUNETIO_LOG=6 "$EX/gpunetio_verbs_write_bw"  -g "$GPU_PCI" -d "$RDEV" -p 1 "${C[@]}");;
  ib_lat)   CMD=(ib_write_lat -d "$RDEV" -x 3 "${P[@]}");;
  ib_bw)    CMD=(ib_write_bw  -d "$RDEV" -x 3 --report_gbits "${P[@]}");;
  ib_lat_all) CMD=(ib_write_lat -d "$RDEV" -x 3 -a "${P[@]}");;
  ib_bw_all)  CMD=(ib_write_bw  -d "$RDEV" -x 3 -a --report_gbits "${P[@]}");;
  *) echo "unknown test $TEST"; exit 1;;
esac
OUT=$RES/${TEST}${RUN:+_run$RUN}_${ROLE}.txt
{ echo "# $(date -u +%FT%TZ) $NODE $ROLE"; echo "\$ ${CMD[*]}"; } | tee "$OUT"
# tee ignores Ctrl+C, so the summary a server prints when you stop it is still saved
"${CMD[@]}" 2>&1 | (trap "" INT; tee -a "$OUT")
echo "(exit ${PIPESTATUS[0]})" | tee -a "$OUT"
echo "saved $OUT"
