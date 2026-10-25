#!/usr/bin/env bash
# Extra Part 7 check: CPU-initiated RDMA write with buffers in GPU memory (perftest built with CUDA).
#   node0: bash part7_cloudlab/perftest_cuda.sh server <lat|bw> [tag] [extra perftest args]
#   node1: bash part7_cloudlab/perftest_cuda.sh client <lat|bw> [tag] [extra perftest args]
# Separates "who initiates" from "where the data lives": GPUNetIO = GPU-initiated + GPU memory,
# stock perftest = CPU-initiated + host memory, this = CPU-initiated + GPU memory.
set -uo pipefail
source "$(dirname "$0")/common.sh"
ROLE=$1; KIND=$2; TAG=${3:-cuda}; shift 3 2>/dev/null || shift $#
PT=$HOME/perftest-src
BIN=$PT/ib_write_$KIND
# MEM=host keeps this side's buffer in host memory (to test one GPU side at a time)
if [ "${MEM:-gpu}" = host ]; then ARGS=(-d "$RDEV" -x 3 ${SWEEP:--a} "$@")
else ARGS=(-d "$RDEV" -x 3 ${SWEEP:--a} --use_cuda=0 --use_cuda_dmabuf "$@"); fi
[ "$KIND" = bw ] && ARGS+=(--report_gbits)
OUT=$RES/ib_${KIND}_${TAG}_$ROLE.txt
{ echo "# $(date -u +%FT%TZ) $NODE $ROLE, buffer in ${MEM:-gpu} memory ($(cd $PT && git log -1 --format=%h) built with CUDA)"
  if [ "$ROLE" = server ]; then echo "\$ $BIN ${ARGS[*]}"; timeout 300 "$BIN" "${ARGS[@]}"
  else echo "\$ $BIN ${ARGS[*]} $PEER"; timeout 300 "$BIN" "${ARGS[@]}" "$PEER"; fi
  echo "(exit $?)"; } 2>&1 | tee "$OUT"
