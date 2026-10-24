#!/usr/bin/env bash
# Pull the ~22 GB DOCA image in the background (needed only for Step 7.5 on node0).
#   bash part7_cloudlab/pull_doca.sh ; tail -f ~/doca_pull.log
source "$(dirname "$0")/common.sh"
nohup sudo docker pull "$IMG" > ~/doca_pull.log 2>&1 &
echo "pulling $IMG in background -> ~/doca_pull.log"
