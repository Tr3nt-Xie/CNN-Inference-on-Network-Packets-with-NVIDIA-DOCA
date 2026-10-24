# Part 7: GPU-Initiated RDMA with GPUNetIO on CloudLab

Owner: Fengmao Xie

PDF Steps 7.1-7.10: CloudLab profile (kept in its own public repository,
https://github.com/Tr3nt-Xie/ee542-lab8-cloudlab-profile), node setup and hardware checks,
`gpunetio_simple_receive` on the d7525 nodes, GPUNetIO latency/bandwidth, and the perftest
CPU-initiated baseline. Logs go to `results/part7/<node>/`.

## Runbook (4-hour slot)

On **both** nodes (two terminals):

```bash
git clone https://github.com/Tr3nt-Xie/CNN-Inference-on-Network-Packets-with-NVIDIA-DOCA.git ~/lab8
cd ~/lab8
bash part7_cloudlab/setup_node.sh 2>&1 | tee ~/setup_node.log   # 7.3 + 7.6 grub, ~20-40 min
sudo reboot
# reconnect, then after EVERY reboot:
cd ~/lab8 && bash part7_cloudlab/after_reboot.sh                 # 7.3 per-boot + 7.4 checks
```

`after_reboot.sh` prints the `-g` (GPU PCI) and `-d` (RDMA device) values; the scripts below
detect them automatically.

node0 only:

```bash
bash part7_cloudlab/pull_doca.sh        # ~22 GB in the background; tail -f ~/doca_pull.log
bash part7_cloudlab/simple_receive.sh   # 7.5 once the pull has finished
```

RDMA tests: **server on node0 first, then client on node1**, one test at a time.

| Step | node0 | node1 |
|---|---|---|
| 7.8 latency | `bash part7_cloudlab/rdma.sh server lat` | `bash part7_cloudlab/rdma.sh client lat` |
| 7.8 `-p 2` once (record the error) | `bash part7_cloudlab/rdma.sh server lat_p2` | `bash part7_cloudlab/rdma.sh client lat_p2` |
| 7.9 put bandwidth | `bash part7_cloudlab/rdma.sh server put_bw` (Ctrl+C after the client ends) | `bash part7_cloudlab/rdma.sh client put_bw` |
| 7.9 write bandwidth | `bash part7_cloudlab/rdma.sh server write_bw` (Ctrl+C after the client ends) | `bash part7_cloudlab/rdma.sh client write_bw` |
| 7.10 CPU latency | `bash part7_cloudlab/rdma.sh server ib_lat` | `bash part7_cloudlab/rdma.sh client ib_lat` |
| 7.10 CPU bandwidth | `bash part7_cloudlab/rdma.sh server ib_bw` | `bash part7_cloudlab/rdma.sh client ib_bw` |

Before the slot ends, copy the logs off (CloudLab erases the nodes), from the laptop:

```bash
scp -r <user>@<node0-host>:lab8/results/part7/ ./results/
scp -r <user>@<node1-host>:lab8/results/part7/ ./results/
```

Then terminate the experiment.

## Files

| File | Step | What it does |
|---|---|---|
| `common.sh` | - | Finds the 10.10.1.x interface, its RDMA device, NIC and GPU PCI addresses; sets paths |
| `setup_node.sh` | 7.3, 7.6 | Packages, NVIDIA open driver 580, CUDA 13.0, Docker + NVIDIA runtime with data-root `/mydata/docker`, open-source GPUNetIO build, `amd_iommu=off iommu=off` |
| `after_reboot.sh` | 7.3, 7.4 | Persistence mode, 2,048 x 2 MB hugepages, GDRCopy v2.5.2 build + gdrdrv, ACS off (setpci), hardware report |
| `pull_doca.sh` | 7.5 | Pulls the DOCA image in the background |
| `simple_receive.sh` | 7.5 | Builds and runs `gpunetio_simple_receive` in the DOCA container |
| `rdma.sh` | 7.8-7.10 | Runs one GPUNetIO or perftest test as server or client and saves the output |

Not done, per the handout: no firmware flashing and no `mlxconfig` changes.
