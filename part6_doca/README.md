# Part 6: NVIDIA DOCA Container

Owner: Fengmao Xie

PDF Steps 6.1-6.3 on the team's AWS instance: Docker with GPU access, `doca_caps`, the
`gpunetio_simple_receive` build and where it stops, and the open-source GPUNetIO build and example.

Run: `bash part6_doca/run_part6.sh` (2026-10-25 02:09 UTC). Every command and its output is in
[`results/part6/20261025T020930Z/`](../results/part6/20261025T020930Z/).

## Instance

| | |
|---|---|
| Instance | AWS g4dn.xlarge, Ubuntu 22.04, kernel 6.8.0-1066-aws |
| GPU | Tesla T4 (sm_75), PCI 00:1e.0, driver 595.91.07 |
| NIC | **Amazon Elastic Network Adapter (ENA)**, PCI 00:05.0; no ConnectX or BlueField |
| Container runtime | Docker 29.9.1, NVIDIA Container Toolkit 1.20.1 |
| DOCA image | `nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host` (DOCA 3.5.0098, CUDA 13.0) |

## Results

| Step | Result | Log |
|---|---|---|
| 6.1 GPU inside Docker | **works**: `docker run --gpus all ubuntu nvidia-smi -L` lists the T4 | `01_gpu_docker.txt` |
| 6.2 `doca_caps --list-libs` | all 21 DOCA libraries installed, including `gpunetio`, `flow`, `eth`, `rdma`, `verbs` | `03_doca_caps.txt` |
| 6.2 `doca_caps --list-devs` | **No DOCA device was found** | `03_doca_caps.txt` |
| 6.3 `gpunetio_simple_receive` build | builds (sm_75 is already in `meson.build`) | `04_simple_receive.txt` |
| 6.3 run, first try | `CUDA initialization failed: system has unsupported display driver / cuda driver combination` (CUDA error 803) | `04_simple_receive.txt` |
| 6.3 run, compat library removed | CUDA starts; stops at `Failed to open NIC device based on PCI address` / `Requested Resource Not Found` | `04b_simple_receive_nocompat.txt` |
| 6.3 open-source GPUNetIO (`NVIDIA-DOCA/gpunetio` b2b7a29) | builds on the host with `CUDA_ARCH=75` | `05_gpunetio_build.txt` |
| 6.3 `ibv_devices` | empty; no `/dev/infiniband` | `06_ib_devices.txt` |
| 6.3 `gpunetio_verbs_write_lat` | `Failed to get RDMA devices list` | `07_gpunetio_example.txt` |

## What happened and why

**CUDA error 803 in the container.** The image's `LD_LIBRARY_PATH` lists
`/usr/local/cuda-13/compat` (forward-compatibility `libcuda.so.580.178.04`) before the host
driver's library, and the host driver here is newer (595.91.07). Loading the older compat
`libcuda` against the newer kernel driver fails with 803 (`04b_compat_check.txt`). Dropping the
compat directory from `LD_LIBRARY_PATH` inside the container fixes it; `run_part6.sh` now does
this, the same fix as Part 7 (`part7_cloudlab/simple_receive.sh`).

**Where it stops, and why that is expected.** After the fix the GPU side initializes, and the
sample stops when it tries to open the NIC: DOCA (and GPUNetIO's RDMA path) need an NVIDIA
ConnectX or BlueField device, and the instance has only Amazon's ENA. `doca_caps --list-devs`,
`ibv_devices` and the GPUNetIO verbs example all agree that there is no such device. So on this
instance the software stack installs, builds and reaches the GPU, but GPU-direct packet I/O cannot
run. That is why Part 7 moves to CloudLab nodes with ConnectX-6 NICs.
