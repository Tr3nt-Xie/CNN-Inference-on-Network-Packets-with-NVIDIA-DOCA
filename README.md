# CNN Inference on Network Packets with NVIDIA DOCA

EE542 Laboratory #8 (Fall 2026), group work. Parts 1 to 6 run on one shared
AWS GPU instance (Ubuntu 22.04, NVIDIA driver 580+); Part 7 runs on two
CloudLab d7525 nodes.

## Team

| Member | Parts |
|---|---|
| Wenxu Zhang | 1 (environment), 2 (TensorRT inference from GPU memory) |
| Fengmao Xie | 3 (images as UDP packets), 4 (DPDK receiver), 6 (DOCA container), 7 (GPUNetIO RDMA on CloudLab) |
| Xiaopeng Wu | 5 (classifying the received images on the GPU) |

## Layout

```text
part1_env/         setup_instance.sh, check_env.sh           (Part 1)
part2_tensorrt/    export_onnx.py, build_engine.py,
                   trt_runner.py, benchmark.py, run_part2.sh  (Part 2)
part3_packets/     packet generator -> images.pcap            (Part 3)
part4_dpdk/        dpdk_rx.c -> frames_rx.bin                 (Part 4)
part5_classify/    GPU preprocessing + FP16 classification    (Part 5)
part6_doca/        DOCA container and GPUNetIO build logs     (Part 6)
part7_cloudlab/    profile.py notes, command lines, logs      (Part 7)
results/           measured outputs, one folder per part
models/            generated ONNX files and engines (not committed)
```

## Data passed between parts

| From | To | What |
|---|---|---|
| Part 2 | Part 5 | `part2_tensorrt/trt_runner.py` (`TRTRunner`) and `models/resnet18_fp16.engine`, built on the instance by `run_part2.sh` |
| Part 3 | Part 4, Part 5 | `images.pcap` (in order), a shuffled PCAP, and the 16 original images |
| Part 4 | Part 5 | the rebuilt images written by the receiver |

Each image is 224 x 224 x 3 bytes (150,528 bytes) in height x width x channel
order. Parts 3 and 4 define the exact file names and layout; see their READMEs.

## Quick start on the GPU instance

```bash
git clone https://github.com/Tr3nt-Xie/CNN-Inference-on-Network-Packets-with-NVIDIA-DOCA.git
cd CNN-Inference-on-Network-Packets-with-NVIDIA-DOCA
bash part1_env/setup_instance.sh      # once per machine
source ~/.bashrc
bash part1_env/check_env.sh           # Part 1.3 versions -> results/part1/
bash part2_tensorrt/run_part2.sh      # Part 2 export, build, benchmark -> results/part2/
```

TensorRT engines depend on the GPU and the TensorRT version, so build them on
the machine that runs them instead of copying `.engine` files.
