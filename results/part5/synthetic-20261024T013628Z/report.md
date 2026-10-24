# Part 5: Classifying the Received Images

Measured at 2026-10-24T01:36:55.152079+00:00 on Tesla T4.
Overall result: **PASS_SYNTHETIC_ONLY**.

**Synthetic development inputs: this run validates Part 5 code only. It is not evidence of real PCAP/DPDK reassembly; integration with Parts 3 and 4 is pending.**

## Method

Raw RGB uint8 bytes are staged in pinned CPU memory and copied to the GPU. The GPU converts NHWC to NCHW float32 and scales to [0, 1]. The ResNet-18 weight preset then resizes to 256, center-crops to 224, and normalizes with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]. The result becomes contiguous FP16 and is passed to the Part 2 TRTRunner on the same CUDA stream.

Original and received images use the same engine and preprocessing. Images are matched by frame ID. Byte equality is checked separately: equal predictions alone do not prove correct reassembly.

GPU preprocessing versus an untimed CPU reference: max absolute difference 0.00195312 after the FP16 cast.

## Classification consistency

| Input | Frames | Byte-identical | Same top-1 class | Maximum logit difference |
|---|---:|---:|---:|---:|
| synthetic_ordered | 16 | 16/16 | 16/16 | 0 |
| synthetic_shuffled | 16 | 16/16 | 16/16 | 0 |

Per-frame classes and differences are in `predictions.csv`. This measures original/received consistency, not ImageNet accuracy; no ground-truth labels are assumed.

## Pinned CPU-to-GPU copy

| Batch | Bytes | Mean (ms) | Median (ms) | P95 (ms) | Mean per image (us) | Effective GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 150528 | 0.026788 | 0.026688 | 0.026912 | 26.788 | 5.619 |
| 8 | 1204224 | 0.195369 | 0.195168 | 0.195650 | 24.421 | 6.164 |
| 16 | 2408448 | 0.387235 | 0.387072 | 0.387779 | 24.202 | 6.220 |

Each size uses 50 warm-up copies and 200 measured copies. CUDA events bracket only nonblocking uint8 H2D copies on a dedicated stream, using already-pinned source memory and a preallocated GPU destination. File loading, pinning/allocation, preprocessing and inference are outside the timed interval. Small-transfer timings can include gaps between host submissions; these are observed copy timings, not a measurement of peak PCIe bandwidth. Every timed destination is also checked byte for byte.

## Interpretation

In this DPDK pipeline the reconstructed image lives in CPU memory, so it must be copied to GPU memory before GPU preprocessing and inference. GPUNetIO can place received network data directly in GPU memory, removing this CPU-to-GPU image copy. It still needs image reassembly, preprocessing and inference. These measurements isolate the copy cost; they do not directly measure an end-to-end GPUNetIO speedup.

## Reproduction

```bash
python part5_classify/classify.py --original part5_classify/test_data/synthetic/original.npy --received synthetic_ordered=part5_classify/test_data/synthetic/ordered.bin --received synthetic_shuffled=part5_classify/test_data/synthetic/shuffled.npz --out results/part5/synthetic-20261024T013628Z
```

Engine SHA-256: `96fb44da051904278837891ffebead43680e7f6077dcac0c1211c968f6683775`

Environment and all timing samples are retained in `results.json`.

References: [ResNet-18 preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) and [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html).
