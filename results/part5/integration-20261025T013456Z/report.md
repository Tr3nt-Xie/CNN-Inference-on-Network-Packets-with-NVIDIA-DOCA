# Part 5: Classifying the Received Images

Measured at 2026-10-25T01:35:10.267305+00:00 on Tesla T4.
Overall result: **PASS**.

This integration run used 16 real cat/dog images from the cached Lab7 dataset, packetized with the teammate Part 3 generator and replayed through the actual Part 4 DPDK pcap PMD on this T4 instance. In-order, shuffled, and stress PCAPs were all reassembled by the compiled C receiver before GPU classification. This is PCAP replay, not a live NIC throughput benchmark. Input filenames and hashes are retained in `original_images.json` and `input_provenance.json`; the adjacent run log records DPDK counters. These newly prepared image bytes differ from the teammate previous run.

## Method

Raw RGB uint8 bytes are staged in pinned CPU memory and copied to the GPU. The GPU converts NHWC to NCHW float32 and scales to [0, 1]. The ResNet-18 weight preset then resizes to 256, center-crops to 224, and normalizes with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]. The result becomes contiguous FP16 and is passed to the Part 2 TRTRunner on the same CUDA stream.

Original and received images use the same engine and preprocessing. Images are matched by frame ID. Byte equality is checked separately: equal predictions alone do not prove correct reassembly.

GPU preprocessing versus an untimed CPU reference: max absolute difference 0.00195312 after the FP16 cast.

## Classification consistency

| Input | Frames | Byte-identical | Same top-1 class | Maximum logit difference |
|---|---:|---:|---:|---:|
| inorder | 16 | 16/16 | 16/16 | 0 |
| shuffled | 16 | 16/16 | 16/16 | 0 |
| stress | 16 | 16/16 | 16/16 | 0 |

Per-frame classes and differences are in `predictions.csv`. This measures original/received consistency, not ImageNet accuracy; no ground-truth labels are assumed.

## Pinned CPU-to-GPU copy

| Batch | Bytes | Mean (ms) | Median (ms) | P95 (ms) | Mean per image (us) | Effective GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 150528 | 0.026954 | 0.026752 | 0.027168 | 26.954 | 5.585 |
| 8 | 1204224 | 0.195430 | 0.195232 | 0.195845 | 24.429 | 6.162 |
| 16 | 2408448 | 0.387400 | 0.387168 | 0.387778 | 24.212 | 6.217 |

Each size uses 50 warm-up copies and 200 measured copies. CUDA events bracket only nonblocking uint8 H2D copies on a dedicated stream, using already-pinned source memory and a preallocated GPU destination. File loading, pinning/allocation, preprocessing and inference are outside the timed interval. Small-transfer timings can include gaps between host submissions; these are observed copy timings, not a measurement of peak PCIe bandwidth. Every timed destination is also checked byte for byte.

## Interpretation

In this DPDK pipeline the reconstructed image lives in CPU memory, so it must be copied to GPU memory before GPU preprocessing and inference. GPUNetIO can place received network data directly in GPU memory, removing this CPU-to-GPU image copy. It still needs image reassembly, preprocessing and inference. These measurements isolate the copy cost; they do not directly measure an end-to-end GPUNetIO speedup.

## Reproduction

```bash
python part5_classify/classify.py --original /home/ubuntu/lab8/data/integration-20261025T013456Z/part3/original.bin --received inorder=/home/ubuntu/lab8/data/integration-20261025T013456Z/part4/frames_rx_inorder.bin --received shuffled=/home/ubuntu/lab8/data/integration-20261025T013456Z/part4/frames_rx_shuffled.bin --received stress=/home/ubuntu/lab8/data/integration-20261025T013456Z/part4/frames_rx_stress.bin --received-ids inorder=/home/ubuntu/lab8/data/integration-20261025T013456Z/part4/frames_rx_inorder.bin.ids.json --received-ids shuffled=/home/ubuntu/lab8/data/integration-20261025T013456Z/part4/frames_rx_shuffled.bin.ids.json --received-ids stress=/home/ubuntu/lab8/data/integration-20261025T013456Z/part4/frames_rx_stress.bin.ids.json --out results/part5/integration-20261025T013456Z
```

Engine SHA-256: `96fb44da051904278837891ffebead43680e7f6077dcac0c1211c968f6683775`

Environment and all timing samples are retained in `results.json`.

References: [ResNet-18 preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) and [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html).
