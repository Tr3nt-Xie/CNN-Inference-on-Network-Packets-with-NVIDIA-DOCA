# Part 5: Classifying the Received Images

Owner: Xiaopeng Wu

PDF Part 5: copy the rebuilt images to GPU memory, convert HWC uint8 to normalized NCHW on the GPU, classify with the FP16 engine through `part2_tensorrt/trt_runner.py`, check each class against the original image, and time the pinned-memory CPU-to-GPU copy.
