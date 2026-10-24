# Part 5 视频讲解提纲

先在真实 GPU 上使用组员的 Part 3 原图和 Part 4 重组结果运行程序。
打开这次运行生成的 `report.md`、`predictions.csv` 和 `copy_timing.csv`。
下面是讲解顺序，不包含尚未测得的数字。

1. **说明输入与目标。** Part 4 在 CPU 内存里重组了 RGB 图片。本部分把图片
   送入 GPU，用 Part 2 的 TensorRT FP16 ResNet-18 分类。原图和重组图使用
   完全相同的预处理与推理流程。
2. **展示 GPU 预处理代码。** 图片是 HWC uint8；先复制原始字节到 GPU，再在
   GPU 上转换为 NCHW float32、缩放到 [0,1]，调用权重对应的 resize、crop、
   ImageNet normalization，最后转换为连续的 FP16 输入。
3. **展示一致性结果。** 打开本次 `predictions.csv`，说明按 frame_id 匹配图片，
   展示顺序与乱序接收数据的逐图类别。引用真实报告中的一致数量。字节一致性
   是独立检查，因为两个不同图片也可能得到同一个预测类别。
4. **展示复制时间。** 打开 `copy_timing.csv`，说明 batch 1、8、16 的实际结果。
   输入使用 pinned memory；计时前已经分配 CPU/GPU 缓冲区；使用 CUDA events，
   预热 50 次、测量 200 次；GPU 工作完成后才读取时间。计时不包括磁盘读取、
   内存分配、归一化和推理。
5. **解释 GPUNetIO 的作用。** 当前 DPDK 路径先在 CPU 内存得到图像，再搬到 GPU。
   GPUNetIO 能让网络数据直接进入 GPU 内存，从而省去这里测量的主机到 GPU 复制。
   图像重组、预处理和推理仍然存在，所以不能把复制时间直接说成完整系统的实测加速。

如果只运行了 synthetic smoke test，要在视频中明确它是程序验证数据。
正式展示需要真实 Part 3/4 的文件和实际运行输出；不要把未运行的项目说成已完成。
