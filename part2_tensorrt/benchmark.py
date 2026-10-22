"""Part 2.4: time PyTorch FP32, TensorRT FP32 and TensorRT FP16 on GPU input.

For each batch size the input is a random CUDA tensor created before timing,
so each measurement covers only the model running on data already in GPU
memory (no host-to-device copy). Every implementation gets warm-up runs, then
each timed run is bracketed by CUDA events and the mean is reported.

Accuracy is checked on the same input: the largest absolute logit difference
from PyTorch FP32 and the fraction of images whose top-1 class matches.

Run on the GPU instance after export_onnx.py and build_engine.py:
    python part2_tensorrt/benchmark.py --models models --out results/part2
"""
import argparse
import datetime
import json
import pathlib
import platform
import statistics
import subprocess

import tensorrt as trt
import torch
import torchvision

from trt_runner import TRTRunner


def time_gpu(fn, warmup, iters):
    """Return per-run GPU times in ms for fn(), measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(iters)]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in events]


def summarize(times_ms, batch):
    mean = statistics.fmean(times_ms)
    return {
        "mean_ms": mean,
        "std_ms": statistics.pstdev(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "images_per_s": batch / (mean / 1000.0),
    }


def environment():
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    return {
        "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "driver": driver,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "tensorrt": trt.__version__,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def markdown_table(results, batches):
    head = "| Implementation | " + " | ".join(f"Batch {b}" for b in batches) + " |"
    lines = [head, "|---" * (len(batches) + 1) + "|"]
    for impl, rows in results.items():
        cells = []
        for b in batches:
            r = rows[str(b)]
            cell = f"{r['mean_ms']:.3f} ms, {r['images_per_s']:.0f} img/s"
            if "speedup" in r:
                cell += f", {r['speedup']:.2f}x"
            cells.append(cell)
        lines.append(f"| {impl} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", type=pathlib.Path, default=pathlib.Path("models"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/part2"))
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model = torchvision.models.resnet18(weights="DEFAULT").eval().cuda()
    trt32 = TRTRunner(args.models / "resnet18_fp32.engine")
    trt16 = TRTRunner(args.models / "resnet18_fp16.engine")

    results = {"PyTorch FP32": {}, "TensorRT FP32": {}, "TensorRT FP16": {}}
    accuracy = {"TensorRT FP32": {}, "TensorRT FP16": {}}
    # Run everything on one non-default stream. On the legacy default stream
    # TensorRT adds a cudaStreamSynchronize to every enqueueV3 call.
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    for batch in args.batches:
        x32 = torch.randn(batch, 3, 224, 224, device="cuda")
        x16 = x32.to(trt16.input_dtype)  # converted once, outside the timed region
        out32 = torch.empty(batch, 1000, device="cuda", dtype=trt32.output_dtype)
        out16 = torch.empty(batch, 1000, device="cuda", dtype=trt16.output_dtype)

        with torch.inference_mode():
            ref = model(x32)
            runs = {
                "PyTorch FP32": lambda: model(x32),
                "TensorRT FP32": lambda: trt32.infer(x32, out32),
                "TensorRT FP16": lambda: trt16.infer(x16, out16),
            }
            for impl, fn in runs.items():
                results[impl][str(batch)] = summarize(time_gpu(fn, args.warmup, args.iters), batch)

            trt32.infer(x32, out32)
            trt16.infer(x16, out16)
            torch.cuda.synchronize()
            for impl, out in (("TensorRT FP32", out32), ("TensorRT FP16", out16)):
                out = out.float()
                accuracy[impl][str(batch)] = {
                    "max_abs_logit_diff": (out - ref).abs().max().item(),
                    "top1_agreement": (out.argmax(1) == ref.argmax(1)).float().mean().item(),
                }

        base = results["PyTorch FP32"][str(batch)]["mean_ms"]
        for impl in ("TensorRT FP32", "TensorRT FP16"):
            r = results[impl][str(batch)]
            r["speedup"] = base / r["mean_ms"]
        print(f"batch {batch}: " + ", ".join(
            f"{impl} {results[impl][str(batch)]['mean_ms']:.3f} ms" for impl in results))

    stamp = datetime.datetime.now().astimezone()
    report = {
        "run_id": stamp.strftime("%Y%m%d-%H%M%S"),
        "time": stamp.isoformat(timespec="seconds"),
        "environment": environment(),
        "settings": {"batches": args.batches, "warmup": args.warmup, "iters": args.iters,
                     "seed": args.seed, "input": "torch.randn on GPU",
                     "timing": "CUDA events around each run; input already on GPU"},
        "latency": results,
        "accuracy_vs_pytorch_fp32": accuracy,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"benchmark-{report['run_id']}.json"
    json_path.write_text(json.dumps(report, indent=2))

    table = markdown_table(results, args.batches)
    acc_lines = ["| Engine | Batch | Max abs logit diff | Top-1 agreement |", "|---|---|---|---|"]
    for impl, rows in accuracy.items():
        for b in args.batches:
            a = rows[str(b)]
            acc_lines.append(f"| {impl} | {b} | {a['max_abs_logit_diff']:.4g} | {a['top1_agreement']:.2%} |")
    md = f"{table}\n\nSpeedup = PyTorch FP32 time / TensorRT time.\n\n" + "\n".join(acc_lines) + "\n"
    (args.out / f"benchmark-{report['run_id']}.md").write_text(md)
    print()
    print(md)
    print(f"saved {json_path}")


if __name__ == "__main__":
    main()
