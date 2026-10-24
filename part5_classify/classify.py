#!/usr/bin/env python3
"""Part 5: GPU preprocessing, FP16 classification, and pinned H2D timing."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from frame_io import FRAME_BYTES, compare_pixels, load_frames

ROOT = Path(__file__).resolve().parents[1]


def named_paths(values, option):
    paths = {}
    for value in values:
        label, separator, path = value.partition('=')
        if not separator or not label or not path or label in paths:
            raise ValueError(f'{option} requires a unique LABEL=PATH, got {value!r}')
        if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
            raise ValueError('labels may contain letters, digits, underscores, and hyphens only')
        paths[label] = Path(path)
    return paths


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'


class GPUClassifier:
    def __init__(self, engine_path):
        # Keep file validation and --help usable on machines without CUDA.
        import torch
        import torchvision
        import tensorrt
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GPU is unavailable; run on the shared T4 instance')
        sys.path.insert(0, str(ROOT / 'part2_tensorrt'))
        from trt_runner import TRTRunner
        self.torch = torch
        self.stream = torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            self.runner = TRTRunner(str(engine_path))
        if self.runner.input_dtype != torch.float16:
            raise ValueError(f'Part 5 requires an FP16 engine, got {self.runner.input_dtype}')
        self.weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.transform = self.weights.transforms()
        self.categories = self.weights.meta['categories']
        self.environment = {
            'gpu': torch.cuda.get_device_name(),
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'cuda_runtime': torch.version.cuda, 'tensorrt': tensorrt.__version__,
            'driver': command_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']),
            'git_commit': command_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
            'input_dtype': str(self.runner.input_dtype),
            'output_dtype': str(self.runner.output_dtype),
            'max_batch': self.runner.max_batch,
        }

    def pinned(self, pixels):
        torch = self.torch
        host = torch.empty(pixels.shape, dtype=torch.uint8, pin_memory=True)
        host.copy_(torch.from_numpy(pixels))
        return host

    def preprocess(self, raw):
        """All layout conversion, scaling, resizing, cropping, normalization on GPU."""
        torch = self.torch
        if not raw.is_cuda or raw.dtype != torch.uint8:
            raise ValueError('preprocessing requires a CUDA uint8 NHWC tensor')
        nchw_float = raw.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
        # The weight preset accepts float tensors in [0, 1], resizes to 256,
        # center-crops to 224, and applies its ImageNet mean and standard deviation.
        return self.transform(nchw_float).to(self.runner.input_dtype).contiguous()

    def check_preprocessing(self, pixels):
        """An untimed CPU reference checks the GPU transform before classification."""
        torch = self.torch
        with torch.inference_mode():
            reference = self.transform(torch.from_numpy(pixels[:1]).permute(0, 3, 1, 2).float() / 255.0)
            reference = reference.to(self.runner.input_dtype)
            host = self.pinned(pixels[:1])
            with torch.cuda.stream(self.stream):
                raw = host.to('cuda', non_blocking=True)
                result = self.preprocess(raw)
            self.stream.synchronize()
            actual = result.cpu()
        error = float((reference.float() - actual.float()).abs().max())
        if not torch.allclose(actual.float(), reference.float(), atol=0.003, rtol=0.001):
            raise RuntimeError(f'GPU preprocessing differs from CPU weight preset; max abs error={error}')
        return error

    def classify(self, pixels, batch_size):
        torch = self.torch
        host = self.pinned(pixels)
        outputs = []
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            for offset in range(0, len(pixels), batch_size):
                raw = host[offset:offset + batch_size].to('cuda', non_blocking=True)
                normalized = self.preprocess(raw)
                outputs.append(self.runner.infer(normalized))
            logits = torch.cat(outputs)
        self.stream.synchronize()
        result = logits.float().cpu().numpy()
        if result.shape != (len(pixels), 1000) or not np.isfinite(result).all():
            raise RuntimeError(f'invalid ResNet-18 output: shape={result.shape}, finite={np.isfinite(result).all()}')
        return result

    def benchmark_copy(self, pixels, batches, warmup, iterations):
        torch = self.torch
        results = []
        for batch in batches:
            host = self.pinned(pixels[:batch])
            with torch.cuda.stream(self.stream):
                destination = torch.empty(host.shape, dtype=torch.uint8, device='cuda')
                for _ in range(warmup):
                    destination.copy_(host, non_blocking=True)
            self.stream.synchronize()
            pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
            with torch.cuda.stream(self.stream):
                for start, end in pairs:
                    start.record(self.stream)
                    destination.copy_(host, non_blocking=True)
                    end.record(self.stream)
            self.stream.synchronize()
            samples = np.array([start.elapsed_time(end) for start, end in pairs], dtype=np.float64)
            if not torch.equal(destination.cpu(), host):
                raise RuntimeError('pinned H2D benchmark copy failed byte-for-byte verification')
            mean_ms = float(samples.mean())
            if mean_ms <= 0:
                raise RuntimeError('CUDA timer returned a nonpositive interval')
            results.append({
                'batch': batch, 'bytes': batch * FRAME_BYTES,
                'warmup': warmup, 'iterations': iterations,
                'mean_ms': mean_ms, 'median_ms': float(np.median(samples)),
                'p95_ms': float(np.percentile(samples, 95)),
                'mean_us_per_image': mean_ms * 1000 / batch,
                'effective_GB_per_s': batch * FRAME_BYTES / (mean_ms * 1e6),
                'samples_ms': samples.tolist(),
            })
        return results


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(out, report, predictions):
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    write_csv(out / 'predictions.csv', predictions)
    copies = [{k: v for k, v in row.items() if k != 'samples_ms'} for row in report['pinned_h2d']]
    write_csv(out / 'copy_timing.csv', copies)
    env = report['environment']
    lines = [
        '# Part 5: Classifying the Received Images', '',
        f"Measured at {report['measured_at_utc']} on {env['gpu']}.",
        f"Overall result: **{report['status']}**.", '',
        ('**Synthetic development inputs: this run validates Part 5 code only. '
         'It is not evidence of real PCAP/DPDK reassembly; integration with Parts 3 and 4 is pending.**'
         if report['synthetic_test'] else 'Input artifacts were provided for this run; their source paths and hashes are recorded.'), '',
        '## Method', '',
        'Raw RGB uint8 bytes are staged in pinned CPU memory and copied to the GPU. '
        'The GPU converts NHWC to NCHW float32 and scales to [0, 1]. '
        'The ResNet-18 weight preset then resizes to 256, center-crops to 224, '
        'and normalizes with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]. '
        'The result becomes contiguous FP16 and is passed to the Part 2 TRTRunner on the same CUDA stream.', '',
        'Original and received images use the same engine and preprocessing. '
        'Images are matched by frame ID. Byte equality is checked separately: '
        'equal predictions alone do not prove correct reassembly.', '',
        f"GPU preprocessing versus an untimed CPU reference: max absolute difference "
        f"{report['preprocessing_reference_max_abs_error']:.6g} after the FP16 cast.", '',
        '## Classification consistency', '',
        '| Input | Frames | Byte-identical | Same top-1 class | Maximum logit difference |',
        '|---|---:|---:|---:|---:|',
    ]
    for label, row in report['datasets'].items():
        lines.append(f"| {label} | {row['frames']} | {row['byte_equal_frames']}/{row['frames']} | "
                     f"{row['class_equal_frames']}/{row['frames']} | {row['max_abs_logit_difference']:.6g} |")
    lines += ['', 'Per-frame classes and differences are in `predictions.csv`. '
              'This measures original/received consistency, not ImageNet accuracy; no ground-truth labels are assumed.', '',
              '## Pinned CPU-to-GPU copy', '',
              '| Batch | Bytes | Mean (ms) | Median (ms) | P95 (ms) | Mean per image (us) | Effective GB/s |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['pinned_h2d']:
        lines.append(f"| {row['batch']} | {row['bytes']} | {row['mean_ms']:.6f} | {row['median_ms']:.6f} | "
                     f"{row['p95_ms']:.6f} | {row['mean_us_per_image']:.3f} | {row['effective_GB_per_s']:.3f} |")
    lines += ['',
              f"Each size uses {report['warmup']} warm-up copies and {report['iterations']} measured copies. "
              'CUDA events bracket only nonblocking uint8 H2D copies on a dedicated stream, '
              'using already-pinned source memory and a preallocated GPU destination. '
              'File loading, pinning/allocation, preprocessing and inference are outside the timed interval. '
              'Small-transfer timings can include gaps between host submissions; these are observed copy timings, '
              'not a measurement of peak PCIe bandwidth. Every timed destination is also checked byte for byte.', '',
              '## Interpretation', '',
              'In this DPDK pipeline the reconstructed image lives in CPU memory, so it must be copied to GPU memory '
              'before GPU preprocessing and inference. GPUNetIO can place received network data directly in GPU memory, '
              'removing this CPU-to-GPU image copy. It still needs image reassembly, preprocessing and inference. '
              'These measurements isolate the copy cost; they do not directly measure an end-to-end GPUNetIO speedup.', '',
              '## Reproduction', '', '```bash', report['command'], '```', '',
              f"Engine SHA-256: `{report['engine_sha256']}`", '',
              'Environment and all timing samples are retained in `results.json`.', '',
              'References: [ResNet-18 preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) '
              'and [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html).', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--received', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--original-ids', type=Path)
    parser.add_argument('--received-ids', action='append', default=[], metavar='LABEL=JSON')
    parser.add_argument('--expected-frames', type=int, default=16)
    parser.add_argument('--engine', type=Path, default=ROOT / 'models/resnet18_fp16.engine')
    parser.add_argument('--batch-size', type=int, default=8, help='classification batch size')
    parser.add_argument('--copy-batches', type=int, nargs='+', default=[1, 8, 16])
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--out', type=Path, help='new or empty result directory')
    parser.add_argument('--validate-only', action='store_true', help='check inputs only; does not complete GPU Part 5')
    args = parser.parse_args()
    if min(args.expected_frames, args.batch_size, args.warmup, args.iterations, *args.copy_batches) < 1:
        parser.error('counts and batch sizes must be positive')
    received_paths = named_paths(args.received, '--received')
    received_ids = named_paths(args.received_ids, '--received-ids')
    if set(received_ids) - set(received_paths):
        parser.error('--received-ids label does not name a --received input')
    if args.batch_size > args.expected_frames or max(args.copy_batches) > args.expected_frames:
        parser.error('batch sizes cannot exceed the number of real input frames; set --copy-batches explicitly for smaller datasets')
    original = load_frames(args.original, args.original_ids, args.expected_frames)
    if any(path.resolve() == args.original.resolve() for path in received_paths.values()):
        raise ValueError('original and received paths must identify separate artifacts from Parts 3 and 4')
    received = {name: load_frames(path, received_ids.get(name), args.expected_frames) for name, path in received_paths.items()}
    differences = {name: compare_pixels(original, frames) for name, frames in received.items()}
    validation = {name: {'frames': len(frames.ids), 'byte_equal_frames': int((differences[name] == 0).sum()),
                         'frame_ids': frames.ids.tolist(), 'different_bytes_per_frame': differences[name].tolist()}
                  for name, frames in received.items()}
    for label, row in validation.items():
        print(f"{label}: {row['byte_equal_frames']}/{row['frames']} frames byte-identical", flush=True)
    out = args.out or ROOT / 'results/part5' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'result directory is not empty: {out}; choose a new directory to preserve previous evidence')
    out.mkdir(parents=True, exist_ok=True)
    validation_evidence = {'datasets': validation, 'gpu_classification_run': False}
    (out / 'input_validation.json').write_text(json.dumps(validation_evidence, indent=2) + '\n', encoding='utf-8')
    byte_ok = all(np.all(diff == 0) for diff in differences.values())
    if args.validate_only:
        print(f'Input validation only; GPU classification and copy timing NOT RUN. Results: {out}', flush=True)
        return 0 if byte_ok else 1
    if not args.engine.is_file():
        raise FileNotFoundError(f'FP16 engine not found: {args.engine}; generate it on this GPU using Part 2')
    classifier = GPUClassifier(args.engine)
    if args.batch_size > classifier.runner.max_batch:
        raise ValueError(f'classification batch exceeds engine profile maximum {classifier.runner.max_batch}')
    preproc_error = classifier.check_preprocessing(original.pixels)
    original_logits = classifier.classify(original.pixels, args.batch_size)
    original_classes = original_logits.argmax(axis=1)
    datasets, predictions = {}, []
    for label, frames in received.items():
        logits = classifier.classify(frames.pixels, args.batch_size)
        classes = logits.argmax(axis=1)
        errors = np.max(np.abs(original_logits - logits), axis=1)
        equal = classes == original_classes
        datasets[label] = {**validation[label], 'source': frames.source,
                           'sha256_aligned_rgb_bytes': hashlib.sha256(frames.pixels.tobytes()).hexdigest(),
                           'class_equal_frames': int(equal.sum()), 'max_abs_logit_difference': float(errors.max())}
        print(f'{label}: {int(equal.sum())}/{len(equal)} top-1 classes match', flush=True)
        for index, frame_id in enumerate(original.ids):
            original_class, received_class = int(original_classes[index]), int(classes[index])
            predictions.append({'dataset': label, 'frame_id': int(frame_id),
                                'original_class_id': original_class, 'original_class': classifier.categories[original_class],
                                'received_class_id': received_class, 'received_class': classifier.categories[received_class],
                                'class_equal': bool(equal[index]), 'different_bytes': int(differences[label][index]),
                                'max_abs_logit_difference': float(errors[index])})
    copy_source_label = next(iter(received))
    copies = classifier.benchmark_copy(received[copy_source_label].pixels, sorted(set(args.copy_batches)), args.warmup, args.iterations)
    passed = byte_ok and all(row['class_equal_frames'] == row['frames'] for row in datasets.values())
    fixture_markers = [str(path.resolve().parent / 'FIXTURE_ONLY.json')
                       for path in [args.original, *received_paths.values()]
                       if (path.resolve().parent / 'FIXTURE_ONLY.json').exists()]
    synthetic_test = bool(fixture_markers)
    import shlex
    report = {
        'status': ('PASS_SYNTHETIC_ONLY' if synthetic_test else 'PASS') if passed else 'FAIL',
        'synthetic_test': synthetic_test, 'fixture_markers': sorted(set(fixture_markers)),
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'environment': classifier.environment, 'engine': str(args.engine.resolve()), 'engine_sha256': digest(args.engine),
        'source_sha256': {name: digest(Path(__file__).parent / name)
                          for name in ['classify.py', 'frame_io.py', 'run_part5.sh']},
        'part2_runner_sha256': digest(ROOT / 'part2_tensorrt/trt_runner.py'),
        'original': original.source, 'original_sha256_aligned_rgb_bytes': hashlib.sha256(original.pixels.tobytes()).hexdigest(),
        'frame_shape': [224, 224, 3], 'frame_dtype': 'uint8', 'classification_batch_size': args.batch_size,
        'preprocessing_reference_max_abs_error': preproc_error, 'datasets': datasets,
        'copy_source_label': copy_source_label, 'warmup': args.warmup, 'iterations': args.iterations,
        'pinned_h2d': copies, 'command': shlex.join(['python', 'part5_classify/classify.py', *sys.argv[1:]]),
    }
    write_report(out, report, predictions)
    validation_evidence['gpu_classification_run'] = True
    (out / 'input_validation.json').write_text(json.dumps(validation_evidence, indent=2) + '\n', encoding='utf-8')
    for row in copies:
        print(f"Pinned H2D batch {row['batch']}: {row['mean_ms']:.6f} ms, {row['effective_GB_per_s']:.3f} GB/s", flush=True)
    print(f"Part 5: {report['status']}. Results: {out}", flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as error:
        print(f'Part 5 error: {error}', file=sys.stderr)
        raise SystemExit(1)
