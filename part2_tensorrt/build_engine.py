"""Part 2.2: build a TensorRT engine from an ONNX file.

The engine gets one optimization profile that covers batch sizes 1 to 32.
TensorRT 11 networks are strongly typed, so the engine precision follows the
ONNX file: resnet18_fp16.onnx gives an FP16 engine. TF32 stays on by default,
so an FP32 engine on Ampere or newer may run some math in TF32.

Run on the GPU instance:
    python part2_tensorrt/build_engine.py models/resnet18_fp32.onnx models/resnet18_fp32.engine
    python part2_tensorrt/build_engine.py models/resnet18_fp16.onnx models/resnet18_fp16.engine
"""
import argparse
import pathlib
import sys
import time

import tensorrt as trt

TRT_MAJOR = int(trt.__version__.split(".")[0])


def create_network(builder):
    if TRT_MAJOR >= 11:
        return builder.create_network()
    # TensorRT 10 needs the flag to honour the ONNX types the same way.
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    return builder.create_network(flag)


def build(onnx_path, engine_path, min_batch, opt_batch, max_batch, workspace_gib):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = create_network(builder)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file finds the external .onnx.data file next to the model.
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise SystemExit(f"failed to parse {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)

    inp = network.get_input(0)
    _, c, h, w = inp.shape  # batch is -1 (dynamic)
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (min_batch, c, h, w), (opt_batch, c, h, w), (max_batch, c, h, w))
    config.add_optimization_profile(profile)

    print(f"TensorRT {trt.__version__}: {onnx_path} -> {engine_path}")
    print(f"  input {inp.name} {inp.dtype} batch min/opt/max = {min_batch}/{opt_batch}/{max_batch}")
    out = network.get_output(0)
    print(f"  output {out.name} {out.dtype}")

    start = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise SystemExit("engine build failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine)
    print(f"  built in {time.perf_counter() - start:.1f} s, {engine_path.stat().st_size / 2**20:.1f} MiB")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("onnx", type=pathlib.Path)
    p.add_argument("engine", type=pathlib.Path)
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--workspace-gib", type=int, default=2)
    args = p.parse_args()
    build(args.onnx, args.engine, args.min_batch, args.opt_batch, args.max_batch, args.workspace_gib)


if __name__ == "__main__":
    main()
