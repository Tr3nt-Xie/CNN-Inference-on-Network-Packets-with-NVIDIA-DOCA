"""Part 2.1: export torchvision ResNet-18 to ONNX in FP32 and FP16.

Both models get a dynamic batch dimension. Recent PyTorch uses the dynamo
exporter, which writes the weights to a separate <name>.onnx.data file; keep it
next to the .onnx file, because build_engine.py loads the model by path.

Run on the GPU instance:
    python part2_tensorrt/export_onnx.py --out-dir models
"""
import argparse
import pathlib

import onnx
import torch
import torchvision

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def export(precision, out_dir, opset):
    dtype = DTYPES[precision]
    # The FP16 export casts weights and input to half, so every tensor in the
    # graph is FP16 and TensorRT 11 (strongly typed) builds an FP16 engine.
    model = torchvision.models.resnet18(weights="DEFAULT").eval().cuda().to(dtype)
    x = torch.randn(1, 3, 224, 224, device="cuda", dtype=dtype)
    path = out_dir / f"resnet18_{precision}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model, (x,), str(path),
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
        )
    return path


def describe(path):
    # Checking by path lets the checker find the external .onnx.data file.
    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)
    for kind, values in (("input", model.graph.input), ("output", model.graph.output)):
        for v in values:
            t = v.type.tensor_type
            dims = [d.dim_param or d.dim_value for d in t.shape.dim]
            dtype = onnx.TensorProto.DataType.Name(t.elem_type)
            print(f"  {kind} {v.name}: {dtype} {dims}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("models"))
    p.add_argument("--precision", choices=["fp32", "fp16", "both"], default="both")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    precisions = ["fp32", "fp16"] if args.precision == "both" else [args.precision]
    for precision in precisions:
        path = export(precision, args.out_dir, args.opset)
        print(f"exported {path}")
        describe(path)


if __name__ == "__main__":
    main()
