"""Part 2.3: run a TensorRT engine on input that is already in GPU memory.

TRTRunner takes a CUDA tensor, hands its address to TensorRT, and returns the
logits as a CUDA tensor. Nothing is copied through CPU memory. Part 5 imports
this class to classify the rebuilt images with the FP16 engine:

    from trt_runner import TRTRunner
    torch.cuda.set_stream(torch.cuda.Stream())   # avoid the legacy default stream
    runner = TRTRunner("models/resnet18_fp16.engine")
    logits = runner.infer(batch)          # batch: CUDA tensor, runner.input_dtype
    torch.cuda.current_stream().synchronize()
"""
import tensorrt as trt
import torch

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}


class TRTRunner:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"cannot load {engine_path}; rebuild it on this GPU and TensorRT version")
        self.context = self.engine.create_execution_context()

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        mode = self.engine.get_tensor_mode
        self.input_name = next(n for n in names if mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(n for n in names if mode(n) == trt.TensorIOMode.OUTPUT)
        self.input_dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(self.input_name)]
        self.output_dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(self.output_name)]
        _, _, max_shape = self.engine.get_tensor_profile_shape(self.input_name, 0)
        self.max_batch = max_shape[0]

    def infer(self, inp, out=None, stream=None):
        """Queue inference on `stream` (default: the current PyTorch stream).

        The call only enqueues work. Reading `out` on the same stream is safe;
        reading it from the CPU needs a stream synchronize first.
        """
        if not inp.is_cuda or not inp.is_contiguous():
            raise ValueError("input must be a contiguous CUDA tensor")
        if inp.dtype != self.input_dtype:
            raise TypeError(f"engine expects {self.input_dtype}, got {inp.dtype}")
        if not 1 <= inp.shape[0] <= self.max_batch:
            raise ValueError(f"batch {inp.shape[0]} is outside the profile (1..{self.max_batch})")

        current = torch.cuda.current_stream(inp.device)
        if stream is None:
            stream = current
        elif stream != current:
            # The input was produced on the current stream; make sure it is
            # ready before TensorRT reads it on the other stream.
            stream.wait_stream(current)

        self.context.set_input_shape(self.input_name, tuple(inp.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if out is None:
            out = torch.empty(out_shape, dtype=self.output_dtype, device=inp.device)
        elif tuple(out.shape) != out_shape or out.dtype != self.output_dtype:
            raise ValueError(f"output must be {self.output_dtype} {out_shape}")

        self.context.set_tensor_address(self.input_name, inp.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("execute_async_v3 failed")
        return out
