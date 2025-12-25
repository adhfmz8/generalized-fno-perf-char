import torch
import torch.nn as nn
import argparse
import sys
import os
import torch.cuda.profiler as profiler
from contextlib import nullcontext
import time

# Imports
try:
    from neuralop.models import FNO, TFNO, UNO
    from neuralop.data.datasets import load_darcy_flow_small
except ImportError as e:
    print(f"Import Error: {e}. Install with: pip install neuraloperator")
    sys.exit(1)

# Try importing fvcore for FLOP counting
try:
    from fvcore.nn import FlopCountAnalysis

    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


def patch_spectral_conv_for_bf16():
    from neuralop.layers.spectral_convolution import SpectralConv
    from torch.cuda.amp import custom_fwd

    original_forward = SpectralConv.forward

    @custom_fwd(cast_inputs=torch.float32)
    def patched_forward(self, x, *args, **kwargs):
        # --- PROFILING DIAGNOSTIC ---
        # Print stride info only once per run to avoid spam
        if not hasattr(self, "has_printed_stride"):
            print(f"\n[DIAGNOSTIC] Input to SpectralConv:")
            print(f"  Shape: {x.shape}")
            print(f"  Strides: {x.stride()}")
            print(f"  Is Contiguous? {x.is_contiguous()}")
            print(
                f"  Is Channels Last? {x.is_contiguous(memory_format=torch.channels_last)}"
            )
            self.has_printed_stride = True
        # ---------------------------
        return original_forward(self, x, *args, **kwargs)

    SpectralConv.forward = patched_forward


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def register_nvtx_hooks(model):
    """
    Registers NVTX ranges for major layers to clean up Nsight visualization.
    Only targets FNO-relevant blocks to avoid timeline clutter.
    """

    def pre_hook(module, input):
        torch.cuda.nvtx.range_push(module.__class__.__name__)

    def post_hook(module, input, output):
        torch.cuda.nvtx.range_pop()

    # Apply only to specific layers of interest
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        # Instrument high-level blocks and heavy compute layers
        if any(
            x in class_name
            for x in ["SpectralConv", "MLP", "FNO", "Block", "SkipConnection"]
        ):
            module.register_forward_pre_hook(pre_hook)
            module.register_forward_hook(post_hook)

    print("NVTX hooks registered for granular profiling.")


# Compute-Bound Baseline: Heavy CNN
class HeavyCNN(nn.Module):
    def __init__(self, in_ch, out_ch, width, dim=2):
        super().__init__()
        Conv = nn.Conv3d if dim == 3 else nn.Conv2d
        fat_width = width * 4
        self.enc1 = Conv(in_ch, fat_width, kernel_size=3, padding=1)
        self.enc2 = Conv(fat_width, fat_width, kernel_size=3, padding=1)
        self.dec1 = Conv(fat_width, out_ch, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.enc1(x))
        for _ in range(3):
            x = self.act(self.enc2(x))
        x = self.dec1(x)
        return x


def get_model(name, res, width, modes, in_ch=1, out_ch=1, dim=2):
    name = name.upper()
    if dim == 2:
        n_modes = (modes, modes)
    elif dim == 3:
        n_modes = (modes, modes, modes)

    if name == "HEAVYCNN":
        return HeavyCNN(in_ch, out_ch, width, dim=dim)
    elif name == "FNO":
        return FNO(
            n_modes=n_modes,
            hidden_channels=width,
            in_channels=in_ch,
            out_channels=out_ch,
        )
    elif name == "TFNO":
        return TFNO(
            n_modes=n_modes,
            hidden_channels=width,
            in_channels=in_ch,
            out_channels=out_ch,
            factorization="tucker",
            rank=0.42,
        )
    else:
        raise ValueError(f"Unknown Model: {name}")


def get_data_batch(args, device):
    """
    Returns: (input_tensor, in_channels, out_channels)
    """
    if args.data == "synthetic":
        shape = (args.batch, 1, *([args.res] * args.dim))
        return torch.randn(*shape, device=device), 1, 1

    if args.data == "real":
        # Force synthetic for 3D to avoid IO bottlenecks during kernel profiling
        if args.dim == 3:
            shape = (args.batch, 1, args.res, args.res, args.res)
            return torch.randn(*shape, device=device), 1, 1

        data_root = os.environ.get("SCRATCH", ".") + "/neuralop_data"
        try:
            train_loader, _, _ = load_darcy_flow_small(
                n_train=args.batch,
                batch_size=args.batch,
                test_resolutions=[args.res],
                resolution=args.res,
                encode_input=False,
                encode_output=False,
                root_dir=data_root,
            )
            batch = next(iter(train_loader))
            x = batch["x"].to(device)
            if x.shape[-1] < 5 and x.ndim == 4:
                x = x.permute(0, 3, 1, 2)
            return x, x.shape[1], 1
        except Exception as e:
            print(f"Failed to load real data: {e}. Fallback to synthetic.")
            shape = (args.batch, 1, args.res, args.res)
            return torch.randn(*shape, device=device), 1, 1


def run_benchmark(args):
    device = "cuda"
    if not torch.cuda.is_available():
        sys.exit(1)

    if args.precision == "bf16":
        patch_spectral_conv_for_bf16()

    # --- Precision Setup ---
    # Default to High (TF32 allowed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.precision == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        amp_ctx = nullcontext()
    elif args.precision == "tf32":
        amp_ctx = nullcontext()
    elif args.precision == "bf16":
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        raise ValueError("Unknown precision")

    try:
        x1, in_ch, out_ch = get_data_batch(args, device)
        x2 = torch.randn_like(x1)

        model = get_model(
            args.model,
            args.res,
            args.width,
            args.modes,
            in_ch=in_ch,
            out_ch=out_ch,
            dim=args.dim,
        )
        model = model.to(device)
        model.eval()

        # --- Metrics: Parameters & FLOPs ---
        param_count_m = count_parameters(model) / 1e6

        flops_g = 0.0
        if HAS_FVCORE:
            try:
                # Calculate FLOPs on a single forward pass
                # Warning: fvcore may warn on FFTs, but gives a decent estimate for the rest
                with amp_ctx:
                    flop_analyzer = FlopCountAnalysis(model, x1)
                    # Ignore warnings for cleaner output
                    flop_analyzer.warn_unsupported_ops = False
                    flops_g = flop_analyzer.total() / 1e9
            except Exception as e:
                # Fallback if fvcore fails (common with complex numbers or custom ops)
                pass

        # --- NVTX Instrumentation ---
        # Apply hooks BEFORE compilation (compiled models usually inherit hooks,
        # but sometimes hooks break graph capture. If so, move this after).
        # We apply it here for granular Eager profiling.
        if not args.compile:
            register_nvtx_hooks(model)

        if args.compile:
            # max-autotune is more aggressive but might take longer to warm up
            # "default" is safer for initial diagnostics
            # Suppress errors to allow fallbacks if graph breaks occur
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="default")

        # --- Warmup ---
        with torch.no_grad(), amp_ctx:
            for _ in range(10):
                _ = model(x1)
        torch.cuda.synchronize()

        # --- Profiling ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Notify Nsight Systems that the "interesting" part is starting
        profiler.start()

        with torch.no_grad(), amp_ctx:
            start_event.record()
            for i in range(args.unroll):
                input_tensor = x1 if i % 2 == 0 else x2
                _ = model(input_tensor)
            end_event.record()

        profiler.stop()
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / args.unroll
        throughput = args.batch / (avg_latency_ms / 1000.0)
        max_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        comp_str = "Yes" if args.compile else "No"

        # Calculate TFLOPs/s utilization if FLOPs available
        # TFLOPs/s = (GFLOPs * batch) / (latency_sec * 1000)
        tflops_per_sec = 0.0
        if flops_g > 0:
            tflops_per_sec = (flops_g * args.batch) / (avg_latency_ms / 1000.0) / 1000.0

        print(
            f"RESULT,{args.model},{args.dim}D,{args.res},{args.batch},{args.precision},"
            f"{avg_latency_ms:.4f},{throughput:.2f},{max_mem_mb:.2f},{comp_str},"
            f"{args.data},{param_count_m:.2f},{flops_g:.2f},{tflops_per_sec:.2f}"
        )

    except Exception as e:
        print(
            f"FAILED: {args.model} Res{args.res} Prec{args.precision}: {e}",
            file=sys.stderr,
        )
        # Dummy CSV line for error handling
        print(
            f"ERR,{args.model},{args.dim}D,{args.res},{args.batch},{args.precision},0,0,0,{args.compile},Error,0,0,0"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--unroll", type=int, default=50)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--data", type=str, default="synthetic", choices=["synthetic", "real"]
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="tf32",
        choices=["fp32", "tf32", "bf16"],
        help="fp32=Strict Float32, tf32=TensorFloat32 (A100 Default), bf16=Mixed Precision",
    )

    args = parser.parse_args()
    run_benchmark(args)
