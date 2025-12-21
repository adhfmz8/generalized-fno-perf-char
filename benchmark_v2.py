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

    # --- Precision Setup ---
    # Default to High (TF32 allowed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.precision == "fp32":
        # Strictly turn off TF32 for pure FP32 comparison
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        amp_ctx = nullcontext()
        dtype_val = torch.float32
    elif args.precision == "tf32":
        # TF32 is enabled by flags above, context remains FP32
        amp_ctx = nullcontext()
        dtype_val = torch.float32
    elif args.precision == "bf16":
        # Use Autocast for Mixed Precision
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        dtype_val = torch.bfloat16
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

        if args.compile:
            # Using 'default' as requested to avoid CUDA Graph/Complex errors
            model = torch.compile(model, mode="default")

        # --- Warmup ---
        with torch.no_grad(), amp_ctx:
            for _ in range(10):
                _ = model(x1)
        torch.cuda.synchronize()

        # --- Profiling ---
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Optional: Nsight Systems Profiler Hooks
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

        print(
            f"RESULT,{args.model},{args.dim}D,{args.res},{args.batch},{args.precision},"
            f"{avg_latency_ms:.4f},{throughput:.2f},{max_mem_mb:.2f},{comp_str},{args.data}"
        )

    except Exception as e:
        print(
            f"FAILED: {args.model} Res{args.res} Prec{args.precision}: {e}",
            file=sys.stderr,
        )
        # Print a dummy CSV line to allow analysis tools to see the failure
        print(
            f"ERR,{args.model},{args.dim}D,{args.res},{args.batch},{args.precision},0,0,0,{args.compile},Error"
        )
        # import traceback
        # traceback.print_exc()


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
