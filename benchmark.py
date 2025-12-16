import torch
import torch.nn as nn
import argparse
import nvtx
import sys
import time

# Imports
try:
    from neuralop.models import FNO, TFNO, UNO
except ImportError as e:
    print(f"Import Error: {e}. Install with: pip install neuraloperator")
    sys.exit(1)


# Compute-Bound Baseline: Heavy CNN (U-Net style)
class HeavyCNN(nn.Module):
    def __init__(self, in_ch, out_ch, width, dim=2):
        super().__init__()
        # Select 2D or 3D convolution
        Conv = nn.Conv3d if dim == 3 else nn.Conv2d

        # We multiply width by 4 to create a "Fat" network that provides
        # high Arithmetic Intensity (lots of FLOPs per byte loaded)
        fat_width = width * 4

        self.enc1 = Conv(in_ch, fat_width, kernel_size=3, padding=1)
        self.enc2 = Conv(fat_width, fat_width, kernel_size=3, padding=1)
        # 1x1 conv to project back
        self.dec1 = Conv(fat_width, out_ch, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.enc1(x))
        # Run enc2 multiple times to simulate "Heavy" compute load
        for _ in range(3):
            x = self.act(self.enc2(x))
        x = self.dec1(x)
        return x


# Model Factory
def get_model(name, res, width, modes, in_ch=1, out_ch=1, dim=2):
    name = name.upper()

    # Configure modes tuple based on dimension
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

    elif name == "UNO":
        if dim == 3:
            raise NotImplementedError("UNO is not configured for 3D in this benchmark.")

        n_layers = 5
        return UNO(
            in_channels=in_ch,
            out_channels=out_ch,
            hidden_channels=width,
            projection_channels=width,
            n_layers=n_layers,
            uno_out_channels=[width] * n_layers,
            uno_n_modes=[n_modes] * n_layers,
            uno_scalings=[[1.0, 1.0]] * n_layers,
            channel_mlp_skip="linear",
        )
    else:
        raise ValueError(f"Unknown Model: {name}")


def run_benchmark(args):
    device = "cuda"
    if not torch.cuda.is_available():
        sys.exit(1)

    # Enable CuDNN benchmark for the CNN baseline
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Data Shapes
    if args.dim == 3:
        input_shape = (args.batch, 1, args.res, args.res, args.res)
    else:
        input_shape = (args.batch, 1, args.res, args.res)

    try:
        model = get_model(args.model, args.res, args.width, args.modes, dim=args.dim)
        model = model.to(device)
        model.eval()

        # OPTIONAL: Compile
        # In a paper, you should probably run this script twice: once with --compile, once without
        if args.compile:
            model = torch.compile(model)

        # Create two buffers to toggle between, forcing memory traffic
        x1 = torch.randn(*input_shape, device=device)
        x2 = torch.randn(*input_shape, device=device)

        # Warmup (Runs compiler if enabled)
        print(f"--- Warmup {args.model} ---", file=sys.stderr)
        with torch.no_grad():
            for _ in range(10):
                _ = model(x1)
                _ = model(x2)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        nvtx.push_range(f"{args.model}_{args.dim}D_R{args.res}")

        with torch.no_grad():
            start_event.record()
            for i in range(args.unroll):
                # Toggle input to prevent L2 Cache residency for small batches
                input_tensor = x1 if i % 2 == 0 else x2
                _ = model(input_tensor)
            end_event.record()

        torch.cuda.synchronize()
        nvtx.pop_range()

        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / args.unroll
        throughput = args.batch / (avg_latency_ms / 1000.0)

        # Calculate Arithmetic Intensity (approximate) or Bandwidth utilization?
        # For now, memory usage is good.
        max_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Add "Compile" status to CSV
        comp_str = "Yes" if args.compile else "No"
        print(
            f"RESULT,{args.model},{args.dim}D,{args.res},{args.batch},{avg_latency_ms:.4f},{throughput:.2f},{max_mem_mb:.2f},{comp_str}"
        )

    except Exception as e:
        # Improved error logging
        print(f"ERR,{args.model},{args.dim}D,{args.res},{args.batch},0,0,0,Error")
        print(f"FAILED: {args.model} Res{args.res} : {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--unroll", type=int, default=50)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")

    args = parser.parse_args()
    run_benchmark(args)
