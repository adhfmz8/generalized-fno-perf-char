import torch
import torch.nn as nn
import argparse
import nvtx
import sys
import os

# Imports
try:
    from neuralop.models import FNO, TFNO, UNO

    # Import Datasets
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
    elif name == "UNO":
        if dim == 3:
            raise NotImplementedError("UNO not configured for 3D")
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


def get_data_batch(args, device):
    """
    Returns: (input_tensor, in_channels, out_channels)
    """
    # 1. Synthetic (Default for clean benchmarking)
    if args.data == "synthetic":
        if args.dim == 3:
            shape = (args.batch, 1, args.res, args.res, args.res)
        else:
            shape = (args.batch, 1, args.res, args.res)
        return torch.randn(*shape, device=device), 1, 1

    # 2. Real Data (Darcy 2D)
    # Note: 3D real data is huge. We usually stick to synthetic for 3D profiling unless dataset is pre-downloaded.
    if args.data == "real":
        if args.dim == 3:
            print(
                "Warning: Real data requested for 3D. NeuralOp built-in loaders are mostly 2D. Reverting to Synthetic 3D."
            )
            shape = (args.batch, 1, args.res, args.res, args.res)
            return torch.randn(*shape, device=device), 1, 1

        # Load Darcy
        # We define a download path to avoid clogging the home directory on clusters
        data_root = os.environ.get("SCRATCH", ".") + "/neuralop_data"

        try:
            # load_darcy_flow_small automatically downloads if not present
            train_loader, _, _ = load_darcy_flow_small(
                n_train=args.batch,  # We only need one batch
                batch_size=args.batch,
                test_resolutions=[args.res],
                resolution=args.res,
                encode_input=False,
                encode_output=False,
                root_dir=data_root,
            )

            # Grab one batch
            batch = next(iter(train_loader))
            x = batch["x"].to(device)
            # NeuralOp datasets often come as (Batch, X, Y, Channels) or (Batch, Channels, X, Y)
            # FNO expects (Batch, Channels, X, Y)
            if x.shape[-1] < 5 and x.ndim == 4:  # Likely (B, X, Y, C)
                x = x.permute(0, 3, 1, 2)

            in_ch = x.shape[1]
            out_ch = 1  # Darcy output is pressure (1 channel)

            return x, in_ch, out_ch

        except Exception as e:
            print(f"Failed to load real data: {e}. Fallback to synthetic.")
            shape = (args.batch, 1, args.res, args.res)
            return torch.randn(*shape, device=device), 1, 1


def run_benchmark(args):
    device = "cuda"
    if not torch.cuda.is_available():
        sys.exit(1)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    try:
        # GET DATA FIRST to determine channels
        x1, in_ch, out_ch = get_data_batch(args, device)
        # Create a second buffer with same shape for cache thrashing simulation
        x2 = torch.randn_like(x1)

        # Instantiate Model
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
            model = torch.compile(model, mode="reduce-overhead")

        # Warmup
        print(
            f"--- Warmup {args.model} [In:{in_ch}, Out:{out_ch}, Res:{args.res}] ---",
            file=sys.stderr,
        )
        with torch.no_grad():
            for _ in range(10):
                _ = model(x1)
                _ = model(x2)
        torch.cuda.synchronize()

        # Profiling Block
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        nvtx.push_range(f"{args.model}_{args.dim}D_R{args.res}")
        with torch.no_grad():
            start_event.record()
            for i in range(args.unroll):
                input_tensor = x1 if i % 2 == 0 else x2
                _ = model(input_tensor)
            end_event.record()
        torch.cuda.synchronize()
        nvtx.pop_range()

        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency_ms = total_time_ms / args.unroll
        throughput = args.batch / (avg_latency_ms / 1000.0)
        max_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        comp_str = "Yes" if args.compile else "No"
        dtype_str = "Real" if args.data == "real" else "Synth"

        print(
            f"RESULT,{args.model},{args.dim}D,{args.res},{args.batch},{avg_latency_ms:.4f},{throughput:.2f},{max_mem_mb:.2f},{comp_str},{dtype_str}"
        )

    except Exception as e:
        print(f"ERR,{args.model},{args.dim}D,{args.res},{args.batch},0,0,0,Error,Error")
        print(f"FAILED: {args.model} Res{args.res} : {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


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
    # New Argument
    parser.add_argument(
        "--data",
        type=str,
        default="synthetic",
        choices=["synthetic", "real"],
        help="Use real Darcy data or random noise",
    )

    args = parser.parse_args()
    run_benchmark(args)
