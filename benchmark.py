import torch
import argparse
import nvtx
import sys
import time

# 1. Imports based on your API reference
try:
    from neuralop.models import FNO, TFNO, UNO, CODANO
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you have installed neuraloperator: pip install neuraloperator")
    sys.exit(1)


def get_model(name, res, width, modes, in_ch=1, out_ch=1):
    """
    Factory to instantiate models with correct API signatures.
    """
    name = name.upper()

    # --- FNO (Baseline) ---
    if name == "FNO":
        # n_modes is a tuple (modes_x, modes_y) for 2D
        return FNO(
            n_modes=(modes, modes),
            hidden_channels=width,
            in_channels=in_ch,
            out_channels=out_ch,
        )

    # --- TFNO (Tensorized FNO) ---
    elif name == "TFNO":
        return TFNO(
            n_modes=(modes, modes),
            hidden_channels=width,
            in_channels=in_ch,
            out_channels=out_ch,
            factorization="tucker",  # Standard default
            rank=0.42,
        )  # Common compression ratio

    # --- LocalNO (Local Fourier / Convolutional) ---
    #    elif name == 'LOCALNO':
    # LocalNO typically behaves like FNO but with local filter operations
    # If specific args differ, check: help(LocalNO)
    #        return LocalNO(n_modes=(modes, modes),
    #                       hidden_channels=width,
    #                       in_channels=in_ch,
    #                       out_channels=out_ch)

    # U-NO (U-Shaped)
    elif name == "UNO":
        return UNO(
            in_channels=in_ch,
            out_channels=out_ch,
            hidden_channels=width,
            projection_channels=width,
            n_layers=5,
            uno_out_channels=width,  # required in newer versions
            uno_n_modes=[(modes, modes)] * 5,  # Defines modes per layer
            uno_scalings=[[1.0, 1.0]] * 5,  # Defines scaling per layer
        )

    # CODANO (Attention-based)
    elif name == "CODANO":
        # CODANO is heavy. It might not use 'n_modes' but 'n_heads' or similar.
        # This is a best-guess init based on typical attention APIs.
        return CODANO(
            in_channels=in_ch,
            out_channels=out_ch,
            hidden_channels=width,
            n_modes=(modes, modes),
        )

    else:
        raise ValueError(f"Model {name} not implemented in factory.")


def run_benchmark(args):
    device = "cuda"
    if not torch.cuda.is_available():
        print("Error: CUDA not found.")
        sys.exit(1)

    # Setup Model
    try:
        model = get_model(args.model, args.res, args.width, args.modes)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error initializing {args.model}: {e}")
        sys.exit(1)

    # Setup Data (GPU Resident)
    # Shape: (Batch, Channels, Height, Width)
    input_shape = (args.batch, 1, args.res, args.res)
    x = torch.randn(*input_shape, device=device)

    # Warmup
    print(f"--- Warming up {args.model} ({args.res}x{args.res}, b={args.batch}) ---")
    with torch.no_grad():
        # Run enough warmup to settle JIT/Autotuner
        for _ in range(20):
            _ = model(x)
    torch.cuda.synchronize()

    # Timed Execution
    print("--- Starting Timed Run ---")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # NVTX Range for Nsight Systems
    nvtx.push_range(f"{args.model}_B{args.batch}_R{args.res}")

    with torch.no_grad():
        start_event.record()
        for i in range(args.unroll):
            y = model(x)
        end_event.record()

    torch.cuda.synchronize()
    nvtx.push_range()

    # 5. Metrics
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / args.unroll
    throughput = args.batch / (avg_latency_ms / 1000.0)
    max_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # CSV Format Output
    print(
        f"RESULT,{args.model},{args.res},{args.batch},{args.modes},{args.width},{avg_latency_ms:.4f},{throughput:.2f},{max_mem_mb:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--unroll", type=int, default=50)

    args = parser.parse_args()
    run_benchmark(args)
