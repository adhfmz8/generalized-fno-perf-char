import torch
import torch.nn as nn
import argparse
import time
import nvtx  # pip install nvtx
import sys

# Try importing neuralop models
try:
    from neuralop.models import FNO, CNO, UNO
    # Note: TFNO is often accessed via FNO with implementation='factorized' in newer versions, 
    # or neuralop.models.TFNO depending on version. I need to test this.
except ImportError:
    print("Error: 'neuralop' library not installed. Install via: pip install neuraloperator")
    sys.exit(1)

def get_model(name, res, width, modes, in_ch=1, out_ch=1):
    """
    Factory function to initialize models based on the experiment matrix.
    Adjust parameters here to match library API versions if needed.
    """
    name = name.upper()
    
    if name == 'FNO':
        # Standard FNO2d
        return FNO(n_modes=(modes, modes), hidden_channels=width, 
                   in_channels=in_ch, out_channels=out_ch)
    
    elif name == 'CNO':
        # Convolutional Neural Operator (LocalNO replacement)
        return CNO(in_channels=in_ch, out_channels=out_ch, 
                   hidden_channels=width)
    
    elif name == 'UNO':
        return UNO(in_channels=in_ch, out_channels=out_ch, 
                   hidden_channels=width)
    
    else:
        raise ValueError(f"Model {name} not implemented in factory.")

def run_benchmark(args):
    device = 'cuda'
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This benchmark requires a GPU.")

    # Setup Model
    model = get_model(args.model, args.res, args.width, args.modes)
    model = model.to(device)
    model.eval()

    # Setup Data (GPU Resident)
    # Shape: (Batch, Channels, Height, Width)
    input_shape = (args.batch, 1, args.res, args.res)
    x = torch.randn(*input_shape, device=device)

    # Warmup
    print(f"--- Warming up {args.model} ({args.res}x{args.res}, b={args.batch}) ---")
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
    torch.cuda.synchronize()

    # Timed Execution
    print("--- Starting Timed Run ---")
    
    # CUDA Events for sub-millisecond precision
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # NVTX Range for Nsys visualization
    nvtx.range_push(f"{args.model}_B{args.batch}_R{args.res}")
    
    with torch.no_grad():
        start_event.record()
        for i in range(args.unroll):
            nvtx.range_push("forward") 
            y = model(x)
            nvtx.range_pop()
        end_event.record()

    torch.cuda.synchronize()
    nvtx.range_pop()

    # Calculate Metrics
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / args.unroll
    throughput = args.batch / (avg_latency_ms / 1000.0) # samples per second
    max_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Output
    # Columns: Model, Res, Batch, Modes, Width, Latency(ms), Throughput(samp/s), Mem(MB)
    print(f"RESULT,{args.model},{args.res},{args.batch},{args.modes},{args.width},{avg_latency_ms:.4f},{throughput:.2f},{max_mem_mb:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name (FNO, CNO, UNO)')
    parser.add_argument('--res', type=int, default=128, help='Resolution (NxN)')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--modes', type=int, default=16, help='Number of spectral modes')
    parser.add_argument('--width', type=int, default=64, help='Hidden channel width')
    parser.add_argument('--unroll', type=int, default=50, help='Iterations for timing averaging')
    
    args = parser.parse_args()
    run_benchmark(args)