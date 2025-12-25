import torch
import sys

try:
    from neuralop.models import TFNO
    from neuralop.layers.spectral_convolution import SpectralConv
    from torch.cuda.amp import custom_fwd
except ImportError as e:
    print(f"Error: {e}. Make sure neuraloperator is installed.")
    sys.exit(1)


def patch_and_spy_on_spectral_conv():
    """
    Patches SpectralConv to print the memory layout of its input.
    """
    original_forward = SpectralConv.forward

    # We use custom_fwd to mimic your BF16 setup, though strictly not needed just to check strides
    @custom_fwd(cast_inputs=torch.float32)
    def diagnostic_forward(self, x, *args, **kwargs):
        # --- THE SPY CODE ---
        if not hasattr(self, "has_printed_stride"):
            print(f"\n[DIAGNOSTIC] Layer: {self.__class__.__name__}")
            print(f"  > Input Shape: {x.shape}")
            print(f"  > Input Strides: {x.stride()}")

            # Check Contiguity
            is_contig = x.is_contiguous()
            print(f"  > Is Contiguous (Default)? {is_contig}")

            # Check Channels Last (often used for optimization)
            is_chan_last = x.is_contiguous(memory_format=torch.channels_last)
            print(f"  > Is Channels Last? {is_chan_last}")

            # Calculate if it's channel first contiguous
            expected_stride = 1
            calculated_strides = []
            for dim in reversed(x.shape):
                calculated_strides.append(expected_stride)
                expected_stride *= dim
            calculated_strides = tuple(reversed(calculated_strides))

            if x.stride() != calculated_strides:
                print(f"  > \033[91mNON-CONTIGUOUS DETECTED\033[0m")
                print(f"  > Expected Dense Strides: {calculated_strides}")
            else:
                print(f"  > Memory is Dense/Contiguous")

            self.has_printed_stride = True
        # --------------------
        return original_forward(self, x, *args, **kwargs)

    SpectralConv.forward = diagnostic_forward
    print(">>> Diagnostic Patch Applied to SpectralConv")


def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Running on {device}")

    # 1. Apply Patch
    patch_and_spy_on_spectral_conv()

    # 2. Setup 3D TFNO Model (Standard Config)
    model = TFNO(
        n_modes=(16, 16, 16),
        hidden_channels=32,
        in_channels=1,
        out_channels=1,
        factorization="tucker",
        rank=0.42,
    ).to(device)

    # 3. Create Dummy Data (Batch 1, 1 Channel, 64^3)
    # Note: Standard PyTorch tensors are created Contiguous (Batch, Channel, D, H, W)
    input_tensor = torch.randn(1, 1, 64, 64, 64, device=device)

    print(
        f">>> Input Tensor Created. Shape: {input_tensor.shape}, Stride: {input_tensor.stride()}"
    )
    print(">>> Starting Forward Pass...")

    # 4. Run Forward Pass
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        _ = model(input_tensor)

    print(">>> Test Complete.")


if __name__ == "__main__":
    run_test()
