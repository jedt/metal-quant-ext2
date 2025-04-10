# test-blockwise-quant.py
import torch
from metal_quant_ext2 import blockwise_quant  # Replace with your actual module name
import os

os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "1"
# os.environ["METAL_XCODE_DEBUG"] = "1" # Optional: Keep if needed for Metal debugging

# Check MPS availability
if not getattr(torch,'has_mps', False):
    print("MPS not available. Skipping test.")
    exit() # Or raise an exception

mps_device = torch.device("mps")
cpu_device = torch.device("cpu") # Define cpu device for clarity

def test_blockwise_quant():
    BLOCK_SIZE = 256
    # Keep input and quantized on MPS as the kernel operates on them there
    input_tensor = torch.randn(1024, device=mps_device, dtype=torch.float32)
    quantized = torch.empty_like(input_tensor, dtype=torch.int8) # Will inherit device from input_tensor (MPS)

    # Create properly sized outputs for scales and offsets
    num_blocks = (input_tensor.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE

    # The C++ code copies results from Metal Shared Buffers to CPU pointers,
    # so these tensors should be on the CPU to receive the data correctly via data_ptr().
    scales = torch.empty(num_blocks, device=cpu_device, dtype=torch.float32)
    offsets = torch.empty(num_blocks, device=cpu_device, dtype=torch.float32)

    # Execute
    print("test_blockwise_quant() executing blockwise_quant()")
    # Pass the tensors (input/quantized on MPS, scales/offsets on CPU)
    blockwise_quant(input_tensor, quantized, scales, offsets)
    print("test_blockwise_quant() done")

    print(f"quantized: {quantized}")
    # Verification
    # Bring quantized tensor to CPU for comparison if needed, or compare on MPS
    assert torch.all(quantized.cpu() >= -127) and torch.all(quantized.cpu() <= 127)
    print("quantized tensors passed")

    # Small tolerance for floating point conversion/quantization effects might be needed for offset check
    assert torch.allclose(offsets, torch.zeros_like(offsets), atol=1e-5), f"Non-zero offsets detected: {offsets}"
    print("Offsets are zero (or close to zero)")
    assert torch.all(scales >= 0), f"Negative scales detected: {scales}" # Allow zero scale if absmax was 0
    print("Verification complete")

    # Reconstruction test (needs all tensors on the same device, move to CPU)
    input_cpu = input_tensor.cpu()
    quantized_cpu = quantized.cpu()
    reconstructed = torch.zeros_like(input_cpu)

    for i in range(num_blocks):
        start = i * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, input_cpu.numel())
        # Ensure slicing works correctly if tensors are multi-dimensional later
        block = quantized_cpu.flatten()[start:end].float()
        # Scales and offsets are already on CPU
        reconstructed.flatten()[start:end] = block * scales[i] + offsets[i]

    error = torch.abs(input_cpu - reconstructed)
    print(f"Max error: {error.max().item():.4f}, Mean error: {error.mean().item():.4f}")

if __name__ == "__main__":
    test_blockwise_quant()