import numpy as np

# Parameters
NUM_BITS = 8                      # Quantization bit-width
BLOCK_SHAPE = (2, 2)              # Size of each quantization block
np.random.seed(42)                # For reproducibility

def symmetric_blockwise_quantization(tensor):
    """
    Perform symmetric blockwise quantization on input tensor.
    Returns: (quantized tensor, scales, original shape)
    """
    # 1. Split tensor into blocks
    blocks, orig_shape = split_blocks(tensor, BLOCK_SHAPE)

    # 2. Quantize each block
    quantized_blocks = []
    scales = []
    max_val = 2**(NUM_BITS-1) - 1  # 127 for 8-bit

    for block in blocks:
        # Find maximum absolute value in block
        abs_max = np.max(np.abs(block))

        # Handle zero block case
        if abs_max == 0:
            scale = 1.0
        else:
            scale = abs_max / max_val  # Calculate scale factor

        # Quantize block
        quantized_block = np.round(block / scale).astype(np.int8)

        scales.append(scale)
        quantized_blocks.append(quantized_block)

    return quantized_blocks, scales, orig_shape

def symmetric_blockwise_dequantization(quantized_blocks, scales, orig_shape):
    """
    Reconstruct original tensor from quantized blocks and scales.
    """
    # Dequantize each block
    dequantized_blocks = []
    for q_block, scale in zip(quantized_blocks, scales):
        deq_block = q_block.astype(np.float32) * scale
        dequantized_blocks.append(deq_block)

    # Reconstruct original shape
    return assemble_blocks(dequantized_blocks, orig_shape, BLOCK_SHAPE)

def split_blocks(tensor, block_shape):
    """Split tensor into blocks of specified shape"""
    orig_shape = tensor.shape
    blocks = []

    # For 2D tensor (expand for higher dimensions)
    for i in range(0, tensor.shape[0], block_shape[0]):
        for j in range(0, tensor.shape[1], block_shape[1]):
            block = tensor[i:i+block_shape[0], j:j+block_shape[1]]
            blocks.append(block)

    return blocks, orig_shape

def assemble_blocks(blocks, orig_shape, block_shape):
    """Reconstruct tensor from blocks"""
    reconstructed = np.zeros(orig_shape, dtype=np.float32)
    block_idx = 0

    # For 2D tensor (expand for higher dimensions)
    for i in range(0, orig_shape[0], block_shape[0]):
        for j in range(0, orig_shape[1], block_shape[1]):
            block_size = (min(block_shape[0], orig_shape[0]-i),
                         min(block_shape[1], orig_shape[1]-j))
            reconstructed[i:i+block_shape[0], j:j+block_shape[1]] = blocks[block_idx]
            block_idx += 1

    return reconstructed

# Create sample tensor
original = np.random.randn(8, 8).astype(np.float32)
print("Original Tensor (First 4x4 corner):\n", original[:4, :4], "\n")

# Perform quantization
quantized, scales, orig_shape = symmetric_blockwise_quantization(original)

# Perform dequantization
reconstructed = symmetric_blockwise_dequantization(quantized, scales, orig_shape)

# Calculate error metrics
mse = np.mean((original - reconstructed)**2)
psnr = 10 * np.log10(np.max(original)**2 / mse)

print("Reconstructed Tensor (First 4x4 corner):\n", reconstructed[:4, :4])
print("\nQuantization Metrics:")
print(f"- MSE: {mse:.6f}")
print(f"- PSNR: {psnr:.2f} dB")
print(f"- Compression Ratio: {original.nbytes / (len(quantized)*quantized[0].nbytes + len(scales)*4):.1f}x")

# Show block-wise scales
print("\nFirst 4 Block Scales:", [f"{s:.4f}" for s in scales[:4]])
