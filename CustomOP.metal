#pragma once

const char *CUSTOM_KERNEL = R"(
#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 256       // Must match host code: Number of floats processed per block/threadgroup
#define THREADGROUP_SIZE 256 // Must match host code: Number of threads working together on a block

kernel void blockwise_quant(
														device const float* input [[buffer(0)]],     // Input array of floats
														// --- CHANGE: Use 'char' (signed 8-bit) instead of 'uchar' ---
														device char* quantized [[buffer(1)]],         // Output array for quantized bytes (int8_t)
														// --- END CHANGE ---
														device half* scales [[buffer(2)]],        // Output array for per-block scales (using half-precision float)
														device half* offsets [[buffer(3)]],       // Output array for per-block offsets (using half-precision float)
														constant uint& total_elements [[buffer(4)]], // Total number of elements in the input array
														uint tid [[thread_index_in_threadgroup]],    // Index of the current thread within its group (0 to THREADGROUP_SIZE-1)
														uint bid [[threadgroup_position_in_grid]] // Index of the current threadgroup/block (0 to numBlocks-1)
														)
{
	// Add bounds checking for thread indices
	if (tid >= THREADGROUP_SIZE) return;  // Prevent invalid thread access
	if (bid >= (total_elements + BLOCK_SIZE - 1)/BLOCK_SIZE) return;

	// Allocate shared memory accessible by all threads within the threadgroup
	// Used for efficient parallel reduction (finding max value)
	threadgroup float shared_data[BLOCK_SIZE];

	// Calculate the starting index for the current block
	const uint block_start = bid * BLOCK_SIZE;

	// Calculate the ending index for the current block (ensuring not to exceed total_elements)
	const uint global_end = min(block_start + BLOCK_SIZE, total_elements);

	// 1. Load data into threadgroup shared memory and compute absolute values
	uint load_pos = block_start + tid; // Global index this thread is responsible for initially
	float val = 0.0f;

	// Check if the calculated position is within the valid range of the input buffer
	if (load_pos < global_end) {
		val = input[load_pos]; // Read the original float value
	}

	// Store the absolute value in shared memory. Even if load_pos was out of bounds,
	// we store 0.0f, which won't affect the max reduction correctly.
	shared_data[tid] = fabs(val);
	// Synchronize threads within the group to ensure all data is loaded before reduction
	threadgroup_barrier(mem_flags::mem_threadgroup);

	// 2. Parallel reduction to find the maximum absolute value (absmax) in the block
	for (uint offset = THREADGROUP_SIZE / 2; offset > 0; offset >>= 1) {
		if (tid < offset) {
			shared_data[tid] = max(shared_data[tid], shared_data[tid + offset]);
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}
	const float absmax = shared_data[0];

	// 3. Calculate Scale & Offset for Symmetric Quantization
	// The scale maps the range [-absmax, +absmax] to the intermediate range [-127, +127].
	const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;

	// Store scale and offset (using only thread 0 to avoid redundant writes)
	if (tid == 0) {
		scales[bid] = half(scale);  // Store the calculated scale for this block
		offsets[bid] = half(0.0f); // Store zero offset for symmetric quantization
	}

	// Synchronize threads (good practice)
	threadgroup_barrier(mem_flags::mem_threadgroup);

	// 4. Perform Symmetric Quantization and Store Result
	if (load_pos < global_end) {
		// Divide the original value (with sign) by the scale.
		// This maps the input value to the approximate range [-127.0, 127.0].
		const float scaled = (scale > 0.0f) ? (val / scale) : 0.0f;

		// Clamp the scaled value to the target range [-127.0, 127.0].
		// Note: rounding might still produce -128 if scaled is e.g. -127.5
		const float clamped = clamp(scaled, -127.0f, 127.0f);

		// To this to prevent -128 values:
		quantized[load_pos] = char(round(clamped));
	}
}

kernel void blockwise_dequant(
    device const char* quantized [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    const uint block_start = bid * BLOCK_SIZE;
    const uint global_end = min(block_start + BLOCK_SIZE, total_elements);
    const uint index = block_start + tid;

    if (index >= global_end) return;

    const float scale = scales[bid];
    output[index] = float(quantized[index]) * scale;
}

)";