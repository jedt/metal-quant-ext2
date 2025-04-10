import numpy as np

def affine_quantize(data_fp32, target_dtype=np.uint8):
    """
    Performs affine quantization on a numpy array.

    Args:
        data_fp32: The input numpy array with float32 data.
        target_dtype: The target integer numpy dtype (e.g., np.uint8, np.int8).

    Returns:
        A tuple containing:
        - quantized_data: The numpy array with quantized integer values.
        - scale: The calculated scale factor.
        - zero_point: The calculated zero-point.
    """
    # 1. Determine target integer range
    q_info = np.iinfo(target_dtype)
    q_min, q_max = q_info.min, q_info.max
    print(f"Target range: [{q_min}, {q_max}] ({target_dtype.__name__})")

    # 2. Determine real data range
    real_min = np.min(data_fp32)
    real_max = np.max(data_fp32)
    # print(f"Real data range: [{real_min:.4f}, {real_max:.4f}]")

    # Handle edge case where min == max
    if real_min == real_max:
        # If all values are the same, scale is somewhat arbitrary (choose 1),
        # zero_point should map the value to something sensible (e.g., middle of range or 0 if possible)
        scale = 1.0
        # Try to map the single real value to the integer 0 if real_value is 0,
        # otherwise map it near the center of the quantized range.
        if real_min == 0.0:
             zero_point = q_min # Map real 0 to q_min if possible (esp. for uint8)
        else:
             zero_point = int((q_min + q_max) / 2) # Or map near center
        zero_point = np.clip(zero_point, q_min, q_max).astype(target_dtype) # Ensure it's in range
    else:
        # 3. Calculate Scale
        scale = (real_max - real_min) / (q_max - q_min)

        # 4. Calculate Zero-Point
        # Initial calculation based on mapping real_min to q_min
        zero_point_float = q_min - real_min / scale
        # Round to nearest integer
        zero_point_rounded = np.round(zero_point_float)
        # Clamp to the target integer range
        zero_point = np.clip(zero_point_rounded, q_min, q_max).astype(target_dtype)

    # 5. Quantize: Apply the formula, round, clamp, and cast
    quantized_float = np.round(data_fp32 / scale + zero_point)
    quantized_clamped = np.clip(quantized_float, q_min, q_max)
    quantized_data = quantized_clamped.astype(target_dtype)

    return quantized_data, scale, zero_point

def affine_dequantize(quantized_data, scale, zero_point):
    """
    Performs affine dequantization.

    Args:
        quantized_data: The numpy array with quantized integer values.
        scale: The scale factor used during quantization.
        zero_point: The zero-point used during quantization.

    Returns:
        dequantized_data: The numpy array with approximated float32 values.
    """
    # Convert integer data to float for calculation
    quantized_float = quantized_data.astype(np.float32)
    zero_point_float = float(zero_point) # Ensure zero_point is float for calculation

    dequantized_data = (quantized_float - zero_point_float) * scale
    return dequantized_data

if __name__ == "__main__":
    # --- Example Usage ---
    # Create some sample float data (e.g., representing activations after ReLU)
    data_fp32 = np.array([0.0, 0.5, 1.0, 1.5, 3.0, 5.0, 7.5, 10.0, 12.0, 15.5], dtype=np.float32)
    # data_fp32 = np.linspace(-5.0, 5.0, 11, dtype=np.float32) # Try symmetric data too
    print("Original float32 data:\n", data_fp32)
    print("-" * 30)

    # Quantize to uint8
    target_type = np.uint8
    quantized_data_uint8, scale, zero_point = affine_quantize(data_fp32, target_dtype=target_type)

    print(f"Quantizing to: {target_type.__name__}")
    print(f"Calculated Scale: {scale:.4f}")
    print(f"Calculated Zero-Point: {zero_point}")
    print("\nQuantized uint8 data:\n", quantized_data_uint8)
    print("-" * 30)

    # Dequantize back to float32
    dequantized_data = affine_dequantize(quantized_data_uint8, scale, zero_point)
    print("Dequantized float32 data (approximation):\n", dequantized_data)
    print("-" * 30)

    # Calculate Quantization Error
    error = np.abs(data_fp32 - dequantized_data)
    print("Quantization Error (Absolute Difference):\n", error)
    print(f"\nMean Absolute Error: {np.mean(error):.4f}")
