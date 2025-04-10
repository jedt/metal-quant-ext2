# Import numpy for numerical operations
import numpy as np

# Example float data (non-negative for unsigned quantization)
data = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

# Define quantization parameters: 8-bit unsigned (0 to 255)
num_bits = 4
quant_min = 0  # Minimum quantized integer value
quant_max = 2 ** num_bits - 1  # Maximum quantized integer value (255 for 8 bits)

# Calculate min and max of input data to determine the range
float_min = np.min(data)
float_max = np.max(data)

# Handle case where all data points are the same (avoid division by zero)
if float_max == float_min:
    scale = 1.0
    zero_point = quant_min
else:
    # Compute the scale factor: (data range) / (quantized range)
    scale = (float_max - float_min) / (quant_max - quant_min)
    # Calculate initial zero point: maps float_min to quant_min
    initial_zero_point = quant_min - (float_min / scale)
    # Round to nearest integer (zero_point must be an integer)
    zero_point = np.round(initial_zero_point).astype(int)
    # Clamp zero_point to ensure it's within the quantized range
    zero_point = np.clip(zero_point, quant_min, quant_max)

# Quantize the data: convert float to integer using affine transformation
# 1. Scale data, add zero_point, round, clamp to valid range
quantized_data = np.round(data / scale + zero_point)
quantized_data = np.clip(quantized_data, quant_min, quant_max).astype(np.int32)

# Dequantize the data: convert back to float using inverse transformation
dequantized_data = (quantized_data - zero_point) * scale

# Print results to show the process and outcomes
print("Original data:", data)
print("Quantized data:", quantized_data)
print("Dequantized data:", dequantized_data)
print("Scale:", scale)
print("Zero point:", zero_point)
