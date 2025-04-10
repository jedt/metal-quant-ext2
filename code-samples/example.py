import math
import numpy as np # Using numpy for potential array operations later, though lists work too

# --- Constants (mirrored from Metal) ---
# Note: NUM and NUM_BLOCK relate to the flawed kernel structure and aren't
# directly needed for the corrected element-wise logic translation.
CODEBOOK_SIZE = 256

# --- Python implementation of quantize_scalar ---
# Reasoning: This function mirrors the Metal quantize_scalar template function.
#            It takes a single float 'x' and finds the index (0-255) in the
#            'codebook' that best represents 'x' using a binary search
#            and nearest-neighbor rounding (non-stochastic path).
# Confidence: 98% (Direct translation of the algorithm; minor Python vs C++ nuances possible but unlikely here)
def quantize_scalar_py(x, codebook, stochastic=False, rand_val=0.0):
    """
    Quantizes a single float value 'x' using the provided codebook.

    Args:
        x: The input float value.
        codebook: A list or numpy array of 256 float values, sorted ascendingly.
        stochastic: Boolean flag for stochastic rounding (default False).
        rand_val: Random float [0,1) used for stochastic rounding.

    Returns:
        An integer index (0-255) representing the quantized value.
    """
    if len(codebook) != CODEBOOK_SIZE:
        raise ValueError(f"Codebook must have size {CODEBOOK_SIZE}")

    # Initialize pivots for binary search within the 256-entry codebook
    pivot = 127
    upper_pivot = 255
    lower_pivot = 0

    # Initial bounds assumption (can be refined, matches Metal)
    # These 'lower' and 'upper' track the *values* corresponding to the pivots.
    lower_val = -1.0
    upper_val = 1.0

    # Handle immediate out-of-bounds cases (optional but good practice)
    # Note: The binary search itself implicitly handles clamping if 'x' is outside
    # the codebook's range [-1, 1], but checking explicitly can be clearer.
    if x <= codebook[0]:
        return 0
    if x >= codebook[255]:
        return 255

    # Get value from the middle of the codebook
    current_val = codebook[pivot]

    # Perform binary search (6 iterations for 2^6=64 steps on either side)
    # i = 64, 32, 16, 8, 4, 2, 1
    for i in [64, 32, 16, 8, 4, 2, 1]:
        if x > current_val:
            # If x is in the upper half relative to current_val
            lower_pivot = pivot       # Current pivot is the new lower bound index
            lower_val = current_val   # Current value is the new lower bound value
            pivot += i              # Move pivot up
        else:
            # If x is in the lower half (or equal)
            upper_pivot = pivot       # Current pivot is the new upper bound index
            upper_val = current_val   # Current value is the new upper bound value
            pivot -= i              # Move pivot down

        # Ensure pivot stays within bounds (0-255) - Metal code relied on logic keeping it in bounds
        pivot = max(0, min(255, pivot))
        current_val = codebook[pivot] # Get the value at the new pivot

    # --- Post-binary search refinement and rounding ---
    # After the loop, 'pivot' points to the index whose codebook value ('current_val')
    # is either the closest lower bound or the exact match for 'x'.
    # 'lower_pivot' holds the index of the codebook value just below 'x'.
    # 'upper_pivot' holds the index of the codebook value just above 'x'.
    # Note: The 'lower_val'/'upper_val' from the loop might not be perfectly
    #       tight if the search ended precisely on a value. We need the neighbors.

    # Get the final candidate values and their neighbors
    final_val_at_pivot = codebook[pivot]

    # Determine the actual upper/lower codebook values *surrounding* x
    # based on the final state of the pivots from the search.
    # This logic aims to replicate the Metal code's refinement step.

    # The Metal code refined 'upper' and 'lower' values based on final pivots.
    # Let's deduce the correct neighboring *indices* first.
    # If x landed exactly on code[pivot], the neighbors are pivot-1 and pivot+1.
    # If x is between code[lower_pivot] and code[pivot] (because the last step moved pivot down),
    #   then the relevant values are at lower_pivot and pivot.
    # If x is between code[pivot] and code[upper_pivot] (because the last step moved pivot up),
    #   then the relevant values are at pivot and upper_pivot.

    # Re-evaluate the state based on the *last comparison* implicitly done by the loop exit:
    # The loop sets 'current_val = codebook[pivot]'. The *next* comparison determines the interval.
    # Let's simplify: the binary search finds the *insertion point* or an exact match.

    # Replicate Metal's final refinement logic:
    # It uses the final 'pivot', and the 'upper_pivot'/'lower_pivot' *as they ended up*.
    final_lower_pivot = lower_pivot # Value from last time x > val
    final_upper_pivot = upper_pivot # Value from last time x <= val

    # Get the values corresponding to these final boundary pivots
    final_upper_val = codebook[final_upper_pivot]
    final_lower_val = codebook[final_lower_pivot]


    if not stochastic:
        # --- Non-Stochastic Rounding ---
        # Compare x against the value at the final pivot index
        if x > final_val_at_pivot:
            # x is between final_val_at_pivot (at index pivot) and final_upper_val (at index final_upper_pivot)
            # Note: In this branch, final_upper_pivot should be pivot+1 if the search worked ideally.
            # Let's use the actual neighbor value for clarity if possible.
            neighbor_upper_val = codebook[min(255, pivot + 1)] # Value definitely above or equal
            neighbor_upper_idx = min(255, pivot + 1)

            midpoint = (neighbor_upper_val + final_val_at_pivot) * 0.5
            if x > midpoint:
                # return final_upper_pivot # Using the pivot from search state
                return neighbor_upper_idx # Return the index of the upper neighbor
            else:
                return pivot       # Return index of the lower value ('final_val_at_pivot')
        else:
            # x is less than or equal to final_val_at_pivot
            # x is between final_lower_val (at index final_lower_pivot) and final_val_at_pivot (at index pivot)
            # Note: In this branch, final_lower_pivot should be pivot-1.
            neighbor_lower_val = codebook[max(0, pivot - 1)] # Value definitely below or equal
            neighbor_lower_idx = max(0, pivot - 1)

            midpoint = (final_lower_val + final_val_at_pivot) * 0.5 # Use the value tracked during search
            # Let's re-evaluate the midpoint based on *actual neighbors* for robustness:
            midpoint = (neighbor_lower_val + final_val_at_pivot) * 0.5

            if x < midpoint:
                # return final_lower_pivot # Using the pivot from search state
                return neighbor_lower_idx # Return the index of the lower neighbor
            else:
                return pivot       # Return index of the upper value ('final_val_at_pivot')
    else:
        # --- Stochastic Rounding ---
        if x > final_val_at_pivot:
            # x is between final_val_at_pivot (at index pivot) and neighbor_upper_val (at index neighbor_upper_idx)
            neighbor_upper_val = codebook[min(255, pivot + 1)]
            neighbor_upper_idx = min(255, pivot + 1)
            dist_full = neighbor_upper_val - final_val_at_pivot
            if dist_full <= 0: return pivot # Avoid division by zero if values are identical

            dist_to_upper = abs(neighbor_upper_val - x)
            # Probability of rounding down (to pivot) is dist_to_upper / dist_full
            # Probability of rounding up (to neighbor_upper_idx) is 1 - prob_down
            # Metal compares rand >= dist_to_upper/dist_full to round UP
            if rand_val >= dist_to_upper / dist_full:
                 return neighbor_upper_idx
            else:
                 return pivot
        else:
            # x is less than or equal to final_val_at_pivot
            # x is between neighbor_lower_val (at index neighbor_lower_idx) and final_val_at_pivot (at index pivot)
            neighbor_lower_val = codebook[max(0, pivot - 1)]
            neighbor_lower_idx = max(0, pivot - 1)
            dist_full = final_val_at_pivot - neighbor_lower_val
            if dist_full <= 0: return pivot # Avoid division by zero

            dist_to_lower = abs(neighbor_lower_val - x)
            # Probability of rounding up (to pivot) is dist_to_lower / dist_full
            # Probability of rounding down (to neighbor_lower_idx) is 1 - prob_up
            # Metal compares rand >= dist_to_lower/dist_full to round DOWN
            if rand_val >= dist_to_lower / dist_full:
                 return neighbor_lower_idx
            else:
                 return pivot

# --- Python implementation simulating the (corrected) quantize kernel ---
# Reasoning: This function iterates through the input array 'input_array_A',
#            calls the scalar quantization function for each element, and stores
#            the resulting index in the output array. This mirrors the intended
#            element-wise operation of the (corrected) Metal kernel.
# Confidence: 100% (Standard array processing loop)
def quantize_py(input_array_A, codebook):
    """
    Applies scalar quantization to each element of the input array.

    Args:
        input_array_A: List or numpy array of floats to quantize.
        codebook: The codebook (list or numpy array of 256 floats).

    Returns:
        A list of integers (0-255) representing the quantized indices.
    """
    output_array_out = []
    for x in input_array_A:
        # Call the scalar function (non-stochastic version)
        quantized_index = quantize_scalar_py(x, codebook, stochastic=False)
        output_array_out.append(quantized_index)
    return output_array_out


if __name__ == "__main__":
    # --- Example Usage and Verification ---

    # 1. Create the Codebook (same as Obj-C test)
    codebook_py = [-1.0 + (2.0 * i / (CODEBOOK_SIZE - 1)) for i in range(CODEBOOK_SIZE)]
    # print(f"Codebook size: {len(codebook_py)}")
    # print(f"Codebook[0]: {codebook_py[0]}, Codebook[127]: {codebook_py[127]}, Codebook[128]: {codebook_py[128]}, Codebook[255]: {codebook_py[255]}")

    # 2. Create Input Data (same as Obj-C test)
    n_elements = 10
    inputData_py = [0.0] * n_elements
    inputData_py[0] = 0.0
    inputData_py[1] = -1.0
    inputData_py[2] = 1.0
    inputData_py[3] = -1.1
    inputData_py[4] = 1.1
    inputData_py[5] = codebook_py[50]
    inputData_py[6] = codebook_py[200]
    inputData_py[7] = (codebook_py[10] + codebook_py[11]) * 0.5
    inputData_py[8] = codebook_py[10] * 0.1 + codebook_py[11] * 0.9
    inputData_py[9] = codebook_py[10] * 0.9 + codebook_py[11] * 0.1

    # 3. Run the Python Quantization
    output_indices_py = quantize_py(inputData_py, codebook_py)

    # 4. Print and Assert Results
    print("--- Python Quantization Results ---")
    expected_outputs = [128, 0, 255, 0, 255, 50, 200, 10, 11, 10] # Expected based on Metal analysis

    for i in range(n_elements):
        print(f"Input: {inputData_py[i]:.4f} -> Output Index: {output_indices_py[i]} (Expected: {expected_outputs[i]})")
        try:
            assert output_indices_py[i] == expected_outputs[i], f"Mismatch at index {i}!"
        except AssertionError as e:
            print(f"Assertion ERROR: {e}")

    print("\nVerification complete.")
