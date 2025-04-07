import torch
from metal_quant_ext2 import custom_fill

# check mps device
assert torch.backends.mps.is_available()

mps_device = torch.device("mps")

if __name__ == "__main__":
    # Create the tensor
    input_tensor = torch.zeros(42, device=mps_device, dtype=torch.float) # Explicitly set dtype=torch.float

    print("Tensor before custom_fill:")
    print(input_tensor)

    # Call the function for its IN-PLACE modification side effect.
    # Do NOT assign the return value, as it will be None.
    custom_fill(input_tensor, 42.0) # Use 42.0 to match float type

    print("\nTensor after custom_fill:")
    # Print the tensor variable itself, which has now been modified.
    print(input_tensor)

    # Verification (Optional)
    expected_tensor = torch.full((42,), 42.0, device=mps_device, dtype=torch.float)
    assert torch.equal(input_tensor, expected_tensor), "Tensor content is not as expected!"
    print("\nVerification successful!")
