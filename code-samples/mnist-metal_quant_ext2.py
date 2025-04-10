# mnist.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import copy
import time
# import numpy as np # No longer needed for quantization

# --- Import Metal Extension ---
try:
    from metal_quant_ext2 import blockwise_quant, dequantize
    print("Successfully imported metal_quant_ext2 extension.")
except ImportError as e:
    print(f"Error importing metal_quant_ext2: {e}")
    print("Please ensure the extension is compiled and installed correctly.")
    exit()

# --- Device Setup ---
try:
    mps_device = torch.device("mps:0")
    x = torch.randn(1, device=mps_device) # Test MPS device
    print(f"MPS device detected and working: {mps_device}")
except Exception as e:
    print(f"Failed to initialize or use MPS device: {e}")
    print("Ensure Metal drivers and PyTorch MPS support are correctly installed.")
    exit()

cpu_device = torch.device("cpu")
BLOCK_SIZE = 256 # Define BLOCK_SIZE used by Metal kernels and Python helpers
NUM_BITS = 8 # Still relevant conceptually for symmetric quantization range

# --- Quantization/Dequantization Helpers (Using Metal Extension) ---

def quantize_tensor_metal(fp32_tensor_mps):
    """
    Applies Metal blockwise quantization using the metal_quant_ext2 extension.
    Input tensor MUST be on the MPS device and contiguous.
    Returns quantized tensor (Int8, MPS), scales (FP32, CPU), offsets (FP32, CPU - always zero).
    """
    if fp32_tensor_mps.device != mps_device:
        raise ValueError(f"Input tensor must be on MPS device ({mps_device}), found {fp32_tensor_mps.device}.")
    if fp32_tensor_mps.dtype != torch.float32:
        raise ValueError("Input tensor must be float32.")

    # Ensure input tensor is contiguous
    fp32_tensor_mps = fp32_tensor_mps.contiguous()

    # Calculate number of blocks based on original tensor size
    numel = fp32_tensor_mps.numel()
    if numel == 0:
         # Handle empty tensors if necessary
         print("Warning: Attempting to quantize an empty tensor.")
         quantized_mps = torch.empty_like(fp32_tensor_mps, dtype=torch.int8)
         scales_cpu = torch.empty((0,), dtype=torch.float32, device=cpu_device)
         offsets_cpu = torch.empty((0,), dtype=torch.float32, device=cpu_device)
         return quantized_mps, scales_cpu, offsets_cpu

    num_blocks = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Allocate output tensors
    # Quantized tensor on MPS (same shape as input)
    quantized_mps = torch.empty_like(fp32_tensor_mps, dtype=torch.int8, device=mps_device)
    # Scales and Offsets on CPU (as populated by the extension)
    scales_cpu = torch.empty(num_blocks, dtype=torch.float32, device=cpu_device)
    offsets_cpu = torch.empty(num_blocks, dtype=torch.float32, device=cpu_device)

    # Call the Metal extension function
    # print(f"  Calling blockwise_quant for tensor size {fp32_tensor_mps.shape} ({numel} elements, {num_blocks} blocks)")
    try:
        blockwise_quant(fp32_tensor_mps, quantized_mps, scales_cpu, offsets_cpu)
    except Exception as e:
        print(f"!!! Metal blockwise_quant execution failed: {e}")
        raise # Re-raise the exception

    # print(f"  blockwise_quant finished. Quantized shape: {quantized_mps.shape}, Scales shape: {scales_cpu.shape}")

    # Offsets should be zero for symmetric quantization, add a check if desired
    # assert torch.allclose(offsets_cpu, torch.zeros_like(offsets_cpu), atol=1e-5), "Offsets are not zero!"

    return quantized_mps, scales_cpu, offsets_cpu


def dequantize_tensor_metal(quantized_tensor_mps, scales_cpu, offsets_cpu, original_dtype=torch.float32):
    """
    Dequantizes using Metal blockwise dequantization via metal_quant_ext2 extension.
    Input quantized tensor MUST be on MPS device and contiguous.
    Scales must be provided on CPU (will be moved to MPS internally).
    Ignores offsets_cpu as dequantization kernel is symmetric.
    """
    if quantized_tensor_mps.device != mps_device:
        raise ValueError(f"Quantized tensor must be on MPS device ({mps_device}), found {quantized_tensor_mps.device}.")
    if scales_cpu.device != cpu_device:
        raise ValueError(f"Scales must be provided on CPU device ({cpu_device}), found {scales_cpu.device}.")
    # if offsets_cpu.device != cpu_device: # Check offsets device if needed
    #     raise ValueError("Offsets must be provided on CPU device.")

    # Ensure input tensor is contiguous
    quantized_tensor_mps = quantized_tensor_mps.contiguous()

    # Move scales to MPS device and ensure contiguity for the kernel
    scales_mps = scales_cpu.to(mps_device).contiguous()

    # Calculate number of blocks needed for potential check (optional)
    numel = quantized_tensor_mps.numel()
    if numel == 0:
         print("Warning: Attempting to dequantize an empty tensor.")
         return torch.empty_like(quantized_tensor_mps, dtype=original_dtype)

    num_blocks_expected = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    if scales_mps.numel() != num_blocks_expected:
        raise ValueError(f"Number of scales ({scales_mps.numel()}) does not match expected blocks ({num_blocks_expected}) for tensor size {numel}")

    # Allocate output tensor on MPS (same shape as quantized input)
    output_mps = torch.empty_like(quantized_tensor_mps, dtype=original_dtype, device=mps_device)

    # Call the Metal extension function
    # print(f"  Calling dequantize for tensor size {quantized_tensor_mps.shape} ({numel} elements)")
    try:
        dequantize(quantized_tensor_mps, scales_mps, output_mps)
    except Exception as e:
        print(f"!!! Metal dequantize execution failed: {e}")
        raise # Re-raise the exception

    # print(f"  dequantize finished. Output shape: {output_mps.shape}")

    return output_mps


# --- MNIST Model Definition ---
# (Net class definition remains the same)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # Added padding
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)# Added padding
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Fully connected layers - adjust input features based on conv/pool output
        # Input: 28x28 -> Conv1(pad 1) -> 28x28 -> Pool -> 14x14
        # -> Conv2(pad 1) -> 14x14 -> Pool -> 7x7
        # Flattened size = 64 channels * 7 * 7 = 3136
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # Output: N x 32 x 14 x 14

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # Output: N x 64 x 7 x 7

        x = self.dropout1(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch: N x (64*7*7) = N x 3136
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# --- Training and Evaluation Functions ---
# (train and test functions remain the same)
def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, description="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\n{description} set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.2f}%)\n')
    return test_loss, accuracy

# --- Size Calculation (Adjusted for Metal Quantization) ---
def calculate_model_size_metal(state_dict, quantized_state, block_size=256):
    original_size = 0
    quantized_size = 0

    for name, param in state_dict.items():
        param_bytes = param.numel() * param.element_size()
        original_size += param_bytes

        if name in quantized_state and 'quantized' in quantized_state[name]:
            # Quantized weights: int8 (1 byte) + scales (FP32, 4 bytes per block)
            q_weights = quantized_state[name]['quantized'] # Int8 tensor
            scales = quantized_state[name]['scales']      # FP32 tensor (on CPU)

            # Calculate num_blocks based on ORIGINAL parameter size, same as quantization step
            numel_original = param.numel() # Use original numel
            if numel_original == 0:
                 num_blocks = 0
            else:
                 num_blocks = (numel_original + block_size - 1) // block_size

            # Verify scales tensor size matches expected blocks (optional but good)
            if scales.numel() != num_blocks:
                 print(f"Warning: Size mismatch in calculate_model_size for '{name}'. Expected {num_blocks} scales, found {scales.numel()}. Using scales.numel().")
                 num_blocks = scales.numel() # Use actual scale count for size calculation

            quantized_size += q_weights.numel() * 1  # int8 weights (size matches original)
            quantized_size += num_blocks * 4         # FP32 scales (one per block)
            # Offsets are not stored/used significantly for symmetric quantization size
        else:
            # Non-quantized parameters (e.g., biases)
            quantized_size += param_bytes

    return original_size, quantized_size


# --- Main Execution ---
if __name__ == '__main__':
    # (Hyperparameters, DataLoaders setup remains the same)
    batch_size = 64
    test_batch_size = 1000
    epochs = 5
    lr = 0.01 # Changed from 1.0 to 0.01 for Adam
    seed = 1
    model_filename = "mnist_cnn_fp32.pt"
    force_retrain = False

    torch.manual_seed(seed)

    # Use MPS backend if available for data loading
    # loader_kwargs = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': False} # Adjust workers as needed
    # test_loader_kwargs = {'batch_size': test_batch_size, 'num_workers': 0, 'pin_memory': False}
    # if mps_device == torch.device("mps:0"):
    #      loader_kwargs['pin_memory'] = True # Might help on MPS? Needs testing.
    #      test_loader_kwargs['pin_memory'] = True

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST('../data', train=False, transform=transform)
    # Use default loader settings for now
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)


    # --- Load or Train FP32 Model ---
    model_fp32 = Net().to(mps_device) # Create model and move to MPS
    trained_model_available = False
    if os.path.exists(model_filename) and not force_retrain:
        print(f"--- Loading Pre-trained FP32 Model from {model_filename} ---")
        try:
            # Load state dict to the device the model is currently on (MPS)
            state_dict = torch.load(model_filename, map_location=mps_device)
            model_fp32.load_state_dict(state_dict)
            # model_fp32.to(mps_device) # Model is already on MPS
            print(f"Model loaded successfully onto {mps_device}.")
            trained_model_available = True
        except Exception as e:
            print(f"Error loading model: {e}. Will proceed to retrain.")
            # model_fp32 = Net().to(mps_device) # Already created above
    else:
        if not force_retrain: print(f"--- Model file '{model_filename}' not found. Training new model. ---")
        else: print(f"--- Retraining forced. Training new model. ---")
        # model_fp32.to(mps_device) # Already on MPS

    if not trained_model_available:
        print("--- Training FP32 Model ---")
        model_fp32.train()
        # Optimizer: Adam is often a good default
        optimizer = optim.Adam(model_fp32.parameters(), lr=lr)
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            train(model_fp32, mps_device, train_loader, optimizer, epoch)
            test(model_fp32, mps_device, test_loader, description=f"FP32 Epoch {epoch}")
        end_time = time.time()
        print(f"FP32 training took: {end_time - start_time:.2f} seconds")
        try:
            # Save state dict (usually saved from CPU)
            torch.save(model_fp32.state_dict(), model_filename)
            print(f"Saved trained FP32 model state_dict to {model_filename}.")
        except Exception as e: print(f"Error saving model '{model_filename}': {e}")

    # --- Evaluate Final FP32 Model ---
    print("\n--- Evaluating Final FP32 Model (Loaded or Newly Trained) ---")
    model_fp32.eval()
    fp32_loss, fp32_acc = test(model_fp32, mps_device, test_loader, description="Final FP32")

    # --- Quantize Model Weights (Using Metal Extension) ---
    print("\n--- Quantizing Model Weights (Using Metal Extension) ---")
    quantized_state = {}
    model_fp32.eval()

    quant_start_time = time.time()
    with torch.no_grad():
        for name, param in model_fp32.named_parameters():
            # Only quantize layers with more than 1 dimension (typically weights)
            if param.dim() > 1 and param.numel() > 0:
                print(f"Quantizing layer: {name} with shape {param.shape} and {param.numel()} elements")
                param_mps = param.detach().to(mps_device) # Ensure it's on MPS

                try:
                    # Call the NEW quantize_tensor_metal function
                    q_param_mps, scales_cpu, offsets_cpu = quantize_tensor_metal(param_mps)

                    # Store results: quantized tensor on MPS, scales/offsets on CPU
                    quantized_state[name] = {
                        'quantized': q_param_mps,    # INT8, MPS
                        'scales': scales_cpu,       # FP32, CPU
                        'offsets': offsets_cpu,     # FP32, CPU (should be ~0)
                        'original_shape': param.shape,
                        'original_dtype': param.dtype
                    }
                    print(f"  Finished quantizing {name}. Quantized shape: {q_param_mps.shape}, Scales shape: {scales_cpu.shape}")

                except Exception as e:
                     print(f"  - ERROR during Metal quantize_tensor_metal call for {name}: {e}")
                     # Decide how to handle errors, e.g., skip quantization for this layer
                     # quantized_state[name] = {'original': param.detach().clone()} # Fallback?
                     continue # Skipping this parameter on error

            elif param.numel() > 0: # Handle non-empty 1D params (biases)
                print(f"Skipping quantization for {name} (bias or 1D parameter)")
                # Store original bias on the target device (MPS)
                quantized_state[name] = {'original': param.detach().clone().to(mps_device)}
            else:
                 print(f"Skipping empty parameter: {name}")


    quant_end_time = time.time()
    print(f"Total quantization time: {quant_end_time - quant_start_time:.2f} seconds")

    # --- Evaluating Quantized Model (using Dequantization - Metal Extension) ---
    print("\n--- Evaluating Quantized Model (using Dequantization - Metal Extension) ---")
    model_quant_eval = Net().to(mps_device) # Create eval model on MPS
    model_quant_eval.eval()
    start_time_dq = time.time()
    with torch.no_grad():
        state_dict_to_load = {}
        for name, stored_data in quantized_state.items():
            target_device = mps_device # All params should end up on MPS
            if 'quantized' in stored_data:
                print(f"Dequantizing layer: {name}")
                try:
                    # Call the NEW dequantize_tensor_metal function
                    # Inputs: quantized (MPS), scales (CPU), offsets (CPU)
                    # Output: dequantized (MPS)
                    dequantized_param_mps = dequantize_tensor_metal(
                        stored_data['quantized'],  # INT8, MPS
                        stored_data['scales'],     # FP32, CPU
                        stored_data['offsets'],    # FP32, CPU (ignored by symmetric kernel)
                        original_dtype=stored_data['original_dtype']
                    )

                    # Dequantized param is returned on MPS
                    if dequantized_param_mps.shape != stored_data['original_shape']:
                         print(f"Warning: Shape mismatch after dequantization for {name}. Expected {stored_data['original_shape']}, got {dequantized_param_mps.shape}")
                         # Attempt reshape if necessary (shouldn't happen if quant/dequant is correct)
                         dequantized_param_mps = dequantized_param_mps.reshape(stored_data['original_shape'])

                    state_dict_to_load[name] = dequantized_param_mps # Already on target_device (MPS)
                    print(f"  Finished dequantizing {name}. Final shape: {state_dict_to_load[name].shape}, device: {state_dict_to_load[name].device}")
                except Exception as e:
                    print(f"  - ERROR during Metal dequantize_tensor_metal call for {name}: {e}")
                    # Handle error, e.g., skip loading this parameter
                    continue
            elif 'original' in stored_data:
                 # Biases or skipped params are already stored on MPS
                 state_dict_to_load[name] = stored_data['original'] # Already on target_device (MPS)
            else:
                 print(f"Skipping loading empty parameter state: {name}")


        try:
            # Load the state dict containing dequantized weights (on MPS) and original biases (on MPS)
            model_quant_eval.load_state_dict(state_dict_to_load, strict=True)
            print("Successfully loaded dequantized state dict into evaluation model.")
        except Exception as e:
            print(f"Error loading state dict into evaluation model: {e}")
            # Debugging: Print keys mismatch if needed
            # print("Keys in state_dict_to_load:", sorted(state_dict_to_load.keys()))
            # print("Keys expected by model:", sorted(model_quant_eval.state_dict().keys()))


    quant_loss, quant_acc = test(model_quant_eval, mps_device, test_loader, description="Quantized (Dequantized Weights - Metal)")
    end_time_dq = time.time()
    print(f"Quantized evaluation (including dequantization) took: {end_time_dq - start_time_dq:.2f} seconds")

    # Calculate sizes using the adjusted function
    original_size, quantized_size = calculate_model_size_metal(
        model_fp32.state_dict(), quantized_state, BLOCK_SIZE
    )

    # --- Final Comparison ---
    print("\n--- Comparison ---")
    print(f"FP32 Model: Loss = {fp32_loss:.4f}, Accuracy = {fp32_acc:.2f}%")
    print(f"Quantized Model (Metal): Loss = {quant_loss:.4f}, Accuracy = {quant_acc:.2f}%")
    print(f"FP32 Model Size: {original_size / 1024:.2f} KB")
    print(f"Quantized Model Size (Metal): {quantized_size / 1024:.2f} KB")
    # Avoid division by zero if quantized size is zero
    if quantized_size > 0:
        compression_ratio = original_size / quantized_size
        print(f"Compression Ratio: {compression_ratio:.2f}x")
    else:
        print("Compression Ratio: N/A (Quantized size is zero)")
    print("-" * 20)
    print(f"Accuracy drop: {fp32_acc - quant_acc:.2f}%")
    print(f"Loss increase: {quant_loss - fp32_loss:.4f}")