# train_eval_quantized_mnist.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import copy
import time
import numpy as np # Import numpy

# --- NumPy Quant/Dequant Implementation (Flat) ---
NUM_BITS = 8
# BLOCK_SIZE is defined below near device setup

def symmetric_blockwise_quantization_flat(tensor_flat_padded_np, block_size):
    """
    Perform symmetric blockwise quantization on a padded flat numpy array.
    Returns: (quantized flat numpy array, scales list)
    """
    numel_padded = tensor_flat_padded_np.size
    if numel_padded % block_size != 0:
        raise ValueError(f"Input numpy array size ({numel_padded}) must be multiple of block_size ({block_size})")

    num_blocks = numel_padded // block_size
    quantized_blocks_list = []
    scales_list = []
    max_val = 2**(NUM_BITS - 1) - 1 # e.g., 127 for 8-bit

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block = tensor_flat_padded_np[start:end]

        abs_max = np.max(np.abs(block))
        # Handle all-zero block or very small values to avoid division by zero or large scales
        scale = abs_max / max_val if abs_max > 1e-9 else 1.0

        # Clamp quantized values to the valid range [-max_val, max_val] although rounding should mostly handle this
        quantized_block_float = np.round(block / scale)
        quantized_block = np.clip(quantized_block_float, -max_val, max_val).astype(np.int8)


        scales_list.append(scale)
        quantized_blocks_list.append(quantized_block)

    # Combine blocks back into a single flat array
    quantized_flat_np = np.concatenate(quantized_blocks_list)
    return quantized_flat_np, scales_list

def symmetric_blockwise_dequantization_flat(quantized_flat_padded_np, scales_list_or_array, block_size):
    """
    Reconstruct flat tensor from a padded flat quantized numpy array and scales.
    """
    numel_padded = quantized_flat_padded_np.size
    if numel_padded % block_size != 0:
        raise ValueError(f"Input quantized numpy array size ({numel_padded}) must be multiple of block_size ({block_size})")

    num_blocks = numel_padded // block_size
    if len(scales_list_or_array) != num_blocks:
         raise ValueError(f"Number of scales ({len(scales_list_or_array)}) must match number of blocks ({num_blocks})")

    dequantized_flat_np = np.empty(numel_padded, dtype=np.float32)

    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        q_block = quantized_flat_padded_np[start:end]
        scale = scales_list_or_array[i]

        # Dequantize: Convert int8 back to float32 for multiplication
        deq_block = q_block.astype(np.float32) * scale
        dequantized_flat_np[start:end] = deq_block

    return dequantized_flat_np

# --- Device Setup ---
try:
    mps_device = torch.device("mps:0")
    x = torch.randn(1, device=mps_device)
    print(f"MPS device detected and working")
except Exception as e:
    print(f"Failed to initialize or use MPS device: {e}")
    print("Ensure Metal drivers and PyTorch MPS support are correctly installed.")
    exit()

cpu_device = torch.device("cpu")
BLOCK_SIZE = 256 # Define BLOCK_SIZE used by numpy functions and PyTorch helpers


# --- Quantization/Dequantization Helpers (Using NumPy) ---

def quantize_tensor(fp32_tensor_mps):
    """
    Applies numpy symmetric blockwise quantization.
    Handles PyTorch <-> NumPy conversion and device transfers.
    Returns quantized tensor (Int8, MPS), scales (FP32, CPU), offsets (FP32, CPU - always zero).
    """
    if fp32_tensor_mps.device != mps_device:
        raise ValueError(f"Input tensor must be on MPS device ({mps_device}), found {fp32_tensor_mps.device}.")
    if fp32_tensor_mps.dtype != torch.float32:
        raise ValueError("Input tensor must be float32.")

    original_shape = fp32_tensor_mps.shape
    fp32_tensor_cpu = fp32_tensor_mps.cpu() # Move to CPU for numpy
    fp32_flat_np = fp32_tensor_cpu.numpy().flatten() # Flatten to 1D numpy array

    numel = fp32_flat_np.size
    padded_numel = ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padding_needed = padded_numel - numel

    # Pad in numpy
    if padding_needed > 0:
        # Use numpy padding. Ensure constant_values=0 for padding.
        fp32_flat_padded_np = np.pad(fp32_flat_np, (0, padding_needed), mode='constant', constant_values=0)
    else:
        fp32_flat_padded_np = fp32_flat_np

    # Call the numpy quantization function
    quantized_flat_np, scales_list = symmetric_blockwise_quantization_flat(fp32_flat_padded_np, BLOCK_SIZE)

    # Convert scales list to tensor
    scales_tensor_cpu = torch.tensor(scales_list, dtype=torch.float32, device=cpu_device)
    # Create zero offsets tensor (as quantization is symmetric)
    offsets_tensor_cpu = torch.zeros_like(scales_tensor_cpu)

    # Convert quantized numpy array back to torch tensor (on CPU first)
    quantized_flat_padded_torch = torch.from_numpy(quantized_flat_np)

    # Remove padding
    quantized_flat_torch = quantized_flat_padded_torch[:numel]

    # Reshape and move to MPS
    quantized_original_shape_torch = quantized_flat_torch.reshape(original_shape)
    quantized_mps = quantized_original_shape_torch.to(mps_device) # Move final result to MPS

    # Return MPS tensor for quantized data, CPU tensors for scales/offsets
    return quantized_mps, scales_tensor_cpu, offsets_tensor_cpu

def dequantize_tensor(quantized_tensor_mps, scales_cpu, offsets_cpu, original_dtype=torch.float32):
    """
    Dequantizes using numpy symmetric blockwise dequantization.
    Handles PyTorch <-> NumPy conversion and device transfers.
    Ignores offsets_cpu as dequantization is symmetric.
    """
    # Check input devices
    if quantized_tensor_mps.device != mps_device:
        raise ValueError(f"Quantized tensor must be on MPS device ({mps_device}), found {quantized_tensor_mps.device}.")
    if scales_cpu.device != cpu_device or offsets_cpu.device != cpu_device:
        raise ValueError("Scales and offsets must be on CPU device.")
    # Optional: Check if offsets are indeed zero if expected
    # if not torch.all(offsets_cpu == 0):
    #     print("Warning: Received non-zero offsets for symmetric dequantization.")

    original_shape = quantized_tensor_mps.shape
    quantized_tensor_cpu = quantized_tensor_mps.cpu() # Move to CPU for numpy
    quantized_flat_np = quantized_tensor_cpu.numpy().flatten() # Flatten to 1D numpy array
    scales_np = scales_cpu.numpy() # Convert scales to numpy array

    numel = quantized_flat_np.size
    padded_numel = ((numel + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padding_needed = padded_numel - numel

    # Pad the INT8 numpy array before dequantization
    if padding_needed > 0:
        quantized_flat_padded_np = np.pad(quantized_flat_np, (0, padding_needed), mode='constant', constant_values=0)
    else:
        quantized_flat_padded_np = quantized_flat_np

    # Call numpy dequantization function
    dequantized_flat_np = symmetric_blockwise_dequantization_flat(quantized_flat_padded_np, scales_np, BLOCK_SIZE)

    # Convert dequantized numpy array back to torch tensor (on CPU first)
    # Ensure dtype matches the original floating point type
    dequantized_flat_padded_torch = torch.from_numpy(dequantized_flat_np).to(original_dtype)

    # Remove padding
    dequantized_flat_torch = dequantized_flat_padded_torch[:numel]

    # Reshape and move to MPS
    dequantized_original_shape_torch = dequantized_flat_torch.reshape(original_shape)
    dequantized_mps = dequantized_original_shape_torch.to(mps_device) # Move final result to MPS

    return dequantized_mps


# --- MNIST Model Definition ---
# (Net class definition remains the same)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
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

# --- Size Comparison ---
def calculate_model_size(state_dict, quantized_state, block_size=256):
    original_size = 0
    quantized_size = 0

    for name, param in state_dict.items():
        param_bytes = param.numel() * param.element_size()
        original_size += param_bytes

        if name in quantized_state and 'quantized' in quantized_state[name]:
            # Quantized weights: int8 (1 byte) + scales (FP32, 4 bytes per block)
            q_weights = quantized_state[name]['quantized']
            scales = quantized_state[name]['scales']

            # Calculate padded blocks during quantization
            numel = param.numel()
            padded_numel = ((numel + block_size - 1) // block_size) * block_size
            num_blocks = padded_numel // block_size

            quantized_size += q_weights.numel() * 1  # int8 weights
            quantized_size += scales.numel() * 4     # FP32 scales
        else:
            # Non-quantized parameters (biases)
            quantized_size += param_bytes

    return original_size, quantized_size

# --- Main Execution ---
if __name__ == '__main__':
    # (Hyperparameters, DataLoaders setup remains the same)
    batch_size = 64
    test_batch_size = 1000
    epochs = 5
    lr = 0.01
    seed = 1
    model_filename = "mnist_cnn_fp32.pt"
    force_retrain = False

    torch.manual_seed(seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)


    # --- Load or Train FP32 Model ---
    # (Loading/Training logic remains the same)
    model_fp32 = Net()
    trained_model_available = False
    if os.path.exists(model_filename) and not force_retrain:
        print(f"--- Loading Pre-trained FP32 Model from {model_filename} ---")
        try:
            state_dict = torch.load(model_filename, map_location=mps_device)
            model_fp32.load_state_dict(state_dict)
            model_fp32.to(mps_device)
            print(f"Model loaded successfully onto {mps_device}.")
            trained_model_available = True
        except Exception as e:
            print(f"Error loading model: {e}. Will proceed to retrain.")
            model_fp32 = Net().to(mps_device)
    else:
        if not force_retrain: print(f"--- Model file '{model_filename}' not found. Training new model. ---")
        else: print(f"--- Retraining forced. Training new model. ---")
        model_fp32.to(mps_device)

    if not trained_model_available:
        print("--- Training FP32 Model ---")
        model_fp32.train()
        optimizer = optim.Adam(model_fp32.parameters(), lr=lr)
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            train(model_fp32, mps_device, train_loader, optimizer, epoch)
            test(model_fp32, mps_device, test_loader, description=f"FP32 Epoch {epoch}")
        end_time = time.time()
        print(f"FP32 training took: {end_time - start_time:.2f} seconds")
        try:
            torch.save(model_fp32.state_dict(), model_filename)
            print(f"Saved trained FP32 model state_dict to {model_filename}.")
        except Exception as e: print(f"Error saving model '{model_filename}': {e}")

    # --- Evaluate Final FP32 Model ---
    print("\n--- Evaluating Final FP32 Model (Loaded or Newly Trained) ---")
    model_fp32.eval()
    fp32_loss, fp32_acc = test(model_fp32, mps_device, test_loader, description="Final FP32")

    # --- Quantize Model Weights (Using NumPy Helpers) ---
    print("\n--- Quantizing Model Weights (Using NumPy) ---")
    quantized_state = {}
    model_fp32.eval()

    # Optional: Add back reconstruction check if needed
    # check_first_layer_reconstruction = True

    with torch.no_grad():
        for name, param in model_fp32.named_parameters():
            if param.dim() > 1:
                print(f"Quantizing layer: {name} with shape {param.shape} and {param.numel()} elements")
                param_mps = param.detach().to(mps_device) # Still start with MPS tensor

                try:
                    # Call the NEW quantize_tensor which uses NumPy internally
                    q_param_mps, scales_cpu, offsets_cpu = quantize_tensor(param_mps)

                    # Optional: Reconstruction check can still be done here using the new dequantize_tensor
                    # if check_first_layer_reconstruction:
                    #     print(f"  Performing reconstruction check for {name}...")
                    #     # ... (add reconstruction check logic from previous step if desired) ...
                    #     check_first_layer_reconstruction = False

                    # Store results (quantized tensor is back on MPS)
                    quantized_state[name] = {
                        'quantized': q_param_mps, 'scales': scales_cpu, 'offsets': offsets_cpu,
                        'original_shape': param.shape, 'original_dtype': param.dtype
                    }
                    print(f"  Finished quantizing {name}. Quantized shape: {q_param_mps.shape}, Scales shape: {scales_cpu.shape}")

                except Exception as e:
                     print(f"  - ERROR during NumPy quantize_tensor call for {name}: {e}")
                     # check_first_layer_reconstruction = False # Ensure check is disabled if quant fails
                     continue # Skipping this parameter on error

            else:
                print(f"Skipping quantization for {name} (bias or 1D parameter)")
                quantized_state[name] = {'original': param.detach().clone()}


    # --- Evaluating Quantized Model (using Dequantization - NumPy Helpers) ---
    print("\n--- Evaluating Quantized Model (using Dequantization - NumPy) ---")
    model_quant_eval = Net().to(mps_device)
    model_quant_eval.eval()
    start_time_dq = time.time()
    with torch.no_grad():
        state_dict_to_load = {}
        for name, stored_data in quantized_state.items():
            target_device = mps_device
            if 'quantized' in stored_data:
                print(f"Dequantizing layer: {name}")
                try:
                    # Call the NEW dequantize_tensor which uses NumPy internally
                    dequantized_param = dequantize_tensor(
                        stored_data['quantized'], stored_data['scales'], stored_data['offsets'],
                        original_dtype=stored_data['original_dtype']
                    )

                    # Dequantized param is returned on MPS
                    if dequantized_param.shape != stored_data['original_shape']:
                         print(f"Warning: Shape mismatch after dequantization for {name}. Expected {stored_data['original_shape']}, got {dequantized_param.shape}")
                         dequantized_param = dequantized_param.reshape(stored_data['original_shape'])
                    state_dict_to_load[name] = dequantized_param # Already on target_device (MPS)
                    print(f"  Finished dequantizing {name}. Final shape: {state_dict_to_load[name].shape}, device: {state_dict_to_load[name].device}")
                except Exception as e:
                    print(f"  - ERROR during NumPy dequantize_tensor call for {name}: {e}")
                    continue # Skip loading this parameter if dequantization fails
            elif 'original' in stored_data:
                 state_dict_to_load[name] = stored_data['original'].to(target_device)

        try:
            model_quant_eval.load_state_dict(state_dict_to_load, strict=True)
            print("Successfully loaded dequantized state dict into evaluation model.")
        except Exception as e:
            print(f"Error loading state dict into evaluation model: {e}")
            # Add debug info if needed

    quant_loss, quant_acc = test(model_quant_eval, mps_device, test_loader, description="Quantized (Dequantized Weights - NumPy)")
    end_time_dq = time.time()
    print(f"Quantized evaluation took: {end_time_dq - start_time_dq:.2f} seconds")

    # Calculate sizes
    original_size, quantized_size = calculate_model_size(
        model_fp32.state_dict(), quantized_state, BLOCK_SIZE
    )

    # --- Updated Comparison ---
    print("\n--- Comparison ---")
    print(f"FP32 Model: Loss = {fp32_loss:.4f}, Accuracy = {fp32_acc:.4f}%")
    print(f"Quantized Model (NumPy): Loss = {quant_loss:.4f}, Accuracy = {quant_acc:.4f}%")
    print(f"FP32 Model Size: {original_size / 1024:.4f} KB")
    print(f"Quantized Model Size: {quantized_size / 1024:.4f} KB")
    print(f"Compression Ratio: {original_size / quantized_size:.4f}x")
    print("-" * 20)
    print(f"Accuracy drop: {fp32_acc - quant_acc:.4f}%")
    print(f"Loss increase: {quant_loss - fp32_loss:.4f}")