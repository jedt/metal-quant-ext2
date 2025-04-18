# metal-quant-ext2
metal-quant-ext2 is a repository of my research on PyTorch MPS kernel extensions using Apple Metal. The name includes quant which implies some quantization.

Goal is to develop mps pytorch extensions for efficient local fine-tuning using pytorch huggingface models and [TRL](https://huggingface.co/docs/trl/en/sft_trainer)

The study includes:
- the bitsandbytes mps kernel https://github.com/bitsandbytes-foundation/bitsandbytes/blob/49609323b0ae4c8222927a628641e9814289c8de/csrc/mps_kernels.metal
- The quanto code:
https://github.com/huggingface/optimum-quanto/blob/caca3cc930050e31f04bfca753b5a93c3aa462aa/optimum/quanto/library/extensions/mps/unpack.mm
- https://developer.apple.com/documentation/metal/metal-sample-code-library


# Requirements
- MacOS 15.3.1 or later
- Python 3.12
- [pytorch](https://pytorch.org/get-started/locally/)


## Usage
```bash
pip3 install -r requirements.txt
pip3 install --ignore-installed .
```

## Blockwise Quantization 8-bit
`blockwise_quant` is a function that applies symmetric blockwise 8-bit quantization to a pytorch tensor

```python
from metal_quant_ext2 import blockwise_quant, dequantize
mps_device = torch.device("mps")

input_tensor = torch.randn(1024, device=mps_device, dtype=torch.float32)

quantized = torch.empty_like(input_tensor, dtype=torch.int8) # Will inherit device from input_tensor (MPS)

scales = torch.empty(num_blocks, device=cpu_device, dtype=torch.float32)

offsets = torch.empty(num_blocks, device=cpu_device, dtype=torch.float32)

# the actual MTL call
blockwise_quant(input_tensor, quantized, scales, offsets)

print(f"quantized: {quantized}")
assert torch.all(quantized.cpu() >= -127) and torch.al(quantized.cpu() <= 127)

# Dequantize MTL call
scales = scales.to(mps_device)
output = torch.empty_like(input_tensor)
dequantize(quantized, scales, output)
```

### Testing
Check out the test file with assertions
`test-blockwise-quant.py`

### Blockwise Quantization Example
Below is a python script that helped me understand blockwise quantization

`code-samples/blockwise-quantization.py`