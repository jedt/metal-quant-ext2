# metal-quant-ext2
metal-quant-ext2 is a repository of my research on PyTorch MPS kernel extensions using Apple Metal. The name includes quant which implies some quantization.

Goal is to develop mps pytorch extensions for efficient local fine-tuning using pytorch huggingface models and [TRL](https://huggingface.co/docs/trl/en/sft_trainer)

The study includes:
- The quanto code:
https://github.com/huggingface/optimum-quanto/blob/caca3cc930050e31f04bfca753b5a93c3aa462aa/optimum/quanto/library/extensions/mps/unpack.mm
- https://developer.apple.com/documentation/metal/metal-sample-code-library


# requirements
- MacOS 15.3.1 or later
- Python 3.12
- [pytorch](https://pytorch.org/get-started/locally/)


## Usage
``` bash
pip3 install -r requirements.txt
pip3 install --ignore-installed . && python test.py
```
