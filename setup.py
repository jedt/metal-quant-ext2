from setuptools import setup
from torch.utils import cpp_extension

pytorch_compile_args = [
    "-std=c++17",
    "-arch", "arm64",
    "-mmacosx-version-min=12.0",
]
pytorch_link_args = [
    "-arch", "arm64",
    "-mmacosx-version-min=12.0",
]

framework_args = [
    '-framework', 'Foundation',
    '-framework', 'Metal',
    '-framework', 'Accelerate'
]

pytorch_link_args.extend(framework_args)

setup(
    name="metal_quant_ext2",
    version="0.0.1",
    description="A Custom Pytorch MPS extension using MLX",
    ext_modules=[
        cpp_extension.CppExtension(
            "metal_quant_ext2",
            ["CustomOP.mm"],
            extra_compile_args=pytorch_compile_args,
            extra_link_args=pytorch_link_args,
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)