// CustomOP.mm - pytorch extension
// compiled with cpp_extension.CppExtension
// pytorch_compile_args = [
//     "-std=c++17",
//     "-ObjC++",
//     "-arch", "arm64",
//     "-mmacosx-version-min=12.0",
// ]
// pytorch_link_args = [
//     "-arch", "arm64",
//     "-mmacosx-version-min=12.0",
// ]

// framework_args = [
//     '-framework', 'Foundation',
//     '-framework', 'Metal',
//     '-framework', 'Accelerate',
//     '-framework', 'MetalPerformanceShaders',
//     '-framework', 'MetalPerformanceShadersGraph'
// ]

#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "CustomOP.metal"

// Define constants used by both Host and Device code if not defined elsewhere
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256 // Size of data chunks processed by one threadgroup
#endif

#ifndef THREADGROUP_SIZE
#define THREADGROUP_SIZE 256 // Number of threads within a threadgroup
#endif

// Helper function
float floatFromHalf(uint16_t half) {
	typedef __attribute__((__ext_vector_type__(1))) unsigned short ushort1;
	ushort1 v = { half };
	float f;
	__fp16 h;
	memcpy(&h, &v, sizeof(h));
	f = (float)h;
	return f;
}

static id<MTLBuffer> getMPSBuffer(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.device().is_mps(), "Tensor not on MPS device");
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static void dequantize(
    torch::Tensor& quantized,
    torch::Tensor& scales,
    torch::Tensor& output
) {
    // Type checks
    TORCH_CHECK(quantized.scalar_type() == torch::kInt8,
        "Quantized tensor must be int8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
        "Scales must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kFloat32,
        "Output must be float32");

    // Shape checks
    const int64_t totalElements = quantized.numel();
    TORCH_CHECK(output.numel() == totalElements,
        "Output tensor must match quantized tensor size");
    const int64_t numBlocks = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    TORCH_CHECK(scales.numel() == numBlocks,
        "Scales tensor size must match number of blocks");

    // Contiguity checks
    TORCH_CHECK(quantized.is_contiguous(),
        "Quantized tensor must be contiguous");
    TORCH_CHECK(output.is_contiguous(),
        "Output tensor must be contiguous");

    // Device checks
    TORCH_CHECK(quantized.device().is_mps(),
        "Quantized tensor must reside on MPS device");
    TORCH_CHECK(scales.device().is_mps(),
        "Scales tensor must reside on MPS device");
    TORCH_CHECK(output.device().is_mps(),
        "Output tensor must reside on MPS device");

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(device, "Failed to get system default Metal device");

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource: [NSString stringWithUTF8String: CUSTOM_KERNEL]
                    options:nil
                    error:&error
                ];
        TORCH_CHECK(library, "Failed to create Metal library: %s", error.localizedDescription.UTF8String);

        id<MTLFunction> dequantFunction = [library newFunctionWithName:@"blockwise_dequant"];
        TORCH_CHECK(dequantFunction, "Failed to find 'blockwise_dequant' Metal function");

        id<MTLComputePipelineState> pipelineState = [device
            newComputePipelineStateWithFunction:dequantFunction
            error:&error
        ];
        TORCH_CHECK(pipelineState, "Failed to create pipeline state: %s", error.localizedDescription.UTF8String);

        // Get MTL buffers
        id<MTLBuffer> quantizedBuffer = getMPSBuffer(quantized);
        id<MTLBuffer> scalesBuffer = getMPSBuffer(scales);
        id<MTLBuffer> outputBuffer = getMPSBuffer(output);

        // Total elements buffer
        uint totalElementsArg = static_cast<uint>(totalElements);
        id<MTLBuffer> totalElementsBuffer = [device
            newBufferWithBytes:&totalElementsArg
            length:sizeof(uint)
            options:MTLResourceStorageModeShared
        ];

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^(){
            id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(encoder, "Failed to create compute command encoder");

            [encoder setComputePipelineState:pipelineState];
            [encoder setBuffer:quantizedBuffer offset:0 atIndex:0];
            [encoder setBuffer:scalesBuffer offset:0 atIndex:1];
            [encoder setBuffer:outputBuffer offset:0 atIndex:2];
            [encoder setBuffer:totalElementsBuffer offset:0 atIndex:3];

            MTLSize gridSize = MTLSizeMake(numBlocks, 1, 1);
            MTLSize threadgroupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            torch::mps::commit();
        });

        torch::mps::synchronize();
    }
}

static void blockwise_quant(
        torch::Tensor& input,
        torch::Tensor& quantized,
        torch::Tensor& scales,
        torch::Tensor& offsets
    )
{
    NSLog(@"starting blockwise_quant");
    // Type checks
    TORCH_CHECK(input.scalar_type() == torch::kFloat,
        "Input must be float32 (MPS tensors use 32-bit floats)");
    TORCH_CHECK(quantized.scalar_type() == torch::kInt8,
        "Quantized tensor must be int8");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat,
        "Scales must be float32");
    TORCH_CHECK(offsets.scalar_type() == torch::kFloat,
        "Offsets must be float32");

    // Shape validation
    const int64_t totalElements = input.numel();
    const int64_t numBlocks = (totalElements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    NSLog(@"totalElements: %lld", totalElements);
    NSLog(@"numBlocks: %lld", numBlocks);

    // Contiguity checks
    // In blockwise_quant function:
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(quantized.scalar_type() == torch::kInt8, "Quantized tensor must be int8");
    TORCH_CHECK(scales.numel() == numBlocks, "Scales tensor size mismatch");

    // Device checks
    TORCH_CHECK(input.device().is_mps(),
        "Input tensor must reside on MPS device");
    TORCH_CHECK(quantized.device().is_mps(),
        "Quantized tensor must reside on MPS device");

    TORCH_CHECK(quantized.sizes() == input.sizes(),
        "Quantized tensor must match input shape");
    TORCH_CHECK(scales.numel() == numBlocks,
        "Scales tensor size mismatch");
    TORCH_CHECK(offsets.numel() == numBlocks,
        "Offsets tensor size mismatch");

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(device, "Failed to get system default metal device");
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource: [NSString stringWithUTF8String: CUSTOM_KERNEL]
                    options:nil
                    error:&error
                ];

        TORCH_CHECK(library, "Failed to create Metal library from source, error: ", error.localizedDescription.UTF8String);
        NSLog(@"library created, now creating function");

        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"blockwise_quant"];

        TORCH_CHECK(kernelFunction, "Failed to find 'custom_fill' Metal function in library.");

        NSLog(@"function created, now creating pipeline state");

        id<MTLComputePipelineState> pipelineState = [device
                                                        newComputePipelineStateWithFunction:kernelFunction
                                                        error:&error
                                                    ];

        TORCH_CHECK(pipelineState, "Failed to create compute pipeline state, error: ", error.localizedDescription.UTF8String);

        // Create input buffer
        id<MTLBuffer> inputBuffer = getMPSBuffer(input);

        // --- Create the output `quantized` buffers using sizeof(int8_t) ---
        id<MTLBuffer> quantizedBuffer = getMPSBuffer(quantized);

        // Scale/Offset buffers remain half (uint16_t) on the Metal side
        id<MTLBuffer> scalesBuffer = [device
                newBufferWithLength:numBlocks * sizeof(uint16_t)
                options:MTLResourceStorageModeShared];

        id<MTLBuffer> offsetsBuffer = [device
                newBufferWithLength:numBlocks * sizeof(uint16_t)
                options:MTLResourceStorageModeShared];

        // Total elements buffer
        uint totalElementsArg = (uint)totalElements;

        id<MTLBuffer> totalElementsBuffer = [device
                newBufferWithBytes:&totalElementsArg
                length:sizeof(uint)
                options:MTLResourceStorageModeShared];

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        uint data_size = quantized.numel();

        NSLog(@"data_size: %d", data_size);

        dispatch_sync(serialQueue, ^(){
            // Create command queue and buffer (remains the same)
            id<MTLCommandQueue> commandQueue = [device newCommandQueue];
            id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(encoder, "Failed to create Metal compute command encoder.");

            NSLog(@"encoder created, now setting pipeline state");

            // Configure the encoder
            [encoder setComputePipelineState:pipelineState];
            [encoder setBuffer:inputBuffer offset:0 atIndex:0];
            [encoder setBuffer:quantizedBuffer offset:0 atIndex:1]; // Pass the int8 buffer
            [encoder setBuffer:scalesBuffer offset:0 atIndex:2];
            [encoder setBuffer:offsetsBuffer offset:0 atIndex:3];
            [encoder setBuffer:totalElementsBuffer offset:0 atIndex:4];

            // Dispatch grid and threadgroups (remains the same)
            MTLSize gridSize = MTLSizeMake(numBlocks, 1, 1);
            // Ensure THREADGROUP_SIZE defined in header matches kernel
            MTLSize threadgroupSize = MTLSizeMake(THREADGROUP_SIZE, 1, 1);
            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
            NSLog(@"command buffer committed, now waiting for completion");
            // Submit the currently active command buffer to run on the MPS device.
            torch::mps::commit();

        });

        NSLog(@"waiting for command buffer to complete");
        torch::mps::synchronize();
        NSLog(@"command buffer completed");

        // --- NOW read results from Metal buffers and populate CPU tensors ---
        uint16_t *rawScales = (uint16_t *)scalesBuffer.contents;
        uint16_t *rawOffsets = (uint16_t *)offsetsBuffer.contents;

        // Get pointers to the CPU tensor data passed from Python
        float* scalesPtr = scales.data_ptr<float>();
        float* offsetsPtr = offsets.data_ptr<float>();

        TORCH_CHECK(scalesPtr != nullptr, "Scales tensor data pointer is null (is it on CPU?)");
        TORCH_CHECK(offsetsPtr != nullptr, "Offsets tensor data pointer is null (is it on CPU?)");

        for (int i = 0; i < numBlocks; ++i) {
            scalesPtr[i] = floatFromHalf(rawScales[i]);
            offsetsPtr[i] = floatFromHalf(rawOffsets[i]);
        }
        NSLog(@"CPU tensors populated successfully");
        // --- End reading results ---
    }

    return;
}

torch::Tensor& custom_fill_mps(torch::Tensor output, float fill_value) {
    TORCH_CHECK(output.device().is_mps(), "Output tensor must be on the MPS device.");
    TORCH_CHECK(output.is_contiguous(), "Output tensor must be contiguous.");
    TORCH_CHECK(output.scalar_type() == torch::kFloat, "Output tensor must be of type float.");


    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(device, "Failed to get system default metal device");

        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource: [NSString stringWithUTF8String: CUSTOM_KERNEL]
                    options:nil
                    error:&error
                ];

        TORCH_CHECK(library, "Failed to create Metal library from source, error: ", error.localizedDescription.UTF8String);

        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"custom_fill"];

        TORCH_CHECK(kernelFunction, "Failed to find 'custom_fill' Metal function in library.");

        id<MTLComputePipelineState> pipelineState = [device
                                                        newComputePipelineStateWithFunction:kernelFunction
                                                        error:&error
                                                    ];

        TORCH_CHECK(pipelineState, "Failed to create compute pipeline state, error: ", error.localizedDescription.UTF8String);

        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to get MPS command buffer from PyTorch.")

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        uint data_size = output.numel();

        dispatch_sync(serialQueue, ^(){
            // Create a compute command encoder to record compute commands into the command buffer.
            id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(commandEncoder, "Failed to create Metal compute command encoder."); // Confidence: 10/10 - Essential check.

            [commandEncoder setComputePipelineState:pipelineState];

            [commandEncoder
                setBuffer:getMPSBuffer(output)
                offset:output.storage_offset() * output.element_size()
                atIndex:0
            ];

            [commandEncoder
                setBytes:&fill_value
                length:sizeof(float)
                atIndex:1
            ];

             // Set the data_size as the kernel's third argument (index 2). Passed by reference via setBytes.
            [commandEncoder
                setBytes:&data_size
                length:sizeof(uint)
                atIndex:2
            ];

            // --- Thread Dispatch Calculation ---
            MTLSize gridSize = MTLSizeMake(data_size, 1, 1);

            // Determine the optimal thread group size. Start with the maximum allowed by the PSO.
            NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;

            if (threadGroupSize > data_size) {
                threadGroupSize = data_size;
            }

            // Define the thread group dimensions (1D in this case).
            MTLSize threadgroupDimensions = MTLSizeMake(threadGroupSize, 1, 1);

            [commandEncoder
                dispatchThreads:gridSize threadsPerThreadgroup:threadgroupDimensions
            ];

            [commandEncoder endEncoding];

            // Submit the currently active command buffer to run on the MPS device.
            torch::mps::commit();
        });

        torch::mps::synchronize();
    }
    return output;
}

// Pybind11 Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blockwise_quant", &blockwise_quant,
        py::arg("input"),
        py::arg("quantized"),
        py::arg("scales"),
        py::arg("offsets"));

    m.def("dequantize", &dequantize,
        py::arg("quantized"),
        py::arg("scales"),
        py::arg("output"));
}
