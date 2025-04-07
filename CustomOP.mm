#include <torch/extension.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "CustomOP.metal"


static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.storage().data() != nullptr, "Tensor storage data pointer is null.");
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
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
                setBuffer:getMTLBufferStorage(output)
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

            torch::mps::commit();
        });

        torch::mps::synchronize();
    }
    return output;
}

// Pybind11 Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_fill", &custom_fill_mps, "Custom Metal kernel to fill an MPS tensor (in-place)");
}
