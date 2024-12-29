#include <webgpu/webgpu.h>
#include <iostream>

int main() {
    // Create descriptor for WebGPU instance
    WGPUInstanceDescriptor desc = {};
    desc.nextInChain = nullptr;

    // Create the WebGPU instance
    WGPUInstance instance = wgpuCreateInstance(&desc);
    
    // Check if the instance was created successfully
    if (!instance) {
        std::cerr << "Failed to initialize WebGPU instance!" << std::endl;
        return 1;
    }

    std::cout << "WebGPU instance created successfully: " << instance << std::endl;

    // Release the WebGPU instance
    wgpuInstanceRelease(instance);
    std::cout << "WebGPU instance destroyed." << std::endl;

    return 0;
}
