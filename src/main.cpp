#include <iostream>
#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>

bool initialize_instance(wgpu::Instance& instance) {
    // Create an instance descriptor
    wgpu::InstanceDescriptor instanceDescriptor = {}; // Default settings for now

    // Create the instance properly
    instance = wgpu::createInstance(instanceDescriptor);

    // Check if the instance was created successfully
    if (!instance) {
        std::cerr << "Failed to create WebGPU instance." << std::endl;
        return false;
    }

    std::cout << "WebGPU instance created successfully!" << std::endl;
    return true;
}

int main() {
    // Declare instance variable
    wgpu::Instance instance = nullptr;

    // Initialize the instance
    if (!initialize_instance(instance)) {
        return 1; // Initialization failed
    }

    wgpuInstanceRelease(instance);
    //wgpuAdapterRelease(adapter);
   // wgpuDeviceRelease(device);
    return 0; // Success
}