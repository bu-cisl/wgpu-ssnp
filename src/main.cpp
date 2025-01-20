#include <iostream>
#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>

bool initialize_instance(wgpu::Instance& instance, wgpu::Adapter& adapter, wgpu::Device& device) {
    // Create an instance
    wgpu::InstanceDescriptor instanceDescriptor = {};
    instance = wgpu::createInstance(instanceDescriptor);
    if (!instance) {
        std::cerr << "Failed to create WebGPU instance." << std::endl;
        return false;
    }
    std::cout << "WebGPU instance created successfully!" << std::endl;

    // Request adapter
    wgpu::RequestAdapterOptions adapterOptions = {};
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;
    adapter = instance.requestAdapter(adapterOptions);
    if (!adapter) {
        std::cerr << "Failed to request a WebGPU adapter." << std::endl;
        return false;
    }
    std::cout << "WebGPU adapter requested successfully!" << std::endl;

    // Request device
    wgpu::DeviceDescriptor deviceDescriptor = {};
    deviceDescriptor.label = "Default Device";
    device = adapter.requestDevice(deviceDescriptor);
    if (!device) {
        std::cerr << "Failed to request a WebGPU device." << std::endl;
        return false;
    }
    std::cout << "WebGPU device requested successfully!" << std::endl;

    return true;
}

int main() {
    wgpu::Instance instance = nullptr;
    wgpu::Adapter adapter = nullptr;
    wgpu::Device device = nullptr;

    if (!initialize_instance(instance, adapter, device)) {
        return 1; // Initialization failed
    }

    wgpuInstanceRelease(instance);
    wgpuAdapterRelease(adapter);
    wgpuDeviceRelease(device);
    std::cout << "WebGPU resources released successfully!" << std::endl;
    return 0; // Success
}