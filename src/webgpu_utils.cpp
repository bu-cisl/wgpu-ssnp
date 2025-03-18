#include "webgpu_utils.h"

// INITIALIZING WEBGPU
bool init_wgpu(wgpu::Instance& instance, wgpu::Adapter& adapter, wgpu::Device& device, wgpu::Queue& queue) {
    // Create an instance
    wgpu::InstanceDescriptor instanceDescriptor = {};
    instance = wgpu::createInstance(instanceDescriptor);
    if (!instance) {
        std::cerr << "Failed to create WebGPU instance." << std::endl;
        return false;
    }

    // Request adapter
    wgpu::RequestAdapterOptions adapterOptions = {};
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;
    adapter = instance.requestAdapter(adapterOptions);
    if (!adapter) {
        std::cerr << "Failed to request a WebGPU adapter." << std::endl;
        return false;
    }

    // Request device
    wgpu::DeviceDescriptor deviceDescriptor = {};
    deviceDescriptor.label = "Default Device";
    device = adapter.requestDevice(deviceDescriptor);
    if (!device) {
        std::cerr << "Failed to request a WebGPU device." << std::endl;
        return false;
    }

    // Retrieve command queue
    queue = device.getQueue();
    if (!queue) {
        std::cerr << "Failed to retrieve command queue." << std::endl;
        return false;
    }

    return true;
}