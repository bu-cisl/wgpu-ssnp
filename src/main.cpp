#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>

bool init_wgpu(wgpu::Instance& instance, wgpu::Adapter& adapter, wgpu::Device& device, wgpu::Queue& queue) {
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

    // Retrieve command queue
    queue = device.getQueue();
    if (!queue) {
        std::cerr << "Failed to retrieve command queue." << std::endl;
        return false;
    }
    std::cout << "Command queue retrieved successfully!" << std::endl;

    return true;
}

bool record_and_submit_commands(wgpu::Device& device, wgpu::Queue& queue) {
    // Create a command encoder
    wgpu::CommandEncoder commandEncoder = device.createCommandEncoder({});
    if (!commandEncoder) {
        std::cerr << "Failed to create command encoder." << std::endl;
        return false;
    }
    std::cout << "Command encoder created successfully!" << std::endl;

    // Record commands (For now, just clear a buffer or perform a no-op)
    // Example: Here we are not actually doing anything specific yet, 
    // but we can clear buffers in the next steps.

    // Finish encoding and get the command buffer
    wgpu::CommandBuffer commandBuffer = commandEncoder.finish({});
    if (!commandBuffer) {
        std::cerr << "Failed to finish command buffer." << std::endl;
        return false;
    }
    std::cout << "Command buffer created successfully!" << std::endl;

    // Submit the command buffer to the queue for execution
    queue.submit(1, &commandBuffer);
    std::cout << "Commands submitted successfully!" << std::endl;

    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(commandEncoder);
    std::cout << "Command buffer and encoder released successfully!" << std::endl;

    return true;
}

wgpu::Buffer createBuffer(wgpu::Device& device, size_t size, wgpu::BufferUsage usage, const void* data = nullptr) {
    wgpu::BufferDescriptor bufferDesc = {};
    bufferDesc.size = size;
    bufferDesc.usage = usage;
    bufferDesc.mappedAtCreation = (data != nullptr);

    wgpu::Buffer buffer = device.createBuffer(bufferDesc);
    if (!buffer) {
        std::cerr << "Failed to create buffer!" << std::endl;
        return nullptr;
    }

    if (data) {
        void* mappedData = buffer.getMappedRange(0, size);
        if (!mappedData) {
            std::cerr << "Failed to map buffer!" << std::endl;
            return nullptr;
        }
        std::memcpy(mappedData, data, size);
        buffer.unmap();
    }

    return buffer;
}

// Function to read the shader file
std::string readShaderFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filename << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Function to create a WebGPU shader module from WGSL code
wgpu::ShaderModule createShaderModule(wgpu::Device& device, const std::string& shaderCode) {
    wgpu::ShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.next = nullptr;  // Set next to null since it's the only descriptor
    wgslDesc.chain.sType = wgpu::SType::ShaderModuleWGSLDescriptor;
    wgslDesc.code = shaderCode.c_str();

    wgpu::ShaderModuleDescriptor shaderModuleDesc = {};
    shaderModuleDesc.nextInChain = reinterpret_cast<const WGPUChainedStruct*>(&wgslDesc);

    return device.createShaderModule(shaderModuleDesc);
}

int main() {
    wgpu::Instance instance = nullptr;
    wgpu::Adapter adapter = nullptr;
    wgpu::Device device = nullptr;
    wgpu::Queue queue = nullptr;

    if (!init_wgpu(instance, adapter, device, queue)) {
        return 1; // Initialization failed
    }

    std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};  // Input array
    std::vector<float> outputData(inputData.size(), 0.0f);    // Output array (empty)
    std::vector<float> params = {10.0f, 0.1f, 1.5f};          // res_z, dz, n0

    // Buffer sizes
    size_t inputSize = inputData.size() * sizeof(float);
    size_t outputSize = outputData.size() * sizeof(float);
    size_t paramSize = params.size() * sizeof(float);

    // Create buffers
    wgpu::Buffer inputBuffer = createBuffer(device, inputSize, 
        static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst), 
        inputData.data());

    wgpu::Buffer outputBuffer = createBuffer(device, outputSize, 
        static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst));

    wgpu::Buffer paramBuffer = createBuffer(device, paramSize, 
        static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst), 
        params.data());

    // Load WGSL shader code
    std::string shaderCode = readShaderFile("src/scatter_factor.wgsl");
    if (shaderCode.empty()) {
        std::cerr << "Failed to read shader file." << std::endl;
        return 1;
    }
    std::cout << "Shader file loaded successfully!" << std::endl;

    // Create shader module
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);
    if (!shaderModule) {
        std::cerr << "Failed to create shader module." << std::endl;
        return 1;
    }
    std::cout << "Shader module created successfully!" << std::endl;

    // Continue with pipeline creation...

    if (!record_and_submit_commands(device, queue)) {
        return 1; // Command recording or submission failed
    }

    inputBuffer.release();
    outputBuffer.release();
    paramBuffer.release();
    wgpuInstanceRelease(instance);
    wgpuAdapterRelease(adapter);
    wgpuDeviceRelease(device);
    wgpuQueueRelease(queue);
    std::cout << "WebGPU resources released successfully!" << std::endl;

    return 0; // Success
}