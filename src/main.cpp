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

wgpu::BindGroup createBindGroup(wgpu::Device& device, wgpu::BindGroupLayout layout, 
                                wgpu::Buffer inputBuffer, wgpu::Buffer outputBuffer, wgpu::Buffer paramBuffer) {
    wgpu::BindGroupEntry entries[3] = {};

    // Bind input buffer
    entries[0].binding = 0;
    entries[0].buffer = inputBuffer;
    entries[0].offset = 0;
    entries[0].size = inputBuffer.getSize();

    // Bind output buffer
    entries[1].binding = 1;
    entries[1].buffer = outputBuffer;
    entries[1].offset = 0;
    entries[1].size = outputBuffer.getSize();

    // Bind param buffer
    entries[2].binding = 2;
    entries[2].buffer = paramBuffer;
    entries[2].offset = 0;
    entries[2].size = paramBuffer.getSize();

    // Create the bind group
    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = layout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

wgpu::ComputePipeline createComputePipeline(wgpu::Device& device, wgpu::ShaderModule shaderModule, 
                                            wgpu::BindGroupLayout bindGroupLayout) {
    // Define pipeline layout
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout*>(&bindGroupLayout);

    wgpu::PipelineLayout pipelineLayout = device.createPipelineLayout(pipelineLayoutDesc);

    // Define compute stage
    wgpu::ProgrammableStageDescriptor computeStage = {};
    computeStage.module = shaderModule;
    computeStage.entryPoint = "main";

    // Define compute pipeline
    wgpu::ComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.compute = computeStage;

    return device.createComputePipeline(pipelineDesc);
}

bool record_and_submit_commands(wgpu::Device& device, wgpu::Queue& queue, 
                                wgpu::ComputePipeline& computePipeline, wgpu::BindGroup& bindGroup) {
    // Create command encoder
    wgpu::CommandEncoder commandEncoder = device.createCommandEncoder({});

    // Begin compute pass
    wgpu::ComputePassEncoder computePass = commandEncoder.beginComputePass({});

    // Bind pipeline and bind group
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup, 0, nullptr);

    // Dispatch workgroups (assuming 256 total elements, adjust accordingly)
    computePass.dispatchWorkgroups(4, 1, 1);  // Example: 4 workgroups of 64 threads each

    // End compute pass
    computePass.end();

    // Finish encoding and get the command buffer
    wgpu::CommandBuffer commandBuffer = commandEncoder.finish({});

    // Submit the command buffer
    queue.submit(1, &commandBuffer);

    std::cout << "Compute commands submitted successfully!" << std::endl;

    // Release resources
    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(commandEncoder);

    return true;
}

wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    // Define the bind group layout entries
    std::vector<wgpu::BindGroupLayoutEntry> bindGroupLayoutEntries(3);

    // Input buffer (storage, read-only)
    bindGroupLayoutEntries[0].binding = 0;
    bindGroupLayoutEntries[0].visibility = wgpu::ShaderStage::Compute;
    bindGroupLayoutEntries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    bindGroupLayoutEntries[0].buffer.minBindingSize = 0;

    // Output buffer (storage, read-write)
    bindGroupLayoutEntries[1].binding = 1;
    bindGroupLayoutEntries[1].visibility = wgpu::ShaderStage::Compute;
    bindGroupLayoutEntries[1].buffer.type = wgpu::BufferBindingType::Storage;
    bindGroupLayoutEntries[1].buffer.minBindingSize = 0;

    // Uniform buffer (params)
    bindGroupLayoutEntries[2].binding = 2;
    bindGroupLayoutEntries[2].visibility = wgpu::ShaderStage::Compute;
    bindGroupLayoutEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
    bindGroupLayoutEntries[2].buffer.minBindingSize = 0;

    // Create the bind group layout descriptor
    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc;
    bindGroupLayoutDesc.entryCount = bindGroupLayoutEntries.size();
    bindGroupLayoutDesc.entries = bindGroupLayoutEntries.data();

    // Create and return the bind group layout
    return device.createBindGroupLayout(bindGroupLayoutDesc);
}

int main() {
    wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device);
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

    // Create bind group layout
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);

    // Create bind group
    wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, inputBuffer, outputBuffer, paramBuffer);

    // Load WGSL shader
    std::string shaderCode = readShaderFile("src/scatter_factor.wgsl");
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // Create compute pipeline
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);

    if (!record_and_submit_commands(device, queue, computePipeline, bindGroup)) {
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