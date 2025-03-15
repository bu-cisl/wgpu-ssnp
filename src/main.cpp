#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>

// INPUT PARAMS
struct Params {
    float res_z;
    float dz;
    float n0;
};

size_t buffer_len;

// INITIALIZING WEBGPU
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

// LOADING AND COMPILING SHADER CODE
std::string readShaderFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filename << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();

    /**/
    std::string shaderCode = buffer.str();
    std::cout << "Shader file read successfully! First 100 chars:\n";
    std::cout << shaderCode.substr(0, 100) << "...\n"; 

    return buffer.str();
}

wgpu::ShaderModule createShaderModule(wgpu::Device& device, const std::string& shaderCode) {
    wgpu::ShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.next = nullptr;
    wgslDesc.chain.sType = wgpu::SType::ShaderModuleWGSLDescriptor;
    wgslDesc.code = shaderCode.c_str();

    wgpu::ShaderModuleDescriptor shaderModuleDesc = {};
    shaderModuleDesc.nextInChain = &wgslDesc.chain;

    wgpu::ShaderModule shaderModule = device.createShaderModule(shaderModuleDesc);

    if (!shaderModule) {
        std::cerr << "Failed to create shader module." << std::endl;
    } else {
        std::cout << "Shader module created successfully!" << std::endl;
    }

    return shaderModule;
}

// CREATING BUFFERS FOR SCATTER_FACTOR
std::string bufferUsageToString(wgpu::BufferUsage usage) {
    std::string usageStr;
    if (usage & wgpu::BufferUsage::CopySrc) usageStr += "CopySrc | ";
    if (usage & wgpu::BufferUsage::CopyDst) usageStr += "CopyDst | ";
    if (usage & wgpu::BufferUsage::Index) usageStr += "Index | ";
    if (usage & wgpu::BufferUsage::Vertex) usageStr += "Vertex | ";
    if (usage & wgpu::BufferUsage::Uniform) usageStr += "Uniform | ";
    if (usage & wgpu::BufferUsage::Storage) usageStr += "Storage | ";
    if (usage & wgpu::BufferUsage::Indirect) usageStr += "Indirect | ";
    if (!usageStr.empty()) {
        usageStr = usageStr.substr(0, usageStr.size() - 3);
    }
    return usageStr;
}

wgpu::Buffer createBuffer(wgpu::Device& device, const void* data, size_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor bufferDesc = {};
    bufferDesc.size = size;
    bufferDesc.usage = usage | wgpu::BufferUsage::CopyDst;
    bufferDesc.mappedAtCreation = false;

    wgpu::Buffer buffer = device.createBuffer(bufferDesc);
    if (!buffer) {
        std::cerr << "Failed to create buffer!" << std::endl;
    }

    if (data) {
        device.getQueue().writeBuffer(buffer, 0, data, size);
    }
    std::cout << "Buffer created successfully! Size: " << size << " bytes, Usage: " << bufferUsageToString(usage) << std::endl;

    return buffer;
}

// CREATING BIND GROUP AND LAYOUT
wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry inputBufferLayout = {};
    inputBufferLayout.binding = 0;
    inputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    inputBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry outputBufferLayout = {};
    outputBufferLayout.binding = 1;
    outputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 2;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {inputBufferLayout, outputBufferLayout, uniformBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 3;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

wgpu::BindGroup createBindGroup(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer inputBuffer, wgpu::Buffer outputBuffer, wgpu::Buffer uniformBuffer) {
    wgpu::BindGroupEntry inputEntry = {};
    inputEntry.binding = 0;
    inputEntry.buffer = inputBuffer;
    inputEntry.offset = 0;
    inputEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 1;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 2;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);

    wgpu::BindGroupEntry entries[] = {inputEntry, outputEntry, uniformEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

// CREATING COMPUTE PIPELINE
wgpu::ComputePipeline createComputePipeline(wgpu::Device& device, wgpu::ShaderModule shaderModule, wgpu::BindGroupLayout bindGroupLayout) {
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

int main() {
    std::vector<float> inputData = {1,2,3};
    buffer_len = inputData.size();
    std::vector<float> outputData(buffer_len, 0.0);
    Params params = {0.1f, 1.0f, 1.0f};

    // INITIALIZING WEBGPU
    wgpu::Instance instance = nullptr;
    wgpu::Adapter adapter = nullptr;
    wgpu::Device device = nullptr;
    wgpu::Queue queue = nullptr;

    if (!init_wgpu(instance, adapter, device, queue)) {
        return 1;
    }

    // LOADING AND COMPILING SHADER CODE
    std::string shaderCode = readShaderFile("src/scatter_factor.wgsl"); // function specified
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS FOR SCATTER_FACTOR
    wgpu::Buffer inputBuffer = createBuffer(device, inputData.data(), buffer_len * sizeof(float), wgpu::BufferUsage::Storage);
    wgpu::Buffer outputBuffer = createBuffer(device, nullptr, outputData.size() * sizeof(float),  static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    wgpu::Buffer uniformBuffer = createBuffer(device, &params, sizeof(Params), wgpu::BufferUsage::Uniform);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, inputBuffer, outputBuffer, uniformBuffer);
    if (!bindGroup) {
        std::cerr << "Failed to create bind group!" << std::endl;
        return 1;
    }
    std::cout << "Bind Group created successfully!" << std::endl;

    // CREATING COMPUTE PIPELINE
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);
    if (!computePipeline) {
        std::cerr << "Failed to create compute pipeline!" << std::endl;
        return 1;
    }
    std::cout << "Compute Pipeline created successfully!" << std::endl;

    // ENCODING AND DISPATCHING COMPUTE COMMANDS
    wgpu::CommandEncoderDescriptor encoderDesc = {};
    wgpu::CommandEncoder commandEncoder = device.createCommandEncoder(encoderDesc);
    std::cout << "Command Encoder created successfully!" << std::endl;

    wgpu::ComputePassDescriptor computePassDesc = {};
    wgpu::ComputePassEncoder computePass = commandEncoder.beginComputePass(computePassDesc);
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup, 0, nullptr);
    computePass.dispatchWorkgroups(buffer_len / 64 + 1, 1, 1);
    computePass.end();
    std::cout << "Compute Pass encoded successfully!" << std::endl;

    wgpu::CommandBufferDescriptor cmdBufferDesc = {};
    wgpu::CommandBuffer commandBuffer = commandEncoder.finish(cmdBufferDesc);
    std::cout << "Command Buffer created successfully!" << std::endl;

    queue.submit(1, &commandBuffer);
    std::cout << "Compute work submitted successfully!" << std::endl;

    // READING BACK RESULTS
    wgpu::BufferDescriptor readbackBufferDesc = {};
    readbackBufferDesc.size = outputData.size() * sizeof(float);
    readbackBufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer readbackBuffer = device.createBuffer(readbackBufferDesc);
    std::cout << "Readback Buffer created successfully!" << std::endl;

    wgpu::CommandEncoder copyEncoder = device.createCommandEncoder(encoderDesc);
    copyEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputData.size() * sizeof(float));
    std::cout << "Copy command encoded!" << std::endl;

    wgpu::CommandBuffer commandBuffer2 = copyEncoder.finish();
    queue.submit(1, &commandBuffer2);
    std::cout << "Copy command submitted!" << std::endl;

    // MAPPING
    std::cout << "Queue flushed, waiting before mapping..." << std::endl;

    bool mappingComplete = false;
    auto handle = readbackBuffer.mapAsync(wgpu::MapMode::Read, 0, outputData.size() * sizeof(float), [&](wgpu::BufferMapAsyncStatus status) {
        if (status == wgpu::BufferMapAsyncStatus::Success) {
            std::cout << "Buffer mapped successfully!" << std::endl;
            void* mappedData = readbackBuffer.getMappedRange(0, outputData.size() * sizeof(float));
            if (mappedData) {
                memcpy(outputData.data(), mappedData, outputData.size() * sizeof(float));
                std::cout << "Output data copied successfully!" << std::endl;
                readbackBuffer.unmap();

                std::cout << "Compute shader output: ";
                for (float value : outputData) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            } else {
                std::cerr << "Failed to get mapped range!" << std::endl;
            }
        } else {
            std::cerr << "Failed to map buffer! Status: " << static_cast<int>(status) << std::endl;
        }
        mappingComplete = true;
    });

    // Wait for the mapping to complete
    while (!mappingComplete) {
       wgpuDevicePoll(device, false, nullptr); 
    }

    // RELEASE RESOURCES
    computePipeline.release();
    bindGroup.release();
    bindGroupLayout.release();
    inputBuffer.release();
    outputBuffer.release();
    uniformBuffer.release();
    shaderModule.release();
    readbackBuffer.release();
    commandBuffer.release();
    commandBuffer2.release();
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    std::cout << "WebGPU resources released successfully!" << std::endl;

    return 0;
}