#include "webgpu_utils.h"

// INITIALIZING WEBGPU
bool initWebGPU(WebGPUContext& context) {
    // Create an instance
    wgpu::InstanceDescriptor instanceDescriptor = {};
    context.instance = wgpu::createInstance(instanceDescriptor);
    if (!context.instance) {
        std::cerr << "Failed to create WebGPU instance." << std::endl;
        return false;
    }

    // Request adapter
    wgpu::RequestAdapterOptions adapterOptions = {};
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;
    context.adapter = context.instance.requestAdapter(adapterOptions);
    if (!context.adapter) {
        std::cerr << "Failed to request a WebGPU adapter." << std::endl;
        return false;
    }

    // Request device
    wgpu::DeviceDescriptor deviceDescriptor = {};
    deviceDescriptor.label = "Default Device";
    context.device = context.adapter.requestDevice(deviceDescriptor);
    if (!context.device) {
        std::cerr << "Failed to request a WebGPU device." << std::endl;
        return false;
    }

    // Retrieve command queue
    context.queue = context.device.getQueue();
    if (!context.queue) {
        std::cerr << "Failed to retrieve command queue." << std::endl;
        return false;
    }

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
    }
    return shaderModule;
}

wgpu::Buffer createBuffer(wgpu::Device& device, const void* data, size_t size, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor bufferDesc = {};
    bufferDesc.size = size;
    bufferDesc.usage = usage | wgpu::BufferUsage::CopyDst;
    bufferDesc.mappedAtCreation = false;

    wgpu::Buffer buffer = device.createBuffer(bufferDesc);
    if (!buffer) {
        std::cerr << "Failed to create buffer." << std::endl;
    }

    if (data) {
        device.getQueue().writeBuffer(buffer, 0, data, size);
    }

    return buffer;
}

wgpu::ComputePipeline createComputePipeline(wgpu::Device& device, wgpu::ShaderModule shaderModule, wgpu::BindGroupLayout bindGroupLayout) {
    // Define pipeline layout
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc = {};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = reinterpret_cast<WGPUBindGroupLayout*>(&bindGroupLayout);

    wgpu::PipelineLayout pipelineLayout = device.createPipelineLayout(pipelineLayoutDesc);
    if (!pipelineLayout) {
        std::cerr << "Failed to create pipeline layout." << std::endl;
        return nullptr;
    }

    // Define compute stage
    wgpu::ProgrammableStageDescriptor computeStage = {};
    computeStage.module = shaderModule;
    computeStage.entryPoint = "main";

    // Define compute pipeline
    wgpu::ComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.layout = pipelineLayout;
    pipelineDesc.compute = computeStage;

    wgpu::ComputePipeline pipeline = device.createComputePipeline(pipelineDesc);
    if (!pipeline) {
        std::cerr << "Failed to create compute pipeline." << std::endl;
    }

    return pipeline;
}