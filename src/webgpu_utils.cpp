#include "webgpu_utils.h"

// INITIALIZING WEBGPU
void initWebGPU(WebGPUContext& context) {
    // Create an instance
    wgpu::InstanceDescriptor instanceDescriptor = {};
    context.instance = wgpu::createInstance(instanceDescriptor);
    if (!context.instance) {
        std::cerr << "Failed to create WebGPU instance." << std::endl;
    }

    // Request adapter
    wgpu::RequestAdapterOptions adapterOptions = {};
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;
    context.adapter = context.instance.requestAdapter(adapterOptions);
    if (!context.adapter) {
        std::cerr << "Failed to request a WebGPU adapter." << std::endl;
    }

    // Get adapter's limits
    WGPUSupportedLimits supportedLimits = {};
    wgpuAdapterGetLimits(context.adapter, &supportedLimits);

    // Request device
    wgpu::DeviceDescriptor deviceDescriptor = {};
    deviceDescriptor.label = "Default Device";

    // Align device limits to that of adapter
    WGPURequiredLimits requiredLimits = {};
    requiredLimits.limits = supportedLimits.limits;
    requiredLimits.limits.maxBufferSize = requiredLimits.limits.maxBufferSize-1; // for some reason defaulted to 256 MB before
    deviceDescriptor.requiredLimits = &requiredLimits;
    context.device = context.adapter.requestDevice(deviceDescriptor);
    if (!context.device) {
        std::cerr << "Failed to request a WebGPU device." << std::endl;
    }

    // Retrieve command queue
    context.queue = context.device.getQueue();
    if (!context.queue) {
        std::cerr << "Failed to retrieve command queue." << std::endl;
    }
}

// Function to fetch the workgroup limits
WorkgroupLimits getWorkgroupLimits(wgpu::Device& device) {
    WGPUSupportedLimits limits = {};
    WorkgroupLimits result;

    bool success = wgpuDeviceGetLimits(device, &limits);
    if (success) {
        result.maxWorkgroupSizeX = double(limits.limits.maxComputeWorkgroupSizeX);
        result.maxWorkgroupSizeY = double(limits.limits.maxComputeWorkgroupSizeY);
        result.maxWorkgroupSizeZ = double(limits.limits.maxComputeWorkgroupSizeZ);
        result.maxWorkgroupsPerDimension = double(limits.limits.maxComputeWorkgroupsPerDimension);
    } else {
        std::cerr << "Error fetching workgroup limits." << std::endl;
        result = { -1.0, -1.0, -1.0, -1.0 }; // Return default error values
    }

    return result;
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

// CREATING BUFFERS
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

// COMPUTE PIPELINE UTILITIES
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

// ENCODING AND DISPATCHING COMPUTE COMMANDS
wgpu::CommandBuffer createComputeCommandBuffer(
    wgpu::Device& device,
    wgpu::ComputePipeline& computePipeline,
    wgpu::BindGroup& bindGroup,
    uint32_t workgroupsX,
    uint32_t workgroupsY,
    uint32_t workgroupsZ
) {
    wgpu::CommandEncoderDescriptor encoderDesc = {};
    wgpu::CommandEncoder commandEncoder = device.createCommandEncoder(encoderDesc);

    wgpu::ComputePassDescriptor computePassDesc = {};
    wgpu::ComputePassEncoder computePass = commandEncoder.beginComputePass(computePassDesc);
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup, 0, nullptr);
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    computePass.end();

    wgpu::CommandBufferDescriptor cmdBufferDesc = {};
    return commandEncoder.finish(cmdBufferDesc);
}

void dispatchComputeCommands(
    wgpu::Queue& queue,
    wgpu::CommandBuffer& commandBuffer
) {
    queue.submit(1, &commandBuffer);
}

// READBACK
std::vector<float> readBack(wgpu::Device& device, wgpu::Queue& queue, size_t buffer_len, wgpu::Buffer& outputBuffer) {
    std::vector<float> output(buffer_len);

    // READING BACK RESULTS
    wgpu::BufferDescriptor readbackBufferDesc = {};
    readbackBufferDesc.size = buffer_len * sizeof(float);
    readbackBufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer readbackBuffer = device.createBuffer(readbackBufferDesc);

    wgpu::CommandEncoderDescriptor encoderDesc = {};
    wgpu::CommandEncoder copyEncoder = device.createCommandEncoder(encoderDesc);
    copyEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, buffer_len * sizeof(float));

    wgpu::CommandBuffer commandBuffer = copyEncoder.finish();
    queue.submit(1, &commandBuffer);

    //MAPPING BACK TO CPU
    bool mappingComplete = false;
    auto handle = readbackBuffer.mapAsync(wgpu::MapMode::Read, 0, buffer_len * sizeof(float), [&](wgpu::BufferMapAsyncStatus status) {
        if (status == wgpu::BufferMapAsyncStatus::Success) {
            void* mappedData = readbackBuffer.getMappedRange(0, buffer_len * sizeof(float));
            if (mappedData) {
                memcpy(output.data(), mappedData, buffer_len * sizeof(float));
                readbackBuffer.unmap();
            } else {
                std::cerr << "Failed to get mapped range!" << std::endl;
            }
        } else {
            std::cerr << "Failed to map buffer! Status: " << int(status) << std::endl;
        }
        mappingComplete = true;
    });

    // Wait for the mapping to complete
    while (!mappingComplete) {
        wgpuDevicePoll(device, false, nullptr); 
    }

    readbackBuffer.release();
    commandBuffer.release();

    return output;
}

// Temporary uint32 version of readBack for binary_pupil
std::vector<uint32_t> readBack2(wgpu::Device& device, wgpu::Queue& queue, size_t buffer_len, wgpu::Buffer& outputBuffer) {
    std::vector<uint32_t> output(buffer_len);

    // READING BACK RESULTS
    wgpu::BufferDescriptor readbackBufferDesc = {};
    readbackBufferDesc.size = buffer_len * sizeof(uint32_t);
    readbackBufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer readbackBuffer = device.createBuffer(readbackBufferDesc);

    wgpu::CommandEncoderDescriptor encoderDesc = {};
    wgpu::CommandEncoder copyEncoder = device.createCommandEncoder(encoderDesc);
    copyEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, buffer_len * sizeof(uint32_t));

    wgpu::CommandBuffer commandBuffer = copyEncoder.finish();
    queue.submit(1, &commandBuffer);

    //MAPPING BACK TO CPU
    bool mappingComplete = false;
    auto handle = readbackBuffer.mapAsync(wgpu::MapMode::Read, 0, buffer_len * sizeof(uint32_t), [&](wgpu::BufferMapAsyncStatus status) {
        if (status == wgpu::BufferMapAsyncStatus::Success) {
            void* mappedData = readbackBuffer.getMappedRange(0, buffer_len * sizeof(uint32_t));
            if (mappedData) {
                memcpy(output.data(), mappedData, buffer_len * sizeof(uint32_t));
                readbackBuffer.unmap();
            } else {
                std::cerr << "Failed to get mapped range!" << std::endl;
            }
        } else {
            std::cerr << "Failed to map buffer! Status: " << int(status) << std::endl;
        }
        mappingComplete = true;
    });

    // Wait for the mapping to complete
    while (!mappingComplete) {
        wgpuDevicePoll(device, false, nullptr); 
    }

    readbackBuffer.release();
    commandBuffer.release();

    return output;
}