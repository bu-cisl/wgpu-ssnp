#include "mult.h"

static size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry inputBuffer1Layout = {};
    inputBuffer1Layout.binding = 0;
    inputBuffer1Layout.visibility = wgpu::ShaderStage::Compute;
    inputBuffer1Layout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry inputBuffer2Layout = {};
    inputBuffer2Layout.binding = 1;
    inputBuffer2Layout.visibility = wgpu::ShaderStage::Compute;
    inputBuffer2Layout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry outputBufferLayout = {};
    outputBufferLayout.binding = 2;
    outputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry entries[] = {inputBuffer1Layout, inputBuffer2Layout, outputBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 3;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

static wgpu::BindGroup createBindGroup(
    wgpu::Device& device, 
    wgpu::BindGroupLayout bindGroupLayout, 
    wgpu::Buffer inputBuffer1, 
    wgpu::Buffer inputBuffer2,
    wgpu::Buffer outputBuffer
) {
    wgpu::BindGroupEntry inputEntry1 = {};
    inputEntry1.binding = 0;
    inputEntry1.buffer = inputBuffer1;
    inputEntry1.offset = 0;
    inputEntry1.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry inputEntry2 = {};
    inputEntry2.binding = 1;
    inputEntry2.buffer = inputBuffer2;
    inputEntry2.offset = 0;
    inputEntry2.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 2;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * buffer_len;;

    wgpu::BindGroupEntry entries[] = {inputEntry1, inputEntry2, outputEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void mult(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    wgpu::Buffer& inputBuffer1, 
    wgpu::Buffer& inputBuffer2,
    size_t bufferlen
) {
    buffer_len = bufferlen;

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile("src/mult/mult.wgsl", limits.maxWorkgroupSizeX);
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(
        device, 
        bindGroupLayout, 
        inputBuffer1,
        inputBuffer2,
        outputBuffer
    );

    // CREATING COMPUTE PIPELINE
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);

    // ENCODING AND DISPATCHING COMPUTE COMMANDS
    uint32_t workgroupsX = std::ceil(double(buffer_len)/limits.maxWorkgroupSizeX);
    wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, computePipeline, bindGroup, workgroupsX);
    queue.submit(1, &commandBuffer);

    // RELEASE RESOURCES
    commandBuffer.release();
    computePipeline.release();
    bindGroup.release();
    bindGroupLayout.release();
    shaderModule.release();
}