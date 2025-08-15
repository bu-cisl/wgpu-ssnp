#include "complex_mult.h"

static size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry inputBufferLayout1 = {};
    inputBufferLayout1.binding = 0;
    inputBufferLayout1.visibility = wgpu::ShaderStage::Compute;
    inputBufferLayout1.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry inputBufferLayout2 = {};
    inputBufferLayout2.binding = 1;
    inputBufferLayout2.visibility = wgpu::ShaderStage::Compute;
    inputBufferLayout2.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
   
    wgpu::BindGroupLayoutEntry outputBufferLayout = {};
    outputBufferLayout.binding = 2;
    outputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry entries[] = {inputBufferLayout1, inputBufferLayout2, outputBufferLayout};

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
    inputEntry1.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry inputEntry2 = {};
    inputEntry2.binding = 1;
    inputEntry2.buffer = inputBuffer2;
    inputEntry2.offset = 0;
    inputEntry2.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 2;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry entries[] = {inputEntry1, inputEntry2, outputEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void complex_mult(
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

    // shader file for complex multiplication
    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile("src/common/complex_mult/complex_mult.wgsl", limits.maxWorkgroupSizeX);
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // bind group/layout for complex multiplication
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(
        device, 
        bindGroupLayout, 
        inputBuffer1,
        inputBuffer2, 
        outputBuffer
    );

    // perform complex multiplication
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);
    uint32_t workgroupsX = std::ceil(double(buffer_len)/limits.maxWorkgroupSizeX);
    wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, computePipeline, bindGroup, workgroupsX);
    queue.submit(1, &commandBuffer);

    // Clean resources
    commandBuffer.release();
    computePipeline.release();
    shaderModule.release();
    bindGroup.release();
    bindGroupLayout.release();
}