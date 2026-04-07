#include "volume_grad.h"
#include <cmath>

static size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry dqBufferLayout = {};
    dqBufferLayout.binding = 0;
    dqBufferLayout.visibility = wgpu::ShaderStage::Compute;
    dqBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry gradBufferLayout = {};
    gradBufferLayout.binding = 1;
    gradBufferLayout.visibility = wgpu::ShaderStage::Compute;
    gradBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry uBufferLayout = {};
    uBufferLayout.binding = 2;
    uBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry outputBufferLayout = {};
    outputBufferLayout.binding = 3;
    outputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry entries[] = {dqBufferLayout, gradBufferLayout, uBufferLayout, outputBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 4;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

// CREATING BIND GROUP
static wgpu::BindGroup createBindGroup(
    wgpu::Device& device,
    wgpu::BindGroupLayout bindGroupLayout,
    wgpu::Buffer dqBuffer,
    wgpu::Buffer gradBuffer,
    wgpu::Buffer uBuffer,
    wgpu::Buffer outputBuffer
) {
    wgpu::BindGroupEntry dqEntry = {};
    dqEntry.binding = 0;
    dqEntry.buffer = dqBuffer;
    dqEntry.offset = 0;
    dqEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry gradEntry = {};
    gradEntry.binding = 1;
    gradEntry.buffer = gradBuffer;
    gradEntry.offset = 0;
    gradEntry.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry uEntry = {};
    uEntry.binding = 2;
    uEntry.buffer = uBuffer;
    uEntry.offset = 0;
    uEntry.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 3;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry entries[] = {dqEntry, gradEntry, uEntry, outputEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 4;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void volume_grad(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& dqBuffer,
    wgpu::Buffer& gradBuffer,
    wgpu::Buffer& uBuffer,
    size_t bufferlen
) {
    buffer_len = bufferlen;

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile("src/ssnp/volume_grad/volume_grad.wgsl", limits.maxWorkgroupSizeX);
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, dqBuffer, gradBuffer, uBuffer, outputBuffer);

    // ENCODING AND DISPATCHING COMPUTE COMMANDS
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);
    uint32_t workgroupsX = std::ceil(double(buffer_len) / limits.maxWorkgroupSizeX);
    wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, computePipeline, bindGroup, workgroupsX);
    queue.submit(1, &commandBuffer);

    // RELEASE RESOURCES
    commandBuffer.release();
    computePipeline.release();
    bindGroup.release();
    bindGroupLayout.release();
    shaderModule.release();
}
