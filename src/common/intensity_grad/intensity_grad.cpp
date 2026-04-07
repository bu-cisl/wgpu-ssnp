#include "intensity_grad.h"
#include <cmath>

struct Params {
    float scale;
};

static size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry fieldBufferLayout = {};
    fieldBufferLayout.binding = 0;
    fieldBufferLayout.visibility = wgpu::ShaderStage::Compute;
    fieldBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry measuredBufferLayout = {};
    measuredBufferLayout.binding = 1;
    measuredBufferLayout.visibility = wgpu::ShaderStage::Compute;
    measuredBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry outputBufferLayout = {};
    outputBufferLayout.binding = 2;
    outputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 3;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {fieldBufferLayout, measuredBufferLayout, outputBufferLayout, uniformBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 4;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

// CREATING BIND GROUP
static wgpu::BindGroup createBindGroup(
    wgpu::Device& device,
    wgpu::BindGroupLayout bindGroupLayout,
    wgpu::Buffer fieldBuffer,
    wgpu::Buffer measuredBuffer,
    wgpu::Buffer outputBuffer,
    wgpu::Buffer uniformBuffer
) {
    wgpu::BindGroupEntry fieldEntry = {};
    fieldEntry.binding = 0;
    fieldEntry.buffer = fieldBuffer;
    fieldEntry.offset = 0;
    fieldEntry.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry measuredEntry = {};
    measuredEntry.binding = 1;
    measuredEntry.buffer = measuredBuffer;
    measuredEntry.offset = 0;
    measuredEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 2;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 3;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);

    wgpu::BindGroupEntry entries[] = {fieldEntry, measuredEntry, outputEntry, uniformEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 4;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void intensity_grad(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& fieldBuffer,
    wgpu::Buffer& measuredBuffer,
    size_t bufferlen,
    float scale
) {
    buffer_len = bufferlen;
    Params params = {scale};

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile("src/common/intensity_grad/intensity_grad.wgsl", limits.maxWorkgroupSizeX);
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);
    wgpu::Buffer uniformBuffer = createBuffer(device, &params, sizeof(Params), wgpu::BufferUsage::Uniform);

    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, fieldBuffer, measuredBuffer, outputBuffer, uniformBuffer);

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
    uniformBuffer.release();
}
