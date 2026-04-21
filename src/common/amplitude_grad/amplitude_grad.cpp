#include "amplitude_grad.h"

#include <cmath>
#include <string>

struct Params {
    float inv_pixels;
};

static size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry fieldLayout = {};
    fieldLayout.binding = 0;
    fieldLayout.visibility = wgpu::ShaderStage::Compute;
    fieldLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry measuredLayout = {};
    measuredLayout.binding = 1;
    measuredLayout.visibility = wgpu::ShaderStage::Compute;
    measuredLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry outputLayout = {};
    outputLayout.binding = 2;
    outputLayout.visibility = wgpu::ShaderStage::Compute;
    outputLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry uniformLayout = {};
    uniformLayout.binding = 3;
    uniformLayout.visibility = wgpu::ShaderStage::Compute;
    uniformLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {
        fieldLayout,
        measuredLayout,
        outputLayout,
        uniformLayout
    };

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

    wgpu::BindGroupEntry entries[] = {
        fieldEntry,
        measuredEntry,
        outputEntry,
        uniformEntry
    };

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 4;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void amplitude_grad(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& fieldBuffer,
    wgpu::Buffer& measuredBuffer,
    size_t bufferlen,
    float inv_pixels
) {
    buffer_len = bufferlen;
    Params params = {inv_pixels};

    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile(
        "src/common/amplitude_grad/amplitude_grad.wgsl",
        limits.maxWorkgroupSizeX
    );
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);
    wgpu::Buffer uniformBuffer = createBuffer(
        device,
        &params,
        sizeof(Params),
        wgpu::BufferUsage::Uniform
    );

    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(
        device,
        bindGroupLayout,
        fieldBuffer,
        measuredBuffer,
        outputBuffer,
        uniformBuffer
    );

    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);
    uint32_t workgroupsX = static_cast<uint32_t>(std::ceil(double(buffer_len) / limits.maxWorkgroupSizeX));
    wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, computePipeline, bindGroup, workgroupsX);
    queue.submit(1, &commandBuffer);

    commandBuffer.release();
    computePipeline.release();
    bindGroup.release();
    bindGroupLayout.release();
    shaderModule.release();
    uniformBuffer.release();
}