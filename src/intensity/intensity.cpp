#include "intensity.h"

struct Params {
    int intensity;
};

static size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
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

static wgpu::BindGroup createBindGroup(
    wgpu::Device& device, 
    wgpu::BindGroupLayout bindGroupLayout, 
    wgpu::Buffer inputBuffer, 
    wgpu::Buffer outputBuffer, 
    wgpu::Buffer uniformBuffer
) {
    wgpu::BindGroupEntry inputEntry = {};
    inputEntry.binding = 0;
    inputEntry.buffer = inputBuffer;
    inputEntry.offset = 0;
    inputEntry.size = sizeof(float) * buffer_len * 2;

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

void intense(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    wgpu::Buffer& inputBuffer, 
    size_t bufferlen,
    bool intensity
) {
    buffer_len = bufferlen;
    Params params = {int(intensity)};

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile("src/intensity/intensity.wgsl", limits.maxWorkgroupSizeX);
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS
    wgpu::Buffer uniformBuffer = createBuffer(device, &params, sizeof(Params), wgpu::BufferUsage::Uniform);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(
        device, 
        bindGroupLayout, 
        inputBuffer, 
        outputBuffer, 
        uniformBuffer
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
    uniformBuffer.release();
}