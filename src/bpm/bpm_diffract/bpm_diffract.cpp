#include "bpm_diffract.h"

// INPUT PARAMS
struct Params {
    float dz;
};

static size_t buffer_len;
static size_t res_buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry outputBufferLayout = {};
    outputBufferLayout.binding = 0;
    outputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry inputBufferLayout = {};
    inputBufferLayout.binding = 1;
    inputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    inputBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry resBufferLayout = {};
    resBufferLayout.binding = 2;
    resBufferLayout.visibility = wgpu::ShaderStage::Compute;
    resBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry cgammaBufferLayout = {};
    cgammaBufferLayout.binding = 3;
    cgammaBufferLayout.visibility = wgpu::ShaderStage::Compute;
    cgammaBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 4;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {outputBufferLayout, inputBufferLayout, resBufferLayout, cgammaBufferLayout, uniformBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 5;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

static wgpu::BindGroup createBindGroup(
    wgpu::Device& device, 
    wgpu::BindGroupLayout bindGroupLayout, 
    wgpu::Buffer outputBuffer, 
    wgpu::Buffer inputBuffer, 
    wgpu::Buffer resBuffer, 
    wgpu::Buffer cgammaBuffer,
    wgpu::Buffer uniformBuffer
) {
    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 0;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * 2 * buffer_len;

    wgpu::BindGroupEntry inputEntry = {};
    inputEntry.binding = 1;
    inputEntry.buffer = inputBuffer;
    inputEntry.offset = 0;
    inputEntry.size = sizeof(float) * 2 * buffer_len;

    wgpu::BindGroupEntry resEntry = {};
    resEntry.binding = 2;
    resEntry.buffer = resBuffer;
    resEntry.offset = 0;
    resEntry.size = sizeof(float) * res_buffer_len;

    wgpu::BindGroupEntry cgammaEntry = {};
    cgammaEntry.binding = 3;
    cgammaEntry.buffer = cgammaBuffer;
    cgammaEntry.offset = 0;
    cgammaEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 4;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);

    wgpu::BindGroupEntry entries[] = {outputEntry, inputEntry, resEntry, cgammaEntry, uniformEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 5;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void diffract(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    wgpu::Buffer& inputBuffer,
    size_t bufferlen,
    std::vector<int> shape,
    std::optional<std::vector<float>> res, 
    std::optional<float> dz
) {
    buffer_len = bufferlen;
    res_buffer_len = res.value().size();
    Params params = {dz.value()};

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile("src/bpm/bpm_diffract/bpm_diffract.wgsl", limits.maxWorkgroupSizeX);
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS
    wgpu::Buffer cgammaBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len, WGPUBufferUsage(wgpu::BufferUsage::Storage));
    c_gamma(context, cgammaBuffer, res.value(), shape);
    wgpu::Buffer resBuffer = createBuffer(device, res.value().data(), sizeof(float) * res_buffer_len, wgpu::BufferUsage::Storage);
    wgpu::Buffer uniformBuffer = createBuffer(device, &params, sizeof(Params), wgpu::BufferUsage::Uniform);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, outputBuffer, inputBuffer, resBuffer, cgammaBuffer, uniformBuffer);

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
    cgammaBuffer.release();
    resBuffer.release();
    uniformBuffer.release();
}