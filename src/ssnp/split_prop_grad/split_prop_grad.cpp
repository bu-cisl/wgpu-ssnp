#include "split_prop_grad.h"
#include <cmath>

static size_t buffer_len;
static size_t res_buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry inputBufferLayout = {};
    inputBufferLayout.binding = 0;
    inputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    inputBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry resBufferLayout = {};
    resBufferLayout.binding = 1;
    resBufferLayout.visibility = wgpu::ShaderStage::Compute;
    resBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry cgammaBufferLayout = {};
    cgammaBufferLayout.binding = 2;
    cgammaBufferLayout.visibility = wgpu::ShaderStage::Compute;
    cgammaBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry uGradBufferLayout = {};
    uGradBufferLayout.binding = 3;
    uGradBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uGradBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry udGradBufferLayout = {};
    udGradBufferLayout.binding = 4;
    udGradBufferLayout.visibility = wgpu::ShaderStage::Compute;
    udGradBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry entries[] = {inputBufferLayout, resBufferLayout, cgammaBufferLayout, uGradBufferLayout, udGradBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 5;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

// CREATING BIND GROUP
static wgpu::BindGroup createBindGroup(
    wgpu::Device& device,
    wgpu::BindGroupLayout bindGroupLayout,
    wgpu::Buffer inputBuffer,
    wgpu::Buffer resBuffer,
    wgpu::Buffer cgammaBuffer,
    wgpu::Buffer uGradBuffer,
    wgpu::Buffer udGradBuffer
) {
    wgpu::BindGroupEntry inputEntry = {};
    inputEntry.binding = 0;
    inputEntry.buffer = inputBuffer;
    inputEntry.offset = 0;
    inputEntry.size = sizeof(float) * 2 * buffer_len;

    wgpu::BindGroupEntry resEntry = {};
    resEntry.binding = 1;
    resEntry.buffer = resBuffer;
    resEntry.offset = 0;
    resEntry.size = sizeof(float) * res_buffer_len;

    wgpu::BindGroupEntry cgammaEntry = {};
    cgammaEntry.binding = 2;
    cgammaEntry.buffer = cgammaBuffer;
    cgammaEntry.offset = 0;
    cgammaEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry uGradEntry = {};
    uGradEntry.binding = 3;
    uGradEntry.buffer = uGradBuffer;
    uGradEntry.offset = 0;
    uGradEntry.size = sizeof(float) * 2 * buffer_len;

    wgpu::BindGroupEntry udGradEntry = {};
    udGradEntry.binding = 4;
    udGradEntry.buffer = udGradBuffer;
    udGradEntry.offset = 0;
    udGradEntry.size = sizeof(float) * 2 * buffer_len;

    wgpu::BindGroupEntry entries[] = {inputEntry, resEntry, cgammaEntry, uGradEntry, udGradEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 5;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void split_prop_grad(
    WebGPUContext& context,
    wgpu::Buffer& uGradBuffer,
    wgpu::Buffer& udGradBuffer,
    wgpu::Buffer& forwardGradBuffer,
    size_t bufferlen,
    std::vector<int> shape,
    std::optional<std::vector<float>> res
) {
    buffer_len = bufferlen;
    res_buffer_len = res.value().size();

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile("src/ssnp/split_prop_grad/split_prop_grad.wgsl", limits.maxWorkgroupSizeX);
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    wgpu::Buffer cgammaBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len, wgpu::BufferUsage::Storage);
    c_gamma(context, cgammaBuffer, res.value(), shape);
    wgpu::Buffer resBuffer = createBuffer(device, res.value().data(), sizeof(float) * res_buffer_len, wgpu::BufferUsage::Storage);

    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, forwardGradBuffer, resBuffer, cgammaBuffer, uGradBuffer, udGradBuffer);

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
    cgammaBuffer.release();
    resBuffer.release();
}
