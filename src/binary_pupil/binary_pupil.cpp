#include "binary_pupil.h"

// INPUT PARAMS
struct Params {
    float na;
};

static size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry cgammaBufferLayout = {};
    cgammaBufferLayout.binding = 0;
    cgammaBufferLayout.visibility = wgpu::ShaderStage::Compute;
    cgammaBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry maskBufferLayout = {};
    maskBufferLayout.binding = 1;
    maskBufferLayout.visibility = wgpu::ShaderStage::Compute;
    maskBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 2;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {cgammaBufferLayout, maskBufferLayout, uniformBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 3;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

static wgpu::BindGroup createBindGroup(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer cgammaBuffer, wgpu::Buffer maskBuffer, wgpu::Buffer uniformBuffer) {
    wgpu::BindGroupEntry cgammaEntry = {};
    cgammaEntry.binding = 0;
    cgammaEntry.buffer = cgammaBuffer;
    cgammaEntry.offset = 0;
    cgammaEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry maskEntry = {};
    maskEntry.binding = 1;
    maskEntry.buffer = maskBuffer;
    maskEntry.offset = 0;
    maskEntry.size = sizeof(uint32_t) * buffer_len;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 2;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);

    wgpu::BindGroupEntry entries[] = {cgammaEntry, maskEntry, uniformEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void binary_pupil(
    WebGPUContext& context,
    wgpu::Buffer& maskBuffer,
    std::optional<std::vector<float>> res,
    std::optional<float> na,
    const std::vector<int>& shape
) {
    buffer_len = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    Params params = {na.value()};

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    std::string shaderCode = readShaderFile("src/binary_pupil/binary_pupil.wgsl");
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS FOR BINARY_PUPIL
    wgpu::Buffer cgammaBuffer = createBuffer(
        device, 
        nullptr, 
        sizeof(float) * buffer_len, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage)
    );
    c_gamma(context, cgammaBuffer, res.value(), shape);
    wgpu::Buffer uniformBuffer = createBuffer(
        device, 
        &params, 
        sizeof(Params), 
        wgpu::BufferUsage::Uniform
    );

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(
        device, 
        bindGroupLayout, 
        cgammaBuffer, 
        maskBuffer, 
        uniformBuffer
    );

    // CREATING COMPUTE PIPELINE
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);

    // ENCODING AND DISPATCHING COMPUTE COMMANDS
    wgpu::CommandEncoderDescriptor encoderDesc = {};
    wgpu::CommandEncoder commandEncoder = device.createCommandEncoder(encoderDesc);

    wgpu::ComputePassDescriptor computePassDesc = {};
    wgpu::ComputePassEncoder computePass = commandEncoder.beginComputePass(computePassDesc);
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup, 0, nullptr);
    computePass.dispatchWorkgroups(std::ceil(double(buffer_len)/256.0),1,1);
    computePass.end();

    wgpu::CommandBufferDescriptor cmdBufferDesc = {};
    wgpu::CommandBuffer commandBuffer = commandEncoder.finish(cmdBufferDesc);

    queue.submit(1, &commandBuffer);

    // RELEASE RESOURCES
    bindGroup.release();
    bindGroupLayout.release();
    cgammaBuffer.release();
    uniformBuffer.release();
    shaderModule.release();
    computePipeline.release();
    commandBuffer.release();
}