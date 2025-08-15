#include "scatter.h"

// INPUT PARAMS
struct Params {
    float res_z;
    float dz;
    float n0;
};

static size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry inputBufferLayout = {};
    inputBufferLayout.binding = 0;
    inputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    inputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry scatterBufferLayout = {};
    scatterBufferLayout.binding = 1;
    scatterBufferLayout.visibility = wgpu::ShaderStage::Compute;
    scatterBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 2;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {inputBufferLayout, scatterBufferLayout, uniformBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 3;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

static wgpu::BindGroup createBindGroup(
    wgpu::Device& device, 
    wgpu::BindGroupLayout bindGroupLayout, 
    wgpu::Buffer inputBuffer,
    wgpu::Buffer scatterBuffer,
    wgpu::Buffer uniformBuffer
) {
    wgpu::BindGroupEntry inputEntry = {};
    inputEntry.binding = 0;
    inputEntry.buffer = inputBuffer;
    inputEntry.offset = 0;
    inputEntry.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry scatterEntry = {};
    scatterEntry.binding = 0;
    scatterEntry.buffer = scatterBuffer;
    scatterEntry.offset = 0;
    scatterEntry.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 1;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);

    wgpu::BindGroupEntry entries[] = {inputEntry, scatterEntry, uniformEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void scatter(
    WebGPUContext& context, 
    wgpu::Buffer& outputBuffer, 
    wgpu::Buffer& inputBuffer, 
    size_t bufferlen,
    std::vector<int> shape,
    std::optional<float> res_z, 
    std::optional<float> dz, 
    std::optional<float> n0
) {
    buffer_len = bufferlen;
    Params params = {res_z.value(), dz.value(), n0.value()};

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile("src/bpm/scatter/scatter.wgsl", limits.maxWorkgroupSizeX);
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS
    wgpu::Buffer scatterBuffer = createBuffer(device, nullptr, sizeof(float) * buffer_len * 2, wgpu::BufferUsage::Storage);
    wgpu::Buffer uniformBuffer = createBuffer(device, &params, sizeof(Params), wgpu::BufferUsage::Uniform);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(
        device, 
        bindGroupLayout, 
        inputBuffer,
        scatterBuffer, 
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

    // ifft(field)
    wgpu::Buffer ifftBuffer = createBuffer(device, nullptr, sizeof(float) * buffer_len * 2, wgpu::BufferUsage::Storage);
    dft(context, ifftBuffer, inputBuffer, buffer_len, shape[0], shape[1], 1); // idft
    
    // result = ifft(field) * scatter
    wgpu::Buffer multBuffer = createBuffer(device, nullptr, sizeof(float) * buffer_len * 2, wgpu::BufferUsage::Storage);
    complex_mult(context, multBuffer, ifftBuffer, scatterBuffer, buffer_len);
    ifftBuffer.release();

    // return fft(result)
    dft(context, outputBuffer, multBuffer, buffer_len, shape[0], shape[1], 0);
    multBuffer.release();
}