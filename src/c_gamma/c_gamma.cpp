#include "c_gamma.h"

// INPUT PARAMS
struct Params {
    std::vector<float> res;
    std::vector<int> shape;
};

static size_t output_buffer_len;
static size_t res_buffer_len;
static size_t shape_buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry shapeBufferLayout = {};
    shapeBufferLayout.binding = 0;
    shapeBufferLayout.visibility = wgpu::ShaderStage::Compute;
    shapeBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry resBufferLayout = {};
    resBufferLayout.binding = 1;
    resBufferLayout.visibility = wgpu::ShaderStage::Compute;
    resBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry outputBufferLayout = {};
    outputBufferLayout.binding = 2;
    outputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry entries[] = {shapeBufferLayout, resBufferLayout, outputBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 3;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

static wgpu::BindGroup createBindGroup(
    wgpu::Device& device, 
    wgpu::BindGroupLayout bindGroupLayout, 
    wgpu::Buffer shapeBuffer, 
    wgpu::Buffer resBuffer, 
    wgpu::Buffer outputBuffer
) {
    wgpu::BindGroupEntry shapeEntry = {};
    shapeEntry.binding = 0;
    shapeEntry.buffer = shapeBuffer;
    shapeEntry.offset = 0;
    shapeEntry.size = sizeof(int) * shape_buffer_len;

    wgpu::BindGroupEntry resEntry = {};
    resEntry.binding = 1;
    resEntry.buffer = resBuffer;
    resEntry.offset = 0;
    resEntry.size = sizeof(float) * res_buffer_len;

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 2;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * output_buffer_len;

    wgpu::BindGroupEntry entries[] = {shapeEntry, resEntry, outputEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void c_gamma(WebGPUContext& context, wgpu::Buffer& outputBuffer, std::vector<float> res, std::vector<int> shape) {
    // Calculate the total number of elements in the output buffer
    output_buffer_len = shape[0] * shape[1];
    res_buffer_len = res.size();
    shape_buffer_len = shape.size();

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;
    
    // LOADING AND COMPILING SHADER CODE
    std::string shaderCode = readShaderFile("src/c_gamma/c_gamma.wgsl");
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS
    wgpu::Buffer resBuffer = createBuffer(device, res.data(), sizeof(float) * res_buffer_len, wgpu::BufferUsage::Storage);
    wgpu::Buffer shapeBuffer = createBuffer(device, shape.data(), sizeof(int) * shape_buffer_len, wgpu::BufferUsage::Storage);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, shapeBuffer, resBuffer, outputBuffer);

    // CREATING COMPUTE PIPELINE
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);

    // ENCODING AND DISPATCHING COMPUTE COMMANDS
    uint32_t workgroupsX = std::ceil(double(output_buffer_len)/256.0);
    wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, computePipeline, bindGroup, workgroupsX);
    queue.submit(1, &commandBuffer);

    // RELEASE RESOURCES
    commandBuffer.release();
    computePipeline.release();
    bindGroup.release();
    bindGroupLayout.release();
    shaderModule.release();
    resBuffer.release();
    shapeBuffer.release();
}