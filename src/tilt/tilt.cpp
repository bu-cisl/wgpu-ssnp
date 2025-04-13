#include "tilt.h"

// INPUT PARAMS
struct Params {
    float NA;
    uint32_t trunc_flag;
};

static size_t angles_buffer_len;
static size_t out_buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry anglesBufferLayout = {};
    anglesBufferLayout.binding = 0;
    anglesBufferLayout.visibility = wgpu::ShaderStage::Compute;
    anglesBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry shapeBufferLayout = {};
    shapeBufferLayout.binding = 1;
    shapeBufferLayout.visibility = wgpu::ShaderStage::Compute;
    shapeBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry resBufferLayout = {};
    resBufferLayout.binding = 2;
    resBufferLayout.visibility = wgpu::ShaderStage::Compute;
    resBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry outBufferLayout = {};
    outBufferLayout.binding = 3;
    outBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry uniformNALayout = {};
    uniformNALayout.binding = 4;
    uniformNALayout.visibility = wgpu::ShaderStage::Compute;
    uniformNALayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry uniformTruncLayout = {};
    uniformTruncLayout.binding = 5;
    uniformTruncLayout.visibility = wgpu::ShaderStage::Compute;
    uniformTruncLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {
        anglesBufferLayout,
        shapeBufferLayout,
        resBufferLayout,
        outBufferLayout,
        uniformNALayout,
        uniformTruncLayout
    };

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 6;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

static wgpu::BindGroup createBindGroup(
    wgpu::Device& device, 
    wgpu::BindGroupLayout bindGroupLayout,
    wgpu::Buffer anglesBuffer,
    wgpu::Buffer shapeBuffer,
    wgpu::Buffer resBuffer,
    wgpu::Buffer outBuffer, 
    wgpu::Buffer uniformNABuffer,
    wgpu::Buffer uniformTruncBuffer
) {
    wgpu::BindGroupEntry anglesEntry = {};
    anglesEntry.binding = 0;
    anglesEntry.buffer = anglesBuffer;
    anglesEntry.offset = 0;
    anglesEntry.size = sizeof(float) * angles_buffer_len;

    wgpu::BindGroupEntry shapeEntry = {};
    shapeEntry.binding = 1;
    shapeEntry.buffer = shapeBuffer;
    shapeEntry.offset = 0;
    shapeEntry.size = sizeof(int) * 2; // Always 2 elements for shape

    wgpu::BindGroupEntry resEntry = {};
    resEntry.binding = 2;
    resEntry.buffer = resBuffer;
    resEntry.offset = 0;
    resEntry.size = sizeof(float) * 3; // Always 3 elements for res

    wgpu::BindGroupEntry outEntry = {};
    factorsEntry.binding = 3;
    factorsEntry.buffer = outBuffer
    factorsEntry.offset = 0;
    factorsEntry.size = sizeof(float) * 2 * out_buffer_len;

    wgpu::BindGroupEntry uniformNAEntry = {};
    uniformNAEntry.binding = 4;
    uniformNAEntry.buffer = uniformNABuffer;
    uniformNAEntry.offset = 0;
    uniformNAEntry.size = sizeof(float);

    wgpu::BindGroupEntry uniformTruncEntry = {};
    uniformTruncEntry.binding = 5;
    uniformTruncEntry.buffer = uniformTruncBuffer;
    uniformTruncEntry.offset = 0;
    uniformTruncEntry.size = sizeof(uint32_t);

    wgpu::BindGroupEntry entries[] = {
        anglesEntry,
        shapeEntry,
        resEntry,
        outEntry,
        uniformNAEntry,
        uniformTruncEntry
    };

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 6;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void tilt(
    WebGPUContext& context,
    wgpu::Buffer& factorsBuffer,
    std::vector<float> angles,
    std::vector<int> shape,
    std::optional<float> NA,
    std::optional<std::vector<float>> res,
    std::optional<bool> trunc
) {
    // Validate inputs
    assert(shape.size() == 2 && "Shape must be 2D (height, width)");
    assert(res.value().size() == 3 && "Resolution must have 3 components");
    
    angles_buffer_len = angles.size();
    factors_buffer_len = 2 * angles.size();
    
    Params params = {
        NA.value(),
        trunc.value() ? 1u : 0u
    };

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    std::string shaderCode = readShaderFile("src/tilt/tilt.wgsl");
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);
    
    // CREATING BUFFERS FOR TILT
    wgpu::Buffer anglesBuffer = createBuffer(device, angles.data(), sizeof(float) * angles_buffer_len, wgpu::BufferUsage::Storage);
    wgpu::Buffer shapeBuffer = createBuffer(device, shape.data(), sizeof(int) * 2, wgpu::BufferUsage::Storage);
    wgpu::Buffer resBuffer = createBuffer(device, res.value().data(), sizeof(float) * 3, wgpu::BufferUsage::Storage);
    wgpu::Buffer uniformNABuffer = createBuffer(device, &params.NA, sizeof(float), wgpu::BufferUsage::Uniform);
    wgpu::Buffer uniformTruncBuffer = createBuffer(device, &params.trunc_flag, sizeof(uint32_t), wgpu::BufferUsage::Uniform);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(
        device, 
        bindGroupLayout,
        anglesBuffer, 
        shapeBuffer, 
        resBuffer,
        factorsBuffer, 
        uniformNABuffer, 
        uniformTruncBuffer
    );

    // CREATING COMPUTE PIPELINE
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);

    // ENCODING AND DISPATCHING COMPUTE COMMANDS
    uint32_t workgroupsX = std::ceil(double(angles_buffer_len)/256.0);
    wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, computePipeline, bindGroup, workgroupsX);
    queue.submit(1, &commandBuffer);
    
    // RELEASE RESOURCES
    commandBuffer.release();
    computePipeline.release();
    bindGroup.release();
    bindGroupLayout.release();
    shaderModule.release();
    anglesBuffer.release();
    shapeBuffer.release();
    resBuffer.release();
    uniformNABuffer.release();
    uniformTruncBuffer.release();
}