#include "tilt.h"

// INPUT PARAMS
struct Params {
    uint32_t trunc_flag;
};

static size_t out_buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry cbaBufferLayout = {};
    cbaBufferLayout.binding = 0;
    cbaBufferLayout.visibility = wgpu::ShaderStage::Compute;
    cbaBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

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

    wgpu::BindGroupLayoutEntry uniformTruncLayout = {};
    uniformTruncLayout.binding = 4;
    uniformTruncLayout.visibility = wgpu::ShaderStage::Compute;
    uniformTruncLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {
        cbaBufferLayout,
        shapeBufferLayout,
        resBufferLayout,
        outBufferLayout,
        uniformTruncLayout
    };

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 5;
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
    wgpu::Buffer uniformTruncBuffer
) {
    wgpu::BindGroupEntry cbaEntry = {};
    cbaEntry.binding = 0;
    cbaEntry.buffer = anglesBuffer;
    cbaEntry.offset = 0;
    cbaEntry.size = sizeof(float) * 2;

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
    outEntry.binding = 3;
    outEntry.buffer = outBuffer;
    outEntry.offset = 0;
    outEntry.size = sizeof(float) * 2 * out_buffer_len;

    wgpu::BindGroupEntry uniformTruncEntry = {};
    uniformTruncEntry.binding = 4;
    uniformTruncEntry.buffer = uniformTruncBuffer;
    uniformTruncEntry.offset = 0;
    uniformTruncEntry.size = sizeof(uint32_t);

    wgpu::BindGroupEntry entries[] = {
        cbaEntry,
        shapeEntry,
        resEntry,
        outEntry,
        uniformTruncEntry
    };

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 5;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void tilt(
    WebGPUContext& context,
    wgpu::Buffer& outBuffer,
    std::vector<float> c_ba,
    std::vector<int> shape,
    std::optional<std::vector<float>> res,
    std::optional<bool> trunc
) {
    // Validate inputs
    assert(shape.size() == 2 && "Shape must be 2D (height, width)");
    assert(res.value().size() == 3 && "Resolution must have 3 components");
    assert(c_ba.size() == 2 && "This tilt function only support's one angle's c_ba tuple at a time");

    out_buffer_len =  shape[0] * shape[1];  
    
    Params params = {
        trunc.value() ? 1u : 0u
    };

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile("src/ssnp/tilt/tilt.wgsl", limits.maxWorkgroupSizeX);
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);
    
    // CREATING BUFFERS FOR TILT
    wgpu::Buffer anglesBuffer = createBuffer(device, c_ba.data(), sizeof(float) * 2, wgpu::BufferUsage::Storage);
    wgpu::Buffer shapeBuffer = createBuffer(device, shape.data(), sizeof(int) * 2, wgpu::BufferUsage::Storage);
    wgpu::Buffer resBuffer = createBuffer(device, res.value().data(), sizeof(float) * 3, wgpu::BufferUsage::Storage);
    wgpu::Buffer uniformTruncBuffer = createBuffer(device, &params.trunc_flag, sizeof(uint32_t), wgpu::BufferUsage::Uniform);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(
        device, 
        bindGroupLayout,
        anglesBuffer, 
        shapeBuffer, 
        resBuffer,
        outBuffer,  
        uniformTruncBuffer
    );

    // CREATING COMPUTE PIPELINE
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);

    // ENCODING AND DISPATCHING COMPUTE COMMANDS
    uint32_t workgroupsX = std::ceil(double(out_buffer_len)/limits.maxWorkgroupSizeX);
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
    uniformTruncBuffer.release();
}