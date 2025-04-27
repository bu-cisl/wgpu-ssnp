#include "dft.h"

static size_t buffer_size;

struct Params {
    int rows;
    int cols;
};

// CREATING BIND GROUP LAYOUT
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

    wgpu::BindGroupLayoutEntry inverseFlagLayout = {};
    inverseFlagLayout.binding = 3;
    inverseFlagLayout.visibility = wgpu::ShaderStage::Compute;
    inverseFlagLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {inputBufferLayout, outputBufferLayout, uniformBufferLayout, inverseFlagLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 4;      
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

// CREATING BIND GROUP
static wgpu::BindGroup createBindGroup(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer inputBuffer, wgpu::Buffer outputBuffer, wgpu::Buffer uniformBuffer, wgpu::Buffer inverseFlagBuffer) {
    wgpu::BindGroupEntry inputEntry = {};
    inputEntry.binding = 0;
    inputEntry.buffer = inputBuffer;
    inputEntry.offset = 0;
    inputEntry.size = sizeof(float) * 2 * buffer_size;

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 1;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * 2 * buffer_size;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 2;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);

    wgpu::BindGroupEntry inverseFlagEntry = {};
    inverseFlagEntry.binding = 3;
    inverseFlagEntry.buffer = inverseFlagBuffer;
    inverseFlagEntry.offset = 0;
    inverseFlagEntry.size = sizeof(uint32_t);  
    
    wgpu::BindGroupEntry entries[] = {inputEntry, outputEntry, uniformEntry, inverseFlagEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 4;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void dft(
    WebGPUContext& context, 
    wgpu::Buffer& finalOutputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows, 
    int cols, 
    uint32_t doInverse
) {
    buffer_size = buffersize;
    Params params = {rows, cols};

    // Retrieve device and queue.
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;
    WorkgroupLimits limits = getWorkgroupLimits(device);
    limits.maxWorkgroupSizeX = std::min(limits.maxWorkgroupSizeX, sqrt(limits.maxInvocationsPerWorkgroup));
    limits.maxWorkgroupSizeY = std::min(limits.maxWorkgroupSizeY, sqrt(limits.maxInvocationsPerWorkgroup));

    // Create the uniform buffer for dimensions.
    wgpu::Buffer uniformBuffer = createBuffer(device, &params, sizeof(Params), wgpu::BufferUsage::Uniform);

    uint32_t inverseFlag = doInverse ? 1 : 0;
    wgpu::Buffer inverseFlagBuffer = createBuffer(device, &inverseFlag, sizeof(uint32_t), wgpu::BufferUsage::Uniform);  

    // ROW DFT PASS -> save output in intermediate buffer before column pass
    wgpu::Buffer intermediateBuffer = createBuffer(device, nullptr, sizeof(float) * 2 * buffer_size, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));

    std::string shaderCodeRow = readShaderFile("src/dft/dft_row.wgsl", limits.maxWorkgroupSizeX, limits.maxWorkgroupSizeY);
    wgpu::ShaderModule shaderModuleRow = createShaderModule(device, shaderCodeRow);

    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroupRow = createBindGroup(device, bindGroupLayout, inputBuffer, intermediateBuffer, uniformBuffer, inverseFlagBuffer);
    wgpu::ComputePipeline computePipelineRow = createComputePipeline(device, shaderModuleRow, bindGroupLayout);

    // Note: same workgroups for row pass & col pass
    uint32_t workgroupsX = std::ceil(double(cols)/limits.maxWorkgroupSizeX);
    uint32_t workgroupsY = std::ceil(double(rows)/limits.maxWorkgroupSizeY);

    wgpu::CommandBuffer commandBufferRow = createComputeCommandBuffer(device, computePipelineRow, bindGroupRow, workgroupsX, workgroupsY);
    queue.submit(1, &commandBufferRow);

    // Clean row pass resources before doing column pass
    commandBufferRow.release();
    computePipelineRow.release();
    bindGroupRow.release();
    shaderModuleRow.release();

    // COLUMN DFT PASS
    std::string shaderCodeCol = readShaderFile("src/dft/dft_col.wgsl", limits.maxWorkgroupSizeX, limits.maxWorkgroupSizeY);
    wgpu::ShaderModule shaderModuleCol = createShaderModule(device, shaderCodeCol);

    wgpu::BindGroup bindGroupCol = createBindGroup(device, bindGroupLayout, intermediateBuffer, finalOutputBuffer, uniformBuffer, inverseFlagBuffer);
    wgpu::ComputePipeline computePipelineCol = createComputePipeline(device, shaderModuleCol, bindGroupLayout);

    wgpu::CommandBuffer commandBufferCol = createComputeCommandBuffer(device, computePipelineCol, bindGroupCol, workgroupsX, workgroupsY);
    queue.submit(1, &commandBufferCol);

    // Clean all resources
    commandBufferCol.release();
    computePipelineCol.release();
    bindGroupCol.release();
    bindGroupLayout.release();
    shaderModuleCol.release();
    uniformBuffer.release();
    intermediateBuffer.release();
    inverseFlagBuffer.release();
}
