#include "fft.h"
#include "../dft/dft.h"
#include "../complex_scale/complex_scale.h"
#include <iostream>
#include <cmath>

static size_t buffer_size;

struct FFTParams {
    int rows;
    int cols;
    int stage;  // Which butterfly stage we're on
};

// CREATING BIND GROUP LAYOUT for FFT
static wgpu::BindGroupLayout createFFTBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry inputBufferLayout = {};
    inputBufferLayout.binding = 0;
    inputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    inputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;  // FFT needs read-write

    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 1;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry inverseFlagLayout = {};
    inverseFlagLayout.binding = 2;
    inverseFlagLayout.visibility = wgpu::ShaderStage::Compute;
    inverseFlagLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {inputBufferLayout, uniformBufferLayout, inverseFlagLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 3;      
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

// CREATING BIND GROUP for FFT
static wgpu::BindGroup createFFTBindGroup(
    wgpu::Device& device, 
    wgpu::BindGroupLayout bindGroupLayout, 
    wgpu::Buffer dataBuffer, 
    wgpu::Buffer uniformBuffer, 
    wgpu::Buffer inverseFlagBuffer
) {
    wgpu::BindGroupEntry inputEntry = {};
    inputEntry.binding = 0;
    inputEntry.buffer = dataBuffer;
    inputEntry.offset = 0;
    inputEntry.size = sizeof(float) * 2 * buffer_size;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 1;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(FFTParams);

    wgpu::BindGroupEntry inverseFlagEntry = {};
    inverseFlagEntry.binding = 2;
    inverseFlagEntry.buffer = inverseFlagBuffer;
    inverseFlagEntry.offset = 0;
    inverseFlagEntry.size = sizeof(uint32_t);
    
    wgpu::BindGroupEntry entries[] = {inputEntry, uniformEntry, inverseFlagEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void fft(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    uint32_t doInverse,
    bool forceDft
) {
    if (forceDft || !isValidFFTDimensions(rows, cols)) {
        dft(context, outputBuffer, inputBuffer, buffersize, rows, cols, doInverse);
        return;
    }

    fftPowerOfTwo(context, outputBuffer, inputBuffer, buffersize, rows, cols, doInverse);
}

void fftPowerOfTwo(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    uint32_t doInverse
) {
    buffer_size = buffersize;
    
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;
    WorkgroupLimits limits = getWorkgroupLimits(device);
    limits.maxWorkgroupSizeX = std::min(limits.maxWorkgroupSizeX, sqrt(limits.maxInvocationsPerWorkgroup));
    limits.maxWorkgroupSizeY = std::min(limits.maxWorkgroupSizeY, sqrt(limits.maxInvocationsPerWorkgroup));

    // Create temporary buffer for in-place FFT computation (copy input to output first)
    wgpu::Buffer workBuffer = createBuffer(device, nullptr, sizeof(float) * 2 * buffer_size, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst));

    // Copy input to work buffer
    wgpu::CommandEncoder encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(inputBuffer, 0, workBuffer, 0, sizeof(float) * 2 * buffer_size);
    wgpu::CommandBuffer cmdBuffer = encoder.finish();
    queue.submit(1, &cmdBuffer);
    cmdBuffer.release();

    uint32_t inverseFlag = doInverse ? 1 : 0;
    wgpu::Buffer inverseFlagBuffer = createBuffer(device, &inverseFlag, sizeof(uint32_t), wgpu::BufferUsage::Uniform);

    // ==================== ROW FFT ====================
    {
        wgpu::BindGroupLayout bindGroupLayout = createFFTBindGroupLayout(device);
        
        // Bit-reversal pass for rows
        std::string bitRevShaderCode = readShaderFile("src/common/fft/fft_bit_reversal.wgsl", limits.maxWorkgroupSizeX, limits.maxWorkgroupSizeY);
        wgpu::ShaderModule bitRevShaderModule = createShaderModule(device, bitRevShaderCode);
        
        FFTParams params = {rows, cols, 0};
        wgpu::Buffer paramsBuffer = createBuffer(device, &params, sizeof(FFTParams), wgpu::BufferUsage::Uniform);
        
        wgpu::BindGroup bindGroup = createFFTBindGroup(device, bindGroupLayout, workBuffer, paramsBuffer, inverseFlagBuffer);
        wgpu::ComputePipeline pipeline = createComputePipeline(device, bitRevShaderModule, bindGroupLayout);
        
        uint32_t workgroupsX = std::ceil(double(cols) / limits.maxWorkgroupSizeX);
        uint32_t workgroupsY = std::ceil(double(rows) / limits.maxWorkgroupSizeY);
        
        wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, pipeline, bindGroup, workgroupsX, workgroupsY);
        queue.submit(1, &commandBuffer);
        
        commandBuffer.release();
        pipeline.release();
        bindGroup.release();
        bitRevShaderModule.release();
        paramsBuffer.release();
        bindGroupLayout.release();
    }

    // Butterfly passes for rows (log2(cols) stages)
    int numStagesRow = log2Int(cols);
    for (int stage = 0; stage < numStagesRow; stage++) {
        wgpu::BindGroupLayout bindGroupLayout = createFFTBindGroupLayout(device);
        
        std::string butterflyShaderCode = readShaderFile("src/common/fft/fft_butterfly.wgsl", limits.maxWorkgroupSizeX, limits.maxWorkgroupSizeY);
        wgpu::ShaderModule butterflyShaderModule = createShaderModule(device, butterflyShaderCode);
        
        FFTParams params = {rows, cols, stage};
        wgpu::Buffer paramsBuffer = createBuffer(device, &params, sizeof(FFTParams), wgpu::BufferUsage::Uniform);
        
        wgpu::BindGroup bindGroup = createFFTBindGroup(device, bindGroupLayout, workBuffer, paramsBuffer, inverseFlagBuffer);
        wgpu::ComputePipeline pipeline = createComputePipeline(device, butterflyShaderModule, bindGroupLayout);
        
        uint32_t workgroupsX = std::ceil(double(cols) / limits.maxWorkgroupSizeX);
        uint32_t workgroupsY = std::ceil(double(rows) / limits.maxWorkgroupSizeY);
        
        wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, pipeline, bindGroup, workgroupsX, workgroupsY);
        queue.submit(1, &commandBuffer);
        
        commandBuffer.release();
        pipeline.release();
        bindGroup.release();
        butterflyShaderModule.release();
        paramsBuffer.release();
        bindGroupLayout.release();
    }

    // ==================== COLUMN FFT ====================
    {
        wgpu::BindGroupLayout bindGroupLayout = createFFTBindGroupLayout(device);
        
        // Bit-reversal pass for columns
        std::string bitRevShaderCode = readShaderFile("src/common/fft/fft_bit_reversal_col.wgsl", limits.maxWorkgroupSizeX, limits.maxWorkgroupSizeY);
        wgpu::ShaderModule bitRevShaderModule = createShaderModule(device, bitRevShaderCode);
        
        FFTParams params = {rows, cols, 0};
        wgpu::Buffer paramsBuffer = createBuffer(device, &params, sizeof(FFTParams), wgpu::BufferUsage::Uniform);
        
        wgpu::BindGroup bindGroup = createFFTBindGroup(device, bindGroupLayout, workBuffer, paramsBuffer, inverseFlagBuffer);
        wgpu::ComputePipeline pipeline = createComputePipeline(device, bitRevShaderModule, bindGroupLayout);
        
        uint32_t workgroupsX = std::ceil(double(cols) / limits.maxWorkgroupSizeX);
        uint32_t workgroupsY = std::ceil(double(rows) / limits.maxWorkgroupSizeY);
        
        wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, pipeline, bindGroup, workgroupsX, workgroupsY);
        queue.submit(1, &commandBuffer);
        
        commandBuffer.release();
        pipeline.release();
        bindGroup.release();
        bitRevShaderModule.release();
        paramsBuffer.release();
        bindGroupLayout.release();
    }

    // Butterfly passes for columns (log2(rows) stages)
    int numStagesCol = log2Int(rows);
    for (int stage = 0; stage < numStagesCol; stage++) {
        wgpu::BindGroupLayout bindGroupLayout = createFFTBindGroupLayout(device);
        
        std::string butterflyShaderCode = readShaderFile("src/common/fft/fft_butterfly_col.wgsl", limits.maxWorkgroupSizeX, limits.maxWorkgroupSizeY);
        wgpu::ShaderModule butterflyShaderModule = createShaderModule(device, butterflyShaderCode);
        
        FFTParams params = {rows, cols, stage};
        wgpu::Buffer paramsBuffer = createBuffer(device, &params, sizeof(FFTParams), wgpu::BufferUsage::Uniform);
        
        wgpu::BindGroup bindGroup = createFFTBindGroup(device, bindGroupLayout, workBuffer, paramsBuffer, inverseFlagBuffer);
        wgpu::ComputePipeline pipeline = createComputePipeline(device, butterflyShaderModule, bindGroupLayout);
        
        uint32_t workgroupsX = std::ceil(double(cols) / limits.maxWorkgroupSizeX);
        uint32_t workgroupsY = std::ceil(double(rows) / limits.maxWorkgroupSizeY);
        
        wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, pipeline, bindGroup, workgroupsX, workgroupsY);
        queue.submit(1, &commandBuffer);
        
        commandBuffer.release();
        pipeline.release();
        bindGroup.release();
        butterflyShaderModule.release();
        paramsBuffer.release();
        bindGroupLayout.release();
    }

    // Copy result to output buffer
    {
        wgpu::CommandEncoder encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(workBuffer, 0, outputBuffer, 0, sizeof(float) * 2 * buffer_size);
        wgpu::CommandBuffer cmdBuffer = encoder.finish();
        queue.submit(1, &cmdBuffer);
        cmdBuffer.release();
    }

    // Cleanup
    workBuffer.release();
    inverseFlagBuffer.release();
}

void fft_adjoint_forward(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    bool forceDft
) {
    wgpu::Buffer tempBuffer = createBuffer(
        context.device,
        nullptr,
        sizeof(float) * 2 * buffersize,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    fft(context, tempBuffer, inputBuffer, buffersize, rows, cols, 1, forceDft);
    complex_scale(context, outputBuffer, tempBuffer, buffersize, static_cast<float>(buffersize));
    tempBuffer.release();
}

void fft_adjoint_inverse(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t buffersize,
    int rows,
    int cols,
    bool forceDft
) {
    wgpu::Buffer tempBuffer = createBuffer(
        context.device,
        nullptr,
        sizeof(float) * 2 * buffersize,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    fft(context, tempBuffer, inputBuffer, buffersize, rows, cols, 0, forceDft);
    complex_scale(context, outputBuffer, tempBuffer, buffersize, 1.0f / static_cast<float>(buffersize));
    tempBuffer.release();
}
