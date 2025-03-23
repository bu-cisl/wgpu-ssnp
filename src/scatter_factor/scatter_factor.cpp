#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>
#include "scatter_factor.h"
#include "../webgpu_utils.h"

// INPUT PARAMS
struct Params {
    float res_z;
    float dz;
    float n0;
};

size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
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

wgpu::BindGroup createBindGroup(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer inputBuffer, wgpu::Buffer outputBuffer, wgpu::Buffer uniformBuffer) {
    wgpu::BindGroupEntry inputEntry = {};
    inputEntry.binding = 0;
    inputEntry.buffer = inputBuffer;
    inputEntry.offset = 0;
    inputEntry.size = sizeof(float) * buffer_len;

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

std::vector<float> scatter_factor(WebGPUContext& context, std::vector<float> inputData, std::optional<float> res_z, std::optional<float> dz, std::optional<float> n0) {
    buffer_len = inputData.size();
    std::vector<float> outputData(buffer_len, 0.0);
    Params params = {res_z.value(), dz.value(), n0.value()};

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    std::string shaderCode = readShaderFile("src/scatter_factor/scatter_factor.wgsl");
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS FOR SCATTER_FACTOR
    wgpu::Buffer inputBuffer = createBuffer(device, inputData.data(), buffer_len * sizeof(float), wgpu::BufferUsage::Storage);
    wgpu::Buffer outputBuffer = createBuffer(device, nullptr, outputData.size() * sizeof(float), static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    wgpu::Buffer uniformBuffer = createBuffer(device, &params, sizeof(Params), wgpu::BufferUsage::Uniform);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, inputBuffer, outputBuffer, uniformBuffer);
    if (!bindGroup) {
        std::cerr << "Failed to create bind group!" << std::endl;
        return {};
    }

    // CREATING COMPUTE PIPELINE
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);
    if (!computePipeline) {
        std::cerr << "Failed to create compute pipeline!" << std::endl;
        return {};
    }

    // ENCODING AND DISPATCHING COMPUTE COMMANDS
    wgpu::CommandEncoderDescriptor encoderDesc = {};
    wgpu::CommandEncoder commandEncoder = device.createCommandEncoder(encoderDesc);

    wgpu::ComputePassDescriptor computePassDesc = {};
    wgpu::ComputePassEncoder computePass = commandEncoder.beginComputePass(computePassDesc);
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, bindGroup, 0, nullptr);
    computePass.dispatchWorkgroups(buffer_len / 64 + 1, 1, 1);
    computePass.end();

    wgpu::CommandBufferDescriptor cmdBufferDesc = {};
    wgpu::CommandBuffer commandBuffer = commandEncoder.finish(cmdBufferDesc);

    queue.submit(1, &commandBuffer);

    // READING BACK RESULTS
    wgpu::BufferDescriptor readbackBufferDesc = {};
    readbackBufferDesc.size = outputData.size() * sizeof(float);
    readbackBufferDesc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer readbackBuffer = device.createBuffer(readbackBufferDesc);

    wgpu::CommandEncoder copyEncoder = device.createCommandEncoder(encoderDesc);
    copyEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputData.size() * sizeof(float));

    wgpu::CommandBuffer commandBuffer2 = copyEncoder.finish();
    queue.submit(1, &commandBuffer2);

    // MAPPING
    std::vector<float> output = {};

    bool mappingComplete = false;
    auto handle = readbackBuffer.mapAsync(wgpu::MapMode::Read, 0, outputData.size() * sizeof(float), [&](wgpu::BufferMapAsyncStatus status) {
        if (status == wgpu::BufferMapAsyncStatus::Success) {
            void* mappedData = readbackBuffer.getMappedRange(0, outputData.size() * sizeof(float));
            if (mappedData) {
                memcpy(outputData.data(), mappedData, outputData.size() * sizeof(float));
                readbackBuffer.unmap();
            
                for (float value : outputData) {
                    output.push_back(value);
                }
            } else {
                std::cerr << "Failed to get mapped range!" << std::endl;
            }
        } else {
            std::cerr << "Failed to map buffer! Status: " << static_cast<int>(status) << std::endl;
        }
        mappingComplete = true;
    });

    // Wait for the mapping to complete
    while (!mappingComplete) {
       wgpuDevicePoll(device, false, nullptr); 
    }

    // RELEASE RESOURCES
    computePipeline.release();
    bindGroup.release();
    bindGroupLayout.release();
    inputBuffer.release();
    outputBuffer.release();
    uniformBuffer.release();
    shaderModule.release();
    readbackBuffer.release();
    commandBuffer.release();
    commandBuffer2.release();

    return output;
}