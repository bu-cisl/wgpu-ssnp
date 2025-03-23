#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <webgpu/webgpu.hpp>
#include "c_gamma.h"
#include "../webgpu_utils.h"

// INPUT PARAMS
struct Params {
    std::vector<float> res;
    std::vector<int> shape;
};

static size_t buffer_len;

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

static wgpu::BindGroup createBindGroup(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer shapeBuffer, wgpu::Buffer resBuffer, wgpu::Buffer outputBuffer, const Params& params) {
    wgpu::BindGroupEntry shapeEntry = {};
    shapeEntry.binding = 0;
    shapeEntry.buffer = shapeBuffer;
    shapeEntry.offset = 0;
    shapeEntry.size = sizeof(int) * params.shape.size();

    wgpu::BindGroupEntry resEntry = {};
    resEntry.binding = 1;
    resEntry.buffer = resBuffer;
    resEntry.offset = 0;
    resEntry.size = sizeof(float) * params.res.size();

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 2;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry entries[] = {shapeEntry, resEntry, outputEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

std::vector<float> c_gamma(WebGPUContext& context, const std::vector<float>& res, const std::vector<int>& shape) {
    // Calculate the total number of elements in the output buffer
    size_t num_elements = 1;
    for (int dim : shape) {
        num_elements *= dim;
    }
    buffer_len = num_elements;
    std::vector<float> outputData(buffer_len, 0.0f);
    Params params = {res, shape};

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;
    if (!queue){
        return {};
    }

    // LOADING AND COMPILING SHADER CODE
    std::string shaderCode = readShaderFile("src/c_gamma/c_gamma.wgsl");
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS FOR C_GAMMA
    wgpu::Buffer shapeBuffer = createBuffer(device, params.shape.data(), sizeof(int) * params.shape.size(), 
    wgpu::BufferUsage::Storage);
    wgpu::Buffer resBuffer = createBuffer(device, params.res.data(), sizeof(float) * params.res.size(), 
    wgpu::BufferUsage::Storage);
    wgpu::Buffer outputBuffer = createBuffer(device, outputData.data(), sizeof(float) * buffer_len, 
    static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, shapeBuffer, resBuffer, outputBuffer, params);
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
    computePass.dispatchWorkgroups(4, 4, 4);
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
    shapeBuffer.release();
    resBuffer.release();
    outputBuffer.release();
    shaderModule.release();
    readbackBuffer.release();
    commandBuffer.release();
    commandBuffer2.release();

    return output;
}