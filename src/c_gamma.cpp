#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <webgpu/webgpu.hpp>
#include "c_gamma.h"
#include "webgpu_utils.h"

// INPUT PARAMS
struct Params {
    std::vector<float> res;
    std::vector<int> shape;
};

size_t buffer_len2;

// CREATING BIND GROUP AND LAYOUT
wgpu::BindGroupLayout createBindGroupLayout2(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 0;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry outputBufferLayout = {};
    outputBufferLayout.binding = 1;
    outputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry entries[] = {uniformBufferLayout, outputBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 2;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

wgpu::BindGroup createBindGroup2(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer uniformBuffer, wgpu::Buffer outputBuffer, const Params& params) {
    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 0;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(float) * params.res.size() + sizeof(int) * params.shape.size();

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 1;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * buffer_len2;

    wgpu::BindGroupEntry entries[] = {uniformEntry, outputEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 2;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

std::vector<float> c_gamma(WebGPUContext& context, const std::vector<float>& res, const std::vector<int>& shape) {
    // (EDIT)
    size_t num_elements = 1;
    for (int dim : shape) {
        num_elements *= dim;
    }
    buffer_len2 = num_elements;
    std::vector<float> outputData(buffer_len2, 0.0f);
    Params params = {res, shape};
    size_t uniformBufferSize = sizeof(float) * params.res.size() + sizeof(int) * params.shape.size();

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    // LOADING AND COMPILING SHADER CODE
    std::string shaderCode = readShaderFile("src/c_gamma.wgsl");
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS FOR C_GAMMA (EDIT)
    wgpu::Buffer uniformBuffer = createBuffer(device, nullptr, uniformBufferSize, wgpu::BufferUsage::Uniform);
    wgpu::Buffer outputBuffer = createBuffer(device, nullptr, outputData.size() * sizeof(float), static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout2(device);
    wgpu::BindGroup bindGroup = createBindGroup2(device, bindGroupLayout, uniformBuffer, outputBuffer, params);
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

    uint32_t workgroupCountX = (shape[0] + 3) / 4;
    uint32_t workgroupCountY = (shape[1] + 3) / 4;
    uint32_t workgroupCountZ = (shape[2] + 3) / 4;
    computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
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
    readbackBuffer.mapAsync(wgpu::MapMode::Read, 0, outputData.size() * sizeof(float), [&](wgpu::BufferMapAsyncStatus status) {
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

    // Release resources
    computePipeline.release();
    bindGroup.release();
    bindGroupLayout.release();
    uniformBuffer.release();
    outputBuffer.release();
    shaderModule.release();
    readbackBuffer.release();
    commandBuffer.release();
    commandBuffer2.release();

    return output;
}