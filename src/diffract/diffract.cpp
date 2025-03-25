#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <webgpu/webgpu.hpp>
#include "diffract.h"
#include "../webgpu_utils.h"
#include "../c_gamma/c_gamma.h"

// INPUT PARAMS
struct Params {
    float dz;
};

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry ufBufferLayout = {};
    ufBufferLayout.binding = 0;
    ufBufferLayout.visibility = wgpu::ShaderStage::Compute;
    ufBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry ubBufferLayout = {};
    ubBufferLayout.binding = 1;
    ubBufferLayout.visibility = wgpu::ShaderStage::Compute;
    ubBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry newUFBufferLayout = {};
    newUFBufferLayout.binding = 2;
    newUFBufferLayout.visibility = wgpu::ShaderStage::Compute;
    newUFBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry newUBBufferLayout = {};
    newUBBufferLayout.binding = 3;
    newUBBufferLayout.visibility = wgpu::ShaderStage::Compute;
    newUBBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 4;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {ufBufferLayout, ubBufferLayout, uniformBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 5;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

// static wgpu::BindGroup createBindGroup(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer ufBuffer, wgpu::Buffer ubBuffer, wgpu::Buffer newUFBuffer, wgpu::Buffer newUBBuffer, wgpu::Buffer uniformBuffer) {
//     wgpu::BindGroupEntry shapeEntry = {};
//     shapeEntry.binding = 0;
//     shapeEntry.buffer = shapeBuffer;
//     shapeEntry.offset = 0;
//     shapeEntry.size = sizeof(int) * params.shape.size();

//     wgpu::BindGroupEntry resEntry = {};
//     resEntry.binding = 1;
//     resEntry.buffer = resBuffer;
//     resEntry.offset = 0;
//     resEntry.size = sizeof(float) * params.res.size();

//     wgpu::BindGroupEntry outputEntry = {};
//     outputEntry.binding = 2;
//     outputEntry.buffer = outputBuffer;
//     outputEntry.offset = 0;
//     outputEntry.size = sizeof(float) * buffer_len;

//     wgpu::BindGroupEntry entries[] = {shapeEntry, resEntry, outputEntry};

//     wgpu::BindGroupDescriptor bindGroupDesc = {};
//     bindGroupDesc.layout = bindGroupLayout;
//     bindGroupDesc.entryCount = 3;
//     bindGroupDesc.entries = entries;

//     return device.createBindGroup(bindGroupDesc);
// }

void diffract(WebGPUContext& context, wgpu::Buffer& newUFBuffer, wgpu::Buffer& newUBBuffer, std::vector<float> uf, std::vector<float> ub, std::optional<std::vector<float>> res, std::optional<float> dz) {
    // cgamma call
    assert(uf.size() == ub.size() && "uf and ub must have the same shape");

    Params params = {dz.value()};

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;
    
    // LOADING AND COMPILING SHADER CODE
    std::string shaderCode = readShaderFile("src/diffract/diffract.wgsl");
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS FOR DIFFRACT
    wgpu::Buffer cgammaBuffer = createBuffer(context.device, nullptr, sizeof(float) * uf.size(), static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc));
    c_gamma(context, cgammaBuffer, res.value(), {float(uf.size())});
    wgpu::Buffer ufBuffer = createBuffer(device, uf.data(), sizeof(float) * uf.size(), wgpu::BufferUsage::Storage);
    wgpu::Buffer ubBuffer = createBuffer(device, ub.data(), sizeof(float) * ub.size(), wgpu::BufferUsage::Storage);
    wgpu::Buffer uniformBuffer = createBuffer(device, &params, sizeof(Params), wgpu::BufferUsage::Uniform);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    // wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, ufBuffer, ubBuffer, newUFBuffer, newUBBuffer, uniformBuffer);

    // CREATING COMPUTE PIPELINE
    // wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);

    // ENCODING AND DISPATCHING COMPUTE COMMANDS
    // wgpu::CommandEncoderDescriptor encoderDesc = {};
    // wgpu::CommandEncoder commandEncoder = device.createCommandEncoder(encoderDesc);

    // wgpu::ComputePassDescriptor computePassDesc = {};
    // wgpu::ComputePassEncoder computePass = commandEncoder.beginComputePass(computePassDesc);
    // computePass.setPipeline(computePipeline);
    // computePass.setBindGroup(0, bindGroup, 0, nullptr);
    // computePass.dispatchWorkgroups(64,1,1);
    // computePass.end();

    // wgpu::CommandBufferDescriptor cmdBufferDesc = {};
    // wgpu::CommandBuffer commandBuffer = commandEncoder.finish(cmdBufferDesc);

    // queue.submit(1, &commandBuffer);

    // RELEASE RESOURCES
    // computePipeline.release();
    // bindGroup.release();
    // bindGroupLayout.release();
    cgammaBuffer.release();
    // shaderModule.release();
    // commandBuffer.release();
}