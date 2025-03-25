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

static size_t uf_buffer_len;
static size_t ub_buffer_len;

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

    wgpu::BindGroupLayoutEntry entries[] = {ufBufferLayout, ubBufferLayout, newUFBufferLayout, newUBBufferLayout, uniformBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 5;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

static wgpu::BindGroup createBindGroup(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer ufBuffer, wgpu::Buffer ubBuffer, wgpu::Buffer newUFBuffer, wgpu::Buffer newUBBuffer, wgpu::Buffer uniformBuffer) {
    wgpu::BindGroupEntry ufEntry = {};
    ufEntry.binding = 0;
    ufEntry.buffer = ufBuffer;
    ufEntry.offset = 0;
    ufEntry.size = sizeof(float) * uf_buffer_len;

    wgpu::BindGroupEntry ubEntry = {};
    ubEntry.binding = 1;
    ubEntry.buffer = ubBuffer;
    ubEntry.offset = 0;
    ubEntry.size = sizeof(float) * ub_buffer_len;

    wgpu::BindGroupEntry newUFEntry = {};
    newUFEntry.binding = 2;
    newUFEntry.buffer = newUFBuffer;
    newUFEntry.offset = 0;
    newUFEntry.size = sizeof(float) * uf_buffer_len;

    wgpu::BindGroupEntry newUBEntry = {};
    newUBEntry.binding = 3;
    newUBEntry.buffer = newUBBuffer;
    newUBEntry.offset = 0;
    newUBEntry.size = sizeof(float) * ub_buffer_len;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 4;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);

    wgpu::BindGroupEntry entries[] = {ufEntry, ubEntry, newUFEntry, newUBEntry, uniformEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 5;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void diffract(WebGPUContext& context, wgpu::Buffer& newUFBuffer, wgpu::Buffer& newUBBuffer, std::vector<float> uf, std::vector<float> ub, std::optional<std::vector<float>> res, std::optional<float> dz) {
    // cgamma call
    assert(uf.size() == ub.size() && "uf and ub must have the same shape");
    uf_buffer_len = uf.size();
    ub_buffer_len = ub.size();
    Params params = {dz.value()};

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    // wgpu::Queue queue = context.queue;
    
    // LOADING AND COMPILING SHADER CODE
    std::string shaderCode = readShaderFile("src/diffract/diffract.wgsl");
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS FOR DIFFRACT
    wgpu::Buffer cgammaBuffer = createBuffer(context.device, nullptr, sizeof(float) * uf.size(), static_cast<WGPUBufferUsage>(wgpu::BufferUsage::Storage));
    c_gamma(context, cgammaBuffer, res.value(), {int(uf.size())});
    wgpu::Buffer ufBuffer = createBuffer(device, uf.data(), sizeof(float) * uf.size(), wgpu::BufferUsage::Storage);
    wgpu::Buffer ubBuffer = createBuffer(device, ub.data(), sizeof(float) * ub.size(), wgpu::BufferUsage::Storage);
    wgpu::Buffer uniformBuffer = createBuffer(device, &params, sizeof(Params), wgpu::BufferUsage::Uniform);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(device, bindGroupLayout, ufBuffer, ubBuffer, newUFBuffer, newUBBuffer, uniformBuffer);

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
    bindGroup.release();
    bindGroupLayout.release();
    cgammaBuffer.release();
    ufBuffer.release();
    ubBuffer.release();
    uniformBuffer.release();
    shaderModule.release();
    // computePipeline.release();
    // commandBuffer.release();
}