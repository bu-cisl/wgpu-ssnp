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

wgpu::BindGroup createBindGroup2(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer shapeBuffer, wgpu::Buffer resBuffer, wgpu::Buffer outputBuffer, const Params& params) {
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
    outputEntry.size = sizeof(float) * buffer_len2;

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
    buffer_len2 = num_elements;
    std::vector<float> outputData(buffer_len2, 0.0f);
    Params params = {res, shape};

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;
    if (!queue){
        return {};
    }

    // LOADING AND COMPILING SHADER CODE
    std::string shaderCode = readShaderFile("src/c_gamma.wgsl");
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS FOR C_GAMMA (EDIT)
    std::vector<float> output = {};

    // Release resources
    shaderModule.release();

    return output;
}