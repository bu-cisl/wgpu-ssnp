#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#define WEBGPU_CPP_IMPLEMENTATION
#include <webgpu/webgpu.hpp>
#include "c_gamma.h"
#include "webgpu_utils.h"

// INPUT PARAMS
struct Params {
    float res;
    int shape;
};

size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
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

wgpu::BindGroup createBindGroup(wgpu::Device& device, wgpu::BindGroupLayout bindGroupLayout, wgpu::Buffer uniformBuffer, wgpu::Buffer outputBuffer) {
    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 0;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 1;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = buffer_len * sizeof(float);

    wgpu::BindGroupEntry entries[] = {uniformEntry, outputEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 2;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

