#ifndef WEBGPU_UTILS_H
#define WEBGPU_UTILS_H
#include <webgpu/webgpu.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

struct WebGPUContext {
    wgpu::Instance instance = nullptr;
    wgpu::Adapter adapter = nullptr;
    wgpu::Device device = nullptr;
    wgpu::Queue queue = nullptr;
};

// Initializes WebGPU
bool initWebGPU(WebGPUContext& context);

// Reads shader source code from a file
std::string readShaderFile(const std::string& filename);

// Creates a WebGPU shader module from WGSL source code
wgpu::ShaderModule createShaderModule(wgpu::Device& device, const std::string& shaderCode);

// Creates a WebGPU buffer
wgpu::Buffer createBuffer(wgpu::Device& device, const void* data, size_t size, wgpu::BufferUsage usage);

// Compute pipeline utilities
wgpu::ComputePipeline createComputePipeline(wgpu::Device& device, wgpu::ShaderModule shaderModule, wgpu::BindGroupLayout bindGroupLayout);

#endif