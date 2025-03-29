#ifndef WEBGPU_UTILS_H
#define WEBGPU_UTILS_H
#include <webgpu/webgpu.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

struct WebGPUContext {
    wgpu::Instance instance = nullptr;
    wgpu::Adapter adapter = nullptr;
    wgpu::Device device = nullptr;
    wgpu::Queue queue = nullptr;
};

struct WorkgroupLimits {
    double maxWorkgroupSizeX;
    double maxWorkgroupSizeY;
    double maxWorkgroupSizeZ;
    double maxWorkgroupsPerDimension;
};

// Initializes WebGPU
void initWebGPU(WebGPUContext& context);

WorkgroupLimits getWorkgroupLimits(wgpu::Device& device);

// Reads shader source code from a file
std::string readShaderFile(const std::string& filename);

// Creates a WebGPU shader module from WGSL source code
wgpu::ShaderModule createShaderModule(wgpu::Device& device, const std::string& shaderCode);

// Creates a WebGPU buffer
wgpu::Buffer createBuffer(wgpu::Device& device, const void* data, size_t size, wgpu::BufferUsage usage);

// Compute pipeline utilities
wgpu::ComputePipeline createComputePipeline(wgpu::Device& device, wgpu::ShaderModule shaderModule, wgpu::BindGroupLayout bindGroupLayout);

// Readback from GPU to CPU
std::vector<float> readBack(wgpu::Device& device, wgpu::Queue& queue, size_t buffer_len, wgpu::Buffer& outputBuffer);
std::vector<uint32_t> readBack2(wgpu::Device& device, wgpu::Queue& queue, size_t buffer_len, wgpu::Buffer& outputBuffer);

#endif