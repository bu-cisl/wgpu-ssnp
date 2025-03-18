#ifndef WEBGPU_UTILS_H
#define WEBGPU_UTILS_H
#include <webgpu/webgpu.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// Initializes WebGPU
bool init_wgpu(wgpu::Instance&, wgpu::Adapter&, wgpu::Device&, wgpu::Queue&);

#endif