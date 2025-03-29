#include "binary_pupil.h"

// INPUT PARAMS
struct Params {
    float na;
};

static size_t buffer_len;

// CREATING BIND GROUP AND LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry cgammaBufferLayout = {};
    cgammaBufferLayout.binding = 0;
    cgammaBufferLayout.visibility = wgpu::ShaderStage::Compute;
    cgammaBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry maskBufferLayout = {};
    maskBufferLayout.binding = 1;
    maskBufferLayout.visibility = wgpu::ShaderStage::Compute;
    maskBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 2;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {cgammaBufferLayout, maskBufferLayout, uniformBufferLayout};

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 3;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

static wgpu::BindGroup createBindGroup(
    wgpu::Device& device, wgpu::BindGroupLayout, 
    wgpu::Buffer cgammaBuffer, wgpu::Buffer maskBuffer, wgpu::Buffer uniformBuffer
) {
    wgpu::BindGroupEntry cgammaEntry = {};
    cgammaEntry.binding = 0;
    cgammaEntry.buffer = cgammaBuffer;
    cgammaEntry.offset = 0;
    cgammaEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry maskEntry = {};
    maskEntry.binding = 1;
    maskEntry.buffer = maskBuffer;
    maskEntry.offset = 0;
    maskEntry.size = sizeof(uint32_t) * buffer_len;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 2;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);

    wgpu::BindGroupEntry entries[] = {cgammaEntry, maskEntry, uniformEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = layout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void binary_pupil(
    WebGPUContext& context,
    wgpu::Buffer& maskBuffer,
    std::optional<std::vector<float>> res,
    std::optional<float> na,
    std::vector<int32_t> shape
) {

}