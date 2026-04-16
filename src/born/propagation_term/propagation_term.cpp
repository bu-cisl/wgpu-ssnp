#include "propagation_term.h"

#include <cmath>

namespace born {

struct Params {
    float dims[4];    // height, width, shift_y, shift_x
    float physics[4]; // res_z, res_y, res_x, depth
    float angle[4];   // c_ba[0], c_ba[1], 0, 0
};

static size_t buffer_len;

static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry inputBufferLayout = {};
    inputBufferLayout.binding = 0;
    inputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    inputBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry outputBufferLayout = {};
    outputBufferLayout.binding = 1;
    outputBufferLayout.visibility = wgpu::ShaderStage::Compute;
    outputBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry uniformBufferLayout = {};
    uniformBufferLayout.binding = 2;
    uniformBufferLayout.visibility = wgpu::ShaderStage::Compute;
    uniformBufferLayout.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry entries[] = {
        inputBufferLayout,
        outputBufferLayout,
        uniformBufferLayout
    };

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 3;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

static wgpu::BindGroup createBindGroup(
    wgpu::Device& device,
    wgpu::BindGroupLayout bindGroupLayout,
    wgpu::Buffer inputBuffer,
    wgpu::Buffer outputBuffer,
    wgpu::Buffer uniformBuffer
) {
    wgpu::BindGroupEntry inputEntry = {};
    inputEntry.binding = 0;
    inputEntry.buffer = inputBuffer;
    inputEntry.offset = 0;
    inputEntry.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry outputEntry = {};
    outputEntry.binding = 1;
    outputEntry.buffer = outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = sizeof(float) * buffer_len * 2;

    wgpu::BindGroupEntry uniformEntry = {};
    uniformEntry.binding = 2;
    uniformEntry.buffer = uniformBuffer;
    uniformEntry.offset = 0;
    uniformEntry.size = sizeof(Params);

    wgpu::BindGroupEntry entries[] = {inputEntry, outputEntry, uniformEntry};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void propagation_term(
    WebGPUContext& context,
    wgpu::Buffer& outputBuffer,
    wgpu::Buffer& inputBuffer,
    size_t bufferlen,
    std::vector<int> shape,
    std::vector<float> res,
    std::vector<float> c_ba,
    float depth
) {
    buffer_len = bufferlen;

    const float shift_y = std::round(c_ba[0] * res[1] * float(shape[0]));
    const float shift_x = std::round(c_ba[1] * res[2] * float(shape[1]));
    Params params = {
        {float(shape[0]), float(shape[1]), shift_y, shift_x},
        {res[0], res[1], res[2], depth},
        {c_ba[0], c_ba[1], 0.0f, 0.0f},
    };

    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;

    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile(
        "src/born/propagation_term/propagation_term.wgsl",
        limits.maxWorkgroupSizeX
    );
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);
    wgpu::Buffer uniformBuffer = createBuffer(
        device,
        &params,
        sizeof(Params),
        wgpu::BufferUsage::Uniform
    );

    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(
        device,
        bindGroupLayout,
        inputBuffer,
        outputBuffer,
        uniformBuffer
    );

    wgpu::ComputePipeline computePipeline = createComputePipeline(
        device,
        shaderModule,
        bindGroupLayout
    );

    uint32_t workgroupsX = std::ceil(double(buffer_len) / limits.maxWorkgroupSizeX);
    wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(
        device,
        computePipeline,
        bindGroup,
        workgroupsX
    );
    queue.submit(1, &commandBuffer);

    commandBuffer.release();
    computePipeline.release();
    bindGroup.release();
    bindGroupLayout.release();
    shaderModule.release();
    uniformBuffer.release();
}

} // namespace born
