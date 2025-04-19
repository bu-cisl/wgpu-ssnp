#include "split_prop.h"

static size_t buffer_len;
static size_t res_buffer_len;

// CREATING BIND GROUP LAYOUT
static wgpu::BindGroupLayout createBindGroupLayout(wgpu::Device& device) {
    wgpu::BindGroupLayoutEntry ufBufferLayout = {};
    ufBufferLayout.binding = 0;
    ufBufferLayout.visibility = wgpu::ShaderStage::Compute;
    ufBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry ubBufferLayout = {};
    ubBufferLayout.binding = 1;
    ubBufferLayout.visibility = wgpu::ShaderStage::Compute;
    ubBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry resBufferLayout = {};
    resBufferLayout.binding = 2;
    resBufferLayout.visibility = wgpu::ShaderStage::Compute;
    resBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry cgammaBufferLayout = {};
    cgammaBufferLayout.binding = 3;
    cgammaBufferLayout.visibility = wgpu::ShaderStage::Compute;
    cgammaBufferLayout.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    wgpu::BindGroupLayoutEntry ufNewBufferLayout = {};
    ufNewBufferLayout.binding = 4;
    ufNewBufferLayout.visibility = wgpu::ShaderStage::Compute;
    ufNewBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry ubNewBufferLayout = {};
    ubNewBufferLayout.binding = 5;
    ubNewBufferLayout.visibility = wgpu::ShaderStage::Compute;
    ubNewBufferLayout.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry entries[] = {
        ufBufferLayout,
        ubBufferLayout,
        resBufferLayout,
        cgammaBufferLayout,
        ufNewBufferLayout,
        ubNewBufferLayout
    };

    wgpu::BindGroupLayoutDescriptor layoutDesc = {};
    layoutDesc.entryCount = 6;
    layoutDesc.entries = entries;

    return device.createBindGroupLayout(layoutDesc);
}

// CREATING BIND GROUP
static wgpu::BindGroup createBindGroup(
    wgpu::Device& device, 
    wgpu::BindGroupLayout bindGroupLayout,
    wgpu::Buffer& ufBuffer,
    wgpu::Buffer& ubBuffer,
    wgpu::Buffer& resBuffer,
    wgpu::Buffer& cgammaBuffer,
    wgpu::Buffer& ufNewBuffer,
    wgpu::Buffer& ubNewBuffer
) {
    wgpu::BindGroupEntry ufEntry = {};
    ufEntry.binding = 0;
    ufEntry.buffer = ufBuffer;
    ufEntry.offset = 0;
    ufEntry.size = sizeof(float) * 2 * buffer_len;  // ×2 for complex numbers

    wgpu::BindGroupEntry ubEntry = {};
    ubEntry.binding = 1;
    ubEntry.buffer = ubBuffer;
    ubEntry.offset = 0;
    ubEntry.size = sizeof(float) * 2 * buffer_len;

    wgpu::BindGroupEntry resEntry = {};
    resEntry.binding = 2;
    resEntry.buffer = resBuffer;
    resEntry.offset = 0;
    resEntry.size = sizeof(float) * res_buffer_len; 

    wgpu::BindGroupEntry cgammaEntry = {};
    cgammaEntry.binding = 3;
    cgammaEntry.buffer = cgammaBuffer;
    cgammaEntry.offset = 0;
    cgammaEntry.size = sizeof(float) * buffer_len;

    wgpu::BindGroupEntry ufNewEntry = {};
    ufNewEntry.binding = 4;
    ufNewEntry.buffer = ufNewBuffer;
    ufNewEntry.offset = 0;
    ufNewEntry.size = sizeof(float) * 2 * buffer_len;

    wgpu::BindGroupEntry ubNewEntry = {};
    ubNewEntry.binding = 5;
    ubNewEntry.buffer = ubNewBuffer;
    ubNewEntry.offset = 0;
    ubNewEntry.size = sizeof(float) * 2 * buffer_len;

    wgpu::BindGroupEntry entries[] = {
        ufEntry,
        ubEntry,
        resEntry,
        cgammaEntry,
        ufNewEntry,
        ubNewEntry
    };

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = bindGroupLayout;
    bindGroupDesc.entryCount = 6;
    bindGroupDesc.entries = entries;

    return device.createBindGroup(bindGroupDesc);
}

void split_prop(
    WebGPUContext& context,
    wgpu::Buffer& ufNewBuffer,
    wgpu::Buffer& ubNewBuffer,
    std::vector<std::complex<float>> uf,
    std::vector<std::complex<float>> ub,
    std::vector<int> shape,
    std::optional<std::vector<float>> res
) {
    assert(uf.size() == ub.size() && "uf and ub must have the same shape");
    buffer_len = uf.size();
    res_buffer_len = res.value().size();

    // INITIALIZING WEBGPU
    wgpu::Device device = context.device;
    wgpu::Queue queue = context.queue;
    
    // LOADING AND COMPILING SHADER CODE
    WorkgroupLimits limits = getWorkgroupLimits(device);
    std::string shaderCode = readShaderFile("src/split_prop/split_prop.wgsl", limits.maxWorkgroupSizeX);
    wgpu::ShaderModule shaderModule = createShaderModule(device, shaderCode);

    // CREATING BUFFERS
    wgpu::Buffer cgammaBuffer = createBuffer(context.device, nullptr, sizeof(float) * buffer_len, wgpu::BufferUsage::Storage);
    c_gamma(context, cgammaBuffer, res.value(), shape);

    // Flatten uf and ub
    std::vector<float> ufFlat(buffer_len * 2);
    std::vector<float> ubFlat(buffer_len * 2);
    for (size_t i = 0; i < buffer_len; ++i) {
        ufFlat[2*i] = uf[i].real();
        ufFlat[2*i + 1] = uf[i].imag();
        ubFlat[2*i] = ub[i].real();
        ubFlat[2*i + 1] = ub[i].imag();
    }

    wgpu::Buffer ufBuffer = createBuffer(device, ufFlat.data(), sizeof(float) * ufFlat.size(), wgpu::BufferUsage::Storage);
    wgpu::Buffer ubBuffer = createBuffer(device, ubFlat.data(), sizeof(float) * ubFlat.size(), wgpu::BufferUsage::Storage);
    wgpu::Buffer resBuffer = createBuffer(device, res.value().data(), sizeof(float)*res_buffer_len, wgpu::BufferUsage::Storage);

    // CREATING BIND GROUP AND LAYOUT
    wgpu::BindGroupLayout bindGroupLayout = createBindGroupLayout(device);
    wgpu::BindGroup bindGroup = createBindGroup(
        device,
        bindGroupLayout,
        ufBuffer,
        ubBuffer,
        resBuffer,
        cgammaBuffer,
        ufNewBuffer,
        ubNewBuffer
    );

    // CREATING COMPUTE PIPELINE
    wgpu::ComputePipeline computePipeline = createComputePipeline(device, shaderModule, bindGroupLayout);

    // ENCODING AND DISPATCHING COMPUTE COMMANDS
    uint32_t workgroupsX = std::ceil(double(buffer_len)/limits.maxWorkgroupSizeX);
    wgpu::CommandBuffer commandBuffer = createComputeCommandBuffer(device, computePipeline, bindGroup, workgroupsX);
    queue.submit(1, &commandBuffer);

    // RELEASE RESOURCES
    commandBuffer.release();
    computePipeline.release();
    bindGroup.release();
    bindGroupLayout.release();
    shaderModule.release();
    cgammaBuffer.release();
    ufBuffer.release();
    ubBuffer.release();
    resBuffer.release();
}