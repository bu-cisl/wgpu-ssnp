#include "pipeline.h"

namespace ssnp {

// FLATTENING A REAL SLICE FOR GPU UPLOAD
std::vector<float> flatten_real_slice(const std::vector<std::vector<float>>& slice) {
    std::vector<float> flatSlice;
    for (const auto& row : slice) {
        flatSlice.insert(flatSlice.end(), row.begin(), row.end());
    }
    return flatSlice;
}

// CREATING A COMPLEX BUFFER FROM A REAL SLICE
wgpu::Buffer create_complex_slice_buffer(WebGPUContext& context, const std::vector<std::vector<float>>& slice) {
    std::vector<float> flatSlice = flatten_real_slice(slice);
    std::vector<float> complexSlice;
    complexSlice.reserve(flatSlice.size() * 2);
    for (float value : flatSlice) {
        complexSlice.push_back(value);
        complexSlice.push_back(0.0f);
    }

    return createBuffer(
        context.device,
        complexSlice.data(),
        sizeof(float) * complexSlice.size(),
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
}

// RELEASING SSNP STATE BUFFERS
void release_state(SSNPState& state) {
    state.U.release();
    state.UD.release();
}

// INITIALIZING THE INCIDENT SSNP STATE FOR ONE ANGLE
SSNPState initialize_angle_state(
    WebGPUContext& context,
    const std::vector<float>& c_ba,
    const std::vector<int>& shape,
    const std::vector<float>& res
) {
    size_t buffer_len = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]);

    wgpu::Buffer tiltResultBuffer = createBuffer(
        context.device,
        nullptr,
        sizeof(float) * buffer_len * 2,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    tilt(context, tiltResultBuffer, c_ba, shape, res);

    wgpu::Buffer forwardBuffer = createBuffer(
        context.device,
        nullptr,
        sizeof(float) * buffer_len * 2,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    fft(context, forwardBuffer, tiltResultBuffer, buffer_len, shape[0], shape[1], 0);
    tiltResultBuffer.release();

    std::vector<float> backward(buffer_len * 2, 0.0f);
    wgpu::Buffer backwardBuffer = createBuffer(
        context.device,
        backward.data(),
        sizeof(float) * buffer_len * 2,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );

    SSNPState state = {
        createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)),
        createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc))
    };
    merge_prop(context, state.U, state.UD, forwardBuffer, backwardBuffer, buffer_len, shape, res);

    forwardBuffer.release();
    backwardBuffer.release();
    return state;
}

// PROPAGATING THE SSNP STATE THROUGH THE VOLUME
SSNPState propagate_to_object_exit(
    WebGPUContext& context,
    SSNPState state,
    const std::vector<std::vector<std::vector<float>>>& n,
    const std::vector<int>& shape,
    const std::vector<float>& res,
    float n0
) {
    size_t buffer_len = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]);

    for (const auto& slice : n) {
        SSNPState diffracted = {
            createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)),
            createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc))
        };
        diffract(context, diffracted.U, diffracted.UD, state.U, state.UD, buffer_len, shape, res, 1.0f);
        release_state(state);

        wgpu::Buffer uBuffer = createBuffer(
            context.device,
            nullptr,
            sizeof(float) * buffer_len * 2,
            WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
        );
        fft(context, uBuffer, diffracted.U, buffer_len, shape[0], shape[1], 1);

        wgpu::Buffer sliceBuffer = create_complex_slice_buffer(context, slice);
        wgpu::Buffer scatterBuffer = createBuffer(
            context.device,
            nullptr,
            sizeof(float) * buffer_len * 2,
            WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
        );
        scatter_factor(context, scatterBuffer, sliceBuffer, buffer_len, res[0], 1.0f, n0);

        wgpu::Buffer scatteredUD = createBuffer(
            context.device,
            nullptr,
            sizeof(float) * buffer_len * 2,
            WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
        );
        scatter_effects(context, scatteredUD, scatterBuffer, uBuffer, diffracted.UD, buffer_len, shape);

        uBuffer.release();
        sliceBuffer.release();
        scatterBuffer.release();
        diffracted.UD.release();

        state = {diffracted.U, scatteredUD};
    }

    return state;
}

// PROJECTING AN OBJECT-EXIT STATE TO THE SENSOR FIELD
wgpu::Buffer project_state_to_sensor_field(
    WebGPUContext& context,
    const SSNPState& state,
    const std::vector<int>& shape,
    const std::vector<float>& res,
    float na,
    float focal_offset
) {
    size_t buffer_len = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]);

    SSNPState focalState = {
        createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)),
        createBuffer(context.device, nullptr, sizeof(float) * buffer_len * 2, WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc))
    };
    diffract(context, focalState.U, focalState.UD, const_cast<wgpu::Buffer&>(state.U), const_cast<wgpu::Buffer&>(state.UD), buffer_len, shape, res, focal_offset);

    wgpu::Buffer forwardBuffer = createBuffer(
        context.device,
        nullptr,
        sizeof(float) * buffer_len * 2,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    wgpu::Buffer backwardBuffer = createBuffer(
        context.device,
        nullptr,
        sizeof(float) * buffer_len * 2,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    split_prop(context, forwardBuffer, backwardBuffer, focalState.U, focalState.UD, buffer_len, shape, res);

    release_state(focalState);
    backwardBuffer.release();

    wgpu::Buffer pupilBuffer = createBuffer(
        context.device,
        nullptr,
        sizeof(int) * buffer_len,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    binary_pupil(context, pupilBuffer, shape, na, res);

    wgpu::Buffer filteredForward = createBuffer(
        context.device,
        nullptr,
        sizeof(float) * buffer_len * 2,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    mult(context, filteredForward, forwardBuffer, pupilBuffer, buffer_len);

    forwardBuffer.release();
    pupilBuffer.release();

    wgpu::Buffer fieldBuffer = createBuffer(
        context.device,
        nullptr,
        sizeof(float) * buffer_len * 2,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    fft(context, fieldBuffer, filteredForward, buffer_len, shape[0], shape[1], 1);
    filteredForward.release();

    return fieldBuffer;
}

}
