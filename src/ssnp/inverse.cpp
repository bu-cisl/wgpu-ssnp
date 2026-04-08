#include "inverse.h"
#include <stdexcept>

namespace ssnp {
namespace {

// COMPUTING THE PER-ANGLE MSE LOSS
float mean_squared_loss(
    const std::vector<float>& predicted,
    const std::vector<std::vector<float>>& measured
) {
    float loss = 0.0f;
    size_t width = measured[0].size();
    for (size_t i = 0; i < predicted.size(); ++i) {
        size_t row = i / width;
        size_t col = i % width;
        float residual = predicted[i] - measured[row][col];
        loss += residual * residual;
    }
    return 0.5f * loss / static_cast<float>(predicted.size());
}

// CREATING A ZERO-INITIALIZED GRADIENT VOLUME
std::vector<std::vector<std::vector<float>>> zeros_like(const std::vector<std::vector<std::vector<float>>>& volume) {
    return std::vector<std::vector<std::vector<float>>>(
        volume.size(),
        std::vector<std::vector<float>>(
            volume[0].size(),
            std::vector<float>(volume[0][0].size(), 0.0f)
        )
    );
}

// ACCUMULATING A FLAT SLICE INTO THE 3D GRADIENT VOLUME
void accumulate_slice(
    std::vector<std::vector<std::vector<float>>>& grad_volume,
    size_t z,
    const std::vector<float>& flat_slice
) {
    size_t width = grad_volume[z][0].size();
    for (size_t row = 0; row < grad_volume[z].size(); ++row) {
        for (size_t col = 0; col < width; ++col) {
            grad_volume[z][row][col] += flat_slice[row * width + col];
        }
    }
}

// CREATING COMPLEX TEMPORARY BUFFERS
wgpu::Buffer make_complex_buffer(WebGPUContext& context, size_t buffer_len) {
    return createBuffer(
        context.device,
        nullptr,
        sizeof(float) * buffer_len * 2,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
}

// CREATING REAL TEMPORARY BUFFERS
wgpu::Buffer make_real_buffer(WebGPUContext& context, size_t buffer_len) {
    return createBuffer(
        context.device,
        nullptr,
        sizeof(float) * buffer_len,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
}

} // namespace

ReconstructionResult reconstruct(
    WebGPUContext& context,
    const std::vector<std::vector<std::vector<float>>>& measured,
    const std::vector<std::vector<float>>& angles,
    std::vector<std::vector<std::vector<float>>> initial_volume,
    const std::vector<float>& res,
    float na,
    float n0,
    int iterations,
    float learning_rate
) {
    if (measured.size() != angles.size()) {
        throw std::runtime_error("Measured intensity stack and angle list must have the same length.");
    }
    if (measured.empty() || initial_volume.empty()) {
        throw std::runtime_error("Measured data and initial volume must be non-empty.");
    }

    std::vector<int> shape = {int(initial_volume[0].size()), int(initial_volume[0][0].size())};
    size_t buffer_len = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]);
    float inv_pixels = 1.0f / static_cast<float>(buffer_len);

    ReconstructionResult result;
    result.volume = std::move(initial_volume);

    // RUNNING THE OUTER GRADIENT-DESCENT LOOP
    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<std::vector<std::vector<float>>> grad_volume = zeros_like(result.volume);
        float total_loss = 0.0f;

        // ACCUMULATING GRADIENTS OVER ANGLES
        for (size_t angle_idx = 0; angle_idx < angles.size(); ++angle_idx) {
            // FORWARD MODEL FOR THE CURRENT VOLUME ESTIMATE
            SSNPState exitState = propagate_to_object_exit(
                context,
                initialize_angle_state(context, angles[angle_idx], shape, res),
                result.volume,
                shape,
                res,
                n0
            );

            wgpu::Buffer fieldBuffer = project_state_to_sensor_field(
                context,
                exitState,
                shape,
                res,
                na,
                -static_cast<float>(result.volume.size()) / 2.0f
            );

            wgpu::Buffer predictedIntensityBuffer = make_real_buffer(context, buffer_len);
            intense(context, predictedIntensityBuffer, fieldBuffer, buffer_len, true);
            std::vector<float> predictedIntensity = readBack(context.device, context.queue, buffer_len, predictedIntensityBuffer);
            total_loss += mean_squared_loss(predictedIntensity, measured[angle_idx]);
            predictedIntensityBuffer.release();

            // FORMING THE SENSOR-PLANE LOSS GRADIENT
            std::vector<float> measuredFlat = flatten_real_slice(measured[angle_idx]);
            wgpu::Buffer measuredBuffer = createBuffer(
                context.device,
                measuredFlat.data(),
                sizeof(float) * buffer_len,
                WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
            );

            wgpu::Buffer fieldGrad = make_complex_buffer(context, buffer_len);
            intensity_grad(context, fieldGrad, fieldBuffer, measuredBuffer, buffer_len, inv_pixels);
            fieldBuffer.release();
            measuredBuffer.release();

            wgpu::Buffer splitForwardGrad = make_complex_buffer(context, buffer_len);
            dft_adjoint_inverse(context, splitForwardGrad, fieldGrad, buffer_len, shape[0], shape[1]);
            fieldGrad.release();

            wgpu::Buffer pupilBuffer = createBuffer(
                context.device,
                nullptr,
                sizeof(int) * buffer_len,
                WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
            );
            binary_pupil(context, pupilBuffer, shape, na, res);
            wgpu::Buffer pupilFilteredGrad = make_complex_buffer(context, buffer_len);
            mult(context, pupilFilteredGrad, splitForwardGrad, pupilBuffer, buffer_len);
            splitForwardGrad.release();
            pupilBuffer.release();

            wgpu::Buffer UGrad = make_complex_buffer(context, buffer_len);
            wgpu::Buffer UDGrad = make_complex_buffer(context, buffer_len);
            split_prop_grad(context, UGrad, UDGrad, pupilFilteredGrad, buffer_len, shape, res);
            pupilFilteredGrad.release();

            // REVERSING THE FINAL FOCAL-PLANE PROPAGATION
            wgpu::Buffer exitUGrad = make_complex_buffer(context, buffer_len);
            wgpu::Buffer exitUDGrad = make_complex_buffer(context, buffer_len);
            diffract_grad(
                context,
                exitUGrad,
                exitUDGrad,
                UGrad,
                UDGrad,
                buffer_len,
                shape,
                res,
                -static_cast<float>(result.volume.size()) / 2.0f
            );
            UGrad.release();
            UDGrad.release();
            UGrad = exitUGrad;
            UDGrad = exitUDGrad;

            // TRAVERSING THE VOLUME BACKWARDS SLICE-BY-SLICE
            for (int z = static_cast<int>(result.volume.size()) - 1; z >= 0; --z) {
                wgpu::Buffer uBuffer = make_complex_buffer(context, buffer_len);
                dft(context, uBuffer, exitState.U, buffer_len, shape[0], shape[1], 1);

                wgpu::Buffer negUDGrad = make_complex_buffer(context, buffer_len);
                complex_scale(context, negUDGrad, UDGrad, buffer_len, -1.0f);

                wgpu::Buffer scatterAdjointSpatial = make_complex_buffer(context, buffer_len);
                dft_adjoint_forward(context, scatterAdjointSpatial, negUDGrad, buffer_len, shape[0], shape[1]);
                negUDGrad.release();

                wgpu::Buffer sliceBuffer = create_complex_slice_buffer(context, result.volume[z]);
                wgpu::Buffer qBuffer = make_complex_buffer(context, buffer_len);
                scatter_factor(context, qBuffer, sliceBuffer, buffer_len, res[0], 1.0f, n0);

                wgpu::Buffer UGradUpdateSpatial = make_complex_buffer(context, buffer_len);
                complex_mult(context, UGradUpdateSpatial, qBuffer, scatterAdjointSpatial, buffer_len);
                wgpu::Buffer UGradUpdate = make_complex_buffer(context, buffer_len);
                dft(context, UGradUpdate, UGradUpdateSpatial, buffer_len, shape[0], shape[1], 0);

                wgpu::Buffer nextUGrad = make_complex_buffer(context, buffer_len);
                complex_add(context, nextUGrad, UGrad, UGradUpdate, buffer_len);
                UGrad.release();
                UGradUpdate.release();
                UGrad = nextUGrad;

                wgpu::Buffer dqBuffer = make_real_buffer(context, buffer_len);
                scatter_derivative(context, dqBuffer, sliceBuffer, buffer_len, res[0], 1.0f, n0);
                wgpu::Buffer dnSliceBuffer = make_real_buffer(context, buffer_len);
                volume_grad(context, dnSliceBuffer, dqBuffer, scatterAdjointSpatial, uBuffer, buffer_len);
                std::vector<float> dnSlice = readBack(context.device, context.queue, buffer_len, dnSliceBuffer);
                accumulate_slice(grad_volume, static_cast<size_t>(z), dnSlice);
                UGradUpdateSpatial.release();
                dnSliceBuffer.release();
                dqBuffer.release();

                // UNDOING THE FORWARD SCATTER UPDATE BEFORE STEPPING BACKWARD
                wgpu::Buffer qTimesU = make_complex_buffer(context, buffer_len);
                complex_mult(context, qTimesU, qBuffer, uBuffer, buffer_len);
                wgpu::Buffer undoScatterFreq = make_complex_buffer(context, buffer_len);
                dft(context, undoScatterFreq, qTimesU, buffer_len, shape[0], shape[1], 0);
                qTimesU.release();

                wgpu::Buffer restoredUD = make_complex_buffer(context, buffer_len);
                complex_add(context, restoredUD, exitState.UD, undoScatterFreq, buffer_len);
                undoScatterFreq.release();

                SSNPState previousState = {make_complex_buffer(context, buffer_len), make_complex_buffer(context, buffer_len)};
                idiffract(context, previousState.U, previousState.UD, exitState.U, restoredUD, buffer_len, shape, res, 1.0f);
                release_state(exitState);
                restoredUD.release();
                exitState = previousState;

                wgpu::Buffer previousUGrad = make_complex_buffer(context, buffer_len);
                wgpu::Buffer previousUDGrad = make_complex_buffer(context, buffer_len);
                diffract_grad(context, previousUGrad, previousUDGrad, UGrad, UDGrad, buffer_len, shape, res, 1.0f);
                UGrad.release();
                UDGrad.release();
                UGrad = previousUGrad;
                UDGrad = previousUDGrad;

                sliceBuffer.release();
                qBuffer.release();
                uBuffer.release();
                scatterAdjointSpatial.release();
            }

            release_state(exitState);
            UGrad.release();
            UDGrad.release();
        }

        // APPLYING THE GRADIENT-DESCENT UPDATE
        float angle_scale = 1.0f / static_cast<float>(angles.size());
        for (size_t z = 0; z < result.volume.size(); ++z) {
            for (size_t row = 0; row < result.volume[z].size(); ++row) {
                for (size_t col = 0; col < result.volume[z][row].size(); ++col) {
                    result.volume[z][row][col] -= learning_rate * grad_volume[z][row][col] * angle_scale;
                }
            }
        }

        result.loss_history.push_back(total_loss * angle_scale);
    }

    return result;
}

} // namespace ssnp
