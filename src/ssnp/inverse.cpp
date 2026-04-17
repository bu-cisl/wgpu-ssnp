#include "inverse.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ssnp {
namespace {

constexpr int kConvergencePatience = 5;

struct AngleGradientResult {
    float loss = 0.0f;
};

wgpu::Buffer make_real_buffer(WebGPUContext& context, size_t buffer_len);

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

// COMPUTING THE FORWARD LOSS FOR ONE ANGLE
float compute_angle_loss(
    WebGPUContext& context,
    const std::vector<std::vector<std::vector<float>>>& volume,
    const std::vector<std::vector<float>>& measured,
    const std::vector<float>& angle,
    const std::vector<int>& shape,
    const std::vector<float>& res,
    float na,
    float n0,
    size_t buffer_len
) {
    SSNPState exit_state = propagate_to_object_exit(
        context,
        initialize_angle_state(context, angle, shape, res),
        volume,
        shape,
        res,
        n0
    );

    wgpu::Buffer field_buffer = project_state_to_sensor_field(
        context,
        exit_state,
        shape,
        res,
        na,
        -static_cast<float>(volume.size()) / 2.0f
    );

    wgpu::Buffer predicted_intensity_buffer = make_real_buffer(context, buffer_len);
    intense(context, predicted_intensity_buffer, field_buffer, buffer_len, true);
    std::vector<float> predicted_intensity = readBack(
        context.device,
        context.queue,
        buffer_len,
        predicted_intensity_buffer
    );

    predicted_intensity_buffer.release();
    field_buffer.release();
    release_state(exit_state);

    return mean_squared_loss(predicted_intensity, measured);
}

// COMPUTING THE AVERAGE MEASUREMENT LOSS FOR THE CURRENT VOLUME
float compute_measurement_loss(
    WebGPUContext& context,
    const std::vector<std::vector<std::vector<float>>>& volume,
    const std::vector<std::vector<std::vector<float>>>& measured,
    const std::vector<std::vector<float>>& angles,
    const std::vector<int>& shape,
    const std::vector<float>& res,
    float na,
    float n0,
    size_t buffer_len
) {
    float total_loss = 0.0f;
    for (size_t angle_idx = 0; angle_idx < angles.size(); ++angle_idx) {
        total_loss += compute_angle_loss(
            context,
            volume,
            measured[angle_idx],
            angles[angle_idx],
            shape,
            res,
            na,
            n0,
            buffer_len
        );
    }
    return total_loss / static_cast<float>(angles.size());
}

// CONVERTING THE STORED LOSS TO MEASUREMENT MSE
float measurement_mse_from_loss(float loss) {
    return 2.0f * loss;
}

// CREATING A ZERO-INITIALIZED GRADIENT VOLUME
std::vector<std::vector<std::vector<float>>> zeros_like(
    const std::vector<std::vector<std::vector<float>>>& volume
) {
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

// APPLYING ONE GRADIENT-DESCENT STEP AND RETURNING THE MAX VOXEL UPDATE
float apply_gradient_step(
    std::vector<std::vector<std::vector<float>>>& volume,
    const std::vector<std::vector<std::vector<float>>>& grad_volume,
    float learning_rate,
    float angle_scale
) {
    float max_voxel_update = 0.0f;
    for (size_t z = 0; z < volume.size(); ++z) {
        for (size_t row = 0; row < volume[z].size(); ++row) {
            for (size_t col = 0; col < volume[z][row].size(); ++col) {
                float voxel_update = learning_rate * grad_volume[z][row][col] * angle_scale;
                max_voxel_update = std::max(max_voxel_update, std::abs(voxel_update));
                volume[z][row][col] -= voxel_update;
            }
        }
    }
    return max_voxel_update;
}

// BACKPROPAGATING ONE ANGLE THROUGH THE VOLUME
void backpropagate_through_volume(
    WebGPUContext& context,
    const std::vector<std::vector<std::vector<float>>>& volume,
    const std::vector<int>& shape,
    const std::vector<float>& res,
    float n0,
    size_t buffer_len,
    SSNPState& exit_state,
    wgpu::Buffer& U_grad,
    wgpu::Buffer& UD_grad,
    std::vector<std::vector<std::vector<float>>>& grad_volume
) {
    for (int z = static_cast<int>(volume.size()) - 1; z >= 0; --z) {
        // CONVERTING THE CURRENT OBJECT-EXIT FIELD BACK TO SPATIAL DOMAIN
        wgpu::Buffer u_buffer = make_complex_buffer(context, buffer_len);
        fft(context, u_buffer, exit_state.U, buffer_len, shape[0], shape[1], 1);

        // FORMING THE SCATTER ADJOINT FROM -UD_GRAD
        wgpu::Buffer neg_UD_grad = make_complex_buffer(context, buffer_len);
        complex_scale(context, neg_UD_grad, UD_grad, buffer_len, -1.0f);

        wgpu::Buffer scatter_adjoint_spatial = make_complex_buffer(context, buffer_len);
        fft_adjoint_forward(
            context,
            scatter_adjoint_spatial,
            neg_UD_grad,
            buffer_len,
            shape[0],
            shape[1]
        );
        neg_UD_grad.release();

        wgpu::Buffer slice_buffer = create_complex_slice_buffer(context, volume[static_cast<size_t>(z)]);
        wgpu::Buffer q_buffer = make_complex_buffer(context, buffer_len);
        scatter_factor(context, q_buffer, slice_buffer, buffer_len, res[0], 1.0f, n0);

        // ACCUMULATING THE U_GRAD UPDATE FROM THE SCATTER TERM
        wgpu::Buffer U_grad_update_spatial = make_complex_buffer(context, buffer_len);
        complex_mult(context, U_grad_update_spatial, q_buffer, scatter_adjoint_spatial, buffer_len);
        wgpu::Buffer U_grad_update = make_complex_buffer(context, buffer_len);
        fft(context, U_grad_update, U_grad_update_spatial, buffer_len, shape[0], shape[1], 0);

        wgpu::Buffer next_U_grad = make_complex_buffer(context, buffer_len);
        complex_add(context, next_U_grad, U_grad, U_grad_update, buffer_len);
        U_grad.release();
        U_grad_update.release();
        U_grad = next_U_grad;

        // ACCUMULATING THE VOLUME GRADIENT FOR THIS SLICE
        wgpu::Buffer dq_buffer = make_real_buffer(context, buffer_len);
        scatter_derivative(context, dq_buffer, slice_buffer, buffer_len, res[0], 1.0f, n0);
        wgpu::Buffer dn_slice_buffer = make_real_buffer(context, buffer_len);
        volume_grad(context, dn_slice_buffer, dq_buffer, scatter_adjoint_spatial, u_buffer, buffer_len);
        std::vector<float> dn_slice = readBack(context.device, context.queue, buffer_len, dn_slice_buffer);
        accumulate_slice(grad_volume, static_cast<size_t>(z), dn_slice);

        // UNDOING THE FORWARD SCATTER STEP BEFORE STEPPING BACKWARD
        wgpu::Buffer q_times_u = make_complex_buffer(context, buffer_len);
        complex_mult(context, q_times_u, q_buffer, u_buffer, buffer_len);
        wgpu::Buffer undo_scatter_freq = make_complex_buffer(context, buffer_len);
        fft(context, undo_scatter_freq, q_times_u, buffer_len, shape[0], shape[1], 0);

        wgpu::Buffer restored_UD = make_complex_buffer(context, buffer_len);
        complex_add(context, restored_UD, exit_state.UD, undo_scatter_freq, buffer_len);

        // REVERSING THE FORWARD DIFFRACTION STEP
        SSNPState previous_state = {
            make_complex_buffer(context, buffer_len),
            make_complex_buffer(context, buffer_len)
        };
        idiffract(
            context,
            previous_state.U,
            previous_state.UD,
            exit_state.U,
            restored_UD,
            buffer_len,
            shape,
            res,
            1.0f
        );
        release_state(exit_state);
        exit_state = previous_state;

        // PROPAGATING THE FIELD GRADIENTS BACKWARD ONE SLICE
        wgpu::Buffer previous_U_grad = make_complex_buffer(context, buffer_len);
        wgpu::Buffer previous_UD_grad = make_complex_buffer(context, buffer_len);
        diffract_grad(
            context,
            previous_U_grad,
            previous_UD_grad,
            U_grad,
            UD_grad,
            buffer_len,
            shape,
            res,
            1.0f
        );
        U_grad.release();
        UD_grad.release();
        U_grad = previous_U_grad;
        UD_grad = previous_UD_grad;

        slice_buffer.release();
        q_buffer.release();
        dq_buffer.release();
        dn_slice_buffer.release();
        q_times_u.release();
        undo_scatter_freq.release();
        restored_UD.release();
        u_buffer.release();
        scatter_adjoint_spatial.release();
        U_grad_update_spatial.release();
    }
}

// COMPUTING ONE ANGLE'S LOSS AND VOLUME GRADIENT CONTRIBUTION
AngleGradientResult compute_angle_gradient(
    WebGPUContext& context,
    const std::vector<std::vector<std::vector<float>>>& volume,
    const std::vector<std::vector<float>>& measured,
    const std::vector<float>& angle,
    const std::vector<int>& shape,
    const std::vector<float>& res,
    float na,
    float n0,
    size_t buffer_len,
    float inv_pixels,
    std::vector<std::vector<std::vector<float>>>& grad_volume
) {
    // FORWARD PROPAGATION TO THE OBJECT EXIT
    SSNPState exit_state = propagate_to_object_exit(
        context,
        initialize_angle_state(context, angle, shape, res),
        volume,
        shape,
        res,
        n0
    );

    // PROJECTING THE OBJECT-EXIT STATE TO THE SENSOR FIELD
    wgpu::Buffer field_buffer = project_state_to_sensor_field(
        context,
        exit_state,
        shape,
        res,
        na,
        -static_cast<float>(volume.size()) / 2.0f
    );

    // COMPUTING THE MEASUREMENT LOSS FOR THIS ANGLE
    wgpu::Buffer predicted_intensity_buffer = make_real_buffer(context, buffer_len);
    intense(context, predicted_intensity_buffer, field_buffer, buffer_len, true);
    std::vector<float> predicted_intensity = readBack(
        context.device,
        context.queue,
        buffer_len,
        predicted_intensity_buffer
    );
    float loss = mean_squared_loss(predicted_intensity, measured);
    predicted_intensity_buffer.release();

    // FORMING THE SENSOR-PLANE LOSS GRADIENT
    std::vector<float> measured_flat = flatten_real_slice(measured);
    wgpu::Buffer measured_buffer = createBuffer(
        context.device,
        measured_flat.data(),
        sizeof(float) * buffer_len,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );

    wgpu::Buffer field_grad = make_complex_buffer(context, buffer_len);
    intensity_grad(context, field_grad, field_buffer, measured_buffer, buffer_len, inv_pixels);
    field_buffer.release();
    measured_buffer.release();

    // MAPPING THE SENSOR GRADIENT BACK TO THE EXIT STATE
    wgpu::Buffer split_forward_grad = make_complex_buffer(context, buffer_len);
    fft_adjoint_inverse(context, split_forward_grad, field_grad, buffer_len, shape[0], shape[1]);
    field_grad.release();

    wgpu::Buffer pupil_buffer = createBuffer(
        context.device,
        nullptr,
        sizeof(int) * buffer_len,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    binary_pupil(context, pupil_buffer, shape, na, res);
    wgpu::Buffer pupil_filtered_grad = make_complex_buffer(context, buffer_len);
    mult(context, pupil_filtered_grad, split_forward_grad, pupil_buffer, buffer_len);
    split_forward_grad.release();
    pupil_buffer.release();

    wgpu::Buffer U_grad = make_complex_buffer(context, buffer_len);
    wgpu::Buffer UD_grad = make_complex_buffer(context, buffer_len);
    split_prop_grad(context, U_grad, UD_grad, pupil_filtered_grad, buffer_len, shape, res);
    pupil_filtered_grad.release();

    // REVERSING THE FINAL FOCAL-PLANE PROPAGATION
    wgpu::Buffer exit_U_grad = make_complex_buffer(context, buffer_len);
    wgpu::Buffer exit_UD_grad = make_complex_buffer(context, buffer_len);
    diffract_grad(
        context,
        exit_U_grad,
        exit_UD_grad,
        U_grad,
        UD_grad,
        buffer_len,
        shape,
        res,
        -static_cast<float>(volume.size()) / 2.0f
    );
    U_grad.release();
    UD_grad.release();
    U_grad = exit_U_grad;
    UD_grad = exit_UD_grad;

    // BACKPROPAGATING THE EXIT-STATE GRADIENT THROUGH THE VOLUME
    backpropagate_through_volume(
        context,
        volume,
        shape,
        res,
        n0,
        buffer_len,
        exit_state,
        U_grad,
        UD_grad,
        grad_volume
    );

    release_state(exit_state);
    U_grad.release();
    UD_grad.release();

    AngleGradientResult result;
    result.loss = loss;
    return result;
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
    const ReconstructionOptions& options
) {
    if (measured.size() != angles.size()) {
        throw std::runtime_error("Measured intensity stack and angle list must have the same length.");
    }
    if (angles.empty()) {
        throw std::runtime_error("Angle list must be non-empty.");
    }
    if (measured.empty() || initial_volume.empty()) {
        throw std::runtime_error("Measured data and initial volume must be non-empty.");
    }
    if (options.max_iterations <= 0) {
        throw std::runtime_error("max_iterations must be positive.");
    }
    if (options.learning_rate <= 0.0f) {
        throw std::runtime_error("learning_rate must be positive.");
    }
    if (options.print_every < 0) {
        throw std::runtime_error("print_every must be non-negative.");
    }

    std::vector<int> shape = {int(initial_volume[0].size()), int(initial_volume[0][0].size())};
    for (const auto& slice : initial_volume) {
        if (slice.size() != static_cast<size_t>(shape[0])) {
            throw std::runtime_error("Initial volume slices must have consistent height.");
        }
        for (const auto& row : slice) {
            if (row.size() != static_cast<size_t>(shape[1])) {
                throw std::runtime_error("Initial volume slices must have consistent width.");
            }
        }
    }
    for (const auto& image : measured) {
        if (image.size() != static_cast<size_t>(shape[0])) {
            throw std::runtime_error("Measured images must match the volume height.");
        }
        for (const auto& row : image) {
            if (row.size() != static_cast<size_t>(shape[1])) {
                throw std::runtime_error("Measured images must match the volume width.");
            }
        }
    }

    size_t buffer_len = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]);
    float inv_pixels = 1.0f / static_cast<float>(buffer_len);
    float angle_scale = 1.0f / static_cast<float>(angles.size());
    float current_learning_rate = options.learning_rate;

    ReconstructionResult result;
    result.volume = std::move(initial_volume);
    result.best_loss = std::numeric_limits<float>::infinity();

    float previous_loss = std::numeric_limits<float>::infinity();
    int stalled_iterations = 0;

    // RUNNING THE OUTER GRADIENT-DESCENT LOOP
    for (int iter = 0; iter < options.max_iterations; ++iter) {
        std::vector<std::vector<std::vector<float>>> grad_volume = zeros_like(result.volume);
        float total_loss = 0.0f;

        // ACCUMULATING LOSS AND GRADIENTS OVER ANGLES
        for (size_t angle_idx = 0; angle_idx < angles.size(); ++angle_idx) {
            AngleGradientResult angle_result = compute_angle_gradient(
                context,
                result.volume,
                measured[angle_idx],
                angles[angle_idx],
                shape,
                res,
                na,
                n0,
                buffer_len,
                inv_pixels,
                grad_volume
            );
            total_loss += angle_result.loss;
        }

        // RECORDING THE CURRENT MEASUREMENT LOSS
        float current_loss = total_loss * angle_scale;

        // APPLYING THE VOLUME UPDATE
        float max_voxel_update = apply_gradient_step(result.volume, grad_volume, current_learning_rate, angle_scale);

        float updated_loss = current_loss;
        if (max_voxel_update != 0.0f) {
            updated_loss = compute_measurement_loss(
                context,
                result.volume,
                measured,
                angles,
                shape,
                res,
                na,
                n0,
                buffer_len
            );
        }

        result.loss_history.push_back(updated_loss);
        result.final_loss = updated_loss;
        result.best_loss = std::min(result.best_loss, updated_loss);

        // PRINTING CONCISE PER-EPOCH MEASUREMENT PROGRESS
        if (options.verbose && options.print_every > 0 && (iter % options.print_every == 0)) {
            std::cout << "iter " << iter
                      << " measurement_mse " << measurement_mse_from_loss(updated_loss)
                      << std::endl;
        }

        // TRACKING SIMPLE CONVERGENCE AND LEARNING-RATE BACKOFF
        if (std::isfinite(previous_loss)) {
            float absolute_improvement = previous_loss - updated_loss;
            float relative_improvement = absolute_improvement / std::max(std::abs(previous_loss), 1e-12f);
            if (updated_loss > previous_loss) {
                current_learning_rate *= 0.5f;
            }
            if (max_voxel_update == 0.0f ||
                absolute_improvement <= options.abs_tol ||
                relative_improvement <= options.rel_tol) {
                ++stalled_iterations;
            } else {
                stalled_iterations = 0;
                }
        }

        previous_loss = updated_loss;
        result.iterations_run = static_cast<int>(result.loss_history.size());
        result.final_learning_rate = current_learning_rate;

        if (stalled_iterations >= kConvergencePatience) {
            break;
        }
    }

    return result;
}

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
    ReconstructionOptions options;
    options.max_iterations = iterations;
    options.learning_rate = learning_rate;
    return reconstruct(context, measured, angles, std::move(initial_volume), res, na, n0, options);
}

} // namespace ssnp
