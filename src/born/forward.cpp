#include "forward.h"

namespace born {

namespace {

wgpu::Buffer create_incident_field(
    WebGPUContext& context,
    const std::vector<int>& shape,
    const std::vector<float>& res,
    const std::vector<float>& c_ba
) {
    const int height = shape[0];
    const int width = shape[1];
    const size_t buffer_len = static_cast<size_t>(height) * static_cast<size_t>(width);

    std::vector<float> incident(buffer_len * 2, 0.0f);
    const int shift_y = static_cast<int>(std::round(c_ba[0] * res[1] * height));
    const int shift_x = static_cast<int>(std::round(c_ba[1] * res[2] * width));
    const int y = ((shift_y % height) + height) % height;
    const int x = ((shift_x % width) + width) % width;
    const size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);

    incident[idx * 2] = static_cast<float>(buffer_len);

    return createBuffer(
        context.device,
        incident.data(),
        sizeof(float) * incident.size(),
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
}

} // namespace

std::vector<std::vector<std::vector<float>>> forward(
    WebGPUContext& context,
    std::vector<std::vector<std::vector<float>>> n,
    std::vector<float> res,
    float na,
    std::vector<std::vector<float>> angles,
    float n0,
    int outputType
) {
    const std::vector<int> shape = {int(n[0].size()), int(n[0][0].size())};
    const size_t buffer_len = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]);
    std::vector<std::vector<std::vector<float>>> result;

    for (const std::vector<float>& c_ba : angles) {
        wgpu::Buffer fieldBuffer = create_incident_field(context, shape, res, c_ba);

        for (size_t z = 0; z < n.size(); ++z) {
            std::vector<float> complexSlice(buffer_len * 2, 0.0f);
            size_t offset = 0;
            for (const auto& row : n[z]) {
                for (float value : row) {
                    complexSlice[offset++] = value;
                    complexSlice[offset++] = 0.0f;
                }
            }

            wgpu::Buffer sliceBuffer = createBuffer(
                context.device,
                complexSlice.data(),
                sizeof(float) * complexSlice.size(),
                WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
            );

            wgpu::Buffer potentialSpatialBuffer = createBuffer(
                context.device,
                nullptr,
                sizeof(float) * buffer_len * 2,
                WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
            );
            scatter_potential(context, potentialSpatialBuffer, sliceBuffer, buffer_len, res[0], n0);
            sliceBuffer.release();

            wgpu::Buffer potentialFourierBuffer = createBuffer(
                context.device,
                nullptr,
                sizeof(float) * buffer_len * 2,
                WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
            );
            fft(
                context,
                potentialFourierBuffer,
                potentialSpatialBuffer,
                buffer_len,
                shape[0],
                shape[1],
                0
            );
            potentialSpatialBuffer.release();

            wgpu::Buffer termBuffer = createBuffer(
                context.device,
                nullptr,
                sizeof(float) * buffer_len * 2,
                WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
            );
            const float depth = float(n.size()) / 2.0f - float(z);
            propagation_term(context, termBuffer, potentialFourierBuffer, buffer_len, shape, res, c_ba, depth);
            potentialFourierBuffer.release();

            wgpu::Buffer nextFieldBuffer = createBuffer(
                context.device,
                nullptr,
                sizeof(float) * buffer_len * 2,
                WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
            );
            complex_add(context, nextFieldBuffer, fieldBuffer, termBuffer, buffer_len);
            fieldBuffer.release();
            termBuffer.release();
            fieldBuffer = nextFieldBuffer;
        }

        wgpu::Buffer pupilBuffer = createBuffer(
            context.device,
            nullptr,
            sizeof(int) * buffer_len,
            WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
        );
        binary_pupil(context, pupilBuffer, shape, na, res);

        wgpu::Buffer filteredFieldBuffer = createBuffer(
            context.device,
            nullptr,
            sizeof(float) * buffer_len * 2,
            WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
        );
        mult(context, filteredFieldBuffer, fieldBuffer, pupilBuffer, buffer_len);
        fieldBuffer.release();
        pupilBuffer.release();

        wgpu::Buffer complexSlice = createBuffer(
            context.device,
            nullptr,
            sizeof(float) * buffer_len * 2,
            WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
        );
        fft(context, complexSlice, filteredFieldBuffer, buffer_len, shape[0], shape[1], 1);
        filteredFieldBuffer.release();

        if (outputType == 2) {
            std::vector<float> complexData = readBack(
                context.device,
                context.queue,
                buffer_len * 2,
                complexSlice
            );
            complexSlice.release();

            std::vector<std::vector<float>> realSlice(shape[0], std::vector<float>(shape[1], 0.0f));
            std::vector<std::vector<float>> imagSlice(shape[0], std::vector<float>(shape[1], 0.0f));
            for (int i = 0; i < shape[0]; ++i) {
                for (int j = 0; j < shape[1]; ++j) {
                    const int idx = i * shape[1] + j;
                    realSlice[i][j] = complexData[idx * 2];
                    imagSlice[i][j] = complexData[idx * 2 + 1];
                }
            }
            result.push_back(realSlice);
            result.push_back(imagSlice);
        } else {
            wgpu::Buffer sliceBuffer = createBuffer(
                context.device,
                nullptr,
                sizeof(float) * buffer_len,
                WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
            );
            intense(context, sliceBuffer, complexSlice, buffer_len, outputType == 1);
            std::vector<float> slice = readBack(
                context.device,
                context.queue,
                buffer_len,
                sliceBuffer
            );
            complexSlice.release();
            sliceBuffer.release();

            std::vector<std::vector<float>> reshapedSlice(shape[0], std::vector<float>(shape[1], 0.0f));
            for (int i = 0; i < shape[0]; ++i) {
                for (int j = 0; j < shape[1]; ++j) {
                    reshapedSlice[i][j] = slice[size_t(i) * size_t(shape[1]) + size_t(j)];
                }
            }
            result.push_back(reshapedSlice);
        }
    }

    return result;
}

} // namespace born
