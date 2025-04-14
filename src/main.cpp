// main.cpp
#define WEBGPU_CPP_IMPLEMENTATION
#include "scatter_factor/scatter_factor.h"
#include "diffract/diffract.h"
#include "binary_pupil/binary_pupil.h"
#include "tilt/tilt.h"
#include "merge_prop/merge_prop.h"
#include "split_prop/split_prop.h"
#include "c_gamma/c_gamma.h"  
#include "webgpu_utils.h"
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

// ---------------------------------------------------------------------
// Helper function to read a matrix of floats from a text file.
// Format: 
//   First line: <rows> <cols>
//   Then one line per row with space‐separated floats.
// ---------------------------------------------------------------------
bool readMatrixFromFile(const string &filename, vector<float>& data, vector<int>& shape) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Error: could not open input file " << filename << endl;
        return false;
    }
    int rows, cols;
    infile >> rows >> cols;
    shape = {rows, cols};
    data.resize(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        if (!(infile >> data[i])) {
            cerr << "Error reading data at index " << i << endl;
            return false;
        }
    }
    infile.close();
    return true;
}

// ---------------------------------------------------------------------
// Helper: Compute and print summary (count, mean, std, min, max) for float data.
// ---------------------------------------------------------------------
void printSummary(const string& label, const vector<float>& data) {
    if (data.empty()){
        cout << label << ": empty data" << endl;
        return;
    }
    float sum = 0.0f;
    float min_val = data[0];
    float max_val = data[0];
    for (float val : data) {
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    float mean = sum / data.size();
    float accum = 0.0f;
    for (float val : data) {
        accum += (val - mean) * (val - mean);
    }
    float std = sqrt(accum / data.size());
    cout << label << ": count=" << data.size() 
         << " mean=" << fixed << setprecision(8) << mean 
         << " std=" << fixed << setprecision(8) << std 
         << " min=" << fixed << setprecision(8) << min_val 
         << " max=" << fixed << setprecision(8) << max_val << endl;
}

// ---------------------------------------------------------------------
// Helper: For outputs of complex functions stored as interleaved floats,
// compute the summary of the magnitudes.
// ---------------------------------------------------------------------
void printComplexSummary(const string& label, const vector<float>& data) {
    vector<float> mags;
    // Assumes data is in [real, imag, real, imag, ...] order.
    for (size_t i = 0; i < data.size(); i += 2) {
        float mag = sqrt(data[i] * data[i] + data[i+1] * data[i+1]);
        mags.push_back(mag);
    }
    printSummary(label, mags);
}

// ---------------------------------------------------------------------
// Helper: For integer arrays (binary pupil mask).
// ---------------------------------------------------------------------
void printIntSummary(const string& label, const vector<uint32_t>& data) {
    if (data.empty()){
        cout << label << ": empty data" << endl;
        return;
    }
    uint32_t sum = 0;
    uint32_t min_val = data[0];
    uint32_t max_val = data[0];
    for (uint32_t val : data) {
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    double mean = double(sum) / data.size();
    cout << label << ": count=" << data.size() 
         << " mean=" << fixed << setprecision(8) << mean 
         << " std=" << fixed << setprecision(8) << sum // intentionally wrong
         << " min=" << fixed << min_val 
         << " max=" << fixed << max_val << endl;
}

int main(int argc, char** argv) {
    // Initialize WebGPU.
    WebGPUContext context;
    initWebGPU(context);

    // -----------------------------------------------------------------
    // Prepare input parameters.
    // Use the same input matrix for all tests.
    // -----------------------------------------------------------------
    vector<int> matrix_shape;
    vector<float> inputMatrix;
    if (argc > 1) {
        if (!readMatrixFromFile(argv[1], inputMatrix, matrix_shape)) {
            cerr << "Failed to read input file. Using default test data." << endl;
            matrix_shape = {3,3};
            inputMatrix = {5.0f, 21.0f, 65.0f, 5.0f, 21.0f, 65.0f, 5.0f, 21.0f, 65.0f};
        }
    } else {
        matrix_shape = {3,3};
        inputMatrix = {5.0f, 21.0f, 65.0f, 5.0f, 21.0f, 65.0f, 5.0f, 21.0f, 65.0f};
    }
    // For functions expecting complex input, convert the float values.
    vector<complex<float>> complexInput;
    for (float val : inputMatrix) {
        complexInput.push_back({val, 0.0f});
    }
    // Fixed resolution.
    vector<float> res = {0.1f, 0.1f, 0.1f};

    // ------------------------- Test: scatter_factor -------------------------
    cout << "Phase: Running scatter_factor test in C++" << endl;
    wgpu::Buffer scatterResultBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * inputMatrix.size(), 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    scatter_factor(context, scatterResultBuffer, inputMatrix);
    vector<float> scatterOutput = readBack(context.device, context.queue, inputMatrix.size(), scatterResultBuffer);
    printSummary("SCATTER_FACTOR", scatterOutput);
    scatterResultBuffer.release();

    // ------------------------- Test: c_gamma -------------------------
    cout << "Phase: Running c_gamma test in C++" << endl;
    size_t c_gamma_len = matrix_shape[0] * matrix_shape[1];
    wgpu::Buffer cGammaResultBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * c_gamma_len, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    c_gamma(context, cGammaResultBuffer, res, matrix_shape);
    vector<float> cGammaOutput = readBack(context.device, context.queue, c_gamma_len, cGammaResultBuffer);
    printSummary("C_GAMMA", cGammaOutput);
    cGammaResultBuffer.release();

    // ------------------------- Test: diffract -------------------------
    cout << "Phase: Running diffract test in C++" << endl;
    size_t numComplex = complexInput.size();
    wgpu::Buffer diffractUFBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * 2 * numComplex, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    wgpu::Buffer diffractUBBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * 2 * numComplex, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    diffract(context, diffractUFBuffer, diffractUBBuffer, complexInput, complexInput, matrix_shape);
    vector<float> diffractUF = readBack(context.device, context.queue, 2 * numComplex, diffractUFBuffer);
    vector<float> diffractUB = readBack(context.device, context.queue, 2 * numComplex, diffractUBBuffer);
    printComplexSummary("DIFRACT_UF", diffractUF);
    printComplexSummary("DIFRACT_UB", diffractUB);
    diffractUFBuffer.release();
    diffractUBBuffer.release();

    // ------------------------- Test: binary_pupil -------------------------
    cout << "Phase: Running binary_pupil test in C++" << endl;
    size_t binaryLen = matrix_shape[0] * matrix_shape[1];
    wgpu::Buffer binaryPupilResultBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(uint32_t) * binaryLen, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    binary_pupil(context, binaryPupilResultBuffer, matrix_shape, 0.9f, res);
    vector<uint32_t> binaryPupilOutput = readBackInt(context.device, context.queue, binaryLen, binaryPupilResultBuffer);
    printIntSummary("BINARY_PUPIL", binaryPupilOutput);
    binaryPupilResultBuffer.release();

    // ------------------------- Test: tilt -------------------------
    cout << "Phase: Running tilt test in C++" << endl;
    vector<float> tilt_angles = {0.1f, 0.5f, 1.0f};
    float tilt_NA = 0.5f;
    vector<float> tilt_res = res;
    size_t tilt_output_size = tilt_angles.size() * matrix_shape[0] * matrix_shape[1];
    size_t tilt_buffer_size = tilt_output_size * 2;
    wgpu::Buffer tiltResultBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * tilt_buffer_size, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    bool trunc = false;
    tilt(context, tiltResultBuffer, tilt_angles, matrix_shape, tilt_NA, tilt_res, trunc);
    vector<float> tiltOutput = readBack(context.device, context.queue, tilt_buffer_size, tiltResultBuffer);
    printComplexSummary("TILT", tiltOutput);
    tiltResultBuffer.release();

    // ------------------------- Test: merge_prop -------------------------
    cout << "Phase: Running merge_prop test in C++" << endl;
    wgpu::Buffer mergeUFBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * 2 * numComplex, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    wgpu::Buffer mergeUBBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * 2 * numComplex, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    merge_prop(context, mergeUFBuffer, mergeUBBuffer, complexInput, complexInput, matrix_shape, res);
    vector<float> mergeUF = readBack(context.device, context.queue, 2 * numComplex, mergeUFBuffer);
    vector<float> mergeUB = readBack(context.device, context.queue, 2 * numComplex, mergeUBBuffer);
    printComplexSummary("MERGE_PROP_UF", mergeUF);
    printComplexSummary("MERGE_PROP_UB", mergeUB);
    mergeUFBuffer.release();
    mergeUBBuffer.release();

    // ------------------------- Test: split_prop -------------------------
    cout << "Phase: Running split_prop test in C++" << endl;
    wgpu::Buffer splitUFBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * 2 * numComplex, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    wgpu::Buffer splitUBBuffer = createBuffer(
        context.device, 
        nullptr, 
        sizeof(float) * 2 * numComplex, 
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    split_prop(context, splitUFBuffer, splitUBBuffer, complexInput, complexInput, matrix_shape, res);
    vector<float> splitUF = readBack(context.device, context.queue, 2 * numComplex, splitUFBuffer);
    vector<float> splitUB = readBack(context.device, context.queue, 2 * numComplex, splitUBBuffer);
    printComplexSummary("SPLIT_PROP_UF", splitUF);
    printComplexSummary("SPLIT_PROP_UB", splitUB);
    splitUFBuffer.release();
    splitUBBuffer.release();

    // Clean up WebGPU resources.
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    
    return 0;
}
