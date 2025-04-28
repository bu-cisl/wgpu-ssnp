#define WEBGPU_CPP_IMPLEMENTATION
#include "scatter_factor/scatter_factor.h"
#include "diffract/diffract.h"
#include "binary_pupil/binary_pupil.h"
#include "tilt/tilt.h"
#include "merge_prop/merge_prop.h"
#include "split_prop/split_prop.h"
#include "c_gamma/c_gamma.h"  
#include "mult/mult.h"
#include "webgpu_utils.h"
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

// Helper to print an array of floats in a parseable format
void printArray(const std::string &label, const std::vector<float>& data) {
    std::cout << label << ":\n" << data.size() << "\n";
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << std::fixed << std::setprecision(8) << data[i] << " ";
    }
    std::cout << "\n";
}

// Helper to print an array of int in a parseable format
void printIntArray(const std::string &label, const std::vector<uint32_t>& data) {
    std::cout << label << ":\n" << data.size() << "\n";
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n";
}

// Read matrix from input matrix txt file
bool readMatrixFromFile(const std::string &filename, std::vector<float>& data, std::vector<int>& shape) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: could not open input file " << filename << "\n";
        return false;
    }
    int rows, cols;
    infile >> rows >> cols;
    shape = {rows, cols};
    data.resize(rows * cols);
    for (int i = 0; i < rows * cols; i++) {
        if (!(infile >> data[i])) {
            std::cerr << "Error reading data at index " << i << "\n";
            return false;
        }
    }
    infile.close();
    return true;
}

int main(int argc, char** argv) {
    // Initialize WebGPU
    WebGPUContext context;
    initWebGPU(context);
    
    // Read input matrix
    std::vector<int> matrix_shape;
    std::vector<float> inputMatrix;
    if(argc > 1) {
        if(!readMatrixFromFile(argv[1], inputMatrix, matrix_shape)) {
            std::cerr << "Failed to read input file. Using default test data.\n";
            matrix_shape = {3, 3};
            inputMatrix = {5.0f, 21.0f, 65.0f, 5.0f, 21.0f, 65.0f, 5.0f, 21.0f, 65.0f};
        }
    } else {
        matrix_shape = {3, 3};
        inputMatrix = {5.0f, 21.0f, 65.0f, 5.0f, 21.0f, 65.0f, 5.0f, 21.0f, 65.0f};
    }

    wgpu::Buffer inputBuffer = createBuffer(
        context.device, inputMatrix.data(),
        sizeof(float) * inputMatrix.size(),
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    
    // For functions expecting complex input, convert the real input to complex numbers with zero imaginary parts
    std::vector<std::complex<float>> complexInput;
    for (float val : inputMatrix) {
        complexInput.push_back({val, 10.0f});
    }

    std::vector<float> complexInputFlat(complexInput.size()*2);
    for (size_t i = 0; i < complexInput.size(); ++i) {
        complexInputFlat[2*i] = complexInput[i].real();
        complexInputFlat[2*i + 1] = complexInput[i].imag();
        complexInputFlat[2*i] = complexInput[i].real();
        complexInputFlat[2*i + 1] = complexInput[i].imag();
    }

    wgpu::Buffer complexInputBuffer = createBuffer(
        context.device, complexInputFlat.data(),
        sizeof(float) * complexInputFlat.size(),
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    
    // Fixed resolution vector.
    std::vector<float> res = {0.1f, 0.1f, 0.1f};
    
    // ------------------------- Test: scatter_factor -------------------------
    std::cout << "Phase: Running scatter_factor test in C++\n";
    wgpu::Buffer scatterResultBuffer = createBuffer(
        context.device, nullptr,
        sizeof(float) * inputMatrix.size(),
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    scatter_factor(context, scatterResultBuffer, inputBuffer, inputMatrix.size());
    std::vector<float> scatterOutput = readBack(context.device, context.queue, inputMatrix.size(), scatterResultBuffer);
    printArray("SCATTER_FACTOR", scatterOutput);
    scatterResultBuffer.release();
    
    // ------------------------- Test: c_gamma -------------------------
    std::cout << "Phase: Running c_gamma test in C++\n";
    size_t c_gamma_len = matrix_shape[0] * matrix_shape[1];
    wgpu::Buffer cGammaResultBuffer = createBuffer(
        context.device, nullptr,
        sizeof(float) * c_gamma_len,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    c_gamma(context, cGammaResultBuffer, res, matrix_shape);
    std::vector<float> cGammaOutput = readBack(context.device, context.queue, c_gamma_len, cGammaResultBuffer);
    printArray("C_GAMMA", cGammaOutput);
    cGammaResultBuffer.release();
    
    // ------------------------- Test: diffract -------------------------
    std::cout << "Phase: Running diffract test in C++\n";
    size_t numComplex = complexInput.size();
    wgpu::Buffer diffractUFBuffer = createBuffer(
        context.device, nullptr,
        sizeof(float) * 2 * numComplex,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    wgpu::Buffer diffractUBBuffer = createBuffer(
        context.device, nullptr,
        sizeof(float) * 2 * numComplex,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    diffract(context, diffractUFBuffer, diffractUBBuffer, complexInputBuffer, complexInputBuffer, numComplex, matrix_shape);
    std::vector<float> diffractUF = readBack(context.device, context.queue, 2 * numComplex, diffractUFBuffer);
    std::vector<float> diffractUB = readBack(context.device, context.queue, 2 * numComplex, diffractUBBuffer);
    printArray("DIFRACT_UF", diffractUF);
    printArray("DIFRACT_UB", diffractUB);
    diffractUFBuffer.release();
    diffractUBBuffer.release();
    
    // ------------------------- Test: binary_pupil -------------------------
    std::cout << "Phase: Running binary_pupil test in C++\n";
    size_t binaryLen = matrix_shape[0] * matrix_shape[1];
    wgpu::Buffer binaryPupilResultBuffer = createBuffer(
        context.device, nullptr,
        sizeof(uint32_t) * binaryLen,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    binary_pupil(context, binaryPupilResultBuffer, matrix_shape, 0.9f, res);
    std::vector<uint32_t> binaryPupilOutput = readBackInt(context.device, context.queue, binaryLen, binaryPupilResultBuffer);
    // Convert to floats for uniform comparison.
    std::vector<float> binaryPupilFloat;
    for (auto x : binaryPupilOutput) {
        binaryPupilFloat.push_back(static_cast<float>(x));
    }
    printArray("BINARY_PUPIL", binaryPupilFloat);
    binaryPupilResultBuffer.release();
    
    // ------------------------- Test: merge_prop -------------------------
    std::cout << "Phase: Running merge_prop test in C++\n";
    wgpu::Buffer mergeUFBuffer = createBuffer(
        context.device, nullptr,
        sizeof(float) * 2 * numComplex,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    wgpu::Buffer mergeUBBuffer = createBuffer(
        context.device, nullptr,
        sizeof(float) * 2 * numComplex,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    merge_prop(context, mergeUFBuffer, mergeUBBuffer, complexInputBuffer, complexInputBuffer, numComplex, matrix_shape, res);
    std::vector<float> mergeUF = readBack(context.device, context.queue, 2 * numComplex, mergeUFBuffer);
    std::vector<float> mergeUB = readBack(context.device, context.queue, 2 * numComplex, mergeUBBuffer);
    printArray("MERGE_PROP_UF", mergeUF);
    printArray("MERGE_PROP_UB", mergeUB);
    mergeUFBuffer.release();
    mergeUBBuffer.release();
    
    // ------------------------- Test: split_prop -------------------------
    std::cout << "Phase: Running split_prop test in C++\n";
    wgpu::Buffer splitUFBuffer = createBuffer(
        context.device, nullptr,
        sizeof(float) * 2 * numComplex,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    wgpu::Buffer splitUBBuffer = createBuffer(
        context.device, nullptr,
        sizeof(float) * 2 * numComplex,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    split_prop(context, splitUFBuffer, splitUBBuffer, complexInputBuffer, complexInputBuffer, numComplex, matrix_shape, res);
    std::vector<float> splitUF = readBack(context.device, context.queue, 2 * numComplex, splitUFBuffer);
    std::vector<float> splitUB = readBack(context.device, context.queue, 2 * numComplex, splitUBBuffer);
    printArray("SPLIT_PROP_UF", splitUF);
    printArray("SPLIT_PROP_UB", splitUB);
    splitUFBuffer.release();
    splitUBBuffer.release();

    // ------------------------- Test: tilt -------------------------
    std::cout << "Phase: Running tilt test in C++\n";
    std::vector<float> c_ba = {0.1f, 0.1f};
    size_t tilt_num_complex = matrix_shape[0] * matrix_shape[1];
    size_t tilt_buffer_size = tilt_num_complex * 2;
    wgpu::Buffer tiltResultBuffer = createBuffer(
        context.device, nullptr,
        sizeof(float) * tilt_buffer_size,
        WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    );
    bool trunc = false;
    tilt(context, tiltResultBuffer, c_ba, matrix_shape, res, trunc);
    std::vector<float> tiltOutput = readBack(context.device, context.queue, tilt_buffer_size, tiltResultBuffer);
    printArray("TILT", tiltOutput);
    tiltResultBuffer.release();

    // test mult
    // wgpu::Buffer multBuffer = createBuffer(
    //     context.device, nullptr,
    //     sizeof(float) * matrix_shape[0]*matrix_shape[1]*2,
    //     WGPUBufferUsage(wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc)
    // );
    // std::cout << "\ntesting mult" << std::endl;
    // mult(context, multBuffer, complexInputBuffer, inputBuffer, matrix_shape[0]*matrix_shape[1]);
    // std::vector<float> multOutput = readBack(context.device, context.queue, matrix_shape[0]*matrix_shape[1]*2, multBuffer);
    // std::vector<float> in1 = readBack(context.device, context.queue, matrix_shape[0]*matrix_shape[1]*2, complexInputBuffer);
    // std::vector<float> in2 = readBack(context.device, context.queue, matrix_shape[0]*matrix_shape[1], inputBuffer);
    // printArray("frwrd", in1);
    // printArray("pupil", in2);
    // printArray("MULT",multOutput);
    // multBuffer.release();

    // Release WebGPU resources.
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);
    inputBuffer.release();
    complexInputBuffer.release();
    
    return 0;
}
