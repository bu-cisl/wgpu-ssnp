#define WEBGPU_CPP_IMPLEMENTATION
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "model_dispatcher.h"

using namespace std;

bool read_input_tensor(const string& filename, vector<vector<vector<float>>>& tensor, int& D, int& H, int& W) {
    ifstream in(filename, ios::binary);
    if (!in) {
        cerr << "Failed to open input file: " << filename << endl;
        return false;
    }

    // Read dimensions (3 x int32)
    in.read(reinterpret_cast<char*>(&D), sizeof(int));
    in.read(reinterpret_cast<char*>(&H), sizeof(int));
    in.read(reinterpret_cast<char*>(&W), sizeof(int));

    size_t total = static_cast<size_t>(D) * H * W;
    vector<float> buffer(total);
    in.read(reinterpret_cast<char*>(buffer.data()), total * sizeof(float));
    in.close();

    // Convert to 3D tensor
    tensor.resize(D, vector<vector<float>>(H, vector<float>(W)));
    size_t idx = 0;
    for (int d = 0; d < D; ++d)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                tensor[d][i][j] = buffer[idx++];

    return true;
}

bool write_output_tensor(const string& filename, const vector<vector<vector<float>>>& tensor) {
    ofstream out(filename, ios::binary);
    if (!out) {
        cerr << "Failed to open output file: " << filename << endl;
        return false;
    }

    int D = tensor.size();
    int H = tensor[0].size();
    int W = tensor[0][0].size();

    out.write(reinterpret_cast<char*>(&D), sizeof(int));
    out.write(reinterpret_cast<char*>(&H), sizeof(int));
    out.write(reinterpret_cast<char*>(&W), sizeof(int));

    for (int d = 0; d < D; ++d)
        for (int i = 0; i < H; ++i)
            out.write(reinterpret_cast<const char*>(tensor[d][i].data()), W * sizeof(float));

    out.close();
    return true;
}

// Main for testing script
int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <model> <input.bin> <output.bin>" << endl;
        return 1;
    }

    string model_type = argv[1];
    string input_filename = argv[2];
    string output_filename = argv[3];

    vector<vector<vector<float>>> input_tensor;
    int D, H, W;

    if (!read_input_tensor(input_filename, input_tensor, D, H, W)) return 1;

    WebGPUContext context;
    initWebGPU(context);

    vector<float> res = {0.1f, 0.1f, 0.1f};
    float na = 0.65f;
    int outputType = 1;
    float n0 = 1.33f;
    vector<vector<float>> angles(1, vector<float>(2, 0.0f)); // default [0, 0]

    auto result = dispatch_model(model_type, context, input_tensor, res, na, angles, n0, outputType);

    if (!write_output_tensor(output_filename, result)) return 1;

    return 0;
}

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <sstream>

EM_JS(void, plot_from_heap, (uintptr_t ptr, int len, int H, int W, float mn, float mx), {
  var view = new Float32Array(HEAPF32.buffer, ptr, len);
  plotSlices(view, H|0, W|0, Number(mn), Number(mx));
});

EM_JS(void, plot_complex_from_heap, (uintptr_t ptr, int len, int H, int W, float magMin, float magMax, float phaseMin, float phaseMax), {
  var view = new Float32Array(HEAPF32.buffer, ptr, len * 2);
  plotComplexSlices(view, H|0, W|0, Number(magMin), Number(magMax), Number(phaseMin), Number(phaseMax));
});

extern "C" {
    EMSCRIPTEN_KEEPALIVE
    void callModel(const char* model, uintptr_t dataPtr, int D, int H, int W, const char* paramsStr) {
        try {
            // Parse params
            std::string fullInput(paramsStr);
            size_t firstPipe = fullInput.find('|');
            size_t secondPipe = fullInput.find('|', firstPipe + 1);
            size_t thirdPipe = fullInput.find('|', secondPipe + 1);
            size_t fourthPipe = fullInput.find('|', thirdPipe + 1);

            std::string anglesStr = fullInput.substr(0, firstPipe);
            std::string resStr = fullInput.substr(firstPipe + 1, secondPipe - firstPipe - 1);
            std::string naStr = fullInput.substr(secondPipe + 1, thirdPipe - secondPipe - 1);
            std::string outputTypeStr = fullInput.substr(thirdPipe + 1, fourthPipe - thirdPipe - 1);
            std::string n0Str = fullInput.substr(fourthPipe + 1);

            // Parse angles
            std::vector<std::vector<float>> angles;
            std::istringstream angleSS(anglesStr);
            std::string token;
            while (std::getline(angleSS, token, ';')) {
                std::istringstream pairStream(token);
                std::string xStr, yStr;
                std::getline(pairStream, xStr, ',');
                std::getline(pairStream, yStr, ',');
                angles.push_back({std::stof(xStr), std::stof(yStr)});
            }

            // Parse resolution
            std::vector<float> res;
            std::istringstream resSS(resStr);
            while (std::getline(resSS, token, ',')) {
                res.push_back(std::stof(token));
            }

            float na = std::stof(naStr);
            int outputType = std::stoi(outputTypeStr);
            float n0 = std::stof(n0Str);

            // Read data from heap & convert to 3D tensor
            float* heapData = reinterpret_cast<float*>(dataPtr);
            vector<vector<vector<float>>> tensor(D, vector<vector<float>>(H, vector<float>(W)));
            size_t idx = 0;
            for (int d = 0; d < D; ++d)
                for (int i = 0; i < H; ++i)
                    for (int j = 0; j < W; ++j)
                        tensor[d][i][j] = heapData[idx++];

            // Init WebGPU
            WebGPUContext context;
            initWebGPU(context);

            // Pass n0 to forward function
            auto result = dispatch_model(std::string(model), context, tensor, res, na, angles, n0, outputType);
            
            // Complex output
            if (outputType == 2) {
                auto realPart = result[0];
                auto imagPart = result[1];
                
                size_t N = (size_t)H * (size_t)W;
                float* complexOut = (float*)malloc(sizeof(float) * N * 2);
                
                // Calculate magnitude and phase
                float magMin = INFINITY, magMax = -INFINITY;
                float phaseMin = INFINITY, phaseMax = -INFINITY;
                
                for (int i = 0; i < H; ++i) {
                    for (int j = 0; j < W; ++j) {
                        float real = realPart[i][j];
                        float imag = imagPart[i][j];
                        float mag = sqrt(real*real + imag*imag);
                        float phase = atan2(imag, real);
                        
                        size_t idx = i * W + j;
                        complexOut[idx * 2] = real;
                        complexOut[idx * 2 + 1] = imag;
                        
                        if (mag < magMin) magMin = mag;
                        if (mag > magMax) magMax = mag;
                        if (phase < phaseMin) phaseMin = phase;
                        if (phase > phaseMax) phaseMax = phase;
                    }
                }
                
                plot_complex_from_heap((uintptr_t)complexOut, N, H, W, magMin, magMax, phaseMin, phaseMax);
                free(complexOut);
            } 
            
            // Amplitude/Intensity outputs
            else {
                auto output = result[0];
                
                size_t N = (size_t)H * (size_t)W;
                float* out = (float*)malloc(sizeof(float) * N);
                size_t k = 0;

                float localMin = output[0][0];
                float localMax = output[0][0];
                for (int i = 0; i < H; ++i) {
                    for (int j = 0; j < W; ++j) {
                        float v = output[i][j];
                        out[k++] = v;
                        if (v < localMin) localMin = v;
                        if (v > localMax) localMax = v;
                    }
                }

                plot_from_heap((uintptr_t)out, N, H, W, localMin, localMax);
                free(out);
            }

        } catch (const std::exception &e) {
            printf("C++ exception: %s\n", e.what());
        } catch (...) {
            printf("Unknown C++ exception\n");
        }
    }
}
#endif