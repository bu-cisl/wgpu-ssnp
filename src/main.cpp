#include "model_dispatcher.h"
#include "ssnp/inverse.h"
#include "utils/testing_io.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// Main for testing script
int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <model> <input.bin> <output.bin>" << endl;
        return 1;
    }

    // PARSING INPUTS
    string model_type = argv[1];
    string input_filename = argv[2];
    string output_filename = argv[3];

    // INITIALIZING WEBGPU
    WebGPUContext context;
    initWebGPU(context);

    if (model_type == "ssnp_reconstruct") {
        testing_io::ReconstructionInput input;
        if (!testing_io::read_reconstruction_input(input_filename, input)) return 1;

        ssnp::ReconstructionOptions options;
        options.max_iterations = input.max_iterations;
        options.learning_rate = input.learning_rate;
        options.abs_tol = input.abs_tol;
        options.rel_tol = input.rel_tol;
        options.print_every = input.print_every;
        options.verbose = input.verbose;

        auto result = ssnp::reconstruct(
            context,
            input.measured,
            input.angles,
            input.initial_volume,
            input.res,
            input.na,
            input.n0,
            options
        );

        if (!testing_io::write_output_tensor(output_filename, result.volume)) return 1;
        return 0;
    }

    vector<vector<vector<float>>> input_tensor;
    int D, H, W;

    if (!testing_io::read_input_tensor(input_filename, input_tensor, D, H, W)) return 1;

    // DISPATCHING THE REQUESTED MODEL
    vector<float> res = {0.1f, 0.1f, 0.1f};
    float na = 0.65f;
    int outputType = 1;
    float n0 = 1.33f;
    vector<vector<float>> angles(1, vector<float>(2, 0.0f)); // default [0, 0]

    auto result = dispatch_model(model_type, context, input_tensor, res, na, angles, n0, outputType);

    if (!testing_io::write_output_tensor(output_filename, result)) return 1;

    return 0;
}

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

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
