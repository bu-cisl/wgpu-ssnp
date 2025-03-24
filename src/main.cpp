#define WEBGPU_CPP_IMPLEMENTATION
#include "scatter_factor/scatter_factor.h"
#include "c_gamma/c_gamma.h"
#include "webgpu_utils.h"
#include <vector>
#include <iostream>
#include <iomanip>

using namespace std;

int main() {
    // Initialize WebGPU
    WebGPUContext context;
    initWebGPU(context);

    // Call scatter_factor
    vector<float> input = {5, 21, 65};
    vector<float> output = scatter_factor(context, input);

    // Print scatter_factor output
    cout << "scatter_factor output: " << endl;
    for (float o : output) cout << fixed << setprecision(8) << o << " ";
    cout << endl;

    // Test c_gamma
    std::vector<float> res = {5.2f, 2.2f};
    std::vector<int> shape = {3, 2};
    vector<float> cgamma = c_gamma(context, res, shape);

    // Print c_gamma output
    cout << "c_gamma output:" << endl;

    for (float c : cgamma) cout << fixed << setprecision(4) << c << " ";
    cout << endl;

    // Release WebGPU resources
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);

    return 0;
}