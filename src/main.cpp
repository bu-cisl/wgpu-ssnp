#include "scatter_factor.h"
#include "webgpu_utils.h"
#include <vector>
#include <iostream>

using namespace std;

int main() {
    // Initialize WebGPU
    WebGPUContext context;
    if (!initWebGPU(context)) {
        cerr << "Failed to initialize WebGPU!" << endl;
        return -1;
    }

    // Call scatter_factor
    vector<float> input = {1, 2, 3};
    vector<float> output = scatter_factor(context, input);

    // Print output
    cout << "Scatter factor output: " << endl;
    for (float o : output) {
        cout << o << " ";
    }
    cout << endl;

    // Release WebGPU resources
    wgpuQueueRelease(context.queue);
    wgpuDeviceRelease(context.device);
    wgpuAdapterRelease(context.adapter);
    wgpuInstanceRelease(context.instance);

    return 0;
}