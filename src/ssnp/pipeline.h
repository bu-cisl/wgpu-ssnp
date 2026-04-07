#ifndef SSNP_PIPELINE_H
#define SSNP_PIPELINE_H

#include "scatter_factor/scatter_factor.h"
#include "ssnp_diffract/ssnp_diffract.h"
#include "../common/binary_pupil/binary_pupil.h"
#include "../common/tilt/tilt.h"
#include "merge_prop/merge_prop.h"
#include "split_prop/split_prop.h"
#include "../common/dft/dft.h"
#include "../common/mult/mult.h"
#include "scatter_effects/scatter_effects.h"
#include "../common/intensity/intensity.h"
#include "../common/webgpu_utils.h"
#include <vector>

namespace ssnp {

struct SSNPState {
    wgpu::Buffer U;
    wgpu::Buffer UD;
};

SSNPState initialize_angle_state(
    WebGPUContext& context,
    const std::vector<float>& c_ba,
    const std::vector<int>& shape,
    const std::vector<float>& res
);

SSNPState propagate_to_object_exit(
    WebGPUContext& context,
    SSNPState state,
    const std::vector<std::vector<std::vector<float>>>& n,
    const std::vector<int>& shape,
    const std::vector<float>& res,
    float n0
);

wgpu::Buffer project_state_to_sensor_field(
    WebGPUContext& context,
    const SSNPState& state,
    const std::vector<int>& shape,
    const std::vector<float>& res,
    float na,
    float focal_offset
);

std::vector<float> flatten_real_slice(const std::vector<std::vector<float>>& slice);
wgpu::Buffer create_complex_slice_buffer(WebGPUContext& context, const std::vector<std::vector<float>>& slice);
void release_state(SSNPState& state);

}

#endif
