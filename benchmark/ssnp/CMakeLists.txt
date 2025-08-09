cmake_minimum_required(VERSION 3.0...3.25)
project(SSNP_BENCHMARK VERSION 0.1.0 LANGUAGES CXX C)

add_executable( 
    benchmark 
    main.cpp 
    ../../src/ssnp/webgpu_utils.cpp
    ../../src/ssnp/scatter_factor/scatter_factor.cpp 
    ../../src/ssnp/c_gamma/c_gamma.cpp 
    ../../src/ssnp/diffract/diffract.cpp
    ../../src/ssnp/binary_pupil/binary_pupil.cpp
    ../../src/ssnp/tilt/tilt.cpp
    ../../src/ssnp/merge_prop/merge_prop.cpp
    ../../src/ssnp/split_prop/split_prop.cpp
    ../../src/ssnp/dft/dft.cpp
    ../../src/ssnp/mult/mult.cpp
    ../../src/ssnp/scatter_effects/scatter_effects.cpp
    ../../src/ssnp/intensity/intensity.cpp
    ../../src/ssnp/forward.cpp
)

set_target_properties(benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_subdirectory(../../webgpu webgpu)
target_link_libraries(benchmark PRIVATE webgpu)
target_copy_webgpu_binaries(benchmark)