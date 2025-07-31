cmake_minimum_required(VERSION 3.0...3.25)
project(SSNP_BENCHMARK VERSION 0.1.0 LANGUAGES CXX C)

add_executable( 
    benchmark 
    main.cpp 
    ../src/webgpu_utils.cpp
    ../src/scatter_factor/scatter_factor.cpp 
    ../src/c_gamma/c_gamma.cpp 
    ../src/diffract/diffract.cpp
    ../src/binary_pupil/binary_pupil.cpp
    ../src/tilt/tilt.cpp
    ../src/merge_prop/merge_prop.cpp
    ../src/split_prop/split_prop.cpp
    ../src/dft/dft.cpp
    ../src/mult/mult.cpp
    ../src/scatter_effects/scatter_effects.cpp
    ../src/intensity/intensity.cpp
    ../src/forward.cpp
)

set_target_properties(benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

add_subdirectory(../webgpu webgpu)
target_link_libraries(benchmark PRIVATE webgpu)
target_copy_webgpu_binaries(benchmark)