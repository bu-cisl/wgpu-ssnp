cmake_minimum_required(VERSION 3.0...3.25)
project(
    SSNP_CPP
    VERSION 0.1.0
    LANGUAGES CXX C
)

add_executable( 
    ssnp_cpp 
    src/main.cpp 
    src/webgpu_utils.cpp
    src/scatter_factor/scatter_factor.cpp 
    src/c_gamma/c_gamma.cpp 
    src/diffract/diffract.cpp
    src/binary_pupil/binary_pupil.cpp
    src/tilt/tilt.cpp
    src/merge_prop/merge_prop.cpp
    src/split_prop/split_prop.cpp
    src/dft/dft.cpp
    src/mult/mult.cpp
)

set_target_properties(ssnp_cpp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    COMPILE_WARNING_AS_ERROR ON
)

if (MSVC)
    target_compile_options(ssnp_cpp PRIVATE /W4)
else()
    target_compile_options(ssnp_cpp PRIVATE -Wall -Wextra -pedantic)
endif()

# Include WebGPU subdirectory
add_subdirectory(webgpu)

# Link WebGPU library to ssnp_cpp
target_link_libraries(ssnp_cpp PRIVATE webgpu)

# Copy necessary runtime binaries
target_copy_webgpu_binaries(ssnp_cpp)