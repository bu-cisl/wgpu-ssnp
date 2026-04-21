cmake_minimum_required(VERSION 3.0...3.25)
project(SSNP_BENCHMARK VERSION 0.1.0 LANGUAGES CXX C)

add_subdirectory(../.. wgpu_ssnp)

add_executable(benchmark main.cpp)

set_target_properties(benchmark PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if (MSVC)
    target_compile_options(benchmark PRIVATE /W4)
else()
    target_compile_options(benchmark PRIVATE -Wall -Wextra -pedantic)
endif()

target_link_libraries(benchmark PRIVATE optics_core)
target_copy_webgpu_binaries(benchmark)
