cmake_minimum_required(VERSION 3.0...3.25)
project(
    OPTICS_SIM
    VERSION 0.1.0
    LANGUAGES CXX C
)

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

add_executable(optics_sim ${SRC_FILES})

set_target_properties(optics_sim PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    COMPILE_WARNING_AS_ERROR ON
)

if (MSVC)
    target_compile_options(optics_sim PRIVATE /W4)
else()
    target_compile_options(optics_sim PRIVATE -Wall -Wextra -pedantic)
endif()

# Include WebGPU subdirectory
add_subdirectory(external/webgpu)

# Link WebGPU library
target_link_libraries(optics_sim PRIVATE webgpu)

# Copy necessary runtime binaries
target_copy_webgpu_binaries(optics_sim)

# Copy web assets to docs/
add_custom_target(copy_web ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/web
            ${CMAKE_BINARY_DIR}
    COMMENT "Copying web assets to build directory"
)

if (EMSCRIPTEN)
    # apply Emscripten flags
    target_link_options(optics_sim PRIVATE
        "-sUSE_WEBGPU"   # hook WebGPU calls into JS API
        "-sASYNCIFY"     # allow sync requestAdapter/Device
        "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/src@/src"  
        "-sALLOW_MEMORY_GROWTH"  # let the browser grow the Wasm heap
        "-sEXPORTED_FUNCTIONS=['_callSSNP']"
        "-sEXPORTED_RUNTIME_METHODS=['ccall']"
        "-sNO_DISABLE_EXCEPTION_CATCHING"  
    )
endif()