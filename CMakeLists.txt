cmake_minimum_required(VERSION 3.0...3.25)
project(
    OPTICS_SIM
    VERSION 0.1.0
    LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# SHARED CORE LIBRARY FOR THE APP AND TESTS
list(REMOVE_ITEM SRC_FILES ${CMAKE_SOURCE_DIR}/src/main.cpp)

add_library(optics_core ${SRC_FILES})
add_executable(optics_sim src/main.cpp)

foreach(target optics_core optics_sim)
    set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
endforeach()

if (MSVC)
    target_compile_options(optics_core PRIVATE /W4)
    target_compile_options(optics_sim PRIVATE /W4)
else()
    target_compile_options(optics_core PRIVATE -Wall -Wextra -pedantic)
    target_compile_options(optics_sim PRIVATE -Wall -Wextra -pedantic)
endif()

add_subdirectory(webgpu)

# LINKING WEBGPU AND SHARED CORE TARGETS
target_link_libraries(optics_core PUBLIC webgpu)
target_link_libraries(optics_sim PRIVATE optics_core)

target_copy_webgpu_binaries(optics_sim)

add_custom_target(copy_web ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/web
            ${CMAKE_BINARY_DIR}
    COMMENT "Copying web assets to build directory"
)

if (EMSCRIPTEN)
  target_link_options(optics_sim PRIVATE
    -sUSE_WEBGPU
    -sASYNCIFY
    "--preload-file=${CMAKE_CURRENT_SOURCE_DIR}/src@/src"
    -sALLOW_MEMORY_GROWTH=1
    -O3
    -msimd128
    "-sEXPORTED_FUNCTIONS=['_callModel','_malloc','_free']"
    "-sEXPORTED_RUNTIME_METHODS=['ccall','HEAPF32','HEAPU8']"
    -sNO_DISABLE_EXCEPTION_CATCHING
  )
endif()
