#!/bin/bash
emcc \
  src/main.cpp \
  src/forward.cpp \
  src/webgpu_utils.cpp \
  src/scatter_factor/scatter_factor.cpp \
  src/c_gamma/c_gamma.cpp \
  src/diffract/diffract.cpp \
  src/binary_pupil/binary_pupil.cpp \
  src/tilt/tilt.cpp \
  src/merge_prop/merge_prop.cpp \
  src/split_prop/split_prop.cpp \
  src/dft/dft.cpp \
  src/mult/mult.cpp \
  src/scatter_effects/scatter_effects.cpp \
  src/intensity/intensity.cpp \
  -o buildJS/index.js \
  -s WASM=1 \
  -s USE_WEBGPU=1 \
  -s ASYNCIFY \
  -s NO_EXIT_RUNTIME=1 \
  -s FORCE_FILESYSTEM=1 \
  -s EXPORTED_FUNCTIONS='["_run_forward_wasm","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='[
     "cwrap","UTF8ToString","HEAPU8","HEAPF32","print",
     "noExitRuntime","FS","FS_createDataFile",
     "FS_createPreloadedFile","PATH","preloadPlugins"
  ]' \
  -s EXPORT_ES6=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME="createModule" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s TOTAL_STACK=16MB \
  -s INITIAL_MEMORY=64MB \
  -s ASSERTIONS=1 \
  -gsource-map \
  --preload-file src/tilt/tilt.wgsl \
  --preload-file src/dft/dft_row.wgsl \
  --preload-file src/dft/dft_col.wgsl \
  --preload-file src/merge_prop/merge_prop.wgsl \
  --preload-file src/c_gamma/c_gamma.wgsl \
  --preload-file src/diffract/diffract.wgsl \
  --preload-file src/scatter_factor/scatter_factor.wgsl \
  --preload-file src/scatter_effects/complex_mult.wgsl \
  --preload-file src/scatter_effects/complex_sub.wgsl \
  --preload-file src/binary_pupil/binary_pupil.wgsl \
  --preload-file src/mult/mult.wgsl \
  --preload-file src/intensity/intensity.wgsl \
  --preload-file src/split_prop/split_prop.wgsl \
  -Iwebgpu/include \
  -Isrc \
  -O3