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
  -s EXPORTED_FUNCTIONS='["_run_forward_wasm"]' \
  -s EXPORTED_RUNTIME_METHODS='["cwrap", "getValue", "setValue", "UTF8ToString"]' \
  -Iwebgpu/include \
  -Isrc \
  -O3