#!/bin/bash

# Create docs/ dir for build
if [ ! -d "docs" ]; then
    source ./emsdk/emsdk_env.sh
    emcmake cmake -B docs
fi

# Build + save html as index for git
cmake --build docs
cp ssnp_cpp.html docs/index.html
python -m http.server -d docs
