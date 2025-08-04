#!/bin/bash

# Create docs/ dir for build
module load python3
source ./emsdk/emsdk_env.sh
emcmake cmake -B docs

# Build + save html as index for git
cmake --build docs
cp ssnp_cpp.html docs/index.html
cp UTIF.js docs/UTIF.js
python -m http.server -d docs
