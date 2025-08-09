#!/bin/bash
module load python3
source ./emsdk/emsdk_env.sh

emcmake cmake -B docs
cmake --build docs

python -m http.server -d docs
