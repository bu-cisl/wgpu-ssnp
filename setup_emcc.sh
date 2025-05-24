#!/usr/bin/env bash
cd emsdk
git pull
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
module load python3
echo 
echo "emcc installed and activated"