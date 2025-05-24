emcmake cmake -B build-web
cmake --build build-web
python -m http.server -d build-web