emcmake cmake -B build-web
cmake --build build-web
cp ssnp_cpp.html build-web
python -m http.server -d build-web