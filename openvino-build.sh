#!/bin/bash

#source /opt/intel/oneapi/setvars.sh
#-DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \

source /opt/intel/openvino_2026/setupvars.sh

rm build -r
mkdir -p build

cmake -B build \
    -DGGML_OPENVINO=ON \
    -DOpenVINO_DIR="/opt/intel/openvino_2026/runtime/cmake" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX_VNNI=ON -DGGML_AVX_BMI=ON -DGGML_FMA=ON \
    -DGGML_SSE42=ON -DGGML_F16C=ON \
    # -DGGML_USE_OPENMP=ON \
    # -DGGML_NATIVE=ON \

cmake --build build --config Release -t llama-server --parallel 16
