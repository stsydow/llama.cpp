#!/bin/bash

source /opt/intel/oneapi/setvars.sh

rm build -r
mkdir -p build

cmake -B build \
    -DGGML_SYCL=ON -DENABLE_TRY_SYCL_COMPILE=ON -DGGML_SYCL_DEVICE_ARCH=acm_g10 \
    -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_DNN=ON \
    -DGGML_SYCL_F16=ON \
    -DGGML_VULKAN=ON \
    -DGGML_USE_OPENMP=ON \
    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX_VNNI=ON -DGGML_AVX_BMI=ON -DGGML_FMA=ON \
    -DGGML_SSE42=ON -DGGML_F16C=ON \
    -DGGML_NATIVE=ON \

cmake --build build --config Release -t llama-server --parallel 16
