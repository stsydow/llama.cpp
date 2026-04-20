#!/bin/bash

rm build -r
mkdir -p build

cmake -B build \
    -DGGML_VULKAN=ON \
    -DGGML_USE_OPENMP=ON \
    -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX_VNNI=ON -DGGML_AVX_BMI=ON -DGGML_FMA=ON \
    -DGGML_SSE42=ON -DGGML_F16C=ON \
    -DGGML_NATIVE=ON \

cmake --build build --config Release -t llama-server --parallel 16
