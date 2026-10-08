#!/bin/bash
# Build one variant of the kv-stream test setup (incremental).
#   ./stream-build.sh            # streaming variant -> build_stream
#   ./stream-build.sh direct     # DIRECT variant -> build_direct
#   ./stream-build.sh clean      # wipe both build dirs
set -e
cd "$(dirname "$0")"

# Every pair the segmented span tile can represent, per
# ggml_kv_stream_span_pair_supported: flash_attn_ext_f16_dequantize_8 converts
# quantized rows for Q8_0 and Q4_0 only, so the reachable set is
# {F16, Q8_0, Q4_0} squared. A pair has to be compiled to be reachable, since
# the KV stream instance template gates on GGML_CUDA_FA_<K>_<V>.
ALL_QUANTS="q4_0-q4_0;q4_1-q4_1;q5_1-q5_1;q8_0-q8_0;f16-f16;bf16-bf16;\
q8_0-q4_0;q4_0-q8_0;\
f16-q8_0;f16-q4_0;q8_0-f16;q4_0-f16"

case "${1:-f16}" in
    clean) rm -rf build_stream build_direct; exit 0 ;;
    f16)    BUILD=build_stream
            QUANTS="$ALL_QUANTS" ;;
    direct) BUILD=build_direct
            QUANTS="$ALL_QUANTS" ;;
    *) echo "usage: $0 [f16|direct|clean]"; exit 1 ;;
esac

LOG="$BUILD/build.log"

# A build that is interrupted leaves a stale binary in place, and the next run
# looks fine while testing code that is not in it. Tie the log to the tree it
# was built from and only write the completion marker on a clean exit.
{
    echo "=== $(date -Is)  HEAD $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    git diff --stat -- . 2>/dev/null | tail -1 | sed 's/^/    dirty: /'
    echo "    quants: $QUANTS"
} > "$LOG"

set -o pipefail
cmake -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_HIP=ON -DGGML_HIP_ROCWMMA_FATTN=ON \
    -DGGML_CUDA_FA_QUANTS="$QUANTS" \
    -DGPU_TARGETS=gfx1101 -DGPU_BUILD_TARGETS=gfx1101 \
    -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX_VNNI=ON -DGGML_AVX_BMI=ON -DGGML_FMA=ON \
    -DGGML_SSE42=ON -DGGML_F16C=ON \
    -DGGML_NATIVE=ON >> "$LOG" 2>&1

cmake --build "$BUILD" -t llama-server -t llama-cli -t llama-completion --parallel 12 2>&1 | tee -a "$LOG"

echo "=== OK $(date -Is)  $BUILD/bin/llama-server $(stat -c %y "$BUILD/bin/llama-server" 2>/dev/null | cut -d. -f1)" >> "$LOG"
echo "log: $LOG"
