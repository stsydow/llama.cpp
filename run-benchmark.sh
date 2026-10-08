#!/bin/bash
# One batch of the pending kv-stream benchmarks, streamed to the console
# and logged separately. Run in a tmux/screen pane:
#   ./run-benchmark.sh 2>&1 | tee kvtrace/bench.out
set -o pipefail
cd "$(dirname "$0")"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="kvtrace/bench-$STAMP"
mkdir -p "$OUT"

echo "== q8_0 matrix (f16/direct x 3 configs, ~25 min) =="
./stream-test.py --sweep --cache-type q8_0 2>&1 | tee "$OUT/q8.log"
echo "q8 rc=$?"

echo "== q5_1 token validation, 30k a400 (~7 min) =="
./stream-test.py --sweep --config 30000:400 2>&1 | tee "$OUT/q5-tokens.log"
echo "q5 rc=$?"

echo "logs in $OUT"
