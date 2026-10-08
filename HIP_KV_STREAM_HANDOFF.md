# HIP KV streaming - open items

Only what is still open on this branch and the context needed to act on it.
Design intent: `MTP_ADAPTIVE_KV_ROADMAP.md`. Memory architecture and invariants:
`DEVICE_MEMORY_INFRASTRUCTURE.md`. Staged delivery plan:
`DEVICE_MEMORY_CONSUMERS_ROADMAP.md`.

## 1. Validate the four remaining span pairs

Cheapest first:

```
F16/F16   Q4_0/Q4_0   F16/Q4_0   Q4_0/F16
```

All four are reachable and compiled on the span path, each needs one A/B. A
failure now means a defect, not a refusal.

Q5_0, Q5_1, Q4_1 and BF16 cannot be validated on the span path on any backend:
the tile's row dequantiser handles only Q8_0 and Q4_0 (exercised by
`convert_mma_rows` in `ggml/src/ggml-cuda/kv-stream-partial.cu`), so the tile
can represent `{F16, Q8_0, Q4_0}` squared - nine pairs.

## 2. 256K with the qualified pair

The roadmap target and the model's reported `context_length`. This is the first
length where Vulkan is a live alternative rather than a reference, because a
resident 256K context does not fit in 16 GB.

If 256K needs arena auto-sizing rather than a hand-tuned value, the candidate is
`--kv-stream-arena-fit`, on the separate `kv-stream-fit-arena` branch.

## 3. MTP at 256K

The reason the shared target/MTP phase arena exists. Context:

- With streaming, `--spec-type` must be `draft-mtp` alone.
  `common_kv_stream_auxiliary_layers` (`common/common.cpp`) throws
  "adaptive KV speculation supports only draft-mtp" for anything else, so the
  preset's `draft-mtp,ngram-mod` combination is not usable with an arena.
  `stream-test.py` rejects it before starting a server.
- The `-mtp` model build is required; the default model reports no MTP layers,
  and the server only says so after loading 11.8 GiB.
- `--kv-stream-auxiliary-layers` is a consistency assertion, not an opt-in
  switch. The layer count comes from `llama_model_n_layer_nextn(model)`.

## 4. Restore the branch's spill scenario

The branch uniquely tested end-to-end spilled-vs-stock logit equivalence.
`tests/test-memory-recurrent-spill.cpp` is layout arithmetic only and never
compares against a stock context. Restore it by extending that test, not by
restoring the deleted 331-line `main()` that superseded master rather than
extending it.

## 5. `--vs-vulkan` has a single data point

It works - 30K q5_1, IDENTICAL 256/256 - but is exercised once. `build/` is the
Vulkan binary and has both HIP and Vulkan enabled. `--vs-vulkan` runs a streamed
number against a resident baseline, not a streaming race; `--backend` selects
`{hip,direct,vulkan}` and the arena flag is dropped for vulkan.

## 6. Q8_0 100K decode shortfall (upstream)

At 100K, q8_0 decode is 60.2% `flash_attn_ext_vec_resume` at 52 GB/s against
q5_1's 169 GB/s on a card that does about 624, with identical call counts and
correct dispatch. Stock Q8_0 FlashAttention efficiency in `ggml-cuda`, upstream
of this branch; streaming is 5.7% of that cost. Profile:
`kvtrace/p-q8b_kernel_trace.csv`.

## Operational notes

- `kvtrace/kvtrace_kernel_trace.csv` is stale v1 data from Sep 14 and is
  dominated by a kernel `build_stream` cannot contain. Do not analyse it.
- `build_stream/build.log` records HEAD, the dirty-file count and an `=== OK`
  marker written only on clean exit. A run without the marker tested a stale
  binary; the dirty count is the case `git log` cannot answer.
- ROCm compiles are slow and not usefully parallel. Batch changes and rebuild
  once.
- `rocm-smi --showtemp --showclocks --json` needs care: keys end in a colon,
  values are parenthesised, and every clock has a sibling `"clock level:"` entry
  holding a letter that matches the same substring. The end-of-run SCLK sample
  is racy; read the start-of-run clock.

## Commands

```sh
python3 stream-test.py --ctx 30000 --cache-type q5_1 --ab 2600 400
python3 stream-test.py --ctx 30000 --cache-type-k q8_0 --cache-type-v q4_0 --vs-vulkan
python3 stream-test.py --ctx 30000 --spec-type draft-mtp --ab 2600 400 \
  --model ~/models/Qwen3.8-27B/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
```

Draft defaults follow `models/preset.ini`: length 2, `q4_0` draft caches,
`--spec-draft-device none`. `--cache-type` sets K and V together; use
`--cache-type-k` / `--cache-type-v` for mixed pairs.
