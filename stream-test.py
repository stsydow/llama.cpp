#!/usr/bin/env python3
"""
kv-stream test: starts llama-server, sends the fill-token prompt, prints
prefill/decode t/s, stops the server, and prints the attention-mode line.

Usage:
    python3 stream-test.py                    # streamed config (arena 400)
    python3 stream-test.py --arena 2600       # resident comparison
    python3 stream-test.py --sweep            # hip/direct x ctx/arena matrix
"""

import argparse
import json
import os
import socket
import signal
import subprocess
import sys
import threading
import time
import urllib.request

HOST = "127.0.0.1"
STARTUP_TIMEOUT = 300
DEFAULT_MODEL = os.path.expanduser(
    "~/models/Qwen3.8-27B/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf")

def _killpg(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass

# backend -> (binary, device arg, supports KV streaming). Vulkan has no KV
# stream, so its rows are resident baselines and the arena flag is dropped.
BACKENDS = {
    "hip":     ("./build_stream/bin/llama-server", ["--device", "Rocm0"], True),
    "direct":  ("./build_direct/bin/llama-server",  ["--device", "Rocm0"], True),
    "vulkan":  ("./build/bin/llama-server",         ["--device", "Vulkan0"], False),
}

def server_args(a):
    binary, device, streams = BACKENDS[a.backend]
    args = [
        binary,
        "-m", a.model,
        "--host", HOST,
        "--port", str(a.port),
        "--ctx-size", str(a.ctx),
        "--parallel", "1",
        "--cache-type-k", a.cache_type_k,
        "--cache-type-v", a.cache_type_v,
        "--flash-attn", "on",
        "--fit", "off",
    ] + device + [
        "-ngl", "all",
        "--no-warmup",
        "-t", str(a.threads),
    ]
    if streams:
        args += ["--kv-stream-arena-mib", str(a.arena)]
    if a.spec_type != "none":
        args += ["--spec-type", a.spec_type,
                 "--spec-draft-n-max", str(a.draft_n_max),
                 "--spec-draft-p-min", str(a.draft_p_min),
                 "--spec-draft-device", "none",
                 "--spec-draft-type-k", a.draft_cache_type,
                 "--spec-draft-type-v", a.draft_cache_type]
    return args

def post(port, path, body):
    req = urllib.request.Request(
        f"http://{HOST}:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=7200))

def port_in_use(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0

def _pump(proc, sink, echo):
    for line in proc.stdout:
        line = line.rstrip("\n")
        sink.append(line)
        if echo:
            print("  " + line, flush=True)

def start_server(a):
    log = []
    cmd = server_args(a)
    if a.profile:
        os.makedirs(a.trace_dir, exist_ok=True)
        cmd = ["rocprofv3", "--kernel-trace", "--output-format", "csv",
               "-o", a.trace_dir, "--"] + cmd
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True)
    threading.Thread(target=_pump, args=(proc, log, not a.quiet),
                     daemon=True).start()
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        time.sleep(1)
        if proc.poll() is not None:
            tail = log[-20:]
            print(f"[ERR] server exited during startup (rc={proc.returncode})")
            print("\n".join(tail))
            sys.exit(1)
        try:
            urllib.request.urlopen(f"http://{HOST}:{a.port}/health", timeout=2)
            return proc, log
        except OSError:
            pass
    print(f"[ERR] server did not become healthy within {STARTUP_TIMEOUT}s")
    sys.exit(1)

def stop_server(proc, log, profile):
    if profile:
        # A single SIGINT finalizes the rocprofv3 trace
        _killpg(proc.pid, signal.SIGINT)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=60 if profile else 10)
    except subprocess.TimeoutExpired:
        _killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    time.sleep(2)
    proc.stdout.close()
    for line in log:
        if "attention mode" in line:
            print("mode:", line.strip())

SWEEP_VARIANTS = ("hip", "direct")
SWEEP_CONFIGS = ((30000, 400), (30000, 2600), (100000, 1000))

def gpu_state():
    # edge temperature and core/memory clocks; a 7800 XT drops clocks well before
    # it is hot, so a benchmark row without this is not comparable. The JSON keys
    # end in a colon and values are parenthesised, and each clock has a matching
    # "clock level:" entry that is a letter, not a speed.
    out = {}
    try:
        raw = subprocess.run(["rocm-smi", "--showtemp", "--showclocks", "--json"],
                             capture_output=True, text=True, timeout=30).stdout
        d = json.loads(raw)
        card = next(iter(d.values())) if isinstance(d, dict) else d[0]
        for k, v in card.items():
            kl = k.lower()
            if "temperature" in kl and "hotspot" not in kl and "mem" not in kl:
                # prefer the edge sensor, but do not depend on JSON key order
                if "edge" in kl or "temp_c" not in out:
                    out["temp_c"] = str(v).strip("()")
            elif "clock speed" in kl:
                name = kl.split()[0]
                if name in ("sclk", "mclk"):
                    out[name] = str(v).strip("()").lower()
    except Exception as e:
        out["error"] = str(e)[:60]
    return out


def run_once(a):
    if port_in_use(HOST, a.port):
        print(f"[ERR] port {a.port} already in use")
        sys.exit(1)

    proc, log = start_server(a)
    cold = gpu_state()
    try:
        tok = post(a.port, "/tokenize",
                   {"content": " the", "add_special": False})["tokens"]
        print("fill token:", tok[0])
        prompt = [tok[0]] * (a.ctx - 256) + tok
        r = post(a.port, "/completion", {
            "prompt": prompt,
            "n_predict": 256,
            "ignore_eos": True,
            "cache_prompt": False,
            "temperature": 0,
            "seed": 1,
            "return_tokens": True,
        })
        hot = gpu_state()
    finally:
        stop_server(proc, log, a.profile)
    t = r.get("timings", {})
    print("prefill:", t.get("prompt_n"), "tok @",
          t.get("prompt_per_second"), "t/s")
    print("decode :", t.get("predicted_n"), "tok @",
          t.get("predicted_per_second"), "t/s")
    print("gpu   :", f"{cold.get('temp_c')}C/{cold.get('sclk')} ->",
          f"{hot.get('temp_c')}C/{hot.get('sclk')}")
    if a.profile:
        print("trace :", os.path.abspath(a.trace_dir))
    tag = f"{a.ctx}-{a.arena}-{a.cache_type_k}-{a.cache_type_v}-{a.backend}"
    if a.spec_type != "none":
        tag += f"-{a.spec_type}"
    path = os.path.join(a.trace_dir, f"tokens-{tag}.txt")
    os.makedirs(a.trace_dir, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("".join(f"{t}\n" for t in r.get("tokens", [])))
    print("tokens :", path)
    return {"prefill": t.get("prompt_per_second"),
            "decode": t.get("predicted_per_second"),
            "gpu_cold": cold, "gpu_hot": hot,
            "tokens": r.get("tokens", [])}

def compare_tokens(x, y, label):
    n = min(len(x["tokens"]), len(y["tokens"]))
    if n == 0:
        print(f"tokens {label}: no tokens to compare")
        return
    same = sum(1 for i in range(n) if x["tokens"][i] == y["tokens"][i])
    first = next((i for i in range(n) if x["tokens"][i] != y["tokens"][i]), None)
    verdict = "IDENTICAL" if same == n else f"DIVERGES at {first}"
    print(f"tokens {label}: {same}/{n} match - {verdict}")

def ab(base, arenas):
    # streaming must reproduce the resident KV bit for bit, so the larger arena
    # is the baseline regardless of the order they were given in
    baseline, streamed = sorted(arenas, reverse=True)
    runs = {}
    for arena in (baseline, streamed):
        a = argparse.Namespace(**vars(base))
        a.arena, a.profile = arena, False
        print(f"== arena {arena} ==")
        runs[arena] = run_once(a)

    r, s = runs[baseline], runs[streamed]
    print(f"\n== arena {baseline} (resident) vs {streamed} (streamed) ==")
    print(f"{'arena':>6} {'prefill':>10} {'decode':>9} {'gpu':>20}")
    for arena in (baseline, streamed):
        g = runs[arena]
        edge = f"{g['gpu_cold'].get('temp_c')}C->{g['gpu_hot'].get('temp_c')}C"
        print(f"{arena:>6} {g['prefill']:>10.1f} {g['decode']:>9.2f} {edge:>20}")
    # ratio of streamed to resident, so above 1.0 means streaming is faster;
    # calling this a "penalty" made a faster streamed run print 1.07x
    print(f"streamed/resident: prefill {s['prefill']/r['prefill']:.3f}x, "
          f"decode {s['decode']/r['decode']:.3f}x   (>1 = streaming faster)")
    compare_tokens(r, s, f"(a{baseline} vs a{streamed})")

def vs_vulkan(base):
    # Vulkan has no KV streaming, so this is our streamed number against a
    # resident baseline on the other backend, not a streaming-vs-streaming race
    runs = {}
    for backend in (base.backend, "vulkan"):
        a = argparse.Namespace(**vars(base))
        a.backend, a.profile = backend, False
        print(f"== {backend} ==")
        runs[backend] = run_once(a)

    ours, vk = runs[base.backend], runs["vulkan"]
    print(f"\n== {base.backend} arena {base.arena} vs vulkan (resident) ==")
    print(f"{'backend':10} {'prefill':>10} {'decode':>9} {'gpu':>20}")
    for name in (base.backend, "vulkan"):
        g = runs[name]
        edge = f"{g['gpu_cold'].get('temp_c')}C->{g['gpu_hot'].get('temp_c')}C"
        print(f"{name:10} {g['prefill']:>10.1f} {g['decode']:>9.2f} {edge:>20}")
    print(f"vulkan / {base.backend}: prefill {vk['prefill']/ours['prefill']:.3f}x, "
          f"decode {vk['decode']/ours['decode']:.3f}x")
    compare_tokens(ours, vk, f"({base.backend} vs vulkan)")

def sweep(base, configs):
    results = {}
    for backend in SWEEP_VARIANTS:
        for ctx, arena in configs:
            key = (backend, ctx, arena)
            binary = BACKENDS[backend][0]
            if not os.path.exists(binary):
                print(f"skip {key}: {binary} missing (./stream-build.sh)")
                continue
            print(f"== {key} ==")
            a = argparse.Namespace(**vars(base))
            a.backend, a.ctx, a.arena, a.profile = backend, ctx, arena, False
            results[key] = run_once(a)

    print("\n== summary ==")
    print(f"{'backend':8} {'ctx':>7} {'arena':>6} {'prefill':>12} {'decode':>10}")
    for backend in SWEEP_VARIANTS:
        for ctx, arena in configs:
            r = results.get((backend, ctx, arena))
            if not r:
                continue
            print(f"{backend:8} {ctx:>7} {arena:>6} "
                  f"{r['prefill']:>12.1f} {r['decode']:>10.2f}")

    ctx0, arena0 = configs[0]
    a, b = results.get(("hip", ctx0, arena0)), results.get(("direct", ctx0, arena0))
    if a and b:
        compare_tokens(a, b, f"({ctx0} a{arena0})")

def check_spec(a):
    # An arena streams only the MTP speculator; common_kv_stream_auxiliary_layers
    # rejects any other type once kv_stream_pool_bytes is set. Fail here rather
    # than after the model loads.
    if not BACKENDS[a.backend][2] or a.spec_type == "none":
        return
    bad = [t for t in a.spec_type.split(",") if t not in ("none", "draft-mtp")]
    if bad:
        sys.exit(f"[ERR] streaming supports only draft-mtp, not {','.join(bad)}; "
                 f"the model must be an MTP build for streamed MTP")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the live server log, print only the summary")
    ap.add_argument("--ctx", type=int, default=30000)
    ap.add_argument("--arena", type=int, default=400)
    ap.add_argument("--cache-type", default="q5_1",
                    help="KV type for both K and V (q5_1, q8_0, q4_0, f16, ...)")
    ap.add_argument("--cache-type-k", default=None,
                    help="K cache type, overrides --cache-type")
    ap.add_argument("--cache-type-v", default=None,
                    help="V cache type, overrides --cache-type. The README "
                         "qualifies V2 for q8_0 K / q4_0 V.")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--profile", action="store_true",
                    help="wrap the server in rocprofv3 --kernel-trace")
    ap.add_argument("--trace-dir", default="kvtrace")
    ap.add_argument("--backend", default="hip", choices=sorted(BACKENDS),
                    help="hip streams, vulkan is a resident baseline (default: hip)")
    ap.add_argument("--spec-type", default="none",
                    help="speculative type; with streaming only draft-mtp is "
                         "legal, and it needs an MTP model build")
    ap.add_argument("--draft-n-max", type=int, default=2)
    ap.add_argument("--draft-p-min", type=float, default=0.75)
    ap.add_argument("--draft-cache-type", default="q4_0")
    ap.add_argument("--threads", type=int, default=8,
                    help="cpu threads (models/preset.ini uses 12; 8 keeps "
                         "earlier measurements comparable)")
    ap.add_argument("--vs-vulkan", action="store_true",
                    help="run the chosen backend and vulkan at the same config")
    ap.add_argument("--sweep", action="store_true",
                    help="run the hip/direct x ctx/arena matrix")
    ap.add_argument("--ab", nargs=2, type=int, metavar=("RESIDENT", "STREAMED"),
                    help="run both arenas and compare tokens; order does not matter")
    ap.add_argument("--config", action="append", default=None,
                    metavar="CTX:ARENA",
                    help="sweep config, repeatable (default: 30000:400 30000:2600 100000:1000)")
    a = ap.parse_args()
    a.cache_type_k = a.cache_type_k or a.cache_type
    a.cache_type_v = a.cache_type_v or a.cache_type

    binary, _, _ = BACKENDS[a.backend]
    if not os.path.exists(binary):
        print(f"[ERR] {binary} missing for backend {a.backend}")
        sys.exit(1)

    check_spec(a)

    if a.vs_vulkan:
        if not os.path.exists(BACKENDS["vulkan"][0]):
            print(f"[ERR] {BACKENDS['vulkan'][0]} missing; need a Vulkan build")
            sys.exit(1)
        vs_vulkan(a)
    elif a.ab:
        ab(a, a.ab)
    elif a.sweep:
        configs = SWEEP_CONFIGS if not a.config else [
            tuple(map(int, c.split(":"))) for c in a.config]
        sweep(a, configs)
    else:
        run_once(a)

if __name__ == "__main__":
    main()
