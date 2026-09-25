# Ternary Bonsai 2 27B

> **Status: BLOCKED.** This model cannot run on a stock install. See below.

PrismML's ternary quantisation of Qwen3.8-27B.

| | |
|---|---|
| Architecture | dense, ternary weights (~1.71 bpw) |
| Quant | PQ2_0 (or PTQ1_0) |
| Size on disk | 6.71 GB (PQ2_0) / 5.54 GB (PTQ1_0) |
| Licence | `apache-2.0` |
| Upstream | [prism-ml/Ternary-Bonsai-2-27B-GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-GGUF) |

## The idea, and why it is interesting

Ternary weights store each parameter as -1, 0, or +1. At 1.71 bits per weight a
27B model fits in ~6 GB instead of ~54 GB at FP16, while retaining **94.6% of
the FP16 reference benchmark score**. On a bandwidth-bound machine that is a
genuinely good trade: ~7 GB read per token instead of ~17 GB for Q4.

## Why it is blocked

**Stock llama.cpp and Ollama cannot load these files.** The ternary kernels and
the Hadamard rotation the architecture depends on are not upstream. As of
September 2026 they are still absent from mainline — see
[ggml-org/llama.cpp discussion #22019](https://github.com/ggml-org/llama.cpp/discussions/22019),
where PrismML's own collaborator was still asking about upstream support.

There are no official Windows release binaries for the fork, so serving this
means building from source.

### To enable it

```powershell
# 1. Install CMake and MSVC Build Tools
# 2.
git clone https://github.com/PrismML-Eng/llama.cpp
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON
cmake --build build --config Release -j
# 3.
copy build\bin\llama-server.exe  ..\.llmlab\prism\
```

Then `llmlab doctor` will stop reporting it as blocked. Note that
`llmlab/runtimes/prism.py` currently stops short of wiring up `start()` — that
is deliberate rather than an oversight, because the fork's flags and pack
format differ enough from stock that it needs its own tested path.

## Verdict for coding specifically

Its parent, Qwen3.8-27B, scores 77.2% on SWE-bench — the best *dense* result in
this catalog. But Bonsai gives that up for a general benchmark, is not
coding-specialised, is dense (so every token reads the full tensor), and needs a
custom toolchain. A 30B MoE delivers more coding capability per unit of speed
for none of that friction.

Worth building if you are curious about ternary inference or want a small
general model. Not the recommended coding pick on this hardware.