# Ternary Bonsai 2 27B

> **Update:** this model was originally marked BLOCKED as needing a from-source
> build of the PrismML llama.cpp fork. **That was wrong.** PrismML publishes a
> full CI matrix, and the latest release ships
> `llama-prism-<tag>-bin-win-cpu-x64.zip`, so `llmlab setup` fetches a prebuilt
> binary automatically. No CMake, no MSVC.

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

## Measured results: unusable on this hardware

Recorded on: Intel Core Ultra 7 265U (12C/14T, DDR5-4800), 32 GB RAM, no
discrete GPU, PrismML fork build 10735.

| Metric | Value |
|---|---|
| Load time | 132.7 s |
| **Prefill** | **0.33-0.48 tok/s** |
| Decode | never reached |
| Peak RSS | 7.1 GB |

The decisive measurement, from the server's own log:

```
prompt processing, n_tokens = 42,  t =  87.88 s / 0.48 tokens per second
prompt processing, n_tokens = 50,  t = 152.60 s / 0.33 tokens per second
19:04  stop: cancel task          <- cancelled after 19 min, still prefill
```

**It never finished prefill.** A 24-token "Say OK" with a 3-token prompt ran for
19 minutes without producing a single generated token. A typical 500-token
system prompt would cost roughly 25 minutes of prefill *before* the model said
anything.

### Why the memory saving did not help

The theory was sound: 6.7 GB of ternary weights reads roughly a third of
Laguna's 18.9 GB, so on a bandwidth-bound machine it should have been ~3x
faster.

What actually happens is that ternary matmul is **compute-bound, not
bandwidth-bound**. Expanding -1/0/+1 weights into a form the CPU can multiply
costs far more arithmetic than it saves in memory traffic. On a GPU that trade
is free — compute is cheap, bandwidth is precious — so ternary wins. On a
low-power U-series CPU with no GPU, the reverse holds.

**Ternary quantisation is a GPU optimisation, exactly like MoE.** Together with
the earlier MoE finding the pattern is consistent: the "efficiency" tricks in
the current model literature all assume hardware with fast compute and scarce
memory. This machine is the reverse, so they are pessimisations here.

### Honest summary

The appeal of Bonsai does not survive contact with this hardware. It is a
genuinely interesting technique and would likely perform well on a CUDA GPU,
but on a DDR5-bound CPU it is roughly **15-50x slower** than the 18.9 GB MoE it
was supposed to beat.

Kept in the catalog as a documented, reproducible negative result rather than
deleted — knowing what does *not* work on this class of machine is part of the
point of the framework.

## How it is served

Stock llama.cpp and Ollama still cannot load these packs: the ternary kernels
and the required Hadamard rotation are not upstream, and as of September 2026
they remain absent from mainline
([discussion #22019](https://github.com/ggml-org/llama.cpp/discussions/22019)).
PrismML's fork is required — but it publishes prebuilt Windows x64 CPU
binaries, so nothing special is needed:

```powershell
llmlab setup ternary-bonsai-2-27b
```

`setup` fetches `llama-prism-*-bin-win-cpu-x64.zip` into `.llmlab/prism/`.
Verified running at build 10735, commit 842b18804. To use a self-built binary
instead, drop `llama-server.exe` into that directory.

## Verdict for coding specifically

Its parent, Qwen3.8-27B, scores 77.2% on SWE-bench — the best *dense* result in
this catalog. But Bonsai trades that for a general benchmark, is not
coding-specialised, and as measured above is impractical on this CPU.

Prior verdict — "not worth the custom toolchain" — was based on the incorrect
assumption that a build was required. The toolchain objection is gone; the
remaining question is purely speed versus quality.