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

## Why it is blocked

**Stock llama.cpp and Ollama cannot load these files.** The ternary kernels and
the Hadamard rotation the architecture depends on are not upstream. As of
September 2026 they are still absent from mainline — see
[ggml-org/llama.cpp discussion #22019](https://github.com/ggml-org/llama.cpp/discussions/22019),
where PrismML's own collaborator was still asking about upstream support.

There are no official Windows release binaries for the fork, so serving this
means building from source.

### How it is served

Nothing special is required:

```powershell
llmlab setup ternary-bonsai-2-27b
llmlab run   ternary-bonsai-2-27b
```

`setup` fetches the prebuilt `llama-prism-*-bin-win-cpu-x64.zip` into
`.llmlab/prism/` and the framework serves the model through the normal
OpenAI-compatible endpoint. Verified running at build 10735, commit 842b18804.

If you would rather use a self-built binary, drop `llama-server.exe` into
`.llmlab/prism/` and the framework will use it instead.

## Verdict for coding specifically

Its parent, Qwen3.8-27B, scores 77.2% on SWE-bench — the best *dense* result in
this catalog. But Bonsai trades that for a general benchmark, is not
coding-specialised, and is dense (so every token reads the full tensor).

The open question is empirical: at ~6.7 GB it reads roughly a third of what
Laguna's 18.9 GB does, so if throughput scales with bytes read it should land
around 3x the tok/s. That is the measurement that matters, and it is why this
model is worth running despite the caveats above.

Prior verdict — "not worth the custom toolchain" — was based on the incorrect
assumption that a build was required. The toolchain objection is gone; the
remaining question is purely speed versus quality.