# Laguna XS 2.1

Poolside's agentic-coding MoE. **The flagship model in this catalog.**

| | |
|---|---|
| Architecture | MoE, 33B total / 3B active |
| Quant | Q4_K_M |
| Size on disk | 18.88 GB |
| Native context | 262,144 |
| Working context | 32,768 |
| Licence | `openmdw-1.1` — **not** standard OSS |
| Upstream | [poolside/Laguna-XS-2.1-GGUF](https://huggingface.co/poolside/Laguna-XS-2.1-GGUF) |

## Published benchmarks

| Benchmark | Score |
|---|---|
| SWE-bench Verified | 70.9% |
| SWE-bench Multilingual | 63.1% |
| SWE-Bench Pro (public) | 47.6% |
| Terminal-Bench 2.0 | 37.5% |

Poolside ran these in their own sandbox; treat cross-vendor comparison with
caution. Terminal-Bench 2.0 is the most relevant of these for agentic use.

## Why it is on this machine

It sits in the one speed class a 32 GB, GPU-less box can actually run.

The MoE trick is the whole point: all 33B of weights must be resident in RAM,
but only ~3B are read to produce a token. That is roughly 2-3 GB of memory
traffic per token instead of the ~17 GB a dense 27B at Q4 would need — about a
6x reduction, which on ~58 GB/s of bandwidth is the difference between ~3 tok/s
and something usable.

At the same speed class as Qwen3-Coder-30B-A3B, it scores roughly 20 points
higher on SWE-bench Verified. That is the whole argument for it.

## Setup

```powershell
llmlab setup laguna-xs-2.1
llmlab run   laguna-xs-2.1
```

Budget: 18.88 GB of weights plus KV cache in a 32 GB machine. The working
context is capped at 32K for that reason. Do not raise `ctx_size` in
`model.toml` without checking free RAM first.

## Measured results

Recorded on: Intel Core Ultra 7 265U (12C/14T, DDR5-4800 dual channel), 32 GB
RAM, Intel iGPU only (no CUDA), llama.cpp `b11178` CPU build, Q4_K_M.

| Metric | ctx=4096 | ctx=8192 |
|---|---|---|
| Load time | 50.1 s | 52.0 s |
| TTFT | 8.08 s | 6.49 s |
| **Decode tok/s** | **5.87** | **6.72** |
| Prefill tok/s | 23.9 | 24.9 |
| Peak RSS | 21.9 GB | 21.9 GB |

Mean decode **6.29 tok/s**.

Cross-checked against the server's own `print_timing` output: the harness
reported 6.72 tok/s where llama-server reported 6.72 tokens/second. The
measurement agrees with ground truth.

### The MoE advantage largely evaporated

This is the headline finding, and it contradicts the reasoning that motivated
putting this model in the catalog.

| | Predicted | Actual |
|---|---|---|
| Bytes read per token | ~2-3 GB (3B active) | ~8-9 GB implied |
| Decode | 20-25 tok/s | **6.3 tok/s** |

At 6.3 tok/s the model is effectively reading close to a *dense* 19 GB model's
worth of memory per token, not the ~3 GB the active-parameter count implies. The
3B-active figure is real, but on CPU it does not translate into the expected
bandwidth saving.

This matches a known llama.cpp anomaly (issue #19480), where a 3B-active MoE
ran 3-4x slower than its active-parameter count predicted, attributed to
expert-routing scatter and poor memory locality.

**MoE is a VRAM optimisation. On a bandwidth-bound CPU it largely does not
apply.** That is the single most useful thing this framework has measured so
far, and it applies to every MoE in the catalog.

Practical consequence: at 6.3 tok/s this is a poor choice for agentic work in
OpenCode, where a turn involves long tool chains and large outputs. It suits
short focused completions and privacy-sensitive snippets.

### Working context was lowered to 16K

`ctx_size` is **16384**, not the 32768 originally set and certainly not the
native 262144. At only 8K the server already held 21.9 GB RSS, leaving roughly
1 GB free on a 31.5 GB machine. 32K would not have fit.

## Notes and gotchas

- **Licence first.** `openmdw-1.1` has use-based restrictions. Read it before
  commercial or hosted use.
- **This is a reasoning model.** It streams into a reasoning field rather than
  `content`, so any client reading only `delta.content` will measure 0 tok/s.
  The framework handles this; ad-hoc scripts probably will not.
- 256K native context is a **spec number, not a usable number** here.
- This is a general agentic model, not a coding-only one. If you want
  code-specialisation with a clean Apache 2.0 licence, see
  `north-mini-code-1.0`.