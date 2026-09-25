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

Populate with `llmlab run laguna-xs-2.1`, then `llmlab rank`.

| Metric | Value |
|---|---|
| Decode tok/s | — |
| TTFT | — |
| Prefill tok/s | — |
| Peak RSS | — |
| Coding pass@1 | — |
| Tool-call validity | — |

## Notes and gotchas

- **Licence first.** `openmdw-1.1` has use-based restrictions. Read it before
  commercial or hosted use.
- 256K native context is a **spec number, not a usable number** here. At 32K
  the weights plus KV cache already consume most of a 32 GB machine.
- This is a general agentic model, not a coding-only one. If you want
  code-specialisation with a clean Apache 2.0 licence, see
  `north-mini-code-1.0`.