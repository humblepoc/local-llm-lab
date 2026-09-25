# Nemotron Cascade 2 30B A3B

NVIDIA's reasoning-focused MoE.

| | |
|---|---|
| Architecture | MoE, 30B total / 3B active |
| Quant | Q4_K_M |
| Size on disk | **23.03 GB** |
| Working context | 8,192 (deliberately low) |
| Licence | `other` — read the model card |
| Upstream | [bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF) |

## Reported benchmarks

Community testing suggests it beats comparable models on most coding
benchmarks **but is weaker on SWE-bench specifically**. It reads as a
reasoning-first model rather than an agentic one.

## ⚠️ This one is tight

23.03 GB of weights in a 32 GB machine leaves roughly 3-4 GB for the OS and KV
cache after overhead. That is why `ctx_size` is pinned to 8K in the manifest
while its siblings run at 32K.

Expect it to be the fastest of the group to *not* work well, purely on memory
pressure. On a machine with more RAM, raise `ctx_size` and re-run.

## Setup

```powershell
llmlab setup nemotron-cascade-2-30b-a3b
llmlab run   nemotron-cascade-2-30b-a3b
```

If the server dies during load or the first request, this model is the reason:
lower `ctx_size` further, or close other memory-heavy applications first.

## Measured results

| Metric | Value |
|---|---|
| Decode tok/s | — |
| TTFT | — |
| Peak RSS | — |
| Coding pass@1 | — |
| Tool-call validity | — |

## Notes and gotchas

- Licence is `other`, not a standard SPDX identifier. Check NVIDIA's terms.
- bartowski is a third-party repackager; the base model is
  `nvidia/Nemotron-Cascade-2-30B-A3B`.