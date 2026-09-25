# Qwen3-Coder-30B-A3B-Instruct

The most widely deployed local coding MoE, and the safest default.

| | |
|---|---|
| Architecture | MoE, 30.5B total / 3.3B active (128 experts, 8 routed) |
| Quant | Q4_K_M |
| Size on disk | 17.28 GB |
| Native context | 262,144 |
| Licence | `apache-2.0` |
| Upstream | [unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) |

## Reported benchmarks

50.3% on SWE-bench Verified per the Qwen model card. That is well below
Laguna XS 2.1's 70.9% — but see the caveat in the main README about
cross-vendor benchmark harnesses before treating that gap as settled.

## Why it is on this machine

It is the reference point for this class: mature, Apache 2.0, native 262K
context, and 3.3B active for CPU-friendly speed. 17.28 GB is the smallest
weight footprint of the four 30B MoE candidates here, which leaves the most
room for KV cache.

Pick it for ecosystem maturity and licence safety. Pick Laguna XS 2.1 for raw
capability per unit of speed.

## Setup

```powershell
llmlab setup qwen3-coder-30b-a3b
llmlab run   qwen3-coder-30b-a3b
```

## Measured results

| Metric | Value |
|---|---|
| Decode tok/s | — |
| TTFT | — |
| Peak RSS | — |
| Coding pass@1 | — |
| Tool-call validity | — |

## Notes and gotchas

- Unsloth recommends sampling at `temperature=0.7, top_p=0.8, top_k=20,
  repetition_penalty=1.05` for best results. This framework evaluates at
  `temperature=0` for determinism, which is correct for pass@1 but is not the
  configuration you would ship.
- This is an **Instruct** model, not a base model. Do not expect raw completion
  behaviour.