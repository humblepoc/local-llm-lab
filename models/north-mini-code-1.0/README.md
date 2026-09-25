# North Mini Code 1.0

Cohere's coding-specialist MoE.

| | |
|---|---|
| Architecture | MoE, 30B total / 3B active |
| Quant | UD-Q4_K_M (Unsloth dynamic) |
| Size on disk | 17.88 GB |
| Licence | `apache-2.0` |
| Upstream | [unsloth/North-Mini-Code-1.0-GGUF](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF) |

## Reported benchmarks

Cohere reports ~67% on SWE-bench Verified using the SWE-agent harness, and 33.4
on the Artificial Analysis Coding Index — ahead of several 120B models. The
harness differs from the one Laguna used, so the two numbers are not directly
comparable.

## Why it is on this machine

Same reasoning as every MoE in this catalog: 3B active means a fraction of the
memory traffic per token, which is the only thing that makes 30B-class quality
viable without a GPU. 17.88 GB leaves slightly more headroom than Laguna.

Choose it over Laguna if the **Apache 2.0 licence** matters for your use, or if
you want a model trained specifically for coding rather than general agentic
work.

## Setup

```powershell
llmlab setup north-mini-code-1.0
llmlab run   north-mini-code-1.0
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

- Unsloth's "UD" quantisation is a dynamic scheme that spends more bits on
  layers that need them. It costs a little CPU at load time and is usually worth
  it.
- Cohere is a newer entrant in open-weight coding models. Ecosystem tooling
  around it is thinner than Qwen3-Coder.