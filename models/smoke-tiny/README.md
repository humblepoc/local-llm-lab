# smoke-tiny

> **This is not a coding model recommendation.** It exists so the framework can
> be verified end to end in under a minute, including in CI, without pulling a
> 19 GB download.

| | |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Architecture | dense, 0.5B |
| Quant | Q4_K_M |
| Size | 469 MB |
| Licence | `apache-2.0` |
| Upstream | [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) |

## What it is for

A 0.5B model cannot code. Expect **low** scores on both suites — the baseline
run scored 21% coding / 75% tools. That is the expected result, and it is the
point: if `smoke-tiny` ever scores highly, something is wrong with the harness.

What it *does* validate, cheaply:

- manifest discovery and validation
- runtime binary fetch and server startup
- streaming, TTFT and decode-rate measurement
- subprocess sandboxing in the coding harness
- tool-call schema validation
- result serialisation and leaderboard generation

## Usage

```powershell
llmlab setup smoke-tiny
llmlab run   smoke-tiny
```

Roughly 470 MB and under a minute on a modern laptop. Use it to check that a
change to the framework has not broken anything before spending an hour on a
real 19 GB model.

## Measured results

Recorded on the development machine: Intel Core Ultra 7 265U (12C/14T,
DDR5-4800), 32 GB RAM, no discrete GPU, llama.cpp `b11178` CPU build.

| Metric | Value |
|---|---|
| Load time | 1.58 s |
| TTFT | 0.21 s |
| Decode tok/s | 25.3 |
| Prefill tok/s | 108.2 |
| Peak RSS | 0.56 GB |
| Coding pass@1 | **21.1%** (4/19) |
| Tool-call validity | **75.0%** (6/8) |

### What this told us about the harness

Two findings worth keeping, both from this run:

**Tool calling is far easier than coding.** A 0.5B model scored 75% on
tool-calling but 21% on coding. Emitting a schema-valid `tool_calls` block is
mostly format imitation; actually writing correct code is not. The two
tiers are measuring genuinely different things, and the tool tier is the
weaker discriminator — do not read a high tool score as "good at coding".

**Truncation shows up as `SyntaxError`, not as a wrong answer.** Several
failures were `unterminated string literal` — the model rambled until it hit
`max_tokens` rather than reasoning badly. Those are counted as failures, which
is the right call for an agent that needs to emit complete code, but it means
the coding score mixes "couldn't solve it" with "didn't stop". Raising
`max_tokens` would separate the two.