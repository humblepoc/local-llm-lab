# Session log — 2026-09-25

Development log for the work that produced this repository, from a single
working session. Rewritten as the session progresses.

**Machine:** Intel Core Ultra 7 265U (12C/14T, U-series), 32 GB DDR5-4800 dual
channel, Intel iGPU only (no CUDA), 842 GB free. OpenCode v2.0.16, Python 3.14.7.

---

## 1. OpenCode V1 → V2 config migration

Started from a `Downloads/opencode.json` that turned out to be **V1-format**
while the installed OpenCode was v2.0.16. A verbatim copy would have been
silently ignored.

| V1 | V2 |
|---|---|
| `provider` | `providers` |
| `.npm` | `.package` |
| `.options` | `.settings` |
| `.options.headers` | `.headers` |
| `@ai-sdk/openai-compatible` | `@opencode/ai/providers/openai-compatible` |
| `@ai-sdk/anthropic` | `@opencode/ai/providers/anthropic` |
| `mcp.<name>` | `mcp.servers.<name>` |
| `enabled: true` | (dropped — default is enabled) |
| `plugin` | `plugins` |

36 models across 7 providers imported to `~/.config/opencode/opencode.json`.
Verified by diffing every provider's model set against the source: no models
lost. (I initially claimed 37 in a plan; the diff proved 36.)

### Why the Claude IDs were wrong

Five model IDs carried an `@` suffix (`claude-sonnet-4-5@20250929`). V2 reserves
`@`/`#` for variants. Probing the Siemens endpoint directly showed the API
rejects them. Fixed to the real Anthropic IDs.

### The reasoning-effort discovery

`code-siemens/qwen-3.8-27b` initially had a `#high` variant. Live probing
against `https://api.siemens.com/llm/v1` returned:

> Supported types are **xhigh (default)**, medium, and low.

`high` is **rejected with HTTP 400** — the configured variant would have failed
on every single request. `xhigh` is valid, lowercase-only (`xHigh` → 400), and
is already the endpoint default. Replaced with custom `xhigh` / `medium` / `low`
variants.

Also relevant: the **root `model` default silently drops variants.** The variant
has to live on an agent (`agents.build.model = "...#xhigh"`) or it is accepted
and ignored.

---

## 2. Local model research

Question: which local coding model suits this machine?

### Hardware finding that reframed everything

**No discrete GPU.** Intel iGPU only, ~2 GB shared, no CUDA. Decode is
**memory-bandwidth bound**, not compute bound. DDR5-4800 dual channel gives
76.8 GB/s theoretical, ~55-62 GB/s achievable.

Rule used throughout: `bytes read per token ÷ ~58 GB/s = tok/s`.

A dense 24B at Q4 reads ~16 GB/token → ~3 tok/s. That killed the idea of any
large dense model.

### Ternary Bonsai — investigated, rejected with reasons

The idea is sound: ternary weights (~1.71 bpw) cut memory traffic 2.4× versus
Q4, which is exactly right for a bandwidth-bound machine. Bonsai 2 27B retains
**94.6% of its FP16 reference score** at ~7 GB.

But it **cannot be loaded** by stock llama.cpp or Ollama — the ternary kernels
and required Hadamard rotation are not upstream, and as of Sept 2026 they are
still absent ([llama.cpp discussion #22019](https://github.com/ggml-org/llama.cpp/discussions/22019),
where PrismML's own collaborator was still asking). No Windows release binaries;
requires building the fork from source with CMake.

It also is dense (every token reads the full tensor) and not coding-specialised.
Shipped as a documented-but-`blocked` manifest rather than dropped.

---

## 3. The framework

`local-llm-lab` — plug-and-play local model evaluation.

A model is **a directory containing `model.toml`**. Drop it in `models/` and
every command sees it. No registry edits, no imports.

```
llmlab doctor | list | new-model | setup | serve | bench | eval | run | rank
```

Measures: load time, TTFT, prefill tok/s, decode tok/s, peak RSS at 3 context
sizes; plus auto-graded coding tasks (subprocess-sandboxed, timed) and
tool-call schema validity.

### Bugs found while building

| Bug | Consequence |
|---|---|
| `/releases/latest` on llama.cpp returns a stable tag whose only asset is `nightly-tag.txt` | found no Windows binary; now scans the rolling `bNNNNN` prereleases |
| UTF-8 BOM in `model.toml` (PowerShell 5.1, Notepad) | `tomllib` rejected every manifest; now read with `utf-8-sig` |
| `llmlab list` early-returned on an empty registry | swallowed all manifest errors |
| Download rate computed in MB/s, labelled `B/s` | cosmetic |
| cp1252 console | `UnicodeEncodeError` on leaderboard glyphs; stdout now forced to UTF-8 |
| `huggingface_hub` 2.0 **deprecated** `HF_HUB_ENABLE_HF_TRANSFER` | switched to `HF_XET_HIGH_PERFORMANCE` for hub ≥ 1.0 |
| **Counted only `delta.content`** | **reported 0.0 tok/s for every reasoning model** |

That last one is the important one — see below.

---

## 4. Measurement: the MoE hypothesis failed

The catalog was built on a theory: 30-33B MoE models with ~3B active should
read only ~2-3 GB per token and therefore run at 20-25 tok/s on this box.

**Measured — Laguna XS 2.1, 33B MoE / 3B active, Q4_K_M:**

| Metric | ctx=4096 | ctx=8192 |
|---|---|---|
| Load | 50.1 s | 52.0 s |
| TTFT | 8.08 s | 6.49 s |
| **Decode** | **5.87 tok/s** | **6.72 tok/s** |
| Prefill | 23.9 tok/s | 24.9 tok/s |
| Peak RSS | 21.9 GB | 21.9 GB |

| | Predicted | Measured |
|---|---|---|
| Bytes read/token | ~2-3 GB | **~8-9 GB implied** |
| Decode | 20-25 tok/s | **6.3 tok/s** |

**MoE is a VRAM optimisation. On a CPU with no VRAM to save, it largely does
not apply.** This matches a documented llama.cpp anomaly
([#19480](https://github.com/ggml-org/llama.cpp/issues/19480)) where a 3B-active
MoE ran 3-4× slower than predicted, from expert-routing scatter and poor memory
locality.

Consequence: the catalog's organising premise was wrong, and the README was
rewritten to say so. A ~4 GB model will beat a 19 GB one on this hardware
regardless of architecture.

### The harness bug that hid this

The first run reported **0.0 tok/s** and a 47.78 s TTFT. Not slow hardware — a
measurement bug. The harness counted only `delta.content`, but reasoning models
stream into a separate field. The server log proved the model had generated 192
tokens at 4.93 tok/s the whole time.

Streaming probe of the actual endpoint:

```
DELTA FIELDS: ['reasoning_content', 'role']
message keys: ['content', 'reasoning_content', 'role']
content repr: ''
```

`content` is **empty**. Fixed to count `content`, `reasoning_content`,
`reasoning`, `reasoning_text`, `thinking`. Re-measured, the harness then
reported 6.72 tok/s where llama-server's own `print_timing` reported 6.72
tokens/second — measurement agreeing with ground truth.

This bug would have silently broken the framework for its entire target class.

### The same trap, in OpenCode

Because these models put text in `reasoning_content`, OpenCode also needs
`compatibility.reasoningField` set per model, or it sees empty responses. This
is configured in `integrations/opencode.json`.

---

## 5. Connection ceiling

| Test | Throughput |
|---|---|
| 1 curl connection | 2.2 MB/s |
| 4 parallel connections | 3.83 MB/s |
| accelerated backend | 4.4 MB/s |

The download accelerator already exceeds 4 parallel curl connections, so the
**connection is the ceiling, not the client**. Accelerators are wired in as an
optional extra (urllib with Range resume remains the zero-dependency default)
but they do not help on this line. Documented rather than claimed as a win.

---

## 6. Run-to-run variance

Two runs of the same model, same context, minutes apart:

```
smoke-tiny:  25.3 tok/s  then  37.1 tok/s     (47% swing)
laguna:      4.93        then  6.72          (36% swing)
```

Larger than any gap between models. The leaderboard is therefore documented as
an **ordering hint**, not a precise figure.

---

## 7. Quality-eval calibration

`smoke-tiny` (Qwen2.5-0.5B, deliberately too small to code) scored:

```
coding 21.1% (4/19)   tool-calling 75.0% (6/8)
```

Two findings, both criticisms of my own eval design:

- **The two tiers are not equally discriminating.** Emitting a schema-valid
  `tool_calls` block is largely format imitation; writing correct code is not.
  A high tool score must not be read as "good at coding".
- **Truncation scores as failure.** Models that ramble to `max_tokens` fail with
  `SyntaxError` rather than a wrong answer, so the coding score blends "couldn't
  solve it" with "didn't stop".

---

## Lessons

1. **Probe, don't trust the docs.** The Siemens `#high` variant would have
   failed every request; only a live 400 revealed it. The Siemens docs URL is
   behind a sign-in wall.
2. **Check your own numbers against ground truth.** The 0.0 tok/s reading looked
   like a slow machine. The server log showed 4.93 tok/s and pointed at my bug.
3. **Publish the hypothesis, then publish its refutation.** The README's MoE
   reasoning was written before any measurement and was wrong. Leaving it
   standing would have been the worst outcome; it is now the headline finding.
4. **A hypothesis-driven framework earns its keep precisely when the hypothesis
   fails.** The first real number was the most informative one.

---

## Open items

- Laguna coding + tool eval never completed (background task was killed). Needs
  re-running.
- `SDC_LLM_API_KEY` unset → 23 `sdc-llm` models cannot authenticate.
- `memory` and `remote-windows-helper` MCP servers still have no code; the
  PrismML fork is not built.
- Catalog leans MoE; a small **dense** coding model (7-9B) is the obvious next
  addition given the bandwidth finding.
- GitHub PAT used for pushing was pasted in plaintext and should be rotated.
