# local-llm-lab

A plug-and-play framework for evaluating local LLMs on modest hardware.

Add a model by dropping a directory into `models/`. Get performance numbers, a
capability score, and a leaderboard.

**Model weights are not in this repository.** Each manifest points at a Hugging
Face repo and `llmlab setup` fetches the file on demand.

---

## Why this exists

Most "best local model" advice assumes a discrete GPU. This framework was built
for the case where there isn't one: a 32 GB laptop with an integrated GPU, where
inference is **memory-bandwidth bound** and the model you pick matters far more
than the quant you pick.

On that hardware the governing rule is:

```
bytes of weights read per generated token  ÷  ~58 GB/s  =  tokens/second
```

---

## The headline finding: MoE does not help on CPU

This repository started with a hypothesis, and the first real measurement
killed it.

The theory was that Mixture-of-Experts models would be dramatically faster here,
because all 33B of weights must be resident in RAM but only ~3B are read per
token — roughly 2-3 GB of traffic instead of ~17 GB for a dense 27B at Q4.

**Measured on Laguna XS 2.1 (33B MoE, 3B active, Q4_K_M):**

| | Predicted | Measured |
|---|---|---|
| Bytes read per token | ~2-3 GB | ~8-9 GB implied |
| Decode | 20-25 tok/s | **6.3 tok/s** |

The active-parameter count is real, but on a bandwidth-bound CPU it does **not**
convert into the expected memory-traffic saving. The model behaves roughly like
a dense model of its full size.

This matches a documented llama.cpp anomaly
([#19480](https://github.com/ggml-org/llama.cpp/issues/19480)), where a
3B-active MoE ran 3-4x slower than predicted, attributed to expert-routing
scatter and poor memory locality.

**MoE is a VRAM optimisation. On a CPU with no VRAM to save, it largely does
not apply.** Every MoE in this catalog is subject to this. Treat the
`active_params` column as a spec, not a performance figure.

The practical upshot: **dense models may be a better bet than expected**, and a
4 GB model will beat a 19 GB one regardless of architecture. The catalog still
leans MoE because the published benchmark quality is there — but measure before
committing, which is what this framework is for.

## The second headline: ternary is a GPU trick too

Ternary quantisation (Bonsai 2, 6.7 GB, ~1.71 bits/weight) looked like the
answer to the bandwidth problem: a third of the bytes of the 18.9 GB MoE, at
94.6% of its FP16 reference quality.

Measured on the same machine:

```
prompt processing, n_tokens = 42,  t =  87.88 s / 0.48 tokens per second
prompt processing, n_tokens = 50,  t = 152.60 s / 0.33 tokens per second
```

**It never finished prefill.** A "Say OK" with a 3-token prompt ran 19 minutes
without producing a single token. Roughly **15-50x slower** than the model it
was supposed to beat.

Ternary matmul is compute-bound: expanding -1/0/+1 weights into something the
CPU can multiply costs more arithmetic than it saves in memory traffic. On a GPU
that trade is free. Here it is a disaster.

### The pattern across both findings

MoE and ternary are both **GPU optimisations**. They trade compute for memory
bandwidth, which is a losing trade when you have neither spare compute nor
much bandwidth — and this machine is a bandwidth-bound CPU with slow cores and
no VRAM.

So the practical guidance for a GPU-less 32 GB box is the opposite of the
usual advice:

- **Ignore `active_params`.** It predicts nothing here.
- **Ignore bits-per-weight.** Same reason.
- **Optimise for total weight size, and little else.** Small, dense, ordinary
  quantisation. A 4 GB model will beat a 19 GB one.

`smoke-tiny` (0.5 B, 469 MB) hitting **37 tok/s** while a 27 B ternary model
could not finish prefilling a 3-token prompt is the whole lesson in two numbers.

---

## Requirements

- Python 3.11+ (developed on 3.14)
- ~30 GB free disk per 4-bit 30B model
- `psutil` is optional but recommended — it gives real peak-RSS numbers
- No compiler needed for the `llamacpp` runtime; prebuilt Windows binaries are
  fetched automatically

```powershell
pip install -e .[better]     # psutil for real RSS numbers
pip install -e .[all]        # also enables the parallel download backend
```

Check your environment at any time:

```powershell
llmlab doctor
```

### Weight download backends

`llmlab setup` picks one automatically:

| Backend | When | Notes |
|---|---|---|
| `huggingface_hub` | installed and importable | Xet backend on hub ≥ 1.0, `hf_transfer` on older |
| `urllib` | otherwise, or `LLMLAB_DISABLE_HF_TRANSFER=1` | **Default. Zero dependencies.** HTTP Range resume |

Both produce an identical on-disk layout, so switching is free and nothing else
in the framework changes.

**The fallback is not decorative — it has already been needed in practice.** On
this machine the `huggingface_hub` Xet path **stalled at 0 bytes** on a 6.9 GB
ternary pack and made no progress at all, while the urllib path pulled the same
file at full rate. If `llmlab setup` ever appears to hang at 0%, run:

```powershell
$env:LLMLAB_DISABLE_HF_TRANSFER = "1"
llmlab setup <model>
```

**Caveat worth stating plainly:** a faster downloader does not help if your
connection is the bottleneck. On the machine this was developed on, 1 curl
connection managed 2.2 MB/s and 4 parallel connections 3.83 MB/s — and the
download backend still only reached 4.4 MB/s. If your line tops out around
4 MB/s, both backends will feel identical, and the real constraint is your
connection, not the client.

---

## Quick start

```powershell
llmlab list                  # what models this repo knows about
llmlab setup laguna-xs-2.1   # download + verify weights (~19 GB)
llmlab run   laguna-xs-2.1   # benchmark + evaluate
llmlab rank                  # rebuild results/leaderboard.md
```

`llmlab run` writes to `results/raw/` (gitignored) and refreshes
`results/leaderboard.md` (committed).

---

## Commands

| Command | What it does |
|---|---|
| `llmlab doctor` | Environment audit: RAM, disk, CPU, toolchain, runtime status |
| `llmlab list` | Every known model with install status and licence |
| `llmlab new-model <id>` | Scaffold a model directory from a template |
| `llmlab setup <id>` | Download weights, verify checksum, fetch runtime |
| `llmlab serve <id>` | Start a server on an OpenAI-compatible endpoint |
| `llmlab bench <id>` | Performance suite at several context sizes |
| `llmlab eval <id>` | Coding + tool-calling quality suite |
| `llmlab run <id>` | `bench` + `eval` + `rank` |
| `llmlab rank` | Regenerate the leaderboard from stored results |

Model ids accept unique fragments: `llmlab bench laguna` works.

---

## Adding a model

This is the whole point — **no code changes required.**

```powershell
llmlab new-model my-model --name "My Model"
```

That creates `models/my-model/` with a `model.toml` to fill in and a `README.md`
to write. Fill in the manifest and the model is immediately available to every
command. The manifest is the contract:

```toml
[model]
id          = "laguna-xs-2.1"   # must match the directory name
name        = "Laguna XS 2.1"
org         = "Poolside"
license     = "openmdw-1.1"     # surfaced in NOTICE.md
arch        = "moe"             # moe | dense | ternary
total_params  = "33B"
active_params = "3B"            # MoE only
context_native = 262144
description = "..."
eval_tiers  = ["perf", "coding", "tools"]
notes       = "..."

[source]
repo    = "poolside/Laguna-XS-2.1-GGUF"
file    = "Laguna-XS-2.1-Q4_K_M.gguf"
sha256  = ""                    # strongly recommended
size_gb = 18.88

[runtime]
kind       = "llamacpp"         # llamacpp | prism
ctx_size   = 32768              # WORKING context, not native
threads    = 0                  # 0 = all cores
extra_args = []
```

Two fields deserve care:

- **`ctx_size` is the working context, not the native one.** A model advertising
  262K will exhaust a 32 GB box. Set what actually fits.
- **`sha256`** is optional but strongly encouraged. Without it the framework
  cannot verify what it downloaded.

---

## What gets measured

### Performance (`bench`)

Run at 4K / 8K / 16K context by default, because the KV cache — not the weights
— is what pushes a memory-bound machine over the edge.

- **load time** — spawn to healthy; CPU weight loading is not instant
- **TTFT** — time to first streamed token
- **prefill tok/s** — prompt processing rate
- **decode tok/s** — generation rate
- **peak RSS** — resident memory of the server process

### Quality (`eval`)

- **Coding** — auto-graded tasks in `tasks/coding/`. Each is a prompt plus
  assertions, executed in a **subprocess with a timeout** so a runaway loop
  cannot take the harness down.
- **Tool calling** — sends an OpenAI-format `tools` array and checks the
  response contains a schema-valid `tool_calls` block. This is the closest
  proxy for "can an agent actually drive this model", which is what matters if
  you intend to use it in an editor or agent harness.

### Honest limitations

Read these before treating the leaderboard as gospel:

- The coding suite is **small** and tests single-function synthesis. It
  separates "can write a function" from "cannot" reliably, and it will
  **systematically underrate agentic ability** because it never exercises
  multi-file editing, tool loops, or debugging.
- **The two quality tiers are not equally discriminating.** The `smoke-tiny`
  baseline (a 0.5B model) scored 21% on coding but 75% on tool calling. Emitting
  a schema-valid `tool_calls` block is largely format imitation; writing correct
  code is not. Do not read a high tool score as "good at coding".
- **Truncation is scored as failure.** A model that rambles until `max_tokens`
  fails with a `SyntaxError` rather than a wrong answer. That is the right call
  for an agent that must emit complete code, but it means the coding score
  blends "couldn't solve it" with "didn't stop".
- **Vendor SWE-bench numbers are not comparable.** Published figures use
  different harnesses (OpenHands, SWE-agent, vendor sandboxes) and the spread
  between harnesses is large enough to reorder models.
- Speed numbers are **single-run measurements on one machine**. They do not
  transfer to other hardware. They are also **noisy**: two runs of the same
  model, same context, minutes apart on the development machine returned 25.3
  and 37.1 tok/s — a 47% swing from ordinary background load. Treat the
  leaderboard as an ordering hint, not a precise figure, and re-run before
  drawing conclusions from small gaps.
- Results in this repo were produced on the machine described in
  `results/leaderboard.md`. Reproduce before trusting them elsewhere.

### Baseline

`smoke-tiny` (Qwen2.5-0.5B-Instruct, 469 MB) is in the catalog as a **pipeline
self-test**, not a recommendation. It verifies the whole framework end to end in
under a minute without a 19 GB download, which makes it usable in CI:

```powershell
llmlab setup smoke-tiny
llmlab run   smoke-tiny
```

---

## Runtimes

| Runtime | Status | Used for |
|---|---|---|
| `llamacpp` | Prebuilt, auto-fetched | Everything in the default catalog |
| `prism` | Prebuilt, auto-fetched | Ternary models (Ternary Bonsai) |

Stock llama.cpp and Ollama **cannot** load PrismML's `PQ2_0`/`PTQ1_0` ternary
packs — the kernels and the required Hadamard rotation are not upstream. As of
September 2026 they are still not in mainline, so ternary models need
PrismML's fork. That fork *does* publish prebuilt Windows x64 CPU binaries, so
`llmlab setup` provisions it automatically: no CMake, no MSVC.

> **Correction.** An earlier version of this README claimed the PrismML fork
> ships no Windows binaries and that serving a ternary model required building
> from source with CMake and MSVC Build Tools. That was wrong, and it made a
> perfectly runnable model look permanently blocked. The `prism` runtime now
> fetches the prebuilt binary exactly the way `llamacpp` does.

---

## Using these models in OpenCode

`integrations/opencode.json` is a ready-to-merge provider block. Add the
`llmlab` entry to `providers{}` in your OpenCode config, then:

```powershell
llmlab serve laguna-xs-2.1     # or any catalog model
opencode run --model "llmlab/laguna-xs-2.1#low" "explain this repo"
```

**If you skip one thing, do not skip `compatibility.reasoningField`.** These
models stream their output into `reasoning_content` and leave `content` empty.
A client that reads only `delta.content` sees nothing at all — which is exactly
the bug this framework hit while measuring them. The setting is already present
in the integration file for every reasoning model.

---

## Layout

```
llmlab/
  manifest.py      model.toml schema + validation
  registry.py      auto-discovery (the plug-and-play mechanism)
  weights.py       download with resume + checksum
  report.py        leaderboard generation
  runtimes/        serving backends behind one interface
  bench/perf.py    performance harness
  eval/            coding + tool-calling harnesses
models/<id>/
  model.toml       the manifest
  README.md        per-model notes, setup, measured results
  weights/         gitignored
tasks/
  coding/*.json    auto-graded coding tasks
  tools/*.json     tool-calling tasks
results/
  leaderboard.md   committed
  raw/             gitignored
```

---

## Licence

Framework code: **MIT** — see [LICENSE](LICENSE).

Model weights are **not** distributed here and each carries its own licence.
See [NOTICE.md](NOTICE.md) for the per-model summary and check each upstream
repository before commercial use.
