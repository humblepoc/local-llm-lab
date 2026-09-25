# Model licences

**This repository distributes no model weights.** Each model is fetched at setup
time from its upstream Hugging Face repository. The licences below were read
from those repositories' model cards at the time this catalog was written and
**may change** — always verify against the upstream source before use.

| Model | Params | Licence | Upstream |
|---|---|---|---|
| `laguna-xs-2.1` | 33B MoE / 3B active | `openmdw-1.1` | [poolside/Laguna-XS-2.1-GGUF](https://huggingface.co/poolside/Laguna-XS-2.1-GGUF) |
| `north-mini-code-1.0` | 30B MoE / 3B active | `apache-2.0` | [unsloth/North-Mini-Code-1.0-GGUF](https://huggingface.co/unsloth/North-Mini-Code-1.0-GGUF) |
| `qwen3-coder-30b-a3b` | 30.5B MoE / 3.3B active | `apache-2.0` | [unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) |
| `nemotron-cascade-2-30b-a3b` | 30B MoE / 3B active | `other` | [bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF) |
| `ternary-bonsai-2-27b` | 27B ternary | `apache-2.0` | [prism-ml/Ternary-Bonsai-2-27B-GGUF](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-GGUF) |

## Licences needing attention

**`openmdw-1.1` (Laguna XS 2.1)** is **not** a standard open-source licence. It
carries use-based restrictions that differ from Apache/MIT. Read it in full
before any commercial or hosted use.

**`other` (Nemotron Cascade 2)** means NVIDIA publishes terms that do not map
onto a standard SPDX identifier. Read the model card directly.

**GGUF repackagings** (`unsloth/*`, `bartowski/*`) are third-party conversions.
The base model's licence governs the weights; the packager adds quantisation
only, but confirm the packager's own terms if that matters to you.

## Adding a model

When you add a model, add a row here. That is part of the contribution, not an
afterthought — a model without a licence note is an incomplete model.
