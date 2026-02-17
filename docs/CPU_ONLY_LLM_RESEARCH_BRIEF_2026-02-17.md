# CPU-Only LLM Research Brief (2026-02-17)

## Purpose

Define a practical path to evolve OpenMeshMind from a routing-centric runtime into a real conversational LLM stack that:

- runs on commodity CPUs
- can be improved with CPU-only training loops
- keeps decentralized and communication-efficient development

## Scope and constraints

- No specialized accelerators are assumed for baseline contributors.
- Reproducibility and verifiable gates are required.
- Current MVP gates must remain green while capability work is added.

## Research findings from primary sources

### 1) CPU inference is mature enough for immediate use

- `llama.cpp` provides CPU-first LLM inference with GGUF and multiple quantization levels, including very low-bit formats.
  - Source: https://github.com/ggml-org/llama.cpp
- ONNX Runtime documents low-bit quantization paths (including int4) for practical deployment optimization.
  - Source: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- `bitnet.cpp` demonstrates dedicated 1-bit-style inference efficiency improvements and validates a CPU-oriented inference direction.
  - Source: https://arxiv.org/abs/2502.11880
  - Source: https://github.com/microsoft/BitNet

Interpretation:
- We can move to a genuine local LLM conversational runtime now, without waiting for new research breakthroughs.

### 2) CPU-only full LLM pretraining remains constrained

- BitNet papers show strong quality/efficiency potential for low-bit networks but the ecosystem is still early for broad, simple CPU-only training pipelines at useful scale.
  - Source: https://arxiv.org/abs/2402.17764
  - Source: https://arxiv.org/abs/2504.12285
- The official BitNet repository currently emphasizes inference frameworks and model support updates rather than a turnkey CPU pretraining stack for broad use.
  - Source: https://github.com/microsoft/BitNet

Interpretation:
- For this project timeline, prioritize CPU-friendly fine-tuning and adapter updates before attempting dense pretraining of large models.

### 3) Parameter-efficient adaptation remains the practical training lever

- LoRA remains foundational for low-cost adaptation.
  - Source: https://arxiv.org/abs/2106.09685
- QLoRA shows how low-bit base weights plus adapters can preserve quality while reducing training resource pressure.
  - Source: https://arxiv.org/abs/2305.14314
- GaLore reduces optimization memory pressure, useful when RAM is the main constraint.
  - Source: https://arxiv.org/abs/2403.03507

Interpretation:
- CPU-only training should focus on PEFT-style adapter updates on smaller backbones, with strict memory/runtime budgeting.

### 4) Small-model quality has improved enough to be useful

- MobileLLM and SmolLM2 indicate strong gains from architecture/data recipe work in small parameter regimes.
  - Source: https://arxiv.org/abs/2402.14905
  - Source: https://arxiv.org/abs/2502.02737

Interpretation:
- Useful local conversational quality is achievable with sub-2B models if the evaluation and adaptation loop is disciplined.

### 5) Federated LLM adaptation research is accelerating

- Recent work focuses on communication-efficient and low-memory federated adapter tuning.
  - Source: https://arxiv.org/abs/2410.13097
  - Source: https://arxiv.org/abs/2602.03019

Interpretation:
- The repo's federated-first direction is aligned with current research, but implementation should begin with simple, auditable adapter aggregation and strict metrics.

## What is feasible now vs later

### Feasible now (recommended)

- CPU-only local inference for a real conversational LLM using GGUF quantized weights.
- CPU-only supervised adapter tuning (small batches, gradient accumulation, LoRA).
- Federated adapter-delta simulation with compression and reproducibility checks.

### Feasible with more time

- Better long-context quality via retrieval-assisted prompting and task-specific distillation.
- More aggressive low-bit or mixed-precision training experiments on CPU clusters.

### Not realistic for near-term MVP

- Fast dense pretraining of multi-billion-parameter models purely on ordinary personal CPUs.

## Proposed build track for OpenMeshMind

### Track A: Local conversational LLM runtime on CPU

- Replace rule-centric response path with a quantized base LLM runtime.
- Keep existing tools (retrieval, calculator, memory), but route them through structured prompting/tool-calling contracts.
- Add regression tasks for instruction fidelity, hallucination resistance, and multi-turn stability.

### Track B: CPU-only adapter training loop

- Add a tiny SFT dataset focused on current runtime tasks (memory, retrieval grounding, tool invocation).
- Train adapters only (LoRA-style), no dense full-model updates.
- Log memory, wall-clock, and quality deltas in machine-readable artifacts.

### Track C: Federated adapter aggregation

- Treat each node as producing compressed adapter deltas and metadata.
- Aggregate centrally only for simulation phase; preserve decentralized submission format.
- Extend current fairness/reproducibility checks with adapter-quality and communication budgets.

## Suggested 8-week execution plan

1. Week 1-2: CPU inference baseline
- Integrate quantized model runtime.
- Add conversation benchmark harness and quality gates.

2. Week 3-4: CPU adapter training baseline
- Add local adapter training/eval scripts.
- Produce first reproducible "before vs after" capability report.

3. Week 5-6: Federated adapter simulation
- Implement compressed adapter exchange and aggregation.
- Add communication-efficiency gates mirroring existing FedAvg style checks.

4. Week 7-8: Hardening and publication
- Reproducibility sweeps across multiple machines.
- Public artifact bundle with strict pass/fail evidence.

## Definition of done for this research-to-build phase

- LLM runtime:
  - A CPU-only command can run a multi-turn conversation with grounded retrieval and tool usage.
- Training:
  - A CPU-only command can fine-tune adapters and produce reproducible quality uplift on held-out tasks.
- Federated:
  - A CPU-only simulation can aggregate adapter updates from multiple nodes and report communication/quality trade-offs.
- Reliability:
  - All new checks are integrated into smoke and status pipelines with machine-readable artifacts.

## Open questions to resolve during implementation

- Which base model family delivers the best quality-per-latency on 8-16 GB RAM machines in this repo's task mix?
- Which adapter rank/target modules maximize gain under CPU wall-clock limits?
- Which compression format for adapter deltas gives the best communication-quality trade-off?
- Which conversation metrics should be mandatory vs informational at first rollout?
