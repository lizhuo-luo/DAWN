# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DAWN (Dependency-Aware Fast Inference for Diffusion LLMs) is a **training-free, plug-and-play** decoding method that speeds up diffusion LLM (dLLM) inference. Instead of unmasking tokens by confidence alone, DAWN builds a sparse dependency graph from the model's attention maps and uses it to select which masked positions are safe to unmask in parallel each step — gaining parallelism with little quality loss.

This repo is a research/evaluation harness, not a library. It wires DAWN (and several baselines) into two open dLLMs — **LLaDA** and **Dream** — and runs them through `lm-eval-harness`.

## Setup & Commands

```bash
pip install -r requirements.txt          # quick start
pip install -r requirements-lock.txt     # reproducible (pins torch==2.5.1+cu121, transformers==4.49.0, lm_eval==0.4.8)
```

Evaluation always requires these env vars (set in the eval `.sh` scripts):
```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

Run a full benchmark sweep (downloads the model from HF on first run, needs a CUDA GPU; main results use an H100):
```bash
cd llada && bash eval_instruct.sh    # LLaDA-8B-Instruct on gsm8k / humaneval / mbpp
cd llada && bash eval_15.sh          # LLaDA-1.5
cd dream && bash eval_instruct.sh    # Dream-7B
cd dream && bash eval_base.sh
```

Each `.sh` runs the **same task several times**, once per decoding method (baseline, parallel-threshold, klass, local_leap, dawn, ...). To run a single method, copy one `accelerate launch` block and edit its `--model_args`. To smoke-test, append `--limit N` to cap the number of examples.

`dream/eval.md` documents the Dream `--model_args` combinations in detail; it is the best reference for valid arg permutations.

### Code-generation post-processing

HumanEval/MBPP results need post-processing before scoring — the raw `samples_*.jsonl` under the output path must be passed through the sanitizer:
```bash
python llada/postprocess_code_humaneval.py <output_path>/samples_*.jsonl   # then sanitize.py
python dream/postprocess_code.py <output_path>/samples_*.jsonl
```
The `.sh` scripts leave a `# NOTICE: use postprocess ...` comment where this applies.

## Architecture

The two model directories (`llada/`, `dream/`) are **parallel, self-contained implementations of the same idea** — they do not import from each other. Pick the directory matching the model you are evaluating; expect to mirror any cross-cutting change in both.

### Entry point: lm-eval harness adapters
- `llada/eval_llada.py` — registers `@register_model("llada_dist")`. The `LLaDAEvalHarness.__init__` signature is the **canonical list of all tunable knobs** (`dawn`, `klass`, `local_leap`, `dual_cache`, `threshold`, the `tau_*` graph params, etc.); everything in `--model_args` maps to it.
- `dream/eval.py` — registers `@register_model("dream")`. Dream selects the method via a single `alg=` string (`entropy`, `confidence_threshold`, `dawn`, `klass`, `local_leap`, ...) rather than boolean flags.

### Decoding methods (where DAWN actually lives)
- **LLaDA**: `llada/generate.py` holds every decoding loop as a separate `generate*` function. The harness dispatches to one based on its flags (`generate_klass`, `generate_with_dual_cache`, or `generate` with `dawn=`/`local_leap=`). The per-step token-selection logic is in the `get_transfer_index_*` functions — `get_transfer_index_dawn` is the core DAWN step.
- **Dream**: `dream/model/generation_utils.py` (`DreamGenerationMixin._sample` / `_sample_block`) contains one big loop that branches on `alg`; the DAWN branch is inline (search for `alg == 'dawn'`).

### Shared DAWN primitives
Both `llada/gdllm_utils.py` and `dream/model/gdllm_utils.py` define the same three building blocks of the DAWN paper's three modules:
- `detect_attn_sinks_` — flags attention-sink columns (high mean incoming attention) to filter from the dependency graph. Threshold = `tau_sink`.
- The graph edges come from thresholding attention with `tau_edge`; anchor-induced positions are relaxed by `tau_induce`; the low-confidence floor is `tau_low`.
- `select_parallel_tokens_conflict_mis` — greedy maximum-independent-set selection over the conflict graph, the *Conflict-Based Scheduling* module. Picks a large non-conflicting set of positions to unmask together.

So a DAWN step is: get attention from the model → remove sinks → build edge/conflict graph → unmask high-confidence + anchor-induced positions + a conflict-free independent set of the rest.

### Vendored models
`llada/model/` and `dream/model/` are HF-style model definitions (`modeling_*.py`, `configuration_*.py`) adapted from the upstream LLaDA/Dream repos. The key local modification is that the forward pass can **return attention scores** (`return_attn_scores=...` / `output_attentions`) so DAWN can read the dependency signal — vanilla HF models don't expose this cheaply.

## Conventions

- Results are written under `evals_results_<model_name>/<method>/<task>-ns0-<length>/` (gitignored). `outp_path` in `--model_args` controls this; `--output_path` is the lm-eval output dir.
- Common generation params in the scripts: `length=256`, `block_length=32`, `steps=length/block_length` for parallel methods vs `steps=length` for the baseline (one token per step).
- Baselines included for comparison: `klass` (KL-divergence based, from Fast-dLLM lineage) and `local_leap` (radius-based relaxed thresholding). When changing DAWN, leave these intact — the paper's tables compare against them.
