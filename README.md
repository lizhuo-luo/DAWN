<div align="center">

![logo](asset/logo.png)

<h1>DAWN: Dependency-Aware Fast Inference for Diffusion LLMs (Paper Coming Soon)</h1>

</div>

DAWN is a **training-free**, **dependency-aware** decoding method for fast dLLM inference.

DAWN leverages a dependency graph to select more reliable unmasking positions at each iteration, achieving high parallelism with negligible loss in generation quality. 

## 🚀 Features
- Fast inference support for Dream and LLaDA model.
- Multiple baseline method realization. 
- Full evaluation provided.
### Key Features
![overview](asset/overview.png)
**DAWN** is composed of three main modules: *Dependency Graph Construction*, *Anchor-Guided Decoding* and *Conflict-Based Scheduling*.

1. *Dependency Graph Construction* extracts a lightweight proxy of token dependencies from the model’s attention maps and builds a sparse directed dependency graph. It mitigates attention-sink bias by filtering positions with abnormal incoming attention mass, then retains only salient high-score attention links to capture meaningful couplings between positions for downstream scheduling.

2. *Anchor-Guided Decoding* first selects high-confidence masked positions that are likely safe to unmask in parallel, then uses previously committed high-confidence positions as anchors to relax the confidence requirement for their dependent (induced) positions. This expands safe parallelism beyond conservative thresholding by leveraging reliable context provided by anchors.

3. *Conflict-Based Scheduling* prevents error-prone joint updates by explicitly avoiding strongly coupled positions for remaining candidates under a lower confidence threshold. Using the dependency graph to define conflicts, it greedily constructs a large non-conflicting update set (an independent set), enabling additional parallel unmasking while reducing inconsistencies caused by non-independent position predictions.

## 🔧 Installation
### Option A: Quick start (recommended)
```bash
pip install -r requirements.txt
```

### Option B: Reproducible install
- Python 3.10.12
- NVIDIA GPU + CUDA 12.1 compatible driver
```bash
pip install -r requirements-lock.txt
```

## ✨ Eval

We provide the eval scripts for the main experiment, you can reproduce it directly. For example:
```bash
cd llada
bash eval_instruct.sh
```
The main experiment is conducted on an Nvidia H100 80 GPU, DAWN exhibits efficiency across multiple models and benchmarks:
![main result](asset/main_result.png)

## 🎓 Citation

Coming Soon...

## 🙏 Acknowledgements

We would like to thank the authors of [LLaDA](https://github.com/llada-project/llada), [Dream](https://github.com/dream-project/dream) and [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) for their excellent work and open-source contributions.
