"""Model registry + backend manager.

The llada/ and dream/ sub-repos both expose top-level module names (`model`,
`generate`, `gdllm_utils`, ...) and use absolute imports like
`from model.modeling_llada import ...`. They can't both be on sys.path at once
without colliding, so this manager keeps exactly ONE backend loaded at a time:
when you switch model family it tears down the previous model, purges the
conflicting cached modules, and points sys.path at the right sub-repo before
importing the new backend.

Single-GPU friendly: only one large model is ever resident.
"""

import gc
import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
LLADA_DIR = os.path.join(REPO_ROOT, "llada")
DREAM_DIR = os.path.join(REPO_ROOT, "dream")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# display name -> (family, hf model path)
MODELS = {
    "LLaDA-8B-Instruct": ("llada", "GSAI-ML/LLaDA-8B-Instruct"),
    "LLaDA-1.5": ("llada", "GSAI-ML/LLaDA-1.5"),
    "Dream-7B (Instruct)": ("dream", "Dream-org/Dream-v0-Instruct-7B"),
}
DEFAULT_MODEL = "LLaDA-8B-Instruct"

# top-level module names owned by the sub-repos that must be purged on a swap
_CONFLICTING = {
    "model", "generate", "sanitize", "gdllm_utils",
    "configuration_llada", "modeling_llada",
    "configuration_dream", "modeling_dream", "generation_utils",
    "tokenization_dream", "postprocess_code", "postprocess_code_humaneval",
}

_active = {"key": None, "backend": None}


def _purge_repo_modules():
    for name in list(sys.modules):
        root = name.split(".")[0]
        if root in _CONFLICTING:
            del sys.modules[name]


def _activate_dir(directory):
    for p in (LLADA_DIR, DREAM_DIR):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, directory)
    _purge_repo_modules()


def get_backend(model_key):
    """Return a loaded backend for `model_key`, swapping the resident model if needed."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    if _active["key"] == model_key and _active["backend"] is not None:
        return _active["backend"]

    # tear down whatever is currently loaded
    if _active["backend"] is not None:
        try:
            _active["backend"].unload()
        finally:
            _active["backend"] = None
            _active["key"] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    family, path = MODELS[model_key]
    if family == "llada":
        _activate_dir(LLADA_DIR)
        from demo.backend_llada import LladaBackend
        backend = LladaBackend(path, DEVICE)
    elif family == "dream":
        _activate_dir(DREAM_DIR)
        from demo.backend_dream import DreamBackend
        backend = DreamBackend(path, DEVICE)
    else:
        raise ValueError(f"Unknown family: {family}")

    _active["key"] = model_key
    _active["backend"] = backend
    return backend
