"""Dream backend for the DAWN speed race.

Dream's DAWN logic lives inside DreamGenerationMixin._sample_block (selected by
`alg`), so instead of forking that loop we capture per-step snapshots through the
built-in `generation_tokens_hook_func(step, x, logits)` hook that the loop calls
after every step. Race methods map to Dream's `alg`:

    vanilla   -> alg='entropy'               (the standard Dream baseline)
    parallel  -> alg='confidence_threshold'  (parallel threshold decoding)
    dawn      -> alg='dawn'                    (dependency-aware decoding)

Dream's loop can't easily be paused mid-flight, so run_stream() generates to
completion (timing the real e2e), then replays the collected frames as a stream
with the e2e/step counters interpolated up to their true final values — so the
UI still shows live-updating timing per lane.

NOTE: heavy imports happen inside __init__, after the registry has put dream/ on
sys.path.
"""

import time

import torch

from demo.viz import frames_from_snapshots, make_decode_token

DREAM_MASK_FALLBACK = 151666

_ALG = {"vanilla": "entropy", "parallel": "confidence_threshold", "dawn": "dawn"}


class DreamBackend:
    family = "dream"

    def __init__(self, model_path, device):
        import types
        from transformers import AutoTokenizer
        from model.modeling_dream import DreamModel
        from model.generation_utils import DreamGenerationMixin

        self.device = device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = (
            DreamModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .to(device)
        )
        # bind the local mixin's generation methods (mirrors eval.py)
        self.model.diffusion_generate = types.MethodType(
            DreamGenerationMixin.diffusion_generate, self.model
        )
        self.model._sample_block = types.MethodType(
            DreamGenerationMixin._sample_block, self.model
        )

        gc = getattr(self.model, "generation_config", None)
        self.mask_id = (
            getattr(gc, "mask_token_id", None)
            or getattr(self.model.config, "mask_token_id", None)
            or DREAM_MASK_FALLBACK
        )
        self.decode_token = make_decode_token(self.tokenizer)
        self.eos_id = self.tokenizer.eos_token_id

    def unload(self):
        del self.model
        self.model = None

    def _build_prompt(self, message, history):
        messages = []
        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        messages.append({"role": "user", "content": message})
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        ids = self.tokenizer(text, return_tensors="pt").input_ids
        return ids.to(self.device)

    @torch.no_grad()
    def run_stream(self, message, history, params):
        """Yield one dict per frame: {state, nfe, e2e, answer, done}."""
        method = params["method"]
        gen_length = int(params["gen_length"])
        block_length = int(params["block_length"])
        temperature = float(params["temperature"])
        top_p = params.get("top_p", 0.95)
        alg = _ALG[method]

        prompt = self._build_prompt(message, history)
        prompt_len = prompt.shape[1]

        xs = []

        def hook(step, x, logits):
            xs.append(x.detach().clone())
            return x

        t0 = time.time()
        out, nfe = self.model.diffusion_generate(
            prompt,
            attention_mask=None,
            max_new_tokens=gen_length,
            steps=gen_length,
            block_length=block_length,
            output_history=False,
            return_dict_in_generate=True,
            temperature=temperature,
            top_p=(top_p if temperature > 0 else None),
            top_k=None,
            alg=alg,
            alg_temp=0.0,
            threshold=float(params["threshold"]),
            tau_induce=float(params["tau_induce"]),
            tau_sink=float(params["tau_sink"]),
            tau_edge=float(params["tau_edge"]),
            conf_threshold=float(params["conf_threshold"]),
            generation_tokens_hook_func=hook,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e = time.time() - t0

        sequences = out.sequences if hasattr(out, "sequences") else out
        frames = frames_from_snapshots(
            xs, prompt_len, gen_length, self.decode_token, self.mask_id, self.eos_id
        )
        answer_ids = sequences[0][prompt_len:].tolist()
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)
        eos = self.tokenizer.eos_token
        if eos and eos in answer:
            answer = answer.split(eos)[0]
        answer = answer.strip()

        n = len(frames)
        for k, (state, eff) in enumerate(frames):
            done = k == n - 1
            yield {
                "state": state,
                "eff": eff,
                "nfe": min(k, nfe),
                "e2e": e2e if done else e2e * (k + 1) / max(n, 1),
                "answer": answer if done else None,
                "done": done,
            }
