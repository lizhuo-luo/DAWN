"""LLaDA backend for the DAWN speed race.

Reuses the exact decoding helpers from llada/generate.py
(get_num_transfer_tokens / get_transfer_index / get_transfer_index_dawn) so the
visualization is faithful to the real algorithm. The three race methods are all
the same generate() loop with different arguments:

    vanilla   -> threshold=None  (one-token-per-step schedule)
    parallel  -> threshold set   (confidence-threshold parallel decoding)
    dawn      -> dawn=True        (dependency-aware decoding)

run_stream() yields one update per decoding step so the UI can show the denoising
and the timing/step counters live as generation proceeds.

NOTE: heavy imports happen inside __init__ (after the registry has put the llada/
directory on sys.path), never at module import time.
"""

import time

import torch

from demo.viz import build_state, make_decode_token

MASK_ID = 126336


class LladaBackend:
    family = "llada"

    def __init__(self, model_path, device):
        # imported here so sys.path is already pointed at llada/
        from transformers import AutoTokenizer
        from model.modeling_llada import LLaDAModelLM
        from generate import (
            get_num_transfer_tokens,
            get_transfer_index,
            get_transfer_index_dawn,
        )

        self._get_num_transfer_tokens = get_num_transfer_tokens
        self._get_transfer_index = get_transfer_index
        self._get_transfer_index_dawn = get_transfer_index_dawn

        self.device = device
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = (
            LLaDAModelLM.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
            .to(device)
            .eval()
        )
        self.decode_token = make_decode_token(self.tokenizer)

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
        ids = self.tokenizer(text)["input_ids"]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def run_stream(self, message, history, params):
        """Yield one dict per step: {state, nfe, e2e, answer, done}.

        `e2e` is cumulative and times only the model + algorithm compute (a CUDA
        sync is forced each step so async kernels are measured fairly — both the
        baseline and DAWN paths get identical treatment).
        """
        method = params["method"]
        gen_length = int(params["gen_length"])
        block_length = int(params["block_length"])
        temperature = float(params["temperature"])
        threshold = float(params["threshold"])
        remasking = "low_confidence"
        dawn = method == "dawn"
        thr = threshold if method == "parallel" else None

        prompt = self._build_prompt(message, history)
        prompt_len = prompt.shape[1]

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps = gen_length
        steps_per_block = steps // num_blocks

        x = torch.full(
            (1, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=self.device
        )
        x[:, :prompt_len] = prompt.clone()

        def window():
            return x[0, prompt_len:prompt_len + gen_length].tolist()

        nfe = 0
        compute_time = 0.0

        # initial all-masked frame
        yield {"state": build_state(window(), self.decode_token, MASK_ID),
               "nfe": 0, "e2e": 0.0, "answer": None, "done": False}

        for num_block in range(num_blocks):
            block_mask_index = (
                x[:, prompt_len + num_block * block_length:
                  prompt_len + (num_block + 1) * block_length] == MASK_ID
            )
            num_transfer_tokens = self._get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )
            i = 0
            while True:
                t0 = time.time()
                nfe += 1
                mask_index = x == MASK_ID
                output, avg_attn_scores = self.model(x, return_attn_scores=dawn)
                logits = output.logits
                mask_index[:, prompt_len + (num_block + 1) * block_length:] = 0

                if dawn:
                    x0, transfer_index = self._get_transfer_index_dawn(
                        logits, temperature, remasking, mask_index, x, None,
                        avg_attn_scores, tau_sink=params["tau_sink"],
                        tau_edge=params["tau_edge"], tau_induce=params["tau_induce"],
                        tau_low=params["tau_low"], num_block=num_block,
                        block_length=block_length, prompt_length=prompt_len,
                    )
                else:
                    x0, transfer_index = self._get_transfer_index(
                        logits, temperature, remasking, mask_index, x,
                        num_transfer_tokens[:, i] if thr is None else None, thr,
                    )

                x[transfer_index] = x0[transfer_index]
                # CUDA is async: force the step's GPU work to finish before timing,
                # otherwise we'd only measure kernel-launch time.
                if x.is_cuda:
                    torch.cuda.synchronize()
                compute_time += time.time() - t0
                i += 1

                block_done = (
                    x[:, prompt_len + num_block * block_length:
                      prompt_len + (num_block + 1) * block_length] == MASK_ID
                ).sum() == 0
                done = bool(block_done) and (num_block == num_blocks - 1)
                answer = (
                    self.tokenizer.decode(
                        x[0, prompt_len:].tolist(), skip_special_tokens=True
                    ).strip()
                    if done else None
                )
                yield {"state": build_state(window(), self.decode_token, MASK_ID),
                       "nfe": nfe, "e2e": compute_time, "answer": answer, "done": done}
                if block_done:
                    break
