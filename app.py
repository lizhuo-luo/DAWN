# DAWN Speed Race — model-agnostic side-by-side demo
#
# Races DAWN (dependency-aware parallel decoding) against a baseline opponent on
# a diffusion LLM. A model selector switches between LLaDA and Dream; only one
# model is resident at a time (single-GPU friendly), generation is sequential,
# and each panel reports its own end-to-end (e2e) wall-clock time, step count
# (NFE) and tokens/sec. DAWN's panel also shows the speedup factor. Each lane is
# animated once, live, as it generates (opponent first, then DAWN).
#
# Run:   python app.py        (from the repo root)
# Needs: a CUDA GPU (~16GB+ for the 7-8B models in bf16) and the demo deps
#        (see requirements-demo.txt).

import time

import gradio as gr

from demo.registry import MODELS, DEFAULT_MODEL, get_backend
from demo.viz import COLOR_MAP, stats_html, status_html

EXAMPLES = [
    # multi-step grade-school math (GSM8K-style)
    "A bakery sells cupcakes in boxes of 6 and cookies in boxes of 8. On Monday "
    "they sold 14 boxes of cupcakes and 9 boxes of cookies. On Tuesday they sold "
    "twice as many cupcakes and half as many cookies as Monday. How many "
    "individual cupcakes and cookies did they sell in total over the two days? "
    "Show your reasoning step by step.",
    # algebra / word problem requiring setup
    "Two trains leave the same station traveling in opposite directions. One "
    "travels at 60 km/h and the other at 90 km/h. After how many hours will they "
    "be 450 km apart, and how far has each train traveled? Explain each step.",
    # # code generation (HumanEval/MBPP-style, non-trivial)
    # "Write a Python function `merge_intervals(intervals)` that takes a list of "
    # "`[start, end]` intervals, merges all overlapping intervals, and returns the "
    # "merged list sorted by start. Include a short docstring and handle the empty "
    # "list case.",
    # # dynamic-programming coding task
    # "Implement a Python function `longest_common_subsequence(a, b)` that returns "
    # "the length of the longest common subsequence of two strings using dynamic "
    # "programming. Add comments explaining the DP table.",
    # # structured reasoning / explanation
    # "Explain the difference between BFS and DFS for graph traversal: give the data "
    # "structure each uses, their time complexity, and one problem where each is the "
    # "better choice.",
]

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
THEME = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="gray",
    font=["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Helvetica", "Arial",
          "sans-serif"],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
).set(
    button_primary_background_fill="#2b6cb0",
    button_primary_background_fill_hover="#245a94",
    button_primary_text_color="#ffffff",
)

CSS = """
.gradio-container {max-width: 1200px !important; margin: 0 auto;}

/* flat, quiet visual language: thin borders, small radii, one steel-blue accent */

/* ---------- header ---------- */
.dawn-header {
  border: 1px solid var(--border-color-primary); border-radius: 8px;
  padding: 14px 18px; margin-bottom: 4px;
  background: var(--block-background-fill);
}
.dawn-header__title {
  font-size: 17px; font-weight: 700; margin: 0; color: var(--body-text-color);
}
.dawn-header__sub {
  font-size: 13px; color: var(--body-text-color-subdued); margin-top: 4px;
  line-height: 1.5;
}

/* ---------- status banner ---------- */
.dawn-status {
  border: 1px solid var(--border-color-primary); border-radius: 8px;
  padding: 8px 14px; font-size: 13px; font-weight: 500;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--body-text-color-subdued); background: var(--block-background-fill);
  display: flex; align-items: center; gap: 8px;
}
.dawn-status::before {content:""; width:8px; height:8px; border-radius:999px;}
.dawn-status--info::before {background:#8b949e;}
.dawn-status--run::before  {background:#d29922;}
.dawn-status--done::before {background:#2da44e;}

/* ---------- racing lanes ---------- */
.dawn-lane {
  border-radius: 8px; padding: 12px !important;
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
}
.dawn-lane--dawn {border: 2px solid #2b6cb0;}

/* ---------- stat card ---------- */
.dawn-stat {padding: 2px 2px 6px;}
.dawn-stat__head {display:flex; align-items:center; gap:8px;}
.dawn-stat__head::before {content:""; width:9px; height:9px; border-radius:999px;}
.dawn-stat--opp  .dawn-stat__head::before {background:#8b949e;}
.dawn-stat--dawn .dawn-stat__head::before {background:#2b6cb0;}
.dawn-stat__title {font-size: 15px; font-weight: 700; color: var(--body-text-color);}
.dawn-stat__chk {color:#2da44e; font-weight:700; font-size:13px;}
.dawn-stat__method {
  display:inline-block; font-family: ui-monospace, monospace; font-size: 12px;
  color: var(--body-text-color-subdued); margin: 4px 0 10px;
}
.dawn-stat__tiles {display:flex; gap:8px;}
.dawn-tile {
  flex:1; text-align:left; padding: 8px 12px; border-radius: 6px;
  background: #f6f8fa; border:1px solid var(--border-color-primary);
}
.dawn-tile__l {
  font-size: 11px; font-family: ui-monospace, monospace;
  color: var(--body-text-color-subdued); margin-bottom: 4px;
}
.dawn-tile__v {
  font-size: 16px; font-weight: 600; line-height: 1.1; color: var(--body-text-color);
  font-variant-numeric: tabular-nums; font-feature-settings: "tnum";
}
.dawn-tile__v span {font-size: 12px; font-weight:500; opacity:.6; margin-left:1px;}
.dawn-speedup {
  margin-top: 10px; font-weight: 600; font-size: 13.5px; color: #2b6cb0;
  padding: 7px 12px; border-radius: 6px;
  border: 1px solid #2b6cb0; background: rgba(43,108,176,.06);
}
"""


# --------------------------------------------------------------------------------------
# Race driver
# --------------------------------------------------------------------------------------
def opponent_label(opponent):
    return "Vanilla (1 token/step)" if opponent == "vanilla" else "Parallel-threshold"


# DAWN hyper-parameters exactly as used in the eval scripts. LLaDA's dawn path
# uses tau_low (conf_threshold unused); Dream's dawn path uses conf_threshold
# (tau_low unused). tau_low is task-dependent in the LLaDA eval (0.7–0.8); 0.75
# is the representative value. tau_sink/edge/induce differ between the two models.
DAWN_PRESETS = {
    "LLaDA-8B-Instruct": dict(tau_sink=0.01, tau_edge=0.07, tau_induce=0.7,  tau_low=0.75, conf_threshold=0.8),
    "LLaDA-1.5":         dict(tau_sink=0.01, tau_edge=0.07, tau_induce=0.7,  tau_low=0.75, conf_threshold=0.8),
    "Dream-7B (Instruct)": dict(tau_sink=0.03, tau_edge=0.10, tau_induce=0.75, tau_low=0.7,  conf_threshold=0.8),
}


def _apply_preset(model_key):
    p = DAWN_PRESETS.get(model_key, DAWN_PRESETS[DEFAULT_MODEL])
    return p["tau_sink"], p["tau_edge"], p["tau_induce"], p["tau_low"], p["conf_threshold"]


def _idle_stats(opp_name):
    return (
        stats_html("Opponent", opp_name, 0, 0.0, 0, accent="opp"),
        stats_html("DAWN", "dependency-aware", 0, 0.0, 0, accent="dawn"),
    )


def _speedup(opp_eff, opp_e2e, dawn_eff, dawn_e2e):
    """Speedup defined by throughput: DAWN tok/s divided by opponent tok/s."""
    opp_tps = (opp_eff / opp_e2e) if opp_e2e > 0 else 0.0
    dawn_tps = (dawn_eff / dawn_e2e) if dawn_e2e > 0 else 0.0
    return (dawn_tps / opp_tps) if opp_tps > 0 else None


def run_race(
    message, model_key, opponent, gen_length, block_length, threshold,
    temperature, top_p, viz_delay, tau_sink, tau_edge, tau_induce, tau_low,
    conf_threshold,
):
    if not message or not message.strip():
        raise gr.Error("Please enter a prompt.")
    if int(gen_length) % int(block_length) != 0:
        raise gr.Error(
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})."
        )

    gen_length = int(gen_length)
    opp_name = opponent_label(opponent)

    common = dict(
        gen_length=gen_length, block_length=int(block_length), threshold=threshold,
        temperature=temperature, top_p=top_p, tau_sink=tau_sink, tau_edge=tau_edge,
        tau_induce=tau_induce, tau_low=tau_low, conf_threshold=conf_threshold,
    )
    empty = []

    def emit(left, right, lstat, rstat, status):
        return left, right, lstat, rstat, status

    idle_l, idle_r = _idle_stats(opp_name)

    # ---- load model (may swap the resident one) -------------------------------
    yield emit(empty, empty, idle_l, idle_r,
               status_html(f"Loading <b>{model_key}</b> …", "run"))
    backend = get_backend(model_key)

    # ---- opponent (streamed live) ---------------------------------------------
    of = []
    opp_nfe, opp_e2e, opp_eff = 0, 0.0, 0
    opp_stat = idle_l
    yield emit(empty, empty, idle_l, idle_r,
               status_html(f"Racing <b>{opp_name}</b> …", "run"))
    for upd in backend.run_stream(message, [], {**common, "method": opponent}):
        of.append(upd["state"])
        opp_nfe, opp_e2e, opp_eff = upd["nfe"], upd["e2e"], upd["eff"]
        opp_stat = stats_html("Opponent", opp_name, opp_nfe, opp_e2e, opp_eff,
                              finished=upd["done"], accent="opp")
        yield emit(upd["state"], empty, opp_stat, idle_r,
                   status_html(f"Opponent (<b>{opp_name}</b>) denoising …", "run"))
        if viz_delay > 0:
            time.sleep(viz_delay)

    # ---- DAWN (streamed live) -------------------------------------------------
    df = []
    dawn_nfe, dawn_e2e, dawn_eff = 0, 0.0, 0
    dawn_stat = idle_r
    opp_final = of[-1] if of else empty
    for upd in backend.run_stream(message, [], {**common, "method": "dawn"}):
        df.append(upd["state"])
        dawn_nfe, dawn_e2e, dawn_eff = upd["nfe"], upd["e2e"], upd["eff"]
        sp = _speedup(opp_eff, opp_e2e, dawn_eff, dawn_e2e)
        dawn_stat = stats_html("DAWN", "dependency-aware", dawn_nfe, dawn_e2e,
                               dawn_eff, speedup=sp, finished=upd["done"], accent="dawn")
        yield emit(opp_final, upd["state"], opp_stat, dawn_stat,
                   status_html("DAWN denoising …", "run"))
        if viz_delay > 0:
            time.sleep(viz_delay)

    speedup = _speedup(opp_eff, opp_e2e, dawn_eff, dawn_e2e)
    opp_stat = stats_html("Opponent", opp_name, opp_nfe, opp_e2e, opp_eff,
                          finished=True, accent="opp")
    dawn_stat = stats_html("DAWN", "dependency-aware", dawn_nfe, dawn_e2e,
                           dawn_eff, speedup=speedup, finished=True, accent="dawn")

    # ---- settle on final state -------------------------------------------------
    winner = "DAWN" if (speedup and speedup >= 1) else "Opponent"
    msg = f"Done — <b>{winner}</b> wins" + (
        f" · {speedup:.2f}× faster" if speedup else ""
    )
    yield emit(of[-1] if of else empty, df[-1] if df else empty,
               opp_stat, dawn_stat, status_html(msg, "done"))


def clear_all():
    idle_l, idle_r = _idle_stats("—")
    return [], [], idle_l, idle_r, "", status_html("Ready when you are.", "info")


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
# Force the light theme regardless of the browser/OS dark-mode preference: Gradio
# only honors the __theme URL param, so redirect once if it isn't set to light.
FORCE_LIGHT_JS = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""


def build_demo():
    with gr.Blocks(title="DAWN Speed Race", theme=THEME, css=CSS,
                   js=FORCE_LIGHT_JS) as demo:
        gr.HTML(
            '<div class="dawn-header">'
            '<div class="dawn-header__title">DAWN Speed Race</div>'
            '<div class="dawn-header__sub">Dependency-Aware Fast Inference for '
            'Diffusion LLMs — DAWN vs. baseline on the same GPU. The two methods '
            'run sequentially (single GPU); each lane reports its own '
            'end-to-end time, steps and tok/s.</div>'
            '</div>'
        )

        with gr.Row():
            model_key = gr.Dropdown(
                choices=list(MODELS.keys()), value=DEFAULT_MODEL, label="Model", scale=2
            )
            opponent = gr.Dropdown(
                choices=[("Vanilla (1 token/step)", "vanilla"),
                         ("Parallel-threshold", "parallel")],
                value="vanilla", label="Opponent", scale=2,
            )

        with gr.Row():
            msg = gr.Textbox(label="Prompt", placeholder="Ask something…",
                             scale=5, autofocus=True)
            send_btn = gr.Button("Race", variant="primary", scale=1,
                                  elem_classes="dawn-race-btn")
            clear_btn = gr.Button("Clear", scale=1)

        gr.Examples(examples=EXAMPLES, inputs=msg)

        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                gen_length = gr.Slider(32, 512, value=256, step=32, label="gen_length")
                block_length = gr.Slider(16, 64, value=32, step=16, label="block_length")
            with gr.Row():
                threshold = gr.Slider(0.5, 1.0, value=0.9, step=0.01,
                                      label="threshold (parallel opponent)")
                temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05,
                                  label="top_p (Dream, temp>0)")
            with gr.Row():
                viz_delay = gr.Slider(0.0, 0.3, value=0.05, step=0.01,
                                      label="animation delay (s/frame)")
            with gr.Accordion("DAWN advanced (tau / conf)", open=False):
                with gr.Row():
                    tau_sink = gr.Slider(0.0, 0.2, value=0.01, step=0.005, label="tau_sink")
                    tau_edge = gr.Slider(0.0, 0.5, value=0.07, step=0.01, label="tau_edge")
                    tau_induce = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="tau_induce")
                    tau_low = gr.Slider(0.0, 1.0, value=0.75, step=0.05,
                                        label="tau_low (LLaDA)")
                    conf_threshold = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                               label="conf_threshold (Dream)")

        status = gr.HTML(status_html("Ready when you are.", "info"))

        idle_l, idle_r = _idle_stats("—")
        with gr.Row(equal_height=True):
            with gr.Column(scale=10, elem_classes="dawn-lane dawn-lane--opp"):
                left_stats = gr.HTML(idle_l)
                left_vis = gr.HighlightedText(
                    label="Opponent — denoising", combine_adjacent=False,
                    show_legend=True, color_map=COLOR_MAP,
                )
            with gr.Column(scale=10, elem_classes="dawn-lane dawn-lane--dawn"):
                right_stats = gr.HTML(idle_r)
                right_vis = gr.HighlightedText(
                    label="DAWN — denoising", combine_adjacent=False,
                    show_legend=True, color_map=COLOR_MAP,
                )

        inputs = [
            msg, model_key, opponent, gen_length, block_length, threshold,
            temperature, top_p, viz_delay, tau_sink, tau_edge, tau_induce, tau_low,
            conf_threshold,
        ]
        outputs = [left_vis, right_vis, left_stats, right_stats, status]

        # switching model resets DAWN's tau/conf to that model's eval preset
        model_key.change(
            _apply_preset, model_key,
            [tau_sink, tau_edge, tau_induce, tau_low, conf_threshold],
        )

        send_btn.click(run_race, inputs=inputs, outputs=outputs).then(
            lambda: "", None, msg
        )
        msg.submit(run_race, inputs=inputs, outputs=outputs).then(lambda: "", None, msg)
        clear_btn.click(
            clear_all, None,
            [left_vis, right_vis, left_stats, right_stats, msg, status],
        )

    return demo


if __name__ == "__main__":
    build_demo().queue().launch(server_name="0.0.0.0", share=True)
