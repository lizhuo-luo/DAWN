# DAWN Speed Race — model-agnostic side-by-side demo
#
# Races three decoding methods on a diffusion LLM, shown side by side: vanilla
# (1 token/step), parallel confidence-threshold, and DAWN (dependency-aware
# parallel decoding). A model selector switches between LLaDA and Dream; only one
# model is resident at a time (single-GPU friendly). The lanes generate
# sequentially (vanilla -> parallel -> dawn), each animated live at its real
# per-step pace; each panel reports its effective tokens and TPS, and the
# parallel/DAWN panels show the throughput speedup vs the vanilla baseline.
#
# Run:   python app.py        (from the repo root)
# Needs: a CUDA GPU (~16GB+ for the 7-8B models in bf16) and the demo deps
#        (see requirements-demo.txt).

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
    # code generation (HumanEval/MBPP-style, non-trivial)
    "Write a Python function `merge_intervals(intervals)` that takes a list of "
    "`[start, end]` intervals, merges all overlapping intervals, and returns the "
    "merged list sorted by start. Include a short docstring and handle the empty "
    "list case.",
    # structured reasoning / explanation
    "Explain the difference between BFS and DFS for graph traversal: give the data "
    "structure each uses, their time complexity, and one problem where each is the "
    "better choice.",
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
  padding: 10px 14px; font-size: 13px; font-weight: 500;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: var(--body-text-color-subdued); background: var(--block-background-fill);
  display: flex; align-items: center; gap: 8px;
}
.dawn-status::before {content:""; width:8px; height:8px; border-radius:999px;}
.dawn-status--info::before {background:#8b949e;}
.dawn-status--run::before  {background:#d29922;}
.dawn-status--done::before {background:#2da44e;}

/* action row: keep the buttons a fixed size — without this they stretch to
   match the status column's height when it shows its loading state */
.dawn-action-row {align-items: flex-start;}
.dawn-action-row button {height: 40px; flex: none;}
.dawn-status {min-height: 40px; box-sizing: border-box;}

/* Gradio 4.44 quirks, verified against its compiled frontend CSS in a headless
   browser: (1) keep hidden per-component status trackers out of the layout;
   (2) while an event runs, gr.HTML wrappers get the `min` class, whose
   min-height:var(--size-24) (96px) inflates the stat cards / status bar and
   opens a blank strip above the lanes until the run finishes. Our HTML always
   sizes itself, so drop the forced minimum. */
.gradio-container .wrap.hide {display: none !important;}
.gradio-container .prose.min {min-height: 0 !important;}

/* Gradio 6.x quirk (this edge env runs 6.17; 4.44 renders gr.HTML as a bare
   div): every gr.HTML gets an .html-container wrapper carrying
   padding: var(--block-padding). That inset makes the header card narrower
   than the rows below it and pushes the status bar down/right out of line
   with the Race/Clear buttons. Flatten the wrapper around our self-styled
   HTML (tagged dawn-flat) — both where elem_classes lands on the container
   itself and where it lands on a parent block. Harmless on 4.44. */
.gradio-container .block.dawn-flat {
  padding: 0 !important; border: none !important;
  background: transparent !important; box-shadow: none !important;
}
.gradio-container .dawn-flat.html-container,
.gradio-container .dawn-flat .html-container {
  padding: 0 !important;
}

/* ---------- racing lanes ---------- */
/* slim separation between the three lanes — just enough to tell them apart */
.dawn-lanes {gap: 4px !important;}
.dawn-lane {
  border-radius: 8px; padding: 8px !important; gap: 6px !important;
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
}
.dawn-lane--dawn {border: 2px solid #2da44e;}

/* ---------- token stream (HighlightedText) ---------- */
/* smaller text and tighter token chips; the longer eased color transition makes
   each burst of committed tokens fade from masked-gray to its lane color instead
   of popping frame by frame (the spans are reused across frames, so this works) */
.dawn-lane .textfield {font-size: 12px !important; line-height: 1.55 !important;}
.dawn-lane .textspan {
  padding: 0.5px 3px !important; border-radius: 3px !important;
  transition: background-color .3s ease, color .3s ease !important;
}
.dawn-lane .category-legend {font-size: 10.5px !important;}
.dawn-lane .category-label {padding: 1px 6px !important;}

/* ---------- token stream, gradio 6.x ----------
   HighlightedText was rewritten in gradio 6 (checked against the compiled CSS
   shipped with 6.17.3): tokens render as .token chips with big default
   margins (4px left / 8px right, plus padding) and a pointer cursor, laid out
   as a flex-wrap list — the stream reads as a sparse grid of tags instead of
   flowing text. Tighten the chips back into a compact stream. The 4.44
   selectors above (.textspan/.category-legend) don't exist in 6.x and vice
   versa, so both blocks coexist harmlessly. */
.dawn-lane .token {
  transition: background-color .3s ease, color .3s ease !important;
  border-radius: 3px !important; cursor: text !important;
}
.dawn-lane .token.highlighted {
  margin: 0 1px 1px 0 !important; padding: 0.5px 3px !important;
}
.dawn-lane .textfield {
  font-size: 12px !important; line-height: 1.55 !important;
  align-content: flex-start;
}
.dawn-lane .legend {font-size: 10.5px !important; gap: 4px !important;}
.dawn-lane .legend-item {padding: 1px 6px !important;}

/* ---------- lane height ----------
   equal_height stretches EVERY direct child of the lane columns equally
   (gradio's .stretch > .column > * {flex-grow: 1}), so the stat card's
   wrapper grew too and opened a blank strip between the Tokens/TPS tiles and
   the token stream. Pin every lane child to its natural height, then hand
   all the remaining lane height to the stream (tagged dawn-stream). */
.dawn-lane > * {flex-grow: 0 !important; flex-shrink: 0 !important;}
/* 472px = 2x the 236px floor gradio's empty placeholder gives the stream
   (.empty.large min-height: calc(var(--size-64) - 20px)). Setting it on the
   stream itself keeps the lane at full height while tokens render too, not
   just when idle. */
.dawn-lane > .dawn-stream {flex-grow: 1 !important; min-height: 472px;}
.dawn-lane .empty.large {min-height: 472px !important;}

/* ---------- stat card ---------- */
.dawn-stat {padding: 2px 2px 6px;}
.dawn-stat__head {display:flex; align-items:center; gap:8px;}
.dawn-stat__head::before {content:""; width:9px; height:9px; border-radius:999px;}
.dawn-stat--opp  .dawn-stat__head::before {background:#8b949e;}
.dawn-stat--par  .dawn-stat__head::before {background:#2b6cb0;}
.dawn-stat--dawn .dawn-stat__head::before {background:#2da44e;}
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
/* the tile background is hardcoded light, so its text must not follow the
   theme vars: in dark mode var(--body-text-color) is near-white and the
   numbers vanish against #f6f8fa. Hardcode matching dark text instead. */
.dawn-tile__l {
  font-size: 11px; font-family: ui-monospace, monospace;
  color: #57606a; margin-bottom: 4px;
}
.dawn-tile__v {
  font-size: 16px; font-weight: 600; line-height: 1.1; color: #1f2328;
  font-variant-numeric: tabular-nums; font-feature-settings: "tnum";
}
.dawn-tile__v span {font-size: 12px; font-weight:500; opacity:.6; margin-left:1px;}
.dawn-stat__speed {
  font-weight: 600; font-size: 12.5px; line-height: 1;
  padding: 3px 8px; border-radius: 999px;
  color: #57606a; border: 1px solid #8b949e; background: rgba(139,148,158,.08);
  font-variant-numeric: tabular-nums;
}
.dawn-stat--par .dawn-stat__speed {
  color: #2b6cb0; border-color: #2b6cb0; background: rgba(43,108,176,.06);
}
.dawn-stat--dawn .dawn-stat__speed {
  color: #2da44e; border-color: #2da44e; background: rgba(45,164,78,.06);
}
"""


# --------------------------------------------------------------------------------------
# Race driver
# --------------------------------------------------------------------------------------
# The three lanes, raced sequentially (single GPU) and shown side by side:
# (method key, panel title, method subtitle, stat-card accent)
LANES = [
    ("vanilla", "Vanilla", "1 token/step", "opp"),
    ("parallel", "Parallel", "confidence-threshold", "par"),
    ("dawn", "DAWN", "dependency-aware", "dawn"),
]


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


def _idle_stats():
    return [
        stats_html(title, sub, 0, 0.0, 0, accent=accent)
        for _, title, sub, accent in LANES
    ]


def _speedup(base_eff, base_e2e, eff, e2e):
    """Throughput speedup vs the vanilla baseline: tok/s divided by tok/s."""
    base_tps = (base_eff / base_e2e) if base_e2e > 0 else 0.0
    tps = (eff / e2e) if e2e > 0 else 0.0
    return (tps / base_tps) if base_tps > 0 else None


def run_race(
    message, model_key, gen_length, block_length, threshold,
    temperature, top_p, tau_sink, tau_edge, tau_induce, tau_low,
    conf_threshold,
):
    if not message or not message.strip():
        raise gr.Error("Please enter a prompt.")
    if int(gen_length) % int(block_length) != 0:
        raise gr.Error(
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})."
        )

    gen_length = int(gen_length)

    common = dict(
        gen_length=gen_length, block_length=int(block_length), threshold=threshold,
        temperature=temperature, top_p=top_p, tau_sink=tau_sink, tau_edge=tau_edge,
        tau_induce=tau_induce, tau_low=tau_low, conf_threshold=conf_threshold,
    )

    states = [[] for _ in LANES]
    stats = _idle_stats()

    def emit(status):
        return (*states, *stats, status)

    # ---- load model (may swap the resident one) -------------------------------
    yield emit(status_html(f"Loading <b>{model_key}</b> …", "run"))
    backend = get_backend(model_key)
    if backend.device == "cpu":
        gr.Warning(
            "CUDA is not available — running on CPU, which is extremely slow. "
            "Check that this container has GPU access and a CUDA build of torch."
        )

    # ---- race the three lanes sequentially (single GPU), each streamed live ---
    base_eff, base_e2e = 0, 0.0  # vanilla baseline for the speedup ratios
    results = {}
    for idx, (method, title, sub, accent) in enumerate(LANES):
        nfe, e2e, eff = 0, 0.0, 0
        yield emit(status_html(f"<b>{title}</b> denoising …", "run"))
        for upd in backend.run_stream(message, [], {**common, "method": method}):
            nfe, e2e, eff = upd["nfe"], upd["e2e"], upd["eff"]
            sp = _speedup(base_eff, base_e2e, eff, e2e) if idx > 0 else 1.0
            states[idx] = upd["state"]
            stats[idx] = stats_html(title, sub, nfe, e2e, eff, speedup=sp,
                                    finished=upd["done"], accent=accent)
            yield emit(status_html(f"<b>{title}</b> denoising …", "run"))
        if method == "vanilla":
            base_eff, base_e2e = eff, e2e
        results[method] = (eff, e2e)

    # ---- final status -----------------------------------------------------------
    def tps(method):
        eff, e2e = results[method]
        return (eff / e2e) if e2e > 0 else 0.0

    winner = max(results, key=tps)
    winner_title = {m: t for m, t, _, _ in LANES}[winner]
    sp = _speedup(base_eff, base_e2e, *results[winner])
    msg = f"Done — <b>{winner_title}</b> wins" + (
        f" · {sp:.2f}× faster than vanilla" if sp and winner != "vanilla" else ""
    )
    yield emit(status_html(msg, "done"))


def clear_all():
    return (*[[] for _ in LANES], *_idle_stats(), "",
            status_html("Ready when you are.", "info"))


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
                   js=FORCE_LIGHT_JS, fill_width=True) as demo:
        gr.HTML(
            elem_classes="dawn-flat",
            value='<div class="dawn-header">'
            '<div class="dawn-header__title">DAWN Speed Race</div>'
            '<div class="dawn-header__sub">Dependency-Aware Fast Inference for '
            'Diffusion LLMs — the three decoding methods run sequentially on the '
            'same GPU and are shown side by side; each lane reports its own '
            'effective tokens and TPS.</div>'
            '</div>'
        )

        # the Model+Prompt row is rendered below the Race row; msg is created
        # unrendered up here only because gr.Examples needs the reference
        msg = gr.Textbox(label="Prompt", placeholder="Ask something…",
                         autofocus=True, scale=4, render=False)

        with gr.Accordion("Examples", open=False):
            gr.Examples(examples=EXAMPLES, inputs=msg)

        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                gen_length = gr.Slider(32, 512, value=256, step=32, label="gen_length")
                block_length = gr.Slider(16, 64, value=32, step=16, label="block_length")
            with gr.Row():
                threshold = gr.Slider(0.5, 1.0, value=0.9, step=0.01,
                                      label="threshold (Parallel lane)")
                temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05,
                                  label="top_p (Dream, temp>0)")
            with gr.Accordion("DAWN advanced (tau / conf)", open=False):
                with gr.Row():
                    tau_sink = gr.Slider(0.0, 0.2, value=0.01, step=0.005, label="tau_sink")
                    tau_edge = gr.Slider(0.0, 0.5, value=0.07, step=0.01, label="tau_edge")
                    tau_induce = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="tau_induce")
                    tau_low = gr.Slider(0.0, 1.0, value=0.75, step=0.05,
                                        label="tau_low (LLaDA)")
                    conf_threshold = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                               label="conf_threshold (Dream)")

        # Race / Clear sit on the left of the status row
        with gr.Row(elem_classes="dawn-action-row"):
            send_btn = gr.Button("Race", variant="primary", scale=0,
                                 min_width=110, elem_classes="dawn-race-btn")
            clear_btn = gr.Button("Clear", scale=0, min_width=90)
            with gr.Column(scale=10):
                status = gr.HTML(status_html("Ready when you are.", "info"),
                                 elem_classes="dawn-flat")

        with gr.Row():
            model_key = gr.Dropdown(
                choices=list(MODELS.keys()), value=DEFAULT_MODEL, label="Model", scale=1
            )
            msg.render()

        idle = _idle_stats()
        vis_boxes, stat_boxes = [], []
        with gr.Row(equal_height=True, elem_classes="dawn-lanes"):
            for (_method, title, _sub, accent), idle_stat in zip(LANES, idle):
                with gr.Column(scale=10, min_width=220,
                               elem_classes=f"dawn-lane dawn-lane--{accent}"):
                    stat_boxes.append(gr.HTML(idle_stat, elem_classes="dawn-flat"))
                    vis_boxes.append(gr.HighlightedText(
                        label=f"{title} — denoising", combine_adjacent=False,
                        show_legend=True, color_map=COLOR_MAP,
                        elem_classes="dawn-stream",
                    ))

        inputs = [
            msg, model_key, gen_length, block_length, threshold,
            temperature, top_p, tau_sink, tau_edge, tau_induce, tau_low,
            conf_threshold,
        ]
        outputs = [*vis_boxes, *stat_boxes, status]

        # switching model resets DAWN's tau/conf to that model's eval preset
        model_key.change(
            _apply_preset, model_key,
            [tau_sink, tau_edge, tau_induce, tau_low, conf_threshold],
        )

        # show_progress="hidden": we render our own status/stats, and Gradio's
        # built-in pending indicator inflates the output components' height
        # (an extra blank strip appears while the model runs).
        send_btn.click(run_race, inputs=inputs, outputs=outputs,
                       show_progress="hidden").then(lambda: "", None, msg)
        msg.submit(run_race, inputs=inputs, outputs=outputs,
                   show_progress="hidden").then(lambda: "", None, msg)
        clear_btn.click(
            clear_all, None,
            [*vis_boxes, *stat_boxes, msg, status],
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    build_demo().queue().launch(server_name="0.0.0.0", share=True)
