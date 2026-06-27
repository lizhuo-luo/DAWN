# DAWN Speed Race — model-agnostic side-by-side demo
#
# Races DAWN (dependency-aware parallel decoding) against a baseline opponent on
# a diffusion LLM. A model selector switches between LLaDA and Dream; only one
# model is resident at a time (single-GPU friendly), generation is sequential,
# and each panel reports its own end-to-end (e2e) wall-clock time, step count
# (NFE) and tokens/sec. DAWN's panel also shows the speedup factor. An optional
# synchronized replay animates both panels on a shared real-time clock so DAWN
# visibly finishes first.
#
# Run:   python app.py        (from the repo root)
# Needs: a CUDA GPU (~16GB+ for the 7-8B models in bf16) and the demo deps
#        (see requirements-demo.txt).

import time

import gradio as gr

from demo.registry import MODELS, DEFAULT_MODEL, get_backend
from demo.viz import COLOR_MAP, stats_html, status_html

EXAMPLES = [
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 "
    "kilometers per hour. How many kilometers can she run in 8 hours?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain why the sky is blue in two sentences.",
]

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

CSS = """
.gradio-container {max-width: 1200px !important; margin: 0 auto;}

/* header */
.dawn-header {
  border-radius: 18px; padding: 26px 30px; margin-bottom: 6px;
  background: linear-gradient(110deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
  color: #fff; box-shadow: 0 10px 30px rgba(79,70,229,.28);
}
.dawn-header__title {font-size: 30px; font-weight: 800; letter-spacing:-.02em; margin:0;}
.dawn-header__sub {font-size: 14.5px; opacity: .92; margin-top: 6px; max-width: 760px;}

/* status banner */
.dawn-status {
  border-radius: 12px; padding: 10px 16px; font-weight: 600; font-size: 14px;
  border: 1px solid transparent;
}
.dawn-status--info {background:#eef2ff; color:#4338ca; border-color:#e0e7ff;}
.dawn-status--run  {background:#fff7ed; color:#c2410c; border-color:#fed7aa;}
.dawn-status--done {background:#ecfdf5; color:#047857; border-color:#a7f3d0;}

/* racing lanes */
.dawn-lane {border-radius: 16px; padding: 14px !important; background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);}
.dawn-lane--dawn {
  border: 1.5px solid #7c3aed;
  box-shadow: 0 8px 24px rgba(124,58,237,.16);
  background: linear-gradient(180deg, rgba(124,58,237,.05), rgba(124,58,237,0) 120px);
}
.dawn-vs {display:flex; align-items:center; justify-content:center;}
.dawn-vs__badge {
  font-weight: 900; font-size: 18px; color:#7c3aed;
  background:#fff; border:2px solid #7c3aed; border-radius: 999px;
  width: 46px; height: 46px; display:flex; align-items:center; justify-content:center;
  box-shadow: 0 4px 14px rgba(124,58,237,.25);
}

/* stat card */
.dawn-stat {padding: 4px 2px 8px;}
.dawn-stat__head {display:flex; align-items:center; gap:8px;}
.dawn-stat__title {font-size: 18px; font-weight: 800; color: var(--body-text-color);}
.dawn-stat__chk {
  color:#fff; background:#22c55e; border-radius:999px; width:20px; height:20px;
  display:inline-flex; align-items:center; justify-content:center; font-size:12px;
}
.dawn-stat__method {
  display:inline-block; font-family: ui-monospace, monospace; font-size: 12px;
  color:#6b7280; background: var(--neutral-100); padding: 2px 8px; border-radius: 6px;
  margin: 6px 0 10px;
}
.dawn-stat__tiles {display:flex; gap:10px;}
.dawn-tile {
  flex:1; text-align:center; padding: 10px 6px; border-radius: 12px;
  background: var(--neutral-100); border:1px solid var(--border-color-primary);
}
.dawn-stat--dawn .dawn-tile {background: rgba(124,58,237,.08); border-color: rgba(124,58,237,.18);}
.dawn-tile__v {font-size: 22px; font-weight: 800; line-height: 1; color: var(--body-text-color);}
.dawn-tile__v span {font-size: 13px; font-weight:600; opacity:.6; margin-left:1px;}
.dawn-tile__l {font-size: 11px; text-transform: uppercase; letter-spacing:.06em;
  color:#9ca3af; margin-top: 5px;}
.dawn-speedup {
  margin-top: 12px; text-align:center; font-weight: 800; font-size: 18px; color:#fff;
  padding: 9px; border-radius: 12px;
  background: linear-gradient(100deg,#7c3aed,#db2777);
  box-shadow: 0 6px 18px rgba(219,39,119,.3);
}
"""


# --------------------------------------------------------------------------------------
# Race driver
# --------------------------------------------------------------------------------------
def opponent_label(opponent):
    return "Vanilla (1 token/step)" if opponent == "vanilla" else "Parallel-threshold"


def _idle_stats(gen_length, opp_name):
    return (
        stats_html("Opponent", opp_name, 0, 0.0, gen_length, accent="opp"),
        stats_html("DAWN", "dependency-aware", 0, 0.0, gen_length, accent="dawn"),
    )


def run_race(
    message, model_key, opponent, gen_length, block_length, threshold,
    temperature, top_p, viz_delay, tau_sink, tau_edge, tau_induce, tau_low,
    conf_threshold, do_replay,
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

    idle_l, idle_r = _idle_stats(gen_length, opp_name)

    # ---- load model (may swap the resident one) -------------------------------
    yield emit(empty, empty, idle_l, idle_r,
               status_html(f"⏳ Loading <b>{model_key}</b> …", "run"))
    backend = get_backend(model_key)

    # ---- opponent (streamed live) ---------------------------------------------
    of = []
    opp_nfe, opp_e2e = 0, 0.0
    opp_stat = idle_l
    yield emit(empty, empty, idle_l, idle_r,
               status_html(f"▶️ Racing <b>{opp_name}</b> …", "run"))
    for upd in backend.run_stream(message, [], {**common, "method": opponent}):
        of.append(upd["state"])
        opp_nfe, opp_e2e = upd["nfe"], upd["e2e"]
        opp_stat = stats_html("Opponent", opp_name, opp_nfe, opp_e2e, gen_length,
                              finished=upd["done"], accent="opp")
        yield emit(upd["state"], empty, opp_stat, idle_r,
                   status_html(f"▶️ Opponent (<b>{opp_name}</b>) denoising …", "run"))

    # ---- DAWN (streamed live) -------------------------------------------------
    df = []
    dawn_nfe, dawn_e2e = 0, 0.0
    dawn_stat = idle_r
    opp_final = of[-1] if of else empty
    for upd in backend.run_stream(message, [], {**common, "method": "dawn"}):
        df.append(upd["state"])
        dawn_nfe, dawn_e2e = upd["nfe"], upd["e2e"]
        sp = (opp_e2e / dawn_e2e) if dawn_e2e > 0 else None
        dawn_stat = stats_html("DAWN", "dependency-aware", dawn_nfe, dawn_e2e,
                               gen_length, speedup=sp, finished=upd["done"], accent="dawn")
        yield emit(opp_final, upd["state"], opp_stat, dawn_stat,
                   status_html("▶️ DAWN denoising …", "run"))

    speedup = (opp_e2e / dawn_e2e) if dawn_e2e > 0 else None
    opp_stat = stats_html("Opponent", opp_name, opp_nfe, opp_e2e, gen_length,
                          finished=True, accent="opp")
    dawn_stat = stats_html("DAWN", "dependency-aware", dawn_nfe, dawn_e2e,
                           gen_length, speedup=speedup, finished=True, accent="dawn")

    # ---- optional synchronized replay on a shared real-time clock --------------
    if do_replay and of and df:
        n_o, n_d = len(of), len(df)
        total = max(opp_e2e, dawn_e2e)
        TICKS = 60
        for t in range(TICKS + 1):
            cur = (t / TICKS) * total
            oi = min(n_o - 1, int(cur / opp_e2e * n_o)) if opp_e2e > 0 else n_o - 1
            di = min(n_d - 1, int(cur / dawn_e2e * n_d)) if dawn_e2e > 0 else n_d - 1
            yield emit(
                of[oi], df[di],
                stats_html("Opponent", opp_name, opp_nfe, opp_e2e, gen_length,
                           finished=(oi == n_o - 1), accent="opp"),
                stats_html("DAWN", "dependency-aware", dawn_nfe, dawn_e2e,
                           gen_length, speedup=speedup, finished=(di == n_d - 1),
                           accent="dawn"),
                status_html("🏁 Racing… <i>(real-time replay)</i>", "run"),
            )
            if viz_delay > 0:
                time.sleep(viz_delay)

    # ---- settle on final state -------------------------------------------------
    winner = "DAWN" if (speedup and speedup >= 1) else "Opponent"
    msg = f"✅ Done — <b>{winner}</b> wins" + (
        f" &nbsp;·&nbsp; {speedup:.2f}× faster" if speedup else ""
    )
    yield emit(of[-1] if of else empty, df[-1] if df else empty,
               opp_stat, dawn_stat, status_html(msg, "done"))


def clear_all():
    idle_l, idle_r = _idle_stats(1, "—")
    return [], [], idle_l, idle_r, "", status_html("Ready when you are.", "info")


# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
def build_demo():
    with gr.Blocks(title="DAWN Speed Race", theme=THEME, css=CSS) as demo:
        gr.HTML(
            '<div class="dawn-header">'
            '<div class="dawn-header__title">🏁 DAWN Speed Race</div>'
            '<div class="dawn-header__sub">Dependency-Aware Fast Inference for '
            'Diffusion LLMs — watch DAWN out-pace the baseline on the same GPU. '
            'Pick a model; on a single GPU the two methods run sequentially and '
            'each lane reports its own end-to-end time.</div>'
            '</div>'
        )

        with gr.Row():
            model_key = gr.Dropdown(
                choices=list(MODELS.keys()), value=DEFAULT_MODEL, label="Model", scale=2
            )
            opponent = gr.Dropdown(
                choices=[("Vanilla (1 token/step)", "vanilla"),
                         ("Parallel-threshold", "parallel")],
                value="parallel", label="Opponent", scale=2,
            )

        with gr.Row():
            msg = gr.Textbox(label="Prompt", placeholder="Ask something…",
                             scale=5, autofocus=True)
            send_btn = gr.Button("🏁 Race", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", scale=1)

        gr.Examples(examples=EXAMPLES, inputs=msg)

        with gr.Accordion("Generation Settings", open=False):
            with gr.Row():
                gen_length = gr.Slider(32, 256, value=128, step=32, label="gen_length")
                block_length = gr.Slider(16, 64, value=32, step=16, label="block_length")
            with gr.Row():
                threshold = gr.Slider(0.5, 1.0, value=0.9, step=0.01,
                                      label="threshold (parallel opponent)")
                temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05,
                                  label="top_p (Dream, temp>0)")
            with gr.Row():
                viz_delay = gr.Slider(0.0, 0.3, value=0.05, step=0.01,
                                      label="replay delay (s/frame)")
                do_replay = gr.Checkbox(value=True, label="Synchronized real-time replay")
            with gr.Accordion("DAWN advanced (tau / conf)", open=False):
                with gr.Row():
                    tau_sink = gr.Slider(0.0, 0.2, value=0.01, step=0.005, label="tau_sink")
                    tau_edge = gr.Slider(0.0, 0.5, value=0.07, step=0.01, label="tau_edge")
                    tau_induce = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="tau_induce")
                    tau_low = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                        label="tau_low (LLaDA)")
                    conf_threshold = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                               label="conf_threshold (Dream)")

        status = gr.HTML(status_html("Ready when you are.", "info"))

        idle_l, idle_r = _idle_stats(1, "—")
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
            conf_threshold, do_replay,
        ]
        outputs = [left_vis, right_vis, left_stats, right_stats, status]

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
