"""Model-agnostic visualization helpers shared by every backend.

A backend produces, per generation, a list of *snapshots* of the answer window
(token-id tensors) plus per-step confidence tensors. These helpers turn those
raw snapshots into the (token_text, label) frames that gr.HighlightedText renders,
and into the per-panel stats markdown.
"""

# Labels -> CSS colors (passed to gr.HighlightedText(color_map=...)).
# Only two states: a position is either still masked, or committed (decoded).
LABEL_MASK = "masked"
LABEL_COMMITTED = "committed"

COLOR_MAP = {
    LABEL_MASK: "#2a2f3a",       # dark slate
    LABEL_COMMITTED: "#6366f1",  # indigo (matches accent)
}


def build_state(now_ids, decode_token, mask_id):
    """One frame: list of (token_text, label) over the answer window.

    Every position is either masked ([MASK]) or committed (its decoded token).
    """
    state = []
    for tid in now_ids:
        if tid == mask_id:
            state.append(("[MASK]", LABEL_MASK))
        else:
            state.append((decode_token(tid), LABEL_COMMITTED))
    return state


def frames_from_snapshots(xs, prompt_len, gen_len, decode_token, mask_id):
    """Build the full frame list from raw snapshots collected during generation.

    xs: list of (1, L) long tensors (one per step, plus an initial all-masked one).
    """
    frames = []
    for x in xs:
        now = x[0, prompt_len:prompt_len + gen_len].tolist()
        frames.append(build_state(now, decode_token, mask_id))
    return frames


def make_decode_token(tokenizer):
    def decode_token(tid):
        txt = tokenizer.decode([tid], skip_special_tokens=True)
        return txt if txt != "" else "·"
    return decode_token


def stats_md(title, method_name, nfe, elapsed, gen_length, speedup=None, finished=False):
    toks_per_s = (gen_length / elapsed) if elapsed > 0 else 0.0
    flag = " ✅" if finished else ""
    lines = [
        f"### {title}{flag}",
        f"`{method_name}`",
        f"⏱ **e2e:** {elapsed:.2f}s &nbsp;&nbsp; 🔁 **steps (NFE):** {nfe}",
        f"⚡ **tok/s:** {toks_per_s:.1f}",
    ]
    if speedup is not None:
        lines.append(f"## 🏁 {speedup:.2f}× faster")
    return "\n\n".join(lines)


def stats_html(title, method_name, nfe, elapsed, gen_length,
               speedup=None, finished=False, accent="opp"):
    """Styled stat card (rendered in a gr.HTML component)."""
    toks_per_s = (gen_length / elapsed) if elapsed > 0 else 0.0
    chk = '<span class="dawn-stat__chk">✓</span>' if finished else ""
    speed = (
        f'<div class="dawn-speedup">🏁 {speedup:.2f}× faster</div>'
        if speedup is not None else ""
    )
    return (
        f'<div class="dawn-stat dawn-stat--{accent}">'
        f'  <div class="dawn-stat__head">'
        f'    <span class="dawn-stat__title">{title}</span>{chk}'
        f'  </div>'
        f'  <div class="dawn-stat__method">{method_name}</div>'
        f'  <div class="dawn-stat__tiles">'
        f'    <div class="dawn-tile"><div class="dawn-tile__v">{elapsed:.2f}<span>s</span></div><div class="dawn-tile__l">e2e</div></div>'
        f'    <div class="dawn-tile"><div class="dawn-tile__v">{nfe}</div><div class="dawn-tile__l">steps</div></div>'
        f'    <div class="dawn-tile"><div class="dawn-tile__v">{toks_per_s:.0f}</div><div class="dawn-tile__l">tok/s</div></div>'
        f'  </div>'
        f'  {speed}'
        f'</div>'
    )


def status_html(text, kind="info"):
    """Styled status banner (rendered in a gr.HTML component). kind: info|run|done."""
    return f'<div class="dawn-status dawn-status--{kind}">{text}</div>'
