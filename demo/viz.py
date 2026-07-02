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
    LABEL_MASK: "#57606a",       # muted gray
    LABEL_COMMITTED: "#2b6cb0",  # steel blue (matches accent)
}


def build_state(now_ids, decode_token, mask_id, eos_id=None):
    """One frame over the answer window, truncated at the first committed EOS.

    Returns (state, eff): the (token_text, label) list to display (only the
    tokens before the first EOS), and the number of effective committed tokens
    shown (non-mask, non-EOS) — used for the tokens/sec metric.
    """
    state = []
    eff = 0
    for tid in now_ids:
        if eos_id is not None and tid == eos_id:
            break  # stop at the first EOS — don't show it or anything after
        if tid == mask_id:
            state.append(("[MASK]", LABEL_MASK))
        else:
            state.append((decode_token(tid), LABEL_COMMITTED))
            eff += 1
    return state, eff


def frames_from_snapshots(xs, prompt_len, gen_len, decode_token, mask_id, eos_id=None):
    """Build the frame list from raw snapshots collected during generation.

    xs: list of (1, L) long tensors (one per step, plus an initial all-masked one).
    Returns a list of (state, eff) tuples.
    """
    frames = []
    for x in xs:
        now = x[0, prompt_len:prompt_len + gen_len].tolist()
        frames.append(build_state(now, decode_token, mask_id, eos_id))
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


def stats_html(title, method_name, nfe, elapsed, n_tokens,
               speedup=None, finished=False, accent="opp"):
    """Styled stat card (rendered in a gr.HTML component).

    n_tokens is the number of *effective* tokens (before the first EOS); tok/s
    is computed from it.
    """
    toks_per_s = (n_tokens / elapsed) if elapsed > 0 else 0.0
    chk = '<span class="dawn-stat__chk">✓ done</span>' if finished else ""
    speed = ""
    if speedup is not None:
        val = f"{speedup:.2f}".rstrip("0").rstrip(".")  # 1.00 -> "1"
        speed = f'<span class="dawn-stat__speed">{val}×</span>'
    return (
        f'<div class="dawn-stat dawn-stat--{accent}">'
        f'  <div class="dawn-stat__head">'
        f'    <span class="dawn-stat__title">{title}</span>{speed}{chk}'
        f'  </div>'
        f'  <div class="dawn-stat__method">{method_name}</div>'
        f'  <div class="dawn-stat__tiles">'
        f'    <div class="dawn-tile"><div class="dawn-tile__l">Tokens</div><div class="dawn-tile__v">{n_tokens}</div></div>'
        f'    <div class="dawn-tile"><div class="dawn-tile__l">TPS</div><div class="dawn-tile__v">{toks_per_s:.1f}</div></div>'
        f'  </div>'
        f'</div>'
    )


def status_html(text, kind="info"):
    """Styled status banner (rendered in a gr.HTML component). kind: info|run|done."""
    return f'<div class="dawn-status dawn-status--{kind}">{text}</div>'
