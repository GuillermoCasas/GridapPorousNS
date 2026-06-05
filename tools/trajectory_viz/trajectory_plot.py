#!/usr/bin/env python3
"""
trajectory_viz — render a nonlinear-solver trajectory (a list of algorithm stages) as one PNG.

SHARED, GENERIC TOOL. It knows nothing about a test's notion of "runs"/"attempts"/eps_pert — those
live in each test's small wrapper. It renders ONE trajectory into a single PNG.

ALL layout/style parameters (fonts, colours, gaps, box sizes, margins, container proportions, dpi,
marker sizes, ...) live in the sibling `plot_params.json` — there are NO hard-coded style literals
in this module. Edit that file to tune by hand, or call `set_params(path)` to point at another one.

Layout is a vertical flow engine (`_VStack`): every element (title, subtitle, the start node, each
algorithm-stage box and its label, the Algorithm-C container, the end node) reserves its own
bounding box in INCHES, and the figure height is computed from the sum BEFORE anything is drawn — so
elements can never overlap and never overflow, regardless of text length or stage count.

Two render paths (auto-dispatched by `plot_attempt`):
  * flat  — a top-to-bottom column of algorithm-stage boxes (ASGS, or any run with no outer loop);
  * nested — for OSGS: a Stage-I box, then an "Algorithm C" container holding (i) the OUTER-loop
    drift convergence vs outer iteration and (ii) one inner-cascade thumbnail per outer iteration,
    with the MMS plateau-verification cycles (outer iters after base convergence) shaded/annotated.

Nomenclature (theory/osgs_algorithm.tex): Alg. O orchestrator; Alg. B RobustNonlinearCascade
(Newton-1 -> Picard -> Newton-2); Alg. C OSGS outer loop; Alg. A ExactNewtonPipeline (the kernel
that produces the per-iteration residual dots). Math symbols use mathtext ($...$); no system LaTeX.
"""
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.lines import Line2D

_DEFAULT_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_params.json")

CONVERGED_REASONS = ("ftol_reached", "initial_ftol", "stagnation_noise_floor_reached")


def _load_params(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"trajectory_viz parameter file not found: {path}. "
                                "It holds ALL plot parameters (no hard-coded defaults in the .py).")
    with open(path) as fh:
        return json.load(fh)


def _apply_params(p):
    """Install a params dict as the module-global P, derive the palette + stop-reason colours,
    and push the matplotlib rcParams it controls."""
    global P, _STOP_COLOR
    P = p
    c = p["colors"]
    _STOP_COLOR = {
        "ftol_reached": c["green"], "initial_ftol": c["green"], "stagnation_noise_floor_reached": c["green"],
        "merit_divergence_escaped": c["red"], "linesearch_failed": c["red"], "linear_solve_nan": c["red"],
        "no_progress_stall": c["red"], "nonfinite": c["red"], "exception": c["red"],
        "max_iters_stagnation": c["amber"], "xtol_stagnation": c["amber"], "max_iters": c["amber"],
        "max_iters_caught": c["amber"],
    }
    plt.rcParams.update({"mathtext.fontset": p["mathtext_fontset"], "font.size": p["base_font_size"]})


def set_params(path):
    """Point the plotter at an alternative parameter JSON (e.g. a per-test override)."""
    _apply_params(_load_params(path))


_apply_params(_load_params(_DEFAULT_PARAMS))

_STEP_NAME = {"N1": "Newton-1 (Alg. A)", "N2": "Newton-2 (Alg. A)", "Picard": "Picard smoother"}


def stage_color(stage):
    state = stage.get("state", "ok")
    if state != "ok":
        return _STOP_COLOR.get(state, P["colors"]["grey"])
    return _STOP_COLOR.get(stage.get("stop_reason", ""), P["colors"]["grey"])


def _parse_stage(code):
    """(algorithm-context, step-name) from a stage code, using documentation nomenclature."""
    parts = code.split(":")
    algo = parts[0]
    mid = parts[1] if len(parts) > 1 else ""
    step = parts[2] if len(parts) > 2 else mid
    if algo == "B" and mid == "StageI":
        context = "Alg. B - Stage I (ASGS init)"
    elif algo == "C" and mid.startswith("OSGS") and step == "Coupled":
        context = "Alg. C - OSGS coupled (single Newton; pi recomputed each iter)"
    elif algo == "C" and mid.startswith("OSGS"):
        idx = mid[mid.find("[") + 1:mid.find("]")] if "[" in mid else "?"
        context = "Alg. C - OSGS outer it. {} (-> Alg. B)".format(idx)
    else:
        context = "{} - {}".format(algo, mid)
    return context, _STEP_NAME.get(step, step)


def fmt_pow(x):
    if x == 0:
        return "0"
    e = math.log10(abs(x))
    if abs(e - round(e)) < 1e-9:
        return rf"10^{{{int(round(e))}}}"
    return f"{x:g}"


def sci(x):
    if x is None or not (isinstance(x, (int, float)) and math.isfinite(x)):
        return r"\mathrm{n/a}"
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    return rf"{x / (10 ** e):.1f}\times10^{{{e}}}"


def _arrow(fig, x0, y0, x1, y1, color=None):
    a = P["arrow"]
    fig.add_artist(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=a["mutation_scale"], lw=a["line_width"],
                                   color=color or P["colors"]["arrow"], shrinkA=a["shrink"], shrinkB=a["shrink"],
                                   zorder=10))


def _finite_pos(v):
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def _residual_xy(stage_list):
    """Concatenate the residual histories of a B-cascade onto a global inner-iteration counter,
    prefixed with the entry residual. Prefers the NORMALIZED residual f_norm = ‖R‖/tol (the
    per-field convergence quantity actually compared to the threshold, ≤1 ⟺ converged); falls back
    to the raw inf-norm ‖R‖_∞ for pre-change traces that lack it. Returns (xs, ys, acc, normalized)."""
    # One scale per panel: normalized iff any history entry of this cascade exposes a finite f_norm.
    normalized = any(_finite_pos(h.get("f_norm"))
                     for s in (stage_list or []) for h in (s.get("history") or []))
    ykey = "f_norm" if normalized else "f_inf"
    inkey = "res_in_norm" if normalized else "res_in"
    xs, ys, acc, gi = [], [], [], 0
    if stage_list:
        ri = stage_list[0].get(inkey)
        if _finite_pos(ri):
            xs.append(0); ys.append(ri); acc.append(True)
    for s in stage_list:
        for h in (s.get("history") or []):
            f = h.get(ykey)
            if _finite_pos(f):
                gi += 1; xs.append(gi); ys.append(f); acc.append(bool(h.get("accepted", True)))
    return xs, ys, acc, normalized


def _start_node_text(stages, override=None):
    """Entry-node label: normalized residual (‖R₀‖/tol) if the trace carries it, else raw ‖R₀‖."""
    s0 = stages[0] if stages else {}
    if override is None:
        rn = s0.get("res_in_norm")
        if _finite_pos(rn):
            return rf"$x_0:\ \|R_0\|/\mathrm{{tol}}={sci(rn)}$"
    r = override if override is not None else s0.get("res_in")
    return rf"$x_0:\ \|R_0\|={sci(r)}$"


def _final_node_text(last, ok):
    """Exit-node label: normalized final residual if available, else raw ‖R‖."""
    mark = r"$\checkmark$" if ok else r"$\times$"
    rn = last.get("res_out_norm")
    if _finite_pos(rn):
        return mark + rf" final $\|R\|/\mathrm{{tol}}={sci(rn)}$"
    return mark + rf" final $\|R\|={sci(last.get('res_out'))}$"


def _fill_stage(ax, stages):
    """Draw a B-cascade's residual-vs-iteration dots (one stage or a list) into the given axes."""
    stages = stages if isinstance(stages, list) else [stages]
    M, F, c, T = P["markers"], P["fonts"], P["colors"], P["threshold"]
    color = stage_color(stages[-1]) if stages else c["grey"]
    xs, ys, acc, normalized = _residual_xy(stages)
    if ys:
        ax.semilogy(xs, ys, "-", color=color, lw=M["stage_line_width"], alpha=M["stage_line_alpha"], zorder=1)
        ax.scatter([x for x, a in zip(xs, acc) if a], [y for y, a in zip(ys, acc) if a],
                   s=M["stage_dot"], color=color, zorder=3)
        rej = [(x, y) for x, y, a in zip(xs, ys, acc) if not a]
        if rej:
            ax.scatter([p[0] for p in rej], [p[1] for p in rej], s=M["stage_rejected"],
                       facecolors="none", edgecolors=c["red"], zorder=3)
        ax.scatter([xs[0]], [ys[0]], s=M["stage_entry"], marker="s", color=color, zorder=4)
        ax.scatter([xs[-1]], [ys[-1]], s=M["stage_exit"], marker="*", color=color, zorder=4)
        if normalized:  # convergence threshold the per-field gate compares against
            ax.axhline(1.0, color=T["color"], lw=T["line_width"], ls=T["linestyle"], zorder=2)
            lo, hi = min(min(ys), 1.0), max(max(ys), 1.0)
            ax.set_ylim(lo / T["ylim_pad"], hi * T["ylim_pad"])
            ax.set_ylabel(r"$\|R\|/\mathrm{tol}$", fontsize=F["axis_label"])
        else:
            ax.set_ylabel(r"$\|R\|_\infty$", fontsize=F["axis_label"])
        ax.set_xlabel(r"$\mathrm{iter}$", fontsize=F["axis_label"])
    else:
        ax.text(0.5, 0.5, r"$\times$", ha="center", va="center", color=color,
                fontsize=F["cross_marker"], transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(labelsize=F["tick_label"])
    for sp in ax.spines.values():
        sp.set_color(color); sp.set_linewidth(P["spines"]["stage_width"])


def _fill_mini(ax, stage_list):
    """Compact residual mini-plot for one B-cascade (the OSGS per-outer-iteration thumbnails)."""
    M, c, T = P["markers"], P["colors"], P["threshold"]
    color = stage_color(stage_list[-1]) if stage_list else c["grey"]
    xs, ys, _, normalized = _residual_xy(stage_list)
    if ys:
        ax.semilogy(xs, ys, "-o", color=color, ms=M["mini_dot"], lw=M["mini_line_width"])
        ax.scatter([xs[-1]], [ys[-1]], s=M["mini_exit"], marker="*", color=color, zorder=4)
        if normalized:  # convergence threshold (y=1) on each thumbnail
            ax.axhline(1.0, color=T["color"], lw=T["mini_line_width"], ls=T["linestyle"], zorder=1)
            lo, hi = min(min(ys), 1.0), max(max(ys), 1.0)
            ax.set_ylim(lo / T["ylim_pad"], hi * T["ylim_pad"])
    else:
        ax.text(0.5, 0.5, r"$\times$", ha="center", va="center", color=color,
                fontsize=P["fonts"]["mini_cross_marker"], transform=ax.transAxes)
    # Sparkline thumbnails carry no axes labels (the quantitative detail lives in the main stage
    # boxes). set_*ticks([]) only clears MAJOR ticks; on a multi-decade log y-axis matplotlib still
    # draws minor ticks that spill outside the box and overlap the plot — kill those too so every
    # tick mark stays within (here: off) the thumbnail's bounding box.
    ax.set_xticks([]); ax.set_yticks([])
    ax.minorticks_off()
    ax.tick_params(which="both", length=0, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_color(color); sp.set_linewidth(P["spines"]["mini_width"])


class _VStack:
    """Top-down vertical flow layout. Elements reserve heights in INCHES; the figure height is the
    sum of all reserved heights + margins, computed before drawing — so nothing overlaps/overflows."""

    def __init__(self, fig_w, x_center=0.5):
        L = P["layout"]
        self.fig_w = fig_w
        self.xc = x_center
        self.top = L["vstack_top_margin_in"]
        self.bottom = L["vstack_bottom_margin_in"]
        self.dgap = L["vstack_default_gap_in"]
        self.items = []

    def _g(self, gap):
        return self.dgap if gap is None else gap

    def text(self, s, fontsize, color="#111", gap=None):
        if not s:
            return
        n = s.count("\n") + 1
        h = fontsize * P["linespacing"] * n / 72.0 + P["layout"]["text_pad_in"]
        self.items.append(("text", {"s": s, "fontsize": fontsize, "color": color, "h": h, "gap": self._g(gap)}))

    def arrow(self, gap=None):
        self.items.append(("arrow", {"h": P["layout"]["arrow_length_in"], "gap": self._g(gap)}))

    def box(self, fill, height, width, xlabel=True, gap=None):
        pad = P["layout"]["box_xlabel_pad_in"] if xlabel else P["layout"]["box_plain_pad_in"]
        self.items.append(("box", {"fill": fill, "box_h": height, "width": width,
                                    "h": height + pad, "gap": self._g(gap)}))

    def custom(self, height, draw, gap=None):
        self.items.append(("custom", {"draw": draw, "h": height, "gap": self._g(gap)}))

    def render(self, out_path):
        total = self.top + self.bottom + sum(it["h"] + it["gap"] for _, it in self.items)
        fig = plt.figure(figsize=(self.fig_w, total))
        y = total - self.top
        for kind, it in self.items:
            y -= it["gap"]
            if kind == "text":
                fig.text(self.xc, y / total, it["s"], ha="center", va="top",
                         fontsize=it["fontsize"], color=it["color"], linespacing=P["linespacing"])
                y -= it["h"]
            elif kind == "arrow":
                _arrow(fig, self.xc, y / total, self.xc, (y - it["h"]) / total)
                y -= it["h"]
            elif kind == "box":
                ax = fig.add_axes([self.xc - it["width"] / 2.0, (y - it["box_h"]) / total,
                                   it["width"], it["box_h"] / total])
                it["fill"](ax)
                y -= it["h"]
            elif kind == "custom":
                it["draw"](fig, total, y, y - it["h"])
                y -= it["h"]
        fig.savefig(out_path, dpi=P["dpi"], bbox_inches="tight")
        plt.close(fig)
        return out_path


def plot_stages(stages, out_path, title="", subtitle="", subtitle_color=None, start_residual=None):
    """Flat render: a top-to-bottom column of algorithm-stage boxes. Used when no outer loop ran."""
    stages = stages or []
    L, F, c = P["layout"], P["fonts"], P["colors"]
    subtitle_color = subtitle_color or c["subtitle_default"]
    vs = _VStack(L["flat_fig_width_in"])
    if title:
        vs.text(title, F["title"], c["title"], gap=L["gap_title_in"])
    if subtitle:
        vs.text(subtitle, F["subtitle"], subtitle_color, gap=L["gap_subtitle_in"])
    vs.text(_start_node_text(stages, override=start_residual), F["start_node"], c["start_node"],
            gap=L["gap_start_node_in"])
    for stage in stages:
        col = stage_color(stage)
        context, step = _parse_stage(stage.get("stage", "?"))
        sr = stage.get("stop_reason", stage.get("state", ""))
        vs.arrow()
        vs.text(context + "\n" + step + "\n[" + sr + "]", F["stage_label"], col, gap=L["gap_stage_label_in"])
        vs.box(lambda ax, s=stage: _fill_stage(ax, [s]), height=L["flat_box_height_in"], width=L["flat_box_width_frac"])
    if stages:
        last = stages[-1]
        ok = last.get("stop_reason", "") in CONVERGED_REASONS
        vs.text(_final_node_text(last, ok), F["end_node"], c["green"] if ok else c["red"],
                gap=L["gap_end_node_in"])
    return vs.render(out_path)


def _osgs_thumb_grid(K):
    cols = min(K, P["container"]["thumb_max_cols"]) or 1
    return cols, math.ceil(K / cols)


def _osgs_container_height(K):
    """Inch height the Algorithm-C container reserves: label + drift plot + gap + note + a thumbnail
    GRID of ceil(K/cols) rows. Computed so the VStack band fits the drawer's internal sections."""
    C = P["container"]
    _, rows = _osgs_thumb_grid(K)
    return (C["top_pad_in"] + C["label_height_in"] + C["drift_height_in"] + C["drift_to_thumb_gap_in"]
            + C["note_height_in"] + C["thumb_grid_top_gap_in"]
            + rows * C["thumb_cell_height_in"] + (rows - 1) * C["thumb_row_gap_in"] + C["bottom_pad_in"])


def _osgs_container_drawer(stages, osgs_outer, bck, mms_relchange, K):
    """draw(fig, total, y_top_in, y_bot_in) for the Algorithm-C container: a top drift panel, a
    clear gap, then a multi-row thumbnail GRID of the inner B-cascades. Laid out top-down in inches
    inside the reserved band (matching _osgs_container_height)."""
    by_k = {}
    for s in stages:
        m = re.search(r"C:OSGS\[(\d+)\]", s.get("stage", ""))
        if m:
            by_k.setdefault(int(m.group(1)), []).append(s)
    C, F, M, c = P["container"], P["fonts"], P["markers"], P["colors"]
    cols, rows = _osgs_thumb_grid(K)

    def draw(fig, total, y_top_in, y_bot_in):
        cx, cw = C["x_frac"], C["width_frac"]
        fig.add_artist(Rectangle((cx, y_bot_in / total), cw, (y_top_in - y_bot_in) / total,
                                 transform=fig.transFigure, fill=False, edgecolor=c["blue"],
                                 linewidth=C["edge_width"], zorder=0))
        cur = y_top_in - C["top_pad_in"]
        fig.text(cx + C["label_inset_frac"], cur / total,
                 f"Algorithm C - OSGS staggered loop  ({K} outer iterations; each runs Algorithm B -> A)",
                 ha="left", va="top", fontsize=F["container_label"], color=c["blue"], fontweight="bold")
        cur -= C["label_height_in"]

        # --- (i) outer-loop drift panel ---
        dx = cx + C["drift_left_pad_frac"]
        dw = cw - C["drift_left_pad_frac"] - C["drift_right_pad_frac"]
        dax = fig.add_axes([dx, (cur - C["drift_height_in"]) / total, dw, C["drift_height_in"] / total])
        kk = list(range(1, K + 1))
        has_drift = any(isinstance(o.get("x_diff_resolved"), (int, float))
                        and math.isfinite(o.get("x_diff_resolved")) for o in osgs_outer)
        if has_drift:
            dm = P["drift_markers"]
            for lbl, key, col, mk in (("state drift", "x_diff_resolved", c["red"], dm["state"]),
                                      (r"$\pi_u$ drift", "pi_u_drift", c["blue"], dm["pi_u"]),
                                      (r"$\pi_p$ drift", "pi_p_drift", c["green"], dm["pi_p"])):
                pts = [(k, osgs_outer[i].get(key)) for i, k in enumerate(kk) if i < len(osgs_outer)]
                pts = [(k, v) for k, v in pts if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
                if pts:
                    dax.semilogy([p[0] for p in pts], [p[1] for p in pts], "-" + mk, color=col,
                                 ms=M["drift_marker"], lw=M["drift_line_width"], label=lbl)
            extra = []   # plateau region + base-convergence marker shown in the LEGEND, not in-plot
            if bck and bck < K:
                dax.axvspan(bck + 0.5, K + 0.5, color=c["plateau_shade"], alpha=P["plateau_shade_alpha"], zorder=0)
                extra.append(Patch(facecolor=c["plateau_shade"], alpha=P["plateau_shade_alpha"],
                                   label="MMS plateau verification"))
            if bck:
                dax.axvline(bck + 0.5 if bck < K else K, color=c["amber"], lw=M["drift_line_width"], ls="--")
                extra.append(Line2D([0], [0], color=c["amber"], lw=M["drift_line_width"], ls="--",
                                    label=rf"base converged ($k={bck}$)"))
            dax.set_xlabel(r"OSGS outer iteration $k$", fontsize=F["drift_axis"])
            dax.set_ylabel("drift", fontsize=F["drift_axis"])
            dax.set_xticks(kk); dax.tick_params(labelsize=F["drift_tick"]); dax.set_xlim(0.5, K + 0.5)
            handles, _ = dax.get_legend_handles_labels()
            dax.legend(handles=handles + extra, fontsize=F["drift_legend"], loc="upper right",
                       ncol=P["drift_legend_ncol"], framealpha=0.9)
            dax.set_title("outer-loop staggered convergence", fontsize=F["drift_title"], color=c["blue"])
        else:
            dax.axis("off")
            dax.text(0.5, 0.5, "outer-loop drift not recorded in this trace\n"
                     "(re-run the sweep with the augmented tracing to populate it)",
                     ha="center", va="center", fontsize=F["missing_data_note"], color=c["note"],
                     style="italic", transform=dax.transAxes)
        cur -= C["drift_height_in"]
        cur -= C["drift_to_thumb_gap_in"]      # clear gap between the drift panel and the grid

        # --- (ii) per-outer-iteration inner-cascade thumbnail GRID (multi-row) ---
        fig.text(cx + C["label_inset_frac"], cur / total,
                 "inner Algorithm-B cascades (one per outer iteration):",
                 ha="left", va="top", fontsize=F["container_note"], color=c["muted"])
        cur -= C["note_height_in"]
        cur -= C["thumb_grid_top_gap_in"]
        grid_top = cur
        usable_w = cw - C["thumb_left_pad_frac"] - C["thumb_right_pad_frac"]
        col_gap = C["thumb_col_gap_frac"]
        tw = max((usable_w - (cols - 1) * col_gap) / cols, 0.02)
        cell_h = C["thumb_cell_height_in"]
        row_pitch = cell_h + C["thumb_row_gap_in"]
        for i in range(K):
            k = i + 1
            row, col = divmod(i, cols)
            x = cx + C["thumb_left_pad_frac"] + col * (tw + col_gap)
            cell_top_in = grid_top - row * row_pitch
            cell_bot_in = cell_top_in - cell_h
            is_plateau = bool(bck and k > bck)
            if is_plateau:
                mfr, min_ = C["plateau_margin_frac"], C["plateau_margin_in"]
                fig.add_artist(Rectangle((x - mfr, (cell_bot_in - min_) / total), tw + 2 * mfr,
                                         (cell_h + 2 * min_) / total, transform=fig.transFigure, fill=True,
                                         facecolor=c["plateau_shade"], alpha=P["plateau_shade_alpha"] - 0.05,
                                         edgecolor="none", zorder=0))
            ax = fig.add_axes([x, cell_bot_in / total, tw, cell_h / total])
            _fill_mini(ax, by_k.get(k, []))
            fig.text(x + tw / 2, (cell_top_in + C["thumb_label_pad_in"]) / total, f"$k={k}$",
                     ha="center", va="bottom", fontsize=F["thumb_label"],
                     color=(c["amber"] if is_plateau else c["blue"]))
            if is_plateau and mms_relchange:
                j = k - (bck + 1)
                if 0 <= j < len(mms_relchange):
                    rc = mms_relchange[j]
                    mr = max(rc) if isinstance(rc, (list, tuple)) else rc
                    fig.text(x + tw / 2, (cell_bot_in - C["thumb_ratio_pad_in"]) / total, rf"$r={sci(mr)}$",
                             ha="center", va="top", fontsize=F["mms_ratio"], color=c["amber"])

    return draw


def _plot_osgs(stages, osgs_outer, base_conv_k, mms_relchange, out_path, title, subtitle, subtitle_color):
    """Nested OSGS render: Stage-I box, then an Algorithm-C container (drift + thumbnails)."""
    L, F, c = P["layout"], P["fonts"], P["colors"]
    subtitle_color = subtitle_color or c["subtitle_default"]
    stage_i = [s for s in stages if s.get("stage", "").startswith("B:StageI")]
    ks = sorted({int(m.group(1)) for s in stages
                 for m in [re.search(r"C:OSGS\[(\d+)\]", s.get("stage", ""))] if m})
    K = max(len(osgs_outer), len(ks), 1)
    bck = base_conv_k if (isinstance(base_conv_k, int) and base_conv_k > 0) else None

    cols, _rows = _osgs_thumb_grid(K)
    fig_w = max(L["osgs_fig_width_min_in"], L["osgs_fig_width_base_in"] + cols * L["osgs_fig_width_per_col_in"])
    vs = _VStack(fig_w)
    if title:
        vs.text(title, F["title"], c["title"], gap=L["gap_title_in"])
    if subtitle:
        vs.text(subtitle, F["subtitle"], subtitle_color, gap=L["gap_subtitle_in"])
    vs.text(_start_node_text(stage_i if stage_i else stages), F["start_node"], c["start_node"],
            gap=L["gap_start_node_in"])

    si_color = stage_color(stage_i[-1]) if stage_i else c["grey"]
    si_stops = " / ".join(s.get("stop_reason", s.get("state", "")) for s in stage_i) or "?"
    vs.arrow()
    vs.text("Algorithm B - Stage I (ASGS init)\n[" + si_stops + "]", F["stage_label"], si_color,
            gap=L["gap_stage_label_in"])
    vs.box(lambda ax: _fill_stage(ax, stage_i), height=L["stage_i_box_height_in"], width=L["stage_i_box_width_frac"])

    vs.arrow()
    vs.custom(_osgs_container_height(K), _osgs_container_drawer(stages, osgs_outer, bck, mms_relchange, K),
              gap=L["gap_container_in"])

    osgs_stages = [(int(m.group(1)), s) for s in stages
                   for m in [re.search(r"C:OSGS\[(\d+)\]", s.get("stage", ""))] if m]
    last = (max(osgs_stages, default=(0, stages[-1] if stages else {}))[1]) if stages else {}
    end_ok = last.get("stop_reason", "") in CONVERGED_REASONS
    vs.text(_final_node_text(last, end_ok), F["end_node"], c["green"] if end_ok else c["red"],
            gap=L["gap_osgs_end_node_in"])
    return vs.render(out_path)


def plot_attempt(stages, out_path, title="", subtitle="", subtitle_color=None,
                 osgs_outer=None, base_conv_k=None, mms_relchange=None):
    """Dispatch: a STAGGERED OSGS run (outer loop ran) -> nested C->B->A diagram; everything else
    (ASGS, or coupled OSGS = Stage I + a single coupled Newton solve with no staggered outer loop)
    -> flat stage list. Coupled mode is detected by the absence of any staggered structure: no
    populated osgs_outer AND no indexed C:OSGS[k] stage. Its trajectory lives in the C:OSGS:Coupled
    stage history and renders as an ordinary flat stage box."""
    stages = stages or []
    osgs_outer = osgs_outer or []
    has_staggered = bool(osgs_outer) or any(re.search(r"C:OSGS\[(\d+)\]", s.get("stage", "")) for s in stages)
    if has_staggered:
        return _plot_osgs(stages, osgs_outer, base_conv_k, mms_relchange or [],
                          out_path, title, subtitle, subtitle_color)
    return plot_stages(stages, out_path, title=title, subtitle=subtitle, subtitle_color=subtitle_color)


if __name__ == "__main__":
    print(__doc__)
    print("This is a library. Call trajectory_plot.plot_attempt(...) from a per-test wrapper "
          "(see tools/trajectory_viz/README.md and test/extended/ManufacturedSolutions/plot_trajectory.py).")
