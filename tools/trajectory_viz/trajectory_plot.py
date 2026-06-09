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

Nomenclature (theory/osgs_algorithm/osgs_algorithm.tex): Alg. O orchestrator; Alg. B RobustNonlinearCascade
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


# Scale-free convergence thresholds the gate decides on (ε_M ≤ tol_M AND ε_C ≤ tol_C). A per-trace
# value (emitted by run_test.jl into the trace top-level) is preferred; set_tols installs it. When a
# trace predates that emission, _tol falls back to the plot_params default (mirrors the solver config
# eps_tol_momentum / eps_tol_mass). Module-global so _fill_stage/_fill_mini need no extra threading,
# matching the P-global pattern; the wrapper calls set_tols() once per trace before rendering.
_TOLS = {"tol_M": None, "tol_C": None}


def set_tols(tol_M=None, tol_C=None):
    """Install the per-trace scale-free thresholds (None -> fall back to the plot_params default)."""
    _TOLS["tol_M"], _TOLS["tol_C"] = tol_M, tol_C


def _tol(which):
    """Resolve a threshold ('tol_M' or 'tol_C'): the per-trace value if finite-positive, else the
    plot_params fallback."""
    v = _TOLS.get(which)
    return v if _finite_pos(v) else P["threshold"][which + "_default"]


def _bold(s):
    r"""Bold-mathtext a plain phrase so a stage label can emphasise just the solver-role words
    (e.g. "Newton (Alg. A)") while the surrounding context and suffix stay regular weight. Spaces
    are escaped to ``\ `` so mathtext keeps them; the cm fontset renders ``\mathbf`` as bold — the
    same math typeface already used for the titles/axis labels elsewhere in the figure."""
    return r"$\mathbf{" + s.replace(" ", r"\ ") + r"}$"


# Solver-role names shown on each stage box. Newton AND Picard are two MODES of ONE kernel,
# ExactNewtonPipeline = Algorithm A (theory/osgs_algorithm.tex, the Alg.↔code table): Newton mode
# uses the exact Jacobian + merit-Armijo, Picard mode a frozen Jacobian + monotone residual descent.
# So every role carries "(Alg. A)" — the Picard smoother is Alg. A in Picard mode, not a separate
# algorithm. The names are bold so they stand out from the (regular-weight) context and stop-reason.
_STEP_NAME = {"N1": _bold("Newton-1 (Alg. A)"), "N2": _bold("Newton-2 (Alg. A)"),
              "Picard": _bold("Picard smoother (Alg. A)")}


def stage_color(stage):
    state = stage.get("state", "ok")
    if state != "ok":
        return _STOP_COLOR.get(state, P["colors"]["grey"])
    return _STOP_COLOR.get(stage.get("stop_reason", ""), P["colors"]["grey"])


def _parse_stage(code):
    """(algorithm-context, step-name) from a stage code, using documentation nomenclature.

    Recognized stage-code forms:
      B:StageI:N1 | B:StageI:Picard | B:StageI:N2     one-way Stage-I cascade
      B:StageI:PP[k]:N | B:StageI:PP[k]:P             Stage-I Newton<->Picard ping-pong, swap k
      C:OSGS:Coupled                                  coupled OSGS solve (no ping-pong fallback)
      C:OSGS:PP[k]:N | C:OSGS:PP[k]:P                 coupled OSGS solve with the ping-pong fallback, swap k
      C:OSGS[k]:N1 | ...                              legacy staggered OSGS, outer iteration k
    In the ping-pong forms the swap index lives in the THIRD field (PP[k]) and the Newton/Picard role
    in the FOURTH (N/P); in every other form the step is the third field itself.
    """
    parts = code.split(":")
    algo = parts[0]
    mid = parts[1] if len(parts) > 1 else ""
    third = parts[2] if len(parts) > 2 else mid
    fourth = parts[3] if len(parts) > 3 else ""
    pp_swap = third[third.find("[") + 1:third.find("]")] if third.startswith("PP[") and "]" in third else None
    step = (fourth or third) if pp_swap is not None else third

    if algo == "B" and mid == "StageI":
        context = "Alg. B - Stage I (ASGS init)"
    elif algo == "C" and mid.startswith("OSGS") and (third == "Coupled" or pp_swap is not None):
        # The coupled OSGS solve is ONE Newton solve (pi recomputed each iter); the PP[k] segments are
        # its Newton<->Picard ping-pong fallback, NOT a staggered outer loop.
        context = "Alg. C - OSGS coupled (single Newton; pi recomputed each iter)"
    elif algo == "C" and mid.startswith("OSGS["):
        idx = mid[mid.find("[") + 1:mid.find("]")]
        context = "Alg. C - OSGS outer it. {} (-> Alg. B)".format(idx)
    else:
        context = "{} - {}".format(algo, mid)

    if pp_swap is not None:
        role = {"N": _bold("Newton (Alg. A)"), "P": _bold("Picard smoother (Alg. A)")}.get(step, step)
        return context, "{} - ping-pong swap {}".format(role, pp_swap)
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


def _rel_drift_series(osgs_outer, kk, drift_key, mag_key):
    """Relative projection drift per outer iteration: ‖Δπ_k‖ / ½(‖π_k‖+‖π_{k-1}‖) — the scale-free
    fraction of its own L²-size the projection still moves. The loop seeds π_0 ≡ 0, so the first
    point is exactly 2 (the symmetric-relative-difference ceiling). Frozen iterations (Δπ=0, the
    freeze_after_k finish) are KEPT as exact-0 points (drawn on a symlog axis), not dropped — they
    mark where π stops updating. Returns [(k, rel), ...], skipping only entries whose magnitude is
    missing/non-finite (legacy traces)."""
    out, prev_mag = [], 0.0
    for i, k in enumerate(kk):
        if i >= len(osgs_outer):
            break
        d, m = osgs_outer[i].get(drift_key), osgs_outer[i].get(mag_key)
        if isinstance(m, (int, float)) and math.isfinite(m):
            denom = 0.5 * (m + prev_mag)
            if isinstance(d, (int, float)) and math.isfinite(d) and d >= 0 and denom > 0:
                out.append((k, d / denom))
            prev_mag = m
    return out


def _freeze_onset_k(osgs_outer, kk):
    """Outer iteration at which the projection is first frozen (Δπ_u == Δπ_p == 0 exactly — the
    freeze_after_k quadratic finish writes literal 0.0 drifts), or None for a staggered run that
    never freezes π. A non-frozen iteration never lands an exact 0.0, so the equality test is safe."""
    for i, k in enumerate(kk):
        if i >= len(osgs_outer):
            break
        o = osgs_outer[i]
        if o.get("pi_u_drift") == 0 and o.get("pi_p_drift") == 0:
            return k
    return None


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


def _eps_xy(stage_list, key):
    """Per-iteration series of a scale-free convergence norm — key 'eps_M' (momentum) or 'eps_C' (mass) —
    concatenated across a cascade's stages onto the inner-iteration counter. These are present only when
    the run was traced with the convergence probe (src/solvers/convergence_criterion.jl); entries with a
    missing/non-finite value are skipped, so probe-off and pre-change traces yield an empty series.
    Returns (xs, ys)."""
    xs, ys, gi = [], [], 0
    for s in (stage_list or []):
        for h in (s.get("history") or []):
            gi += 1
            v = h.get(key)
            if _finite_pos(v):
                xs.append(gi)
                ys.append(v)
    return xs, ys


def _legend_handles(stages):
    """Build the figure's ONE shared key (replacing per-box legends). Entries are emitted only for
    quantities this trajectory actually carries. When the run was probe-traced (eps_M/eps_C present)
    the key leads with the PRIMARY ε_M/ε_C curves and the scale-free gate lines tol_M/tol_C (what the
    solver decides convergence on), then labels the algebraic residual ‖R‖/ftol as the de-emphasised
    right-axis context (box colour = outcome). 'ftol' (the linear-solve floor) is named distinctly from
    the gate 'tol_M'/'tol_C' on purpose — they are different tolerances. Otherwise only the residual
    key and its y=1 threshold appear (the legacy residual-primary layout)."""
    M, c, T = P["markers"], P["colors"], P["threshold"]
    ms = P["legend"]["marker_size"]
    hist = [h for s in (stages or []) for h in (s.get("history") or [])]
    has_eps_m = any(_finite_pos(h.get("eps_M")) for h in hist)
    has_eps_c = any(_finite_pos(h.get("eps_C")) for h in hist)
    H = []
    if has_eps_m or has_eps_c:
        if has_eps_m:
            H.append(Line2D([], [], color=c["blue"], marker="o", ms=ms, lw=M["drift_line_width"],
                            label=r"$\varepsilon_M$ : momentum residual / force scale (primary)"))
            H.append(Line2D([], [], color=c["blue"], ls=T["linestyle"], lw=T["line_width"] + 0.4,
                            label=rf"$\mathrm{{tol}}_M={sci(_tol('tol_M'))}$ : momentum gate ($\varepsilon_M\leq\mathrm{{tol}}_M$)"))
        if has_eps_c:
            H.append(Line2D([], [], color=c["green"], marker="s", ms=ms, lw=M["drift_line_width"],
                            label=r"$\varepsilon_C$ : mass, $\|\nabla\!\cdot(\alpha u)\|/\|\nabla(\alpha u)\|$"))
            H.append(Line2D([], [], color=c["green"], ls=T["linestyle"], lw=T["line_width"] + 0.4,
                            label=rf"$\mathrm{{tol}}_C={sci(_tol('tol_C'))}$ : mass gate ($\varepsilon_C\leq\mathrm{{tol}}_C$)"))
        H.append(Line2D([], [], color=c["muted"], marker="o", ms=ms, lw=M["mini_line_width"],
                        alpha=T["residual_demph_alpha"],
                        label=r"$\|R\|/\mathrm{ftol}$ : algebraic residual / linear-solve floor (right axis, $\neq$ gate; box colour = outcome)"))
    else:
        H.append(Line2D([], [], color=c["grey"], marker="o", ms=ms, lw=M["stage_line_width"],
                        label=r"$\|R\|/\mathrm{ftol}$ : per-field residual (box colour = outcome)"))
        H.append(Line2D([], [], color=T["color"], ls=T["linestyle"], lw=T["line_width"] + 0.4,
                        label=r"$\|R\|/\mathrm{ftol}=1$ : convergence threshold ($\|R\|=$ effective ftol)"))
    return H


def _start_node_text(stages, override=None):
    """Entry-node label: normalized residual (‖R₀‖/ftol) if the trace carries it, else raw ‖R₀‖."""
    s0 = stages[0] if stages else {}
    if override is None:
        rn = s0.get("res_in_norm")
        if _finite_pos(rn):
            return rf"$x_0:\ \|R_0\|/\mathrm{{ftol}}={sci(rn)}$"
    r = override if override is not None else s0.get("res_in")
    return rf"$x_0:\ \|R_0\|={sci(r)}$"


def _final_node_text(last, ok):
    """Exit-node label: normalized final residual if available, else raw ‖R‖."""
    mark = r"$\checkmark$" if ok else r"$\times$"
    rn = last.get("res_out_norm")
    if _finite_pos(rn):
        return mark + rf" final $\|R\|/\mathrm{{ftol}}={sci(rn)}$"
    return mark + rf" final $\|R\|={sci(last.get('res_out'))}$"


def _fill_stage(ax, stages):
    """Draw a B-cascade's convergence trajectory.

    When the run was probe-traced, the scale-free norms ε_M (momentum) and ε_C (mass) are the PRIMARY
    curves on the left axis — with the actual gate lines tol_M / tol_C — because ε_M ≤ tol_M AND
    ε_C ≤ tol_C is what the solver decides convergence on. The algebraic residual ‖R‖/ftol is kept as
    faint context on a right (twin) axis; its denominator ftol is the linear-solve floor (dynamic_ftol
    in the harness), NOT the gate, so it typically stays well above 1 even at convergence — the
    nomenclature point, made explicit by naming it ftol rather than tol. With no probe data (the
    default, and all pre-change traces) the residual stays primary, exactly as before."""
    stages = stages if isinstance(stages, list) else [stages]
    M, F, c, T = P["markers"], P["fonts"], P["colors"], P["threshold"]
    color = stage_color(stages[-1]) if stages else c["grey"]
    xs, ys, acc, normalized = _residual_xy(stages)
    em_x, em_y = _eps_xy(stages, "eps_M")
    ec_x, ec_y = _eps_xy(stages, "eps_C")

    if em_y or ec_y:
        # ----- ε PRIMARY (left axis): the scale-free gate the solver decides on -----
        prim_x, prim_y = (em_x, em_y) if em_y else (ec_x, ec_y)
        if em_y:
            ax.semilogy(em_x, em_y, "-o", color=c["blue"], ms=M["mini_dot"],
                        lw=M["drift_line_width"], zorder=3)
            ax.axhline(_tol("tol_M"), color=c["blue"], lw=T["line_width"], ls=T["linestyle"], zorder=2)
        if ec_y:
            ax.semilogy(ec_x, ec_y, "-s", color=c["green"], ms=M["mini_dot"],
                        lw=M["drift_line_width"], zorder=3)
            ax.axhline(_tol("tol_C"), color=c["green"], lw=T["line_width"], ls=T["linestyle"], zorder=2)
        # entry square + exit star on the primary ε curve, in STAGE colour (ties the curve to its stage)
        ax.scatter([prim_x[0]], [prim_y[0]], s=M["stage_entry"], marker="s", color=color, zorder=4)
        ax.scatter([prim_x[-1]], [prim_y[-1]], s=M["stage_exit"], marker="*", color=color, zorder=4)
        # y-range: include both ε curves AND both gate lines, with a minimum decade span so a
        # barely-moving ε axis still reads on clean ticks (mirrors the legacy ε-overlay logic).
        yall = [v for v in (em_y + ec_y) if v > 0] + [_tol("tol_M"), _tol("tol_C")]
        lo, hi = min(yall), max(yall)
        if math.log10(hi / lo) < T["eps_min_decades"]:
            ctr, half = math.sqrt(lo * hi), 10 ** (T["eps_min_decades"] / 2.0)
            lo, hi = ctr / half, ctr * half
        ax.set_ylim(lo / T["ylim_pad"], hi * T["ylim_pad"])
        ax.set_ylabel(r"$\varepsilon_M,\ \varepsilon_C$", fontsize=F["axis_label"], labelpad=F["axis_label_pad"])
        ax.set_xlabel(r"$\mathrm{iter}$", fontsize=F["axis_label"], labelpad=F["axis_label_pad"])
        ax.tick_params(labelsize=F["tick_label"])
        # algebraic residual kept as faint right-axis context. Its denominator is ftol (the linear-solve
        # floor = dynamic_ftol in the harness), NOT the convergence gate — labelled ‖R‖/ftol to keep it
        # distinct from the left-axis tol_M / tol_C. Because the ε-gate fires first, this curve typically
        # stays well above 1 even at convergence (that is the nomenclature point, made explicit).
        if ys:
            dem = T["residual_demph_alpha"]
            rax = ax.twinx()
            rax.semilogy(xs, ys, "-o", color=c["muted"], ms=M["mini_dot"],
                         lw=M["mini_line_width"], alpha=dem, zorder=1)
            if normalized:
                rax.axhline(1.0, color=c["muted"], lw=T["mini_line_width"], ls=T["linestyle"],
                            alpha=dem, zorder=1)
                lo2, hi2 = min(min(ys), 1.0), max(max(ys), 1.0)
                rax.set_ylim(lo2 / T["ylim_pad"], hi2 * T["ylim_pad"])
            rax.set_ylabel(r"$\|R\|/\mathrm{ftol}$", fontsize=F["axis_label"],
                           labelpad=F["axis_label_pad"], color=c["muted"])
            rax.tick_params(labelsize=F["tick_label"], colors=c["muted"])
    elif ys:
        # ----- RESIDUAL PRIMARY (no probe data): unchanged legacy behaviour -----
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
            ax.set_ylabel(r"$\|R\|/\mathrm{ftol}$", fontsize=F["axis_label"], labelpad=F["axis_label_pad"])
        else:
            ax.set_ylabel(r"$\|R\|_\infty$", fontsize=F["axis_label"], labelpad=F["axis_label_pad"])
        ax.set_xlabel(r"$\mathrm{iter}$", fontsize=F["axis_label"], labelpad=F["axis_label_pad"])
        ax.tick_params(labelsize=F["tick_label"])
    else:
        ax.text(0.5, 0.5, r"$\times$", ha="center", va="center", color=color,
                fontsize=F["cross_marker"], transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(color); sp.set_linewidth(P["spines"]["stage_width"])


def _fill_mini(ax, stage_list):
    """Compact convergence mini-plot for one B-cascade (the OSGS per-outer-iteration thumbnails).
    Foregrounds ε_M/ε_C vs the gate lines tol_M/tol_C when probe-traced; falls back to the residual
    sparkline otherwise. No twin axis here — the tiny cells stay legible and the residual detail lives
    in the main Stage-I box."""
    M, c, T = P["markers"], P["colors"], P["threshold"]
    color = stage_color(stage_list[-1]) if stage_list else c["grey"]
    em_x, em_y = _eps_xy(stage_list, "eps_M")
    ec_x, ec_y = _eps_xy(stage_list, "eps_C")
    if em_y or ec_y:
        prim_x, prim_y = (em_x, em_y) if em_y else (ec_x, ec_y)
        if em_y:
            ax.semilogy(em_x, em_y, "-o", color=c["blue"], ms=M["mini_dot"], lw=M["mini_line_width"])
            ax.axhline(_tol("tol_M"), color=c["blue"], lw=T["mini_line_width"], ls=T["linestyle"], zorder=1)
        if ec_y:
            ax.semilogy(ec_x, ec_y, "-s", color=c["green"], ms=M["mini_dot"], lw=M["mini_line_width"])
            ax.axhline(_tol("tol_C"), color=c["green"], lw=T["mini_line_width"], ls=T["linestyle"], zorder=1)
        ax.scatter([prim_x[-1]], [prim_y[-1]], s=M["mini_exit"], marker="*", color=color, zorder=4)
        yall = [v for v in (em_y + ec_y) if v > 0] + [_tol("tol_M"), _tol("tol_C")]
        lo, hi = min(yall), max(yall)
        if math.log10(hi / lo) < T["eps_min_decades"]:
            ctr, half = math.sqrt(lo * hi), 10 ** (T["eps_min_decades"] / 2.0)
            lo, hi = ctr / half, ctr * half
        ax.set_ylim(lo / T["ylim_pad"], hi * T["ylim_pad"])
    else:
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

    def legend(self, handles, gap=None):
        """Reserve a band for the shared key. Height auto-sizes to the number of rows the entries
        wrap to at the configured column count, so probe-off traces (fewer entries) don't reserve
        the whitespace of a full key."""
        Lg = P["legend"]
        rows = math.ceil(len(handles) / Lg["ncol"]) if handles else 0
        h = rows * P["fonts"]["legend"] * P["linespacing"] / 72.0 + Lg["pad_in"]
        self.items.append(("legend", {"handles": handles, "h": h, "gap": self._g(gap)}))

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
            elif kind == "legend":
                Lg = P["legend"]
                lax = fig.add_axes([self.xc - Lg["width_frac"] / 2.0, (y - it["h"]) / total,
                                    Lg["width_frac"], it["h"] / total])
                lax.axis("off")
                lax.legend(handles=it["handles"], loc="center", ncol=Lg["ncol"],
                           fontsize=P["fonts"]["legend"], frameon=True, framealpha=0.92,
                           handlelength=Lg["handlelength"], columnspacing=Lg["columnspacing"],
                           labelspacing=Lg["labelspacing"], borderpad=Lg["borderpad"])
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


def plot_stages(stages, out_path, title="", subtitle="", subtitle_color=None, start_residual=None,
                success=None):
    """Flat render: a top-to-bottom column of algorithm-stage boxes. Used when no outer loop ran.

    `success`, when given, is the attempt's actual converged/failed verdict and decides the end-node
    mark; otherwise the mark falls back to whether the last stage's stop-reason is a converged one (a
    weaker heuristic that can misread a trailing 0-iteration `initial_ftol` Picard segment as success)."""
    stages = stages or []
    L, F, c = P["layout"], P["fonts"], P["colors"]
    subtitle_color = subtitle_color or c["subtitle_default"]
    vs = _VStack(L["flat_fig_width_in"])
    if title:
        vs.text(title, F["title"], c["title"], gap=L["gap_title_in"])
    if subtitle:
        vs.text(subtitle, F["subtitle"], subtitle_color, gap=L["gap_subtitle_in"])
    vs.legend(_legend_handles(stages), gap=L["gap_legend_in"])
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
        ok = success if success is not None else (last.get("stop_reason", "") in CONVERGED_REASONS)
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
        # The projection drift is reported RELATIVE to the projection's own size,
        # ‖Δπ_k‖ / ½(‖π_k‖+‖π_{k-1}‖): a scale-free "fraction of itself the projection still moves
        # per outer iteration" (π_0 ≡ 0 ⇒ the first point is exactly 2). Absolute drift is misleading —
        # a small-magnitude π_p whose increments are tiny in absolute terms can still be moving ~100%
        # of its (shrinking) size each iteration. Needs the per-iteration magnitudes ‖π‖ (pi_u_l2 /
        # pi_p_l2) from the enriched trace; legacy traces lacking them fall back to absolute drift.
        have_mag = any(_finite_pos(o.get("pi_u_l2")) or _finite_pos(o.get("pi_p_l2")) for o in osgs_outer)
        sax = None
        if has_drift and have_mag:
            dm = P["drift_markers"]
            proj_pos, proj_has_zero = [], False
            for lbl, dkey, mkey, col, mk in (
                    (r"$\pi_u$ rel. drift", "pi_u_drift", "pi_u_l2", c["blue"], dm["pi_u"]),
                    (r"$\pi_p$ rel. drift", "pi_p_drift", "pi_p_l2", c["green"], dm["pi_p"])):
                pts = _rel_drift_series(osgs_outer, kk, dkey, mkey)
                if pts:
                    # plot()+explicit yscale (not semilogy) so the frozen Δπ=0 points survive.
                    dax.plot([p[0] for p in pts], [p[1] for p in pts], "-" + mk, color=col,
                             ms=M["drift_marker"], lw=M["drift_line_width"], label=lbl)
                    proj_pos += [v for _, v in pts if v > 0]
                    proj_has_zero = proj_has_zero or any(v == 0 for _, v in pts)
            # symlog (linear through 0, knee at the smallest real drift) iff a frozen Δπ=0 point is
            # present; otherwise plain log. Lets the exact-0 frozen points render at the axis floor.
            if proj_has_zero and proj_pos:
                dax.set_yscale("symlog", linthresh=min(proj_pos))
            else:
                dax.set_yscale("log")
            dax.set_ylabel(r"rel. proj. drift  $\|\Delta\pi_k\|/\langle\|\pi\|\rangle$", fontsize=F["drift_axis"])
            # State drift stays ABSOLUTE on a twin right axis (no ‖U_h‖ recorded to relativise against).
            sax = dax.twinx()
            spts = [(k, osgs_outer[i].get("x_diff_resolved")) for i, k in enumerate(kk) if i < len(osgs_outer)]
            spts = [(k, v) for k, v in spts if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
            if spts:
                sax.semilogy([p[0] for p in spts], [p[1] for p in spts], "-" + dm["state"], color=c["red"],
                             ms=M["drift_marker"], lw=M["drift_line_width"], label="state drift (abs)")
            sax.set_ylabel(r"state drift $\|\Delta U_k\|$", fontsize=F["drift_axis"], color=c["red"])
            sax.tick_params(axis="y", labelsize=F["drift_tick"], colors=c["red"])
            title = r"outer-loop staggered convergence (relative $\pi$ drift)"
        elif has_drift:
            dm = P["drift_markers"]
            for lbl, key, col, mk in (("state drift", "x_diff_resolved", c["red"], dm["state"]),
                                      (r"$\pi_u$ drift", "pi_u_drift", c["blue"], dm["pi_u"]),
                                      (r"$\pi_p$ drift", "pi_p_drift", c["green"], dm["pi_p"])):
                pts = [(k, osgs_outer[i].get(key)) for i, k in enumerate(kk) if i < len(osgs_outer)]
                pts = [(k, v) for k, v in pts if isinstance(v, (int, float)) and math.isfinite(v) and v > 0]
                if pts:
                    dax.semilogy([p[0] for p in pts], [p[1] for p in pts], "-" + mk, color=col,
                                 ms=M["drift_marker"], lw=M["drift_line_width"], label=lbl)
            dax.set_ylabel("drift", fontsize=F["drift_axis"])
            title = "outer-loop staggered convergence"
        if has_drift:
            extra = []   # plateau region + base-convergence marker shown in the LEGEND, not in-plot
            if bck and bck < K:
                dax.axvspan(bck + 0.5, K + 0.5, color=c["plateau_shade"], alpha=P["plateau_shade_alpha"], zorder=0)
                extra.append(Patch(facecolor=c["plateau_shade"], alpha=P["plateau_shade_alpha"],
                                   label="MMS plateau verification"))
            if bck:
                dax.axvline(bck + 0.5 if bck < K else K, color=c["amber"], lw=M["drift_line_width"], ls="--")
                extra.append(Line2D([0], [0], color=c["amber"], lw=M["drift_line_width"], ls="--",
                                    label=rf"base converged ($k={bck}$)"))
            fz = _freeze_onset_k(osgs_outer, kk)
            if fz is not None:
                dax.axvline(fz - 0.5, color=c["grey"], lw=M["drift_line_width"], ls=":")
                extra.append(Line2D([0], [0], color=c["grey"], lw=M["drift_line_width"], ls=":",
                                    label=rf"$\pi$ frozen ($k\geq{fz}$)"))
            dax.set_xlabel(r"OSGS outer iteration $k$", fontsize=F["drift_axis"])
            dax.set_xticks(kk); dax.tick_params(labelsize=F["drift_tick"]); dax.set_xlim(0.5, K + 0.5)
            handles, _ = dax.get_legend_handles_labels()
            if sax is not None:
                handles += sax.get_legend_handles_labels()[0]
            dax.legend(handles=handles + extra, fontsize=F["drift_legend"], loc="upper right",
                       ncol=P["drift_legend_ncol"], framealpha=0.9)
            dax.set_title(title, fontsize=F["drift_title"], color=c["blue"])
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


def _plot_osgs(stages, osgs_outer, base_conv_k, mms_relchange, out_path, title, subtitle, subtitle_color,
               success=None):
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
    vs.legend(_legend_handles(stages), gap=L["gap_legend_in"])
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
    end_ok = success if success is not None else (last.get("stop_reason", "") in CONVERGED_REASONS)
    vs.text(_final_node_text(last, end_ok), F["end_node"], c["green"] if end_ok else c["red"],
            gap=L["gap_osgs_end_node_in"])
    return vs.render(out_path)


def plot_attempt(stages, out_path, title="", subtitle="", subtitle_color=None,
                 osgs_outer=None, base_conv_k=None, mms_relchange=None, success=None):
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
                          out_path, title, subtitle, subtitle_color, success=success)
    return plot_stages(stages, out_path, title=title, subtitle=subtitle, subtitle_color=subtitle_color,
                       success=success)


if __name__ == "__main__":
    print(__doc__)
    print("This is a library. Call trajectory_plot.plot_attempt(...) from a per-test wrapper "
          "(see tools/trajectory_viz/README.md and test/extended/ManufacturedSolutions/plot_trajectory.py).")
