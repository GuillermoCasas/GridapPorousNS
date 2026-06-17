#!/usr/bin/env python3
"""
make_results_tables.py — generate the paper's 2D manufactured-solution convergence tables,
filled with the LATEST Gridap simulation results, as a standalone compilable LaTeX file.

Reproduces the four tables of article.tex in their exact format (booktabs, velocity+pressure
sub-blocks, ASGS/OSGS columns, `\\num{}` values, "slope (N)" theoretical-rate annotations):

    tab:Linear2DL2     P1 (TRI k=1)   L2-norm     slope (2 / 1)
    tab:Linear2DH1     P1 (TRI k=1)   H1-seminorm slope (1 / -)
    tab:Quadratic2DL2  Q2 (QUAD k=2)  L2-norm     slope (3 / 2)
    tab:Quadratic2DH1  Q2 (QUAD k=2)  H1-seminorm slope (2 / 1)

Data sources (per cell, in priority order):
  1. corner JSONs  — the rescued high-Re/low-alpha corner cells (Re=1e6, alpha_0=0.05),
                     solved at N=512->768 (see run_corner_article.jl / run_corner_osgs.jl);
  2. the sweep HDF5 — every other cell, finest two meshes N=160->320 of the [10..320] ladder.
A cell with no data anywhere is rendered as the literal `n.c.` (as in the paper), and reported.

Re-run any time to refresh the view as new simulations land. Only the numbers change; the table
structure is byte-faithful to the paper.

Usage:
    python3 make_results_tables.py                 # writes results/paper_tables.tex
    python3 make_results_tables.py --out X.tex     # custom output path
"""
import argparse
import datetime
import json
import math
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DEBUG = os.path.join(RESULTS, "debug_results")

# ---- nominal parameter grid (matches the article tables) -----------------------------------
RE_VALS = [1e-6, 1.0, 1e6]
DA_VALS = [1e-6, 1.0, 1e6]
ALPHA_VALS = [0.5, 0.05]
# Row order in every table block: Da outer, Re middle, alpha inner (18 rows).
ROW_KEYS = [(re, da, al) for da in DA_VALS for re in RE_VALS for al in ALPHA_VALS]

# metric field names carried per (Re,Da,alpha,method) record
NORM_FIELDS = ["slope_uL2", "FME_uL2", "slope_uH1", "FME_uH1",
               "slope_pL2", "FME_pL2", "slope_pH1", "FME_pH1"]


def approx(a, b, rtol=1e-6):
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1.0)


def canon(x, vals):
    for v in vals:
        if approx(float(x), v):
            return v
    return float(x)


def _decode(x):
    return x.decode() if isinstance(x, (bytes, bytes)) else str(x)


# ---- HDF5 sweep reader ---------------------------------------------------------------------
def _finest_pair(g, errname):
    """Return (slope, FME, (N_coarse, N_fine)) from the two finest meshes for one norm, or None."""
    h = np.asarray(g["h"][:], dtype=float)
    e = np.asarray(g[errname][:], dtype=float)
    order = np.argsort(h)  # ascending h: finest first
    idx = [i for i in order if np.isfinite(e[i]) and e[i] > 0]
    if len(idx) < 2:
        return None
    i_fine, i_coarse = idx[0], idx[1]
    hf, hc, ef, ec = h[i_fine], h[i_coarse], e[i_fine], e[i_coarse]
    slope = math.log(ec / ef) / math.log(hc / hf)
    return slope, ef, (int(round(1.0 / hc)), int(round(1.0 / hf)))


def load_h5(path):
    """Return {(Re,Da,alpha,method): record} for alpha in {0.5,0.05}, with all 4 norms."""
    data = {}
    if not os.path.isfile(path):
        return data
    norm_map = {"uL2": "err_u_l2", "uH1": "err_u_h1", "pL2": "err_p_l2", "pH1": "err_p_h1"}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            g = f[key]
            a = g.attrs
            try:
                re = canon(float(a["Re"]), RE_VALS)
                da = canon(float(a["Da"]), DA_VALS)
                al = canon(float(a["alpha_0"]), ALPHA_VALS)
            except KeyError:
                continue
            if al not in ALPHA_VALS:
                continue
            method = _decode(a["method"]).upper()
            rec = {"source": os.path.basename(path), "Npair": None}
            ok = True
            for tag, ds in norm_map.items():
                if ds not in g:
                    ok = False
                    break
                r = _finest_pair(g, ds)
                if r is None:
                    ok = False
                    break
                slope, fme, npair = r
                rec[f"slope_{tag}"] = slope
                rec[f"FME_{tag}"] = fme
                rec["Npair"] = npair
            if ok:
                data[(re, da, al, method)] = rec
    return data


# ---- corner JSON reader --------------------------------------------------------------------
# corner JSONs use l2u/l2p/h1u/h1p naming; map to the unified uL2/pL2/uH1/pH1 fields.
_CORNER_MAP = {
    "slope_uL2": "slope_l2u", "FME_uL2": "FME_l2u",
    "slope_uH1": "slope_h1u", "FME_uH1": "FME_h1u",
    "slope_pL2": "slope_l2p", "FME_pL2": "FME_l2p",
    "slope_pH1": "slope_h1p", "FME_pH1": "FME_h1p",
}


def load_corner(path_method_pairs):
    """Read corner result JSONs. Each pair is (path, default_method)."""
    data = {}
    for path, default_method in path_method_pairs:
        if not os.path.isfile(path):
            continue
        with open(path) as fh:
            recs = json.load(fh)
        for rec in recs:
            if any(rec.get(k) is None for k in _CORNER_MAP.values()):
                continue  # incomplete (e.g. base_only) record
            re = canon(float(rec["Re"]), RE_VALS)
            da = canon(float(rec["Da"]), DA_VALS)
            al = canon(float(rec["alpha_0"]), ALPHA_VALS)
            method = str(rec.get("method", default_method)).upper()
            out = {"source": os.path.basename(path),
                   "Npair": tuple(rec.get("Ns", [None, None]))}
            for uni, jk in _CORNER_MAP.items():
                out[uni] = float(rec[jk])
            # corner cells were solved DIRECTLY from the exact guess (no perturbation homotopy):
            # eps_pert = 0; iters are Newton-mode (Picard not used). `iters` is per-mesh; sum = total.
            if rec.get("iters") is not None:
                out["iters_eps"] = 0.0
                out["iters_ns"] = int(sum(int(i) for i in rec["iters"]))
                out["iters_pic"] = 0
            data[(re, da, al, method)] = out
    return data


def load_iters_traces(traces_dir, common_n):
    """Per-cell solver effort at the finest mesh from the trajectory sidecars:
    eps_pert (= successful perturbation) and Newton/Picard iteration totals (sum of `:N` / `:P`
    stage `iters` in the successful attempt). Returns {(Re,Da,alpha,method): (eps, n_ns, n_pic)}."""
    import glob
    out = {}
    for path in glob.glob(os.path.join(traces_dir, f"traj_*_N{common_n}.json")):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        cell = d.get("cell", {})
        try:
            re = canon(float(cell["Re"]), RE_VALS)
            da = canon(float(cell["Da"]), DA_VALS)
            al = canon(float(cell["alpha_0"]), ALPHA_VALS)
        except (KeyError, TypeError):
            continue
        if al not in ALPHA_VALS:
            continue
        method = str(cell.get("method", "")).upper()
        atts = d.get("attempts", [])
        succ = [a for a in atts if a.get("success")]
        a = succ[-1] if succ else (atts[-1] if atts else None)
        if a is None:
            continue
        n_ns = n_pic = 0
        for s in a.get("stages", []):
            nm = s.get("stage", "")
            it = s.get("iters", 0) or 0
            if nm.endswith(":N"):
                n_ns += it
            elif nm.endswith(":P"):
                n_pic += it
        eps = a.get("eps_pert", d.get("successful_eps"))
        out[(re, da, al, method)] = (eps, n_ns, n_pic)
    return out


def alias_forchheimer_da(data):
    """Under Forchheimer, Da is a no-op for the reaction-negligible corner: Da=1 == Da=1e-6.
    Fill a missing (Re=1e6, alpha=0.05, Da=1, method) from its Da=1e-6 twin if present."""
    for method in ("ASGS", "OSGS"):
        k_src = (1e6, 1e-6, 0.05, method)
        k_dst = (1e6, 1.0, 0.05, method)
        if k_src in data and k_dst not in data:
            aliased = dict(data[k_src])
            aliased["aliased_from_Da1em6"] = True
            data[k_dst] = aliased
    return data


# ---- LaTeX emission ------------------------------------------------------------------------
NC = "n.c."          # cell with no datum at all (paper's literal for non-converged)
ITER_NA = "--"       # no iteration record for this cell yet
DAGGER = r"$^{\dagger}$"  # marks an FME extrapolated to the common N (extra meshes were run)


def _fmt_slope(x):
    return "%.2f" % x          # plain fixed-point, never scientific (e.g. 2.01, 0.89)


def _fmt_fme(x):
    return r"\num[scientific-notation=true]{%.2e}" % x


def _eps_tex(eps):
    """Perturbation size as a compact power of ten ($1$, $10^{-1}$, $0$ for the direct corner solve)."""
    if eps is None:
        return None
    if eps == 0:
        return "0"
    k = int(round(-math.log10(eps)))
    return "1" if k == 0 else r"10^{-%d}" % k


def _fmt_iter(rec, compact=True):
    """eps_pert with the iteration sum N_NS+N_Pic. Default (compact): the sum is a SUBSCRIPT of
    eps_pert with no parentheses (a subscript can't be misread as a power), e.g. ${10^{-1}}_{3{+}0}$.
    Inline alternative: $10^{-1}\\,(3{+}0)$."""
    if rec is None or rec.get("iters_ns") is None:
        return ITER_NA
    eps = _eps_tex(rec.get("iters_eps"))
    ns, pic = rec["iters_ns"], rec["iters_pic"]
    if compact:
        return r"${%s}_{%d{+}%d}$" % (eps, ns, pic)
    return r"$%s\,(%d{+}%d)$" % (eps, ns, pic)


def _fme_at_common_N(rec, field, norm, common_n):
    """Error of `rec` at the common reference mesh `common_n`, with an extrapolation flag.

    Standard cells already have their finest mesh == common_n -> literal value, no flag.
    Corner cells (rescued at finer meshes, finest N=768) have no FE solution at N<=320, so the
    value is extrapolated from the cell's finest mesh using its own measured slope for this norm:
        err(common_n) = err(N_finest) * (N_finest / common_n) ** slope.
    Returns (value, is_extrapolated)."""
    fme = rec[f"FME_{field}{norm}"]
    n_finest = int(rec["Npair"][1])
    if n_finest == common_n:
        return fme, False
    slope = rec[f"slope_{field}{norm}"]
    return fme * (float(n_finest) / float(common_n)) ** slope, True


def _tex_re_da(v):
    if approx(v, 1e-6):
        return r"$10^{-6}$"
    if approx(v, 1.0):
        return r"$1$"
    if approx(v, 1e6):
        return r"$10^{6}$"
    return f"${v:g}$"


def _tex_alpha(v):
    return r"$0.5$" if approx(v, 0.5) else r"$0.05$"


def _colspec(fme_ascol_mm, iter_col_mm="17"):
    # 9 cols: Re Da alpha | slope(ASGS,OSGS) | FME(ASGS,OSGS) | eps(NS+Pic)(ASGS,OSGS)
    cols = [r">{\raggedleft\arraybackslash}p{7.5mm}"] * 5
    cols.append(r">{\raggedleft\arraybackslash}p{%smm}" % fme_ascol_mm)
    cols.append(r">{\raggedleft\arraybackslash}p{18mm}")
    cols.append(r">{\raggedleft\arraybackslash}p{%smm}" % iter_col_mm)
    cols.append(r">{\raggedleft\arraybackslash}p{%smm}" % iter_col_mm)
    return "\n                ".join(cols)


def emit_table(dataset, *, norm, label, caption, vel_rate, prs_rate, fme_col6_mm, common_n,
               iter_compact=True):
    """norm in {'L2','H1'}; vel_rate/prs_rate are the parenthesized theoretical orders.

    9 columns: keys | slope(ASGS,OSGS) | FME(ASGS,OSGS) | eps_pert (N_NS+N_Pic) (ASGS,OSGS).
    FME is at the common mesh `common_n`; corner cells are extrapolated there and flagged with a
    dagger. Returns (latex, n_extrapolated, n_nc)."""
    n_ext = 0
    n_nc = 0
    L = []
    L.append(r"\begin{table}[!htbp]")
    L.append(r"\centering")
    L.append(r"\caption{%s}" % caption)
    L.append(r"\label{%s}" % label)
    L.append(r"\small")
    L.append(r"\begin{tabular}{%s}" % _colspec(fme_col6_mm))
    for block, field, rate in (("velocity", "u", vel_rate), ("pressure", "p", prs_rate)):
        L.append(r"\toprule" if block == "velocity" else r"\midrule")
        L.append(r"\multicolumn{9}{c}{%s} \\" % block)
        L.append(r"\midrule")
        L.append(r"{} & {} & {} & \multicolumn{2}{c}{slope (%s)} & \multicolumn{2}{c}{FME} & "
                 r"\multicolumn{2}{c}{${\varepsilon_{\text{pert}}}_{N_{\text{NS}}{+}N_{\text{Pic}}}$}\\" % rate)
        L.append(r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
        L.append(r"$\Reyn$ & $\Damk$ & $\alpha_0$ & \multicolumn{1}{c}{ASGS} & "
                 r"\multicolumn{1}{c}{OSGS} & \multicolumn{1}{c}{ASGS} & \multicolumn{1}{c}{OSGS} & "
                 r"\multicolumn{1}{c}{ASGS} & \multicolumn{1}{c}{OSGS} \\")
        L.append(r"\midrule")
        for (re, da, al) in ROW_KEYS:
            slopes, fmes, iters = [], [], []
            for method in ("ASGS", "OSGS"):
                rec = dataset.get((re, da, al, method))
                if rec is None:
                    slopes.append(NC)
                    fmes.append(NC)
                    iters.append(ITER_NA)
                    n_nc += 1
                    continue
                slopes.append(_fmt_slope(rec[f"slope_{field}{norm}"]))
                val, ext = _fme_at_common_N(rec, field, norm, common_n)
                fmes.append(_fmt_fme(val) + (DAGGER if ext else ""))
                iters.append(_fmt_iter(rec, compact=iter_compact))
                if ext:
                    n_ext += 1
            L.append("%-9s & %-9s & %-7s & %s & %s & %s & %s & %s & %s \\\\" % (
                _tex_re_da(re), _tex_re_da(da), _tex_alpha(al),
                slopes[0], slopes[1], fmes[0], fmes[1], iters[0], iters[1]))
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    if n_ext:
        L.append(r"\par\vspace{2pt}")
        L.append(
            r"{\footnotesize $^{\dagger}$\,FME extrapolated to $N=%d$ from the cell's $N=512\to768$ "
            r"data: at the high-$\Reyn$/low-$\alpha_0$ corner ($\Reyn=10^{6}$, $\alpha_0=0.05$) the "
            r"discrete branch folds for $N\le320$, so no finest-mesh datum exists there and extra "
            r"finer meshes ($N=512,768$) were computed; the value shown is therefore not the literal "
            r"finest-mesh error. Slopes for these cells are measured on $N{=}512{\to}768$, and they were "
            r"solved directly from the exact-solution guess ($\varepsilon_{\text{pert}}=0$), not via the "
            r"perturbation homotopy.}" % common_n)
    L.append(r"\end{table}")
    return "\n".join(L), n_ext, n_nc


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{siunitx}
\sisetup{round-mode=none, exponent-product=\cdot}  % slopes are plain text; FME forces scientific per-cell
% paper macros (standalone definitions so this file compiles on its own)
\newcommand{\Reyn}{\mathrm{Re}}
\newcommand{\Damk}{\mathrm{Da}}
\setlength{\heavyrulewidth}{1.5pt}
\setlength{\abovetopsep}{4pt}
\setlength{\tabcolsep}{4pt}
\begin{document}
"""


def build():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tri-h5", default=os.path.join(RESULTS, "phase1_tri_k1.h5"),
                    help="P1 (TRI k=1) sweep HDF5")
    ap.add_argument("--quad-h5", default=os.path.join(RESULTS, "phase1_quad_k2.h5"),
                    help="Q2 (QUAD k=2) sweep HDF5")
    ap.add_argument("--out", default=os.path.join(RESULTS, "paper_tables.tex"),
                    help="output LaTeX file")
    ap.add_argument("--common-n", type=int, default=320,
                    help="common reference mesh N for the FME column (default 320); cells whose "
                         "finest mesh differs are extrapolated to this N and flagged")
    ap.add_argument("--no-document", action="store_true",
                    help="emit only the table environments (no documentclass wrapper)")
    ap.add_argument("--iter-inline", action="store_true",
                    help="render iters inline as eps (N_NS+N_Pic) instead of the default superscript form")
    args = ap.parse_args()
    common_n = args.common_n

    def _attach_iters(data, traces_dir):
        it = load_iters_traces(traces_dir, common_n)
        for key, (eps, ns, pic) in it.items():
            if key in data:
                data[key]["iters_eps"] = eps
                data[key]["iters_ns"] = ns
                data[key]["iters_pic"] = pic
        return data

    # ---- P1 family: TRI k=1 sweep + per-cell iters from traces + corner JSON overrides ----
    p1 = load_h5(args.tri_h5)
    _attach_iters(p1, os.path.join(RESULTS, "k1", "TRI", "traces"))
    corner = load_corner([
        (os.path.join(DEBUG, "corner_tri_k1_a005.json"), "ASGS"),
        (os.path.join(DEBUG, "corner_tri_k1_a005_osgs.json"), "OSGS"),
        (os.path.join(DEBUG, "corner_tri_k1_a005_osgs_da1e6.json"), "OSGS"),
    ])
    corner = alias_forchheimer_da(corner)
    p1.update(corner)  # corner rescued cells override / fill (carry their own iters)

    # ---- Q2 family: QUAD k=2 sweep + iters from traces + corner JSON overrides ----
    q2 = load_h5(args.quad_h5)
    _attach_iters(q2, os.path.join(RESULTS, "k2", "QUAD", "traces"))
    corner_q2 = load_corner([
        (os.path.join(DEBUG, "corner_quad_k2_a005.json"), "ASGS"),
        (os.path.join(DEBUG, "corner_quad_k2_a005_osgs.json"), "OSGS"),
    ])
    corner_q2 = alias_forchheimer_da(corner_q2)
    q2.update(corner_q2)  # Q2/QUAD-k2 corner: converges directly at the standard 160→320 (no fold)

    specs = [
        dict(data=p1, norm="L2", label="tab:Linear2DL2", vel_rate="2", prs_rate="1", fme_col6_mm="18",
             caption=r"Observed convergence rates and normalized finest mesh error (FME) for the 2D "
                     r"problem ($\mathbb{P}_1$ elements), calculated from the $L^2$-norm of the error "
                     r"obtained with the two finest meshes (theoretical convergence rates in parentheses)"),
        dict(data=p1, norm="H1", label="tab:Linear2DH1", vel_rate="1", prs_rate="-", fme_col6_mm="18",
             caption=r"Observed convergence rates and normalized finest mesh error (FME) for the 2D "
                     r"problem ($\mathbb{P}_1$ elements), calculated from the $H^1$-seminorm of the error "
                     r"obtained with the two finest meshes (theoretical convergence rates in parentheses)"),
        dict(data=q2, norm="L2", label="tab:Quadratic2DL2", vel_rate="3", prs_rate="2", fme_col6_mm="20",
             caption=r"Observed convergence rates and normalized finest mesh error (FME) for the 2D "
                     r"problem ($\mathbb{Q}_2$ elements), calculated from the $L^2$-norm of the error "
                     r"obtained with the two finest meshes (theoretical convergence rates in parentheses)"),
        dict(data=q2, norm="H1", label="tab:Quadratic2DH1", vel_rate="2", prs_rate="1", fme_col6_mm="18",
             caption=r"Observed convergence rates and normalized finest mesh error (FME) for the 2D "
                     r"problem ($\mathbb{Q}_2$ elements), calculated from the $H^1$-seminorm of the error "
                     r"obtained with the two finest meshes (theoretical convergence rates in parentheses)"),
    ]
    tables = []
    total_ext = total_nc = 0
    for s in specs:
        data = s.pop("data")
        tex, n_ext, n_nc = emit_table(data, common_n=common_n,
                                      iter_compact=not args.iter_inline, **s)
        tables.append(tex)
        total_ext += n_ext
        total_nc += n_nc

    # ---- provenance: per-cell source + missing/pending report ----
    stamp = datetime.date.today().isoformat()
    missing = []
    for fam, data, label in (("P1/TRI-k1", p1, "Linear"), ("Q2/QUAD-k2", q2, "Quadratic")):
        for (re, da, al) in ROW_KEYS:
            for method in ("ASGS", "OSGS"):
                if (re, da, al, method) not in data:
                    missing.append((fam, re, da, al, method))

    header_note = (
        "%% Generated by make_results_tables.py on %s\n"
        "%% P1 tables <- %s (+ corner JSONs);  Q2 tables <- %s\n"
        "%% Slopes: two finest meshes of each cell's ladder (standard N=160->320; corner N=512->768).\n"
        "%% FME: reported at a COMMON reference mesh N=%d. Standard cells -> literal N=%d error.\n"
        "%%   Corner cells (Re=1e6, alpha=0.05, no FE root for N<=320) -> extrapolated to N=%d via\n"
        "%%   err(%d)=err(N_finest)*(N_finest/%d)^slope, marked with a dagger.\n"
        "%% Cells with no datum at all are rendered n.c. (see stderr report at generation time).\n"
        % (stamp, os.path.basename(args.tri_h5), os.path.basename(args.quad_h5),
           common_n, common_n, common_n, common_n, common_n)
    )

    body = "\n\n".join(tables)
    if args.no_document:
        out_text = header_note + "\n" + body + "\n"
    else:
        note_tex = (
            r"\noindent\textbf{Latest Gridap MMS results} (generated %s). Convergence slopes use the "
            r"two finest meshes of each cell's ladder ($N{=}160{\to}320$ for the standard sweep; "
            r"$N{=}512{\to}768$ for the rescued high-$\Reyn$/low-$\alpha_0$ corner). FME is reported at "
            r"a common mesh $N{=}%d$; corner cells (no FE solution for $N\le320$) carry an FME "
            r"extrapolated to $N{=}%d$, flagged $^{\dagger}$. The last column gives the perturbation "
            r"size $\varepsilon_{\text{pert}}$ at the finest mesh with the Newton$+$Picard iteration "
            r"counts $N_{\text{NS}}{+}N_{\text{Pic}}$ as a subscript. \texttt{n.c.} marks a cell not "
            r"yet computed; \texttt{-{}-} an iteration count not yet recorded."
            % (stamp, common_n, common_n)
        )
        out_text = (header_note + PREAMBLE + note_tex + "\n\n" + body
                    + "\n\n" + r"\end{document}" + "\n")

    with open(args.out, "w") as fh:
        fh.write(out_text)

    # ---- stderr report ----
    print(f"[make_results_tables] wrote {args.out}", file=sys.stderr)
    print(f"[make_results_tables]   common FME mesh N={common_n}; "
          f"{total_ext} value(s) extrapolated (marked with a dagger)", file=sys.stderr)
    print(f"[make_results_tables]   P1 cells present: {sum(1 for k in p1)} ; "
          f"Q2 cells present: {sum(1 for k in q2)}", file=sys.stderr)
    aliased = [k for k, v in p1.items() if v.get("aliased_from_Da1em6")]
    if aliased:
        print(f"[make_results_tables]   Forchheimer Da-alias (Da=1<-Da=1e-6): "
              f"{[(k[0], k[3]) for k in aliased]}", file=sys.stderr)
    if missing:
        print(f"[make_results_tables]   {len(missing)} cell(s) rendered n.c. (no datum):", file=sys.stderr)
        for fam, re, da, al, method in missing:
            print(f"      {fam:12s} Re={re:<7g} Da={da:<7g} alpha={al:<5g} {method}", file=sys.stderr)
    else:
        print("[make_results_tables]   all cells populated.", file=sys.stderr)


if __name__ == "__main__":
    build()
