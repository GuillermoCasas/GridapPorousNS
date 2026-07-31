#!/usr/bin/env python3
"""
Transcribe the 3D manufactured-solution test results DIRECTLY into the LaTeX table
(tab:3D of theory/paper/article.tex), so no number is ever hand-copied.  Since 2026-07-31
the article reports both norms in ONE table per example, so this emits a single table
whose two vertical blocks (velocity, pressure) each carry the L2 columns followed by the
H1 columns; the former tab:3DL2 / tab:3DH1 pair no longer exists.

Sources (single source of truth):
  * solver rows  -> results/k{1,2}/TET/{structured,nested_red}/convergence3d_results.json
                    (the canonical convergence3d outputs written by smoke3d.jl)
  * interp rows  -> results/interp_reference3d.json
                    (written by run_interpolation_reference3d.jl; optional -- if it is
                     absent the interp rows are emitted as placeholders with a reminder)

Each reported value is:
  * slope = two-finest-mesh order  log(e[-2]/e[-1]) / log(h[-2]/h[-1])
  * FME   = finest-mesh error e[-1]   ("normalized finest mesh error"; the manufactured
            field has O(1) norm, and the errors are already normalized in calc_errors3d)
matching exactly how the paper defines them and how smoke3d.jl / the interp reference compute them.

Usage:
  python3 make_3d_tables.py                 # print the two complete LaTeX tables
  python3 make_3d_tables.py --check ../../../theory/paper/article.tex
                                            # diff every \\num value in the article's 3D
                                            # tables against the data; exit 1 on mismatch
"""
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
INTERP_JSON = os.path.join(RESULTS, "interp_reference3d.json")

# (family label in table, results-dir mesh_sequence)
FAMILIES = [("regular", "structured"), ("irregular", "nested_red")]
FIELDS = ["l2u", "l2p", "h1u", "h1p"]  # order used within a table half


# ---------------------------------------------------------------- formatting
def fmt_slope(x):
    return f"{x:.2f}"


def fmt_fme(x):
    """Solver-row convention: plain 1-decimal for |x|>=1 (the O(1) pressure-H1 errors),
    2-significant-figure scientific otherwise (e.g. 1.9e-3).  The article dropped one
    significant figure from every FME on 2026-07-31 to narrow the merged tables."""
    return f"{x:.1f}" if abs(x) >= 1.0 else f"{x:.1e}"


def fmt_fme4(x):
    """Interp-row convention: the paper prints the interpolation FME at one more
    significant figure than the solver rows (e.g. 1.29e-3)."""
    return f"{x:.2f}" if abs(x) >= 1.0 else f"{x:.2e}"


def _strip_exp(s):
    s = re.sub(r"e([+-])0*(\d)", r"e\1\2", s)  # e-03 -> e-3, e+00 -> e+0
    s = re.sub(r"e\+?0$", "", s)               # 1.29e0 -> 1.29 (guards O(1) sci)
    return s


def num(x, kind):
    """Emit the numeric body of a siunitx \\num{...} token in the paper's style.
    kind: 'slope' (2 dp), 'fme' (solver, 2 s.f.), 'fme4' (interp, 3 s.f.)."""
    s = {"slope": fmt_slope, "fme": fmt_fme, "fme4": fmt_fme4}[kind](x)
    return _strip_exp(s)


# ---------------------------------------------------------------- data loading
def load_solver():
    """data[(field, family)] = {'slope': {method:val}, 'fme': {method:val}}"""
    data = {}
    for kv in (1, 2):
        for fam, seq in FAMILIES:
            path = os.path.join(RESULTS, f"k{kv}", "TET", seq, "convergence3d_results.json")
            with open(path) as fh:
                recs = json.load(fh)
            by_m = {r["method"]: r for r in recs}
            for field in FIELDS:
                slp, fme = {}, {}
                for method, r in by_m.items():
                    e, h = r[field], r["hs"]
                    slp[method] = math.log(e[-2] / e[-1]) / math.log(h[-2] / h[-1])
                    fme[method] = e[-1]
                data[(kv, field, fam)] = {"slope": slp, "fme": fme}
    return data


def load_interp():
    if not os.path.exists(INTERP_JSON):
        return None
    with open(INTERP_JSON) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- table emission
# Since 2026-07-31 the article carries ONE 3D table (tab:3D) holding both norms side by
# side, in place of the former tab:3DL2 / tab:3DH1 pair.  Each of its two vertical blocks
# (velocity, pressure) spans eight data columns: L2 (slope ASGS/OSGS, FME ASGS/OSGS) then
# H1 (idem).  BLOCKS maps the block word to its (L2 field, H1 field) and to the pair of
# worst-case bound rates printed in parentheses after the element label -- L2 first, H1
# second, matching the caption's "for the two norms in that order".
LABEL = "tab:3D"
BLOCKS = [
    ("velocity", ("l2u", "h1u"), {1: ("2", "1"), 2: ("3", "2")}),
    ("pressure", ("l2p", "h1p"), {1: ("1", "$-$"), 2: ("2", "1")}),
]


def solver_row(data, kv, fields, fam, opt):
    cells = []
    for field in fields:                      # L2 half, then H1 half
        d = data[(kv, field, fam)]
        cells += [num(d["slope"]["ASGS"], "slope"), num(d["slope"]["OSGS"], "slope"),
                  num(d["fme"]["ASGS"], "fme"), num(d["fme"]["OSGS"], "fme")]
    body = " & ".join(f"\\num{{{c}}}" for c in cells)
    return f"$\\mathbb{{P}}_{kv}$ ({opt[0]}, {opt[1]}) & {body} \\\\"


def interp_row(interp, kv, fields, fam):
    if interp is None:
        spans = " & ".join([r"\multicolumn{2}{c}{\num{--}}"] * 4)
        return (f"$\\mathbb{{P}}_{kv}$ interp. & {spans} \\\\  "
                f"% TODO run run_interpolation_reference3d.jl")
    s = interp[fam][str(kv)]
    cells = []
    for field in fields:
        cells += [num(s["slope"][field], "slope"), num(s["fme"][field], "fme4")]
    spans = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\num{{{c}}}}}" for c in cells)
    return f"$\\mathbb{{P}}_{kv}$ interp. & {spans} \\\\"


def emit_table(data, interp):
    L = []
    L.append(r"\begin{table}[!htbp]")
    L.append(r"\centering")
    L.append(r"\caption{Observed convergence rates and normalized finest mesh error (FME) for the 3D "
             r"problem, calculated from the $L^2${\hyp}norm and the $H^1${\hyp}seminorm of the error "
             r"obtained with the two finest meshes. The pair in parentheses after each element type "
             r"gives the worst{\hyp}case bound rates for the two norms in that order, in the convention "
             r"of \cref{tab:Linear2D}: $k_u+1$ and $k_u$ for the velocity, $k_p$ and $k_p-1$ for the "
             r"pressure, a dash where the latter is not positive. The nodal{\hyp}interpolant rows "
             r"converge one order faster in the pressure, so the parenthetical bounds the analysis "
             r"rather than the approximability of the space}")
    L.append(r"\label{" + LABEL + r"}")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{2pt}")
    L.append(r"\begin{tabular}{l*{8}{r}}")
    L.append(r"\toprule")
    # the column-header deck is emitted once for the whole table, not once per block
    L.append(r"{} & \multicolumn{4}{c}{$L^2${\hyp}norm} & \multicolumn{4}{c}{$H^1${\hyp}seminorm} \\")
    L.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}")
    L.append(r"{} & \multicolumn{2}{c}{slope} & \multicolumn{2}{c}{FME} "
             r"& \multicolumn{2}{c}{slope} & \multicolumn{2}{c}{FME} \\")
    L.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}")
    L.append(r"element type & \multicolumn{1}{c}{ASGS} & \multicolumn{1}{c}{OSGS} "
             r"& \multicolumn{1}{c}{ASGS} & \multicolumn{1}{c}{OSGS} "
             r"& \multicolumn{1}{c}{ASGS} & \multicolumn{1}{c}{OSGS} "
             r"& \multicolumn{1}{c}{ASGS} & \multicolumn{1}{c}{OSGS} \\")
    for word, fields, opt in BLOCKS:
        L.append(r"\midrule")
        L.append(r"\multicolumn{9}{c}{" + word + r"} \\")
        L.append(r"\midrule")
        for fi, (fam, _seq) in enumerate(FAMILIES):
            if fi > 0:
                L.append(r"\addlinespace")
            L.append(r"\multicolumn{9}{l}{\textit{" + fam + r" mesh}} \\")
            for kv in (1, 2):
                L.append(solver_row(data, kv, fields, fam, opt[kv]))
            for kv in (1, 2):
                L.append(interp_row(interp, kv, fields, fam))
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")
    return "\n".join(L)


# ---------------------------------------------------------------- check mode
def check(article_path, data, interp):
    """Parse tab:3D in article.tex and diff every \\num value against the data: the solver
    rows against the convergence JSONs, and (if interp_reference3d.json is present) the
    interp rows against it.  Both norms now live in ONE table, so each row carries eight
    solver values (L2 slope/FME ASGS+OSGS, then H1) or four interpolant spans."""
    with open(article_path) as fh:
        tex = fh.read()

    mismatches = []
    # nearest preceding \begin{table}: forbid another \begin{table} inside the span
    m = re.search(r"\\begin\{table\}(?:(?!\\begin\{table\}).)*?\\label\{" + re.escape(LABEL) + r"\}"
                  r"(?:(?!\\end\{table\}).)*", tex, re.DOTALL)
    if not m:
        return [f"[{LABEL}] table not found in {article_path}"]
    block = m.group(0)

    for word, fields, _opt in BLOCKS:
        # locate the sub-block for this velocity/pressure word header
        wm = re.search(r"\\multicolumn\{9\}\{c\}\{" + word + r"\}", block)
        sub = block[wm.end():] if wm else block
        for fam, _seq in FAMILIES:
            fm = re.search(r"\\textit\{" + fam + r" mesh\}", sub)
            seg = sub[fm.end():] if fm else sub
            for kv in (1, 2):
                # match the P_kv solver row (not the interp row, which has \multicolumn spans)
                rx = (r"\$\\mathbb\{P\}_" + str(kv) + r"\$ \([^)]*\)"
                      + r" & \\num\{([^}]*)\}" * 8 + r" \\\\")
                rm = re.search(rx, seg)
                if not rm:
                    mismatches.append(f"[{LABEL}/{word}/{fam}/P{kv}] solver row not found")
                    continue
                got = rm.groups()
                want = []
                for field in fields:
                    d = data[(kv, field, fam)]
                    want += [num(d["slope"]["ASGS"], "slope"), num(d["slope"]["OSGS"], "slope"),
                             num(d["fme"]["ASGS"], "fme"), num(d["fme"]["OSGS"], "fme")]
                cols = [f"{n}{c}" for n in ("L2", "H1")
                        for c in ("slopeASGS", "slopeOSGS", "fmeASGS", "fmeOSGS")]
                for col, g, w in zip(cols, got, want):
                    if g != w:
                        mismatches.append(
                            f"[{LABEL}/{word}/{fam}/P{kv}/{col}] article={g!r}  data={w!r}")

                if interp is None:
                    continue
                # interp row: P_kv interp. & \multicolumn{2}{c}{\num{v}} x4  (L2 slope/FME, H1 slope/FME)
                irx = (r"\$\\mathbb\{P\}_" + str(kv) + r"\$ interp\."
                       + r" & \\multicolumn\{2\}\{c\}\{\\num\{([^}]*)\}\}" * 4)
                im = re.search(irx, seg)
                if not im:
                    mismatches.append(f"[{LABEL}/{word}/{fam}/P{kv}/interp] interp row not found")
                    continue
                s = interp[fam][str(kv)]
                iw = []
                for field in fields:
                    iw += [num(s["slope"][field], "slope"), num(s["fme"][field], "fme4")]
                icols = [f"{n}interp{c}" for n in ("L2", "H1") for c in ("Slope", "FME")]
                for col, g, w in zip(icols, im.groups(), iw):
                    if g != w:
                        mismatches.append(
                            f"[{LABEL}/{word}/{fam}/P{kv}/{col}] article={g!r}  data={w!r}")
    return mismatches


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", metavar="ARTICLE_TEX",
                    help="diff the article's 3D solver-row values against the data instead of emitting tables")
    args = ap.parse_args()

    data = load_solver()
    interp = load_interp()

    if args.check:
        if interp is None:
            sys.stderr.write("NOTE: results/interp_reference3d.json not found; interp rows not checked. "
                             "Run run_interpolation_reference3d.jl to enable them.\n")
        mismatches = check(args.check, data, interp)
        if not mismatches:
            scope = "solver + interp" if interp is not None else "solver"
            print(f"OK: every {scope} \\num value in {LABEL} matches the data.")
            return 0
        print(f"MISMATCH ({len(mismatches)}):")
        for m in mismatches:
            print("  " + m)
        return 1

    if interp is None:
        sys.stderr.write("NOTE: results/interp_reference3d.json not found; interp rows emitted as "
                         "placeholders. Run run_interpolation_reference3d.jl to populate them.\n")
    print(emit_table(data, interp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
