#!/usr/bin/env python3
"""
Transcribe the 3D manufactured-solution test results DIRECTLY into the LaTeX tables
(tab:3DL2 and tab:3DH1 of theory/paper/article.tex), so no number is ever hand-copied.

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
    """Solver-row convention: plain 2-decimals for |x|>=1 (the O(1) pressure-H1 errors),
    3-significant-figure scientific otherwise (e.g. 1.90e-3)."""
    return f"{x:.2f}" if abs(x) >= 1.0 else f"{x:.2e}"


def fmt_fme4(x):
    """Interp-row convention: the paper prints the interpolation FME at one more
    significant figure than the solver rows (e.g. 1.291e-3)."""
    return f"{x:.3f}" if abs(x) >= 1.0 else f"{x:.3e}"


def _strip_exp(s):
    s = re.sub(r"e([+-])0*(\d)", r"e\1\2", s)  # e-03 -> e-3, e+00 -> e+0
    s = re.sub(r"e\+?0$", "", s)               # 1.29e0 -> 1.29 (guards O(1) sci)
    return s


def num(x, kind):
    """Emit the numeric body of a siunitx \\num{...} token in the paper's style.
    kind: 'slope' (2 dp), 'fme' (solver, 3 s.f.), 'fme4' (interp, 4 s.f.)."""
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
# Which (norm, field) pairs belong to which table, and the theoretical rate shown
# in parentheses next to each element label.
TABLES = {
    "L2": {  # tab:3DL2
        "label": "tab:3DL2",
        "norm_word": "$L^2$-norm",
        "blocks": [("velocity", "l2u"), ("pressure", "l2p")],
        "opt": {("l2u", 1): "2", ("l2u", 2): "3", ("l2p", 1): "1", ("l2p", 2): "2"},
    },
    "H1": {  # tab:3DH1
        "label": "tab:3DH1",
        "norm_word": "$H^1$-seminorm",
        "blocks": [("velocity", "h1u"), ("pressure", "h1p")],
        "opt": {("h1u", 1): "1", ("h1u", 2): "2", ("h1p", 1): "-", ("h1p", 2): "1"},
    },
}


def solver_row(data, kv, field, fam, opt):
    d = data[(kv, field, fam)]
    a_s, o_s = d["slope"]["ASGS"], d["slope"]["OSGS"]
    a_f, o_f = d["fme"]["ASGS"], d["fme"]["OSGS"]
    return (f"    $\\mathbb{{P}}_{kv}$ ({opt}) & \\num{{{num(a_s,'slope')}}} & \\num{{{num(o_s,'slope')}}} "
            f"& \\num{{{num(a_f,'fme')}}} & \\num{{{num(o_f,'fme')}}} \\\\")


def interp_row(interp, kv, field, fam):
    fam_key = {"regular": "regular", "irregular": "irregular"}[fam]
    if interp is None:
        return (f"    \\amend{{$\\mathbb{{P}}_{kv}$ interp.}} & \\multicolumn{{2}}{{c}}{{\\amend{{\\num{{--}}}}}} "
                f"& \\multicolumn{{2}}{{c}}{{\\amend{{\\num{{--}}}}}} \\\\  "
                f"% TODO run run_interpolation_reference3d.jl")
    s = interp[fam_key][str(kv)]
    slope_v = s["slope"][field]
    fme_v = s["fme"][field]
    return (f"    \\amend{{$\\mathbb{{P}}_{kv}$ interp.}} "
            f"& \\multicolumn{{2}}{{c}}{{\\amend{{\\num{{{num(slope_v,'slope')}}}}}}} "
            f"& \\multicolumn{{2}}{{c}}{{\\amend{{\\num{{{num(fme_v,'fme4')}}}}}}} \\\\")


def emit_table(which, data, interp):
    spec = TABLES[which]
    L = []
    L.append(r"\begin{table}[!htbp]")
    L.append(r"    \centering")
    L.append(r"    \caption{Observed convergence rates and normalized finest mesh error (FME) for the 3D "
             r"problem, calculated from the " + spec["norm_word"] + r" of the error obtained with the two "
             r"finest meshes (theoretical convergence rates in parentheses)}")
    L.append(r"    \begin{tabular}{>{\raggedleft\arraybackslash}p{20mm}")
    L.append(r"                    >{\raggedleft\arraybackslash}p{7.5mm}")
    L.append(r"                    >{\raggedleft\arraybackslash}p{7.5mm}")
    L.append(r"                    >{\raggedleft\arraybackslash}p{18mm}")
    L.append(r"                    >{\raggedleft\arraybackslash}p{18mm}}")
    L.append(r"    \toprule")
    for bi, (word, field) in enumerate(spec["blocks"]):
        if bi > 0:
            L.append(r"    \midrule")
        L.append(r"    \multicolumn{5}{c}{" + word + r"} \\")
        L.append(r"    \midrule")
        L.append(r"    {} & \multicolumn{2}{c}{slope} & \multicolumn{2}{c}{FME}\\")
        L.append(r"    \cmidrule(lr){2-5}")
        L.append(r"    element type & \multicolumn{1}{c}{ASGS} & \multicolumn{1}{c}{OSGS} "
                 r"& \multicolumn{1}{c}{ASGS} & \multicolumn{1}{c}{OSGS} \\")
        L.append(r"    \midrule")
        for fi, (fam, _seq) in enumerate(FAMILIES):
            if fi > 0:
                L.append(r"    \addlinespace")
            L.append(r"    \multicolumn{5}{l}{\textit{" + fam + r" mesh}} \\")
            for kv in (1, 2):
                L.append(solver_row(data, kv, field, fam, spec["opt"][(field, kv)]))
            for kv in (1, 2):
                L.append(interp_row(interp, kv, field, fam))
    L.append(r"    \bottomrule")
    L.append(r"    \end{tabular}")
    L.append(r"    \label{" + spec["label"] + r"}")
    L.append(r"\end{table}")
    return "\n".join(L)


# ---------------------------------------------------------------- check mode
def check(article_path, data, interp):
    """Parse tab:3DL2/tab:3DH1 in article.tex and diff every \\num value against the data:
    the solver rows against the convergence JSONs, and (if interp_reference3d.json is
    present) the interp rows against it."""
    with open(article_path) as fh:
        tex = fh.read()

    mismatches = []
    for which in ("L2", "H1"):
        spec = TABLES[which]
        label = spec["label"]
        # nearest preceding \begin{table}: forbid another \begin{table} inside the span
        m = re.search(r"\\begin\{table\}(?:(?!\\begin\{table\}).)*?\\label\{" + re.escape(label) + r"\}",
                      tex, re.DOTALL)
        if not m:
            mismatches.append(f"[{label}] table not found in {article_path}")
            continue
        block = m.group(0)
        # split into the velocity / pressure halves at the block \midrule separators
        for word, field in spec["blocks"]:
            # locate the sub-block for this field's velocity/pressure word header
            wm = re.search(r"\\multicolumn\{5\}\{c\}\{" + word + r"\}", block)
            sub = block[wm.end():] if wm else block
            for fam, _seq in FAMILIES:
                fm = re.search(r"\\textit\{" + fam + r" mesh\}", sub)
                seg = sub[fm.end():] if fm else sub
                for kv in (1, 2):
                    # match the P_kv solver row (not the interp row, which has \amend + \multicolumn)
                    rx = (r"\$\\mathbb\{P\}_" + str(kv) + r"\$ \([^)]*\) & \\num\{([^}]*)\} & \\num\{([^}]*)\} "
                          r"& \\num\{([^}]*)\} & \\num\{([^}]*)\} \\\\")
                    rm = re.search(rx, seg)
                    if not rm:
                        mismatches.append(f"[{label}/{word}/{fam}/P{kv}] solver row not found")
                        continue
                    got = rm.groups()  # (slope ASGS, slope OSGS, FME ASGS, FME OSGS)
                    d = data[(kv, field, fam)]
                    want = (num(d["slope"]["ASGS"], "slope"), num(d["slope"]["OSGS"], "slope"),
                            num(d["fme"]["ASGS"], "fme"), num(d["fme"]["OSGS"], "fme"))
                    for col, (g, w) in zip(("slopeASGS", "slopeOSGS", "fmeASGS", "fmeOSGS"), zip(got, want)):
                        if g != w:
                            mismatches.append(
                                f"[{label}/{word}/{fam}/P{kv}/{col}] article={g!r}  data={w!r}")

                    if interp is None:
                        continue
                    # interp row: \amend{P_kv interp.} & \multicolumn{2}{c}{\amend{\num{slope}}} & ...{\num{fme}}
                    irx = (r"\\amend\{\$\\mathbb\{P\}_" + str(kv) + r"\$ interp\.\} & "
                           r"\\multicolumn\{2\}\{c\}\{\\amend\{\\num\{([^}]*)\}\}\} & "
                           r"\\multicolumn\{2\}\{c\}\{\\amend\{\\num\{([^}]*)\}\}\}")
                    im = re.search(irx, seg)
                    if not im:
                        mismatches.append(f"[{label}/{word}/{fam}/P{kv}/interp] interp row not found")
                        continue
                    ig = im.groups()  # (slope, FME)
                    s = interp[fam][str(kv)]
                    iw = (num(s["slope"][field], "slope"), num(s["fme"][field], "fme4"))
                    for col, (g, w) in zip(("interpSlope", "interpFME"), zip(ig, iw)):
                        if g != w:
                            mismatches.append(
                                f"[{label}/{word}/{fam}/P{kv}/{col}] article={g!r}  data={w!r}")
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
            print(f"OK: every {scope} \\num value in tab:3DL2 and tab:3DH1 matches the data.")
            return 0
        print(f"MISMATCH ({len(mismatches)}):")
        for m in mismatches:
            print("  " + m)
        return 1

    if interp is None:
        sys.stderr.write("NOTE: results/interp_reference3d.json not found; interp rows emitted as "
                         "placeholders. Run run_interpolation_reference3d.jl to populate them.\n")
    print(emit_table("L2", data, interp))
    print()
    print(emit_table("H1", data, interp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
