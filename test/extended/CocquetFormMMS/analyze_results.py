#!/usr/bin/env python3
"""CocquetFormMMS analysis entry point — thin wrapper around ManufacturedSolutions/analyze_results.py.

The CocquetFormMMS h5 schema is identical to ManufacturedSolutions
(config_<idx>_<METHOD> groups; same datasets and attributes), so the canonical
analyzer runs unchanged. This wrapper just bakes in the local paths and switches
between the two configured variants.

Variants:
  (default)       cocquet_form_mms              — mixed Taylor-Hood P2/P1, ASGS, TRI, N ≤ 80
  --equalorder    cocquet_form_mms_equalorder   — equal-order P1/P1 + P2/P2, ASGS, N ≤ 80
  --comparison    cocquet_form_mms_comparison   — 3-way at Re=Da=1, α=0.5: P1/P1 ASGS + P2/P2 ASGS + P2/P1 Galerkin

Usage:
  python analyze_results.py                       # Taylor-Hood variant
  python analyze_results.py --equalorder          # equal-order variant
  python analyze_results.py --comparison          # 3-way comparison variant
  python analyze_results.py --no-plots            # tables only (any extra flag is passed through)

Outputs (in results/, variant-specific filenames):
  - convergence_config*.png                       per-cell log-log curves with slope annotations
  - merged_convergence_report_<variant>.md        paper-style status table
  - flagged_<variant>.json                        sub-optimal-rate / fold cells for Phase-2 rescue
  - convergence_report.md / summary_tables.txt    detailed per-config tables (shared filenames)
"""
import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CANONICAL = os.path.normpath(os.path.join(HERE, '..', 'ManufacturedSolutions', 'analyze_results.py'))


def main():
    ap = argparse.ArgumentParser(
        description="Analyze CocquetFormMMS h5 output (delegates to the canonical analyze_results.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--equalorder', action='store_true',
                    help='Use the equal-order variant (cocquet_form_mms_equalorder.h5)')
    ap.add_argument('--comparison', action='store_true',
                    help='Use the 3-way comparison variant (cocquet_form_mms_comparison.h5)')
    ap.add_argument('--report-N', type=int, default=80,
                    help='Mesh whose error is reported for verified cells (default 80, the finest in the ladder)')
    args, passthrough = ap.parse_known_args()

    if args.comparison and args.equalorder:
        sys.exit("--comparison and --equalorder are mutually exclusive")
    variant = ('cocquet_form_mms_comparison' if args.comparison
               else 'cocquet_form_mms_equalorder' if args.equalorder
               else 'cocquet_form_mms')
    h5 = os.path.join(HERE, 'results', f'{variant}.h5')
    # The comparison variant uses two leg configs (_stab + _galerkin) that share the same
    # solver/numerical block. The analyzer reads the solver constants from any one; pick the
    # _stab leg.
    config_basename = (f'{variant}_stab' if args.comparison else variant)
    config = os.path.join(HERE, 'data', f'{config_basename}.json')
    flagged = os.path.join(HERE, 'results', f'flagged_{variant}.json')
    out = os.path.join(HERE, 'results', f'merged_convergence_report_{variant}.md')
    outdir = os.path.join(HERE, 'results')

    if not os.path.exists(h5):
        sys.exit(f"h5 file not found: {h5}\n(run run_test.jl first to produce it)")
    if not os.path.exists(CANONICAL):
        sys.exit(f"canonical analyze_results.py not found at {CANONICAL}")

    sys.argv = [CANONICAL, '--h5', h5, '--config', config, '--flagged', flagged,
                '--out', out, '--outdir', outdir, '--report-N', str(args.report_N)] + passthrough
    runpy.run_path(CANONICAL, run_name='__main__')


if __name__ == '__main__':
    main()
