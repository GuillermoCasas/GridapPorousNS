"""Shared MMS convergence helpers for the two-phase sweep workflow.

Both `detect_flagged_cells.py` and the merged report in `plot_results.py` import these so the
"true root" and rate definitions are identical everywhere. The key idea (see
diagnostics/c24_resolution_and_continuation.md): the solver's own "converged" flag is fooled by
the noise floor, so a mesh counts as a TRUE root only if its recorded residual sits within
`k_nf · dynamic_ftol(N)` — the same gate the honest-exit core fix applies.
"""
import os
import json
import time
import numpy as np
import h5py


def robust_open_h5(filepath, mode='r', retries=10, delay=2.0):
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    for attempt in range(retries):
        try:
            return h5py.File(filepath, mode, swmr=True)
        except (OSError, RuntimeError) as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise


# base_config.json fallbacks (used only if a phase-1 config omits a field; the phase-1 configs
# set ftol/ceiling/safety/k_nf explicitly, so these rarely apply).
_BASE = dict(ftol=1e-8, dynamic_ftol_ceiling=1e-4, dynamic_ftol_spatial_safety_factor=1e-2,
             noise_floor_success_max_ftol_multiple=1e30)


def load_solver_constants(config_path):
    """Read the dynamic_ftol / honest-exit constants from a phase-1 JSON config."""
    with open(config_path) as fh:
        cfg = json.load(fh)
    sol = cfg.get("numerical_method", {}).get("solver", {})
    return {k: float(sol.get(k, v)) for k, v in _BASE.items()}


def dynamic_ftol(N, kv, c):
    """Mirror run_test.jl: max(ftol, min(ceiling, safety * h^(kv+1)))."""
    h = 1.0 / N
    return max(c["ftol"], min(c["dynamic_ftol_ceiling"],
                              c["dynamic_ftol_spatial_safety_factor"] * h ** (kv + 1)))


def is_true_root(residual, N, kv, c):
    """A mesh reached a TRUE root iff its residual is finite and within k_nf*dynamic_ftol(N)."""
    return bool(np.isfinite(residual) and
                residual <= c["noise_floor_success_max_ftol_multiple"] * dynamic_ftol(N, kv, c))


def consecutive_slope(h_coarse, e_coarse, h_fine, e_fine):
    """log-log slope between two meshes; NaN if either error is non-finite or non-positive."""
    if not (np.isfinite(e_coarse) and np.isfinite(e_fine) and e_coarse > 0 and e_fine > 0):
        return float('nan')
    return float(np.log(e_coarse / e_fine) / np.log(h_coarse / h_fine))


def decode(x):
    return x.decode('utf-8') if isinstance(x, bytes) else str(x)
