# Plan review + MMS-decoupling design (addendum to the solver-simplification plan)

> **STATUS: IMPLEMENTED as of 2026-07-04 — retained for history, being archived.** The
> `SolutionVerifier` seam (`NoVerification` / `MMSPlateauVerifier`, `on_asgs_converged!` /
> `on_osgs_converged!`, `mms_verification.jl`) and the file split **shipped**; the realized design is
> documented in [`docs/solver/algorithm-code-mapping.md`](algorithm-code-mapping.md). File-path refs to
> `porous_solver.jl` below are stale (now `asgs_solver.jl` / `osgs_solver.jl` / `solver_core.jl`).

This addendum (1) confirms/critiques the execution plan and (2) specifies the MMS-decoupling the
author asked for (take the MMS-specific logic out of the main solver body). It assumes the plan
in front of it. Everything here was re-verified against the live code; the load-bearing facts are
cited inline.

---

## 1. Verdict on the plan

**Endorsed as written, with the four refinements in §2.** The plan correctly preserves all
safeguard control flow (robustness-first), removes only verified-dead code, and — importantly —
**reverses my proposal's Part C, correctly.**

### The Part C reversal is right (verified, and stronger than the plan states)

My proposal floated "wire `osgs_iterations`/`osgs_tolerance` into the coupled solve." The plan
rejects that as a robustness regression. I re-checked `docs/mms/convergence-baseline.md` and the
plan is **correct, with a wider margin than it quotes**:

- The coupled inexact-Newton's OSGS iteration counts per cell are **9–98**, summarised in the doc
  as **"OSGS iteration cost is high (30–104)"** — because the dropped dense `∂π/∂U` block gives a
  *linear* rate (this is the stated JFNK target, not an inner-solver defect).
- The baseline ran a **common Newton budget of 150**, which covers the 104 worst case.
- `osgs_iterations` is `3` (base_config) / `5` (MMS) / `15` (Cocquet). Wiring it as the cap would
  truncate a 30–104-step descent at 3–15 steps, silently degenerating OSGS → ASGS and reporting
  ASGS's optimal rate under the OSGS label.
- The per-field gate (`effective_ftol = max(ftol, rel·‖R₀‖)`, `nonlinear.jl`
  `_initialize_effective_thresholds`) makes a scalar `osgs_tolerance` override inert in the MMS
  harness regardless.

**Removing both knobs is number-neutral** (the coupled solve already runs at
`newton_iterations`/`ftol`; removal changes nothing in the solve, only the config/schema/doc).
Endorsed.

---

## 2. Refinements to the plan

### R1 (consistency — fix before running) — clean *all* now-dead JSON keys in the same stage, not as a follow-up

The plan's gating rule says "treat any tier-warning message as a failure, not noise," but defers
the `~25 test data/*.json` `"accelerator"` cleanup (B.1) to a follow-up sweep and similarly leaves
the test-JSON copies of `osgs_iterations`/`osgs_tolerance` (Part C) and
`osgs_plateau_machine_floor_shortcut` (B.3) for later. These two statements conflict: once the
**schema** drops those keys, `_check_unknown_keys` `@warn`s on every test JSON that still carries
them (`config.jl` `_check_unknown_keys` is non-fatal but *does* warn), so the very next Extended
run trips the gating rule.

**Fix:** in whichever stage removes a key from the schema, remove it from **all** JSON in the same
commit — `config/base_config.json`, the sweep configs, **and** the `test/extended/**/data/*.json`.
Use one grep checklist per key:

```bash
grep -rln '"accelerator"'                         --include='*.json' .   # B.1
grep -rln '"osgs_iterations"\|"osgs_tolerance"'   --include='*.json' .   # Part C
grep -rln '"osgs_plateau_machine_floor_shortcut"' --include='*.json' .   # B.3
```

Because three of the stages (B.1, B.3, Part C) all touch the same `data/*.json` files, batch the
JSON edits: do the *.jl/struct/schema removals per their stages, but make one consolidated JSON
sweep that strips all three key families at once, then re-parse every JSON through the strict
loader and confirm zero new warnings. This keeps "warnings = failure" honest at every gate.

### R2 (sequencing) — do the MMS decoupling *before* the file split, so `solve_osgs_stage!` is authored once

The plan's Stage 3 splits the file and authors `solve_osgs_stage!` with an `mms_cfg` parameter.
The MMS decoupling (§3 below) then changes that signature to take an observer instead of
`mms_cfg`, rewriting `solve_osgs_stage!` a second time and re-touching the orchestrator's two MMS
branches.

**Fix:** insert the MMS decoupling as a stage **between Part C (Stage 2) and the file split.**
Renumbered sequence:

1. Stage 1 — dead code (Part B) — unchanged.
2. Stage 2 — Part C removal — unchanged.
3. **Stage 3 (NEW) — MMS decoupling** (§3). Operates on the still-single `porous_solver.jl`;
   extracts MMS into one verifier file behind a no-op-default seam. Verify with Extended.
4. Stage 4 — file split (was Stage 3). Now a **pure relocation** of MMS-free code;
   `solve_osgs_stage!` is assembled once in its final (observer) form. The "diff is pure
   relocation + orchestrator collapse" property the plan wants is *preserved* because the MMS
   content is already gone before the move.
5. Stage 5 — names/comments + merit de-dup (was Stage 4).
6. Stage 6 — doc sync (was Stage 5).

Doing MMS first is preferable to merging it into the split: merging would break the split's
"pure relocation" guarantee (the safety property that makes the split's Extended gate
interpretable).

### R3 (observation to file — NOT part of this cleanup) — confirm no production OSGS config runs short

Part-C removal is number-neutral, but it makes the OSGS coupled solve permanently share
`newton_iterations` with the Stage-I ASGS boot, with **no** independent budget. The baseline shows
the coupled solve needs **30–104** steps. The MMS harness sets `newton_iterations = 150` (fine),
but `base_config.json` sets `newton_iterations = 20`.

**Action:** check `base_config.json`'s `stabilization.method` (and any production OSGS config). If
any production config runs OSGS at `newton_iterations` well below ~100, the coupled solve is
already truncating today (this is pre-existing, not introduced by the plan). File a separate
ticket; do **not** address it in this cleanup — the fix is not resurrecting `osgs_iterations` (the
old magnitude is an outer-loop count, wrong for an inner cap) but, if independent budgeting is
ever wanted, a *new* appropriately-sized coupled-solve budget distinct from both. Out of scope
here; flagged so it is not lost.

### R4 (safety constraint on the MMS extraction) — relocate verbatim, do not "improve"

The current ASGS plateau loop uses a **raw** `solve!` with a `try/catch` for the legacy
max-iters exception (`_run_asgs_mms_extension!` lines 539–557), *not* `safe_fe_solve!`. When you
move it into the verifier (§3), relocate that body **verbatim** — same raw `solve!`, same
try/catch, same `nls_cache.result` extraction, same per-cycle `diag_cache` writes, same plateau
arithmetic and `eps_*` h-scaling (lines 568–604). Do **not** route it through `safe_fe_solve!`
during this refactor: `safe_fe_solve!` adds a finite-check + backup restore that would change
behavior. The required outcome is **bit-identical MMS plateau results**; refactoring the loop's
internals is a separate exercise.

---

## 3. MMS-decoupling design

### 3.1 Goal and where MMS currently bleeds into the solver

The author wants the MMS-specific logic out of the main algorithm body, via either (a) MMS-specific
solver variants or (b) abstract hooks the MMS layer fills in. **Recommendation: (b), an observer
seam with a no-op default.** Rationale below (§3.5).

MMS currently appears in `porous_solver.jl` in four places, all reading a duck-typed `mms_cfg`:

| Site | Lines | What it does |
|---|---|---|
| `solve_system` kwarg + `_mms_plateau()` | 804, 885–886 | accepts `mms_cfg`; derives the returned `mms_plateau_success` from `diag_cache["mms_plateau_reached"]` |
| ASGS branch | 896–911 | `if mms_cfg.enabled` → `_run_asgs_mms_extension!`; else plain exit |
| `_run_asgs_mms_extension!` | 503–616 | **the plateau loop**: extra 1-iter Newton cycles, `mms_cfg.oracle`, consecutive-pass detection, rate check, `diag_cache["mms_*"]` writes |
| OSGS tail (inside `_run_osgs_relaxation!`) | 775–784 | single `mms_cfg.oracle` eval on the coupled result + `mms_stop_reason`/rate flag (**no** stepping) |

Plus the already-dead `eff_osgs_iters += mms_cfg.max_extra_cycles` (684–687), which Stage 2/Part-C
deletes anyway.

`mms_cfg` is **not** a struct — the harness builds it as a NamedTuple
(`run_test.jl` lines 359–367) with fields:
`enabled, oracle, max_extra_cycles, require_consecutive_passes, tau_err,
eps_u_l2, eps_u_h1, eps_p_l2, h_local, kv, rate_check_factor`.
Every one of these is MMS-only. The `oracle` is the irreducibly MMS-specific capability — it knows
the manufactured exact solution; the solver does not and must not.

**Blast radius (verified):** `mms_cfg` is referenced in exactly three files — `porous_solver.jl`
and the two harnesses `test/extended/ManufacturedSolutions/run_test.jl` and
`test/extended/CocquetFormMMS/run_test.jl`. Of the ~15 `solve_system` call sites, **only those two
pass `mms_cfg`**; the rest (production `run_simulation.jl`, the Cocquet `run_convergence.jl`
drivers, `osgs_orthogonality_quick_test.jl`, the equilibration probe) call it without `mms_cfg`
and will inherit the no-op default unchanged.

### 3.2 The seam: an observer with a no-op default

Define, in the **shared-core section of `asgs_solver.jl`** (it is part of the core's public
interface, and must be parsed before the core signatures that mention it):

```julia
abstract type SolutionVerifier end

"""Production default: the solver runs no post-convergence verification."""
struct NoVerification <: SolutionVerifier end

# No-op hooks (the core only ever calls THESE; dispatch makes them free for production).
on_asgs_converged!(::NoVerification, x, step_once!, diag, iter_count_ref) = nothing
on_osgs_converged!(::NoVerification, x, diag) = nothing
verification_result(::SolutionVerifier) = nothing   # generic; MMS overrides
```

`solve_system`'s signature changes from `; mms_cfg=nothing` to
`; verifier::SolutionVerifier = NoVerification()`. The 5-tuple **return shape is unchanged** — the
second element becomes `verification_result(verifier)`, which is `nothing` for `NoVerification`.
**This is the key low-churn decision:** because the tuple shape and the `nothing`-for-production
semantics are preserved, none of the ~15 destructuring sites
(`success, _mms_plateau_unused, final_x0, … = solve_system(...)`) change.

### 3.3 The two hook points in the orchestrator

The orchestrator calls a hook at each convergence point and is otherwise MMS-blind. It never names
MMS, never reads an oracle, never writes an `mms_*` key.

**ASGS branch** (replaces lines 896–911):

```julia
if method == "ASGS"
    # `step_once!` advances the converged ASGS state by one Newton iteration on the SAME
    # operator the boot used, and returns the SafeSolverResult (iters, residual_norm, …).
    # Built here because operator construction lives in the core; the verifier owns the loop.
    local_solver = FESolver(_with_overrides(solver_newton.nls; max_iters = 1))
    step_once! = () -> _solve_one_step!(final_x0, local_solver, op_newton_init)

    eval_time += @elapsed on_asgs_converged!(verifier, final_x0, step_once!,
                                             diag_cache, iter_count_ref)
    diag_cache["base_convergence_reached"] = true        # core key, set regardless
    diag_cache["pi_u"] = pi_u; diag_cache["pi_p"] = pi_p
    return success, verification_result(verifier), final_x0, iter_count_ref[], eval_time
end
```

- `_solve_one_step!` is a tiny core helper wrapping the existing raw-`solve!` + tuple-unwrap +
  `nls_cache.result` extraction (the body currently at `_run_asgs_mms_extension!` 539–544). For
  `NoVerification` the closure is never invoked, so it costs nothing beyond one allocation of the
  closure object per solve (negligible next to a nonlinear solve; if even that is unwanted, guard
  with `verifier isa NoVerification || (…)`, but pure dispatch is cleaner).
- The cosmetic non-MMS `diag_cache["mms_stop_reason"] = "base_convergence_only"` (line 906) — an
  MMS-named key written on the *non*-MMS path — is dropped from the core. If any consumer reads
  it, have `NoVerification` set it; otherwise delete it (it is diagnostic only).

**OSGS branch** (replaces the `if mms_cfg.enabled` block at OSGS-tail lines 775–784, inside what
becomes `solve_osgs_stage!`):

```julia
# after the coupled solve + the final self-consistent π projection:
diag_cache["base_convergence_reached"] = true          # core key (was line 773)
diag_cache["base_convergence_outer_iter"] = 1          # core key (was line 774)
on_osgs_converged!(verifier, final_x0, diag_cache)      # MMS: oracle eval + rate flag; no stepping
```

`on_osgs_converged!` takes **no** `step_once!` — the coupled solve is a single solve, so the OSGS
verification is stateless (one oracle eval + a suboptimal-rate flag). This asymmetry in the two
hook signatures honestly reflects the asymmetry already in the code (ASGS plateaus by iterating;
OSGS does not).

> The OSGS verification is trivial enough that it *could* instead be done post-hoc in the harness
> (the harness already has the returned state and the oracle), eliminating `on_osgs_converged!`
> entirely. Recommend keeping the hook for symmetry and to keep the `mms_*` diag-key contract in
> one file — but note the post-hoc alternative if even-smaller core surface is wanted.

### 3.4 The MMS verifier (new file `src/solvers/mms_verification.jl`)

A concrete `SolutionVerifier` holding the former `mms_cfg` fields as a real struct, implementing
the two hooks. Included **after** `asgs_solver.jl` and `osgs_solver.jl` (it is referenced only by
harnesses, never by core signatures, so its position is free as long as the abstract type from
§3.2 precedes it).

```julia
Base.@kwdef mutable struct MMSPlateauVerifier{F} <: SolutionVerifier
    oracle::F                              # (uh, ph) -> (E_u_L2, E_p_L2, E_u_H1, E_p_H1)
    max_extra_cycles::Int
    require_consecutive_passes::Int
    tau_err::Float64
    eps_u_l2::Float64; eps_u_h1::Float64; eps_p_l2::Float64
    h_local::Float64; kv::Int
    rate_check_factor::Float64
    # results (the verifier OWNS its outcome; no longer parasitic on diag_cache):
    plateau_reached::Union{Bool,Nothing} = nothing
    stop_reason::String = ""
    error_history::Vector = []
    relative_change_history::Vector = []
end

verification_result(v::MMSPlateauVerifier) = v.plateau_reached

function on_asgs_converged!(v::MMSPlateauVerifier, x, step_once!, diag, iter_count_ref)
    # EXACT relocation of _run_asgs_mms_extension!'s body (R4), with two substitutions:
    #   * the inline `solve!(final_x0, local_fesolver, op_newton_init)` becomes `step_once!()`
    #   * results land in v.* fields; mirror into diag["mms_*"] only if the harness still reads them
    ...
end

function on_osgs_converged!(v::MMSPlateauVerifier, x, diag)
    # EXACT relocation of the OSGS-tail block (lines 776–783):
    #   E_u2, _, _, _ = v.oracle(x...); v.plateau_reached = true
    #   v.stop_reason = E_u2 > v.rate_check_factor * v.h_local^(v.kv+1) ?
    #                       "coupled_at_suboptimal_rate" : "coupled_single_solve"
end
```

**Results channel.** Cleanest is for the verifier to own its results (`v.plateau_reached`,
`v.error_history`, …) and for the harness to read `v.*` after `solve_system` returns — this fully
removes the `diag_cache["mms_*"]` keys from the solver's concern. If you prefer **zero** harness
read-site churn, have the verifier *also* write the legacy `diag_cache["mms_*"]` keys (the harness
keeps reading them). Recommend verifier-owned fields; the harness edit is small and contained.

### 3.5 Why hooks (b), not MMS solver variants (a)

Option (a) — a separate `solve_system_mms` — must either duplicate the whole orchestration (Stage I
boot + dispatch), or call `solve_system` and then continue iterating; but continuing requires the
ASGS operator and a 1-iter solver, which `solve_system` builds internally and does not return. So
(a) either duplicates orchestration (drift risk) or needs the *same* seam as (b) plus a leak of
solver internals into the return value. Option (b) has none of that: no duplication, no MMS in the
core, idiomatic multiple-dispatch, and it matches the theory doc's existing framing of MMS as an
**optional extension** (Algorithm D, typeset separately) — the doc and code converge.

### 3.6 Harness changes (the only call-site edits)

In each of the two harnesses (`ManufacturedSolutions/run_test.jl`, `CocquetFormMMS/run_test.jl`):

1. Replace the `mms_cfg = ( … NamedTuple … )` construction with
   `verifier = PorousNSSolver.MMSPlateauVerifier(oracle=…, max_extra_cycles=…, …)` (same field
   values), and `verifier = PorousNSSolver.NoVerification()` where `mms_cfg` was `nothing`.
2. Change the call `solve_system(…; mms_cfg=mms_cfg)` to `solve_system(…; verifier=verifier)`.
3. Read results from `verifier.*` (or keep reading `diag_cache["mms_*"]` if §3.4's mirror option
   is taken). The destructured return tuple is unchanged.

No other call site touches `mms_cfg`, so no other harness/driver/test changes.

### 3.7 Verification for the MMS-decoupling stage

This stage must be **behaviour-preserving for MMS**, so its gate is the Extended tier:

- Blitz + Quick must stay green (the seam does not touch the kernel or any safeguard; `mms_cfg` is
  absent from blitz/quick — confirmed — so those tiers only confirm no collateral breakage).
- Run `test/extended/ManufacturedSolutions/run_test.jl` (`phase1_quad_k1.json`) and
  `test/extended/CocquetFormMMS/run_test.jl`; diff against `docs/mms/convergence-baseline.md`:
  per-cell H¹(u) rates, OSGS iteration ranges, **and** the MMS plateau outcomes
  (`plateau_reached`, `stop_reason`, error/relative-change histories). All must be **identical**
  to pre-refactor — the loop body was relocated verbatim (R4), only its call site and result
  channel moved.
- Red flag: any change in plateau `stop_reason`, cycle count, or `mms_error_history` ⇒ the
  relocation was not verbatim.

### 3.8 Doc note (folds into Stage 6 / Part F)

In `theory/osgs_algorithm.tex`, Algorithm D ("VerifyMMSPlateau") is already a separate, separately
-coloured extension. Add one sentence to its preamble: it is now implemented as a pluggable
`SolutionVerifier` (`MMSPlateauVerifier`) that the orchestrator invokes through the
`on_asgs_converged!` / `on_osgs_converged!` hooks; the production path uses `NoVerification`. Add
`mms_verification.jl` to the §"Module Hierarchy" file map (Algorithm D → `mms_verification.jl`).

---

## 4. Net effect on the plan's stage list

```
Stage 1  Dead code (Part B)                         [unchanged]
Stage 2  Part C removal                             [unchanged] + R1 JSON sweep
Stage 3  MMS decoupling (NEW, §3)                   [Extended gate]   ← inserted (R2)
Stage 4  File split (asgs_solver.jl + osgs_solver.jl)  [Extended gate; now pure relocation]
Stage 5  Names / comments / merit de-dup (D, E.1)   [unchanged]
Stage 6  Theory-doc sync + ledger (Part F)          [+ Algorithm D / mms_verification.jl note]
```

Everything in the plan's "Explicitly NOT doing" list stands. The MMS decoupling adds **no** change
to any safeguard control flow: the Newton loop, `eval_safeguard_termination_bounds!`, the cascade,
and the ping-pong are untouched; only the *post-convergence* verification layer is relocated behind
a default-off seam.
