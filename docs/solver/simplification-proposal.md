# Solver simplification proposal (`src/solvers/`)

**Status:** specification for an implementing AI. Nothing here is applied yet.
**Scope:** primarily `src/solvers/`, with the unavoidable spillover into `src/config.jl`,
`src/PorousNSSolver.jl`, the JSON config/schema, and `theory/osgs_algorithm.tex`.

**How to read this document.** Every claim is grounded in a specific file/line so you can
re-verify it before touching anything. Where a change has a numerical consequence (Part C in
particular), the proposal states the *fact* and the *options* rather than asserting a single
"correct" answer — the author (an expert in this method) makes the final call. Verify first,
edit second.

A reproduction of the audit greps is collected in the final **Verification** section.

---

## 0. Executive summary

The `solvers/` folder works and is, on the whole, well-organised. The complexity the author
senses is real but localised, and it has three distinct sources, which this proposal treats
separately:

1. **No ASGS/OSGS file boundary.** `porous_solver.jl` (974 lines) holds *everything*: the
   shared Newton-cascade infrastructure, the ASGS Stage-I boot, the ASGS-MMS extension, the
   OSGS coupled solve, and the orchestrator. The requested split is feasible and clean
   (Part A).

2. **Dead code and dead config left by the 2026-06-08 "coupled-only leaning."** The leaning
   removed the staggered OSGS machinery but left behind: an entire unused acceleration
   subsystem, several dead locals/parameters inside the OSGS function, two dead config knobs,
   two dead `CholeskyNumericalSetup` methods, and two dead `_with_overrides` keywords
   (Part B).

3. **One genuine code↔doc divergence that is probably a latent bug:** the coupled OSGS solve
   silently ignores `osgs_iterations` and `osgs_tolerance` and instead runs at the *Newton*
   budget/tolerance. The theory doc says the opposite (Part C).

On top of those, Part D (comments/naming), Part E (small structural/efficiency simplifications)
and Part F (theory-doc structure) collect the rest.

The single most important item to resolve before anything else is **Part C** — decide what the
coupled OSGS solve is *supposed* to do, because that decision determines whether
`osgs_iterations`/`osgs_tolerance` are kept-and-wired or removed-as-vestigial, which in turn
changes Part B.

---

## Part A — Split ASGS and OSGS into two files

### A.1 What is actually in `porous_solver.jl` today

Function/struct inventory, with the natural target file for each. "Shared core" means it is
method-agnostic and is used by *both* the ASGS Stage-I boot and the OSGS coupled solve.

| Lines | Symbol | Nature | Target |
|------:|--------|--------|--------|
| 33–39 | `inner_projection_u` | OSGS — L² projection of the momentum strong residual | **OSGS** |
| 41–45 | `inner_projection_p` | OSGS — L² projection of the mass strong residual | **OSGS** |
| 47–63 | `check_porous_solver_parameters` | validation (touches `osgs_*` too) | core |
| 66–79 | `discrete_l2_projection` (6-arg, cached) | OSGS — hot-path mass solve | **OSGS** |
| 81–93 | `discrete_l2_projection` (4-arg, self-factoring) | OSGS — diagnostic, test-only | **OSGS** |
| 95–107 | `struct FETopology` | shared | core |
| 109–113 | `struct VMSFormulation` | shared | core |
| 115–118 | `struct IterativeSolvers` | shared (rename — see D.4) | core |
| 120–178 | `safe_fe_solve!` | shared — single-attempt FE-solver wrapper | core |
| 180–199 | `_record_stage!` | shared — trajectory diagnostics | core |
| 225–260 | `CascadePolicy` + 3 consts + `cascade_step_outcome` | shared | core |
| 262–343 | `_pingpong_cascade!` | shared — Newton↔Picard scheduler | core |
| 345–501 | `_initialize_asgs_state!` | **ASGS** — Stage I (Algorithm B) | **ASGS** |
| 503–616 | `_run_asgs_mms_extension!` | **ASGS** — Algorithm D (ASGS branch) | **ASGS** |
| 618–789 | `_run_osgs_relaxation!` | **OSGS** — Stage II (Algorithm C) | **OSGS** |
| 791–975 | `solve_system` | orchestrator (dispatches to both) | core |

Verified call-graph facts that make the split clean:

- The OSGS code's *only* dependencies on "core" are `safe_fe_solve!`, `_record_stage!`,
  `cascade_step_outcome`, `_pingpong_cascade!`, `OSGS_INNER_POLICY`, the three structs, and
  `_with_overrides` (the last lives in `nonlinear.jl`, not here). All are method-agnostic.
- `inner_projection_u/p` and `discrete_l2_projection` are used only by the OSGS path and by
  `test/quick/osgs_orthogonality_quick_test.jl` (which calls them as `PorousNSSolver.…`, so a
  same-module move does not break it).
- Nothing in the ASGS path calls anything OSGS-specific.

### A.2 Target layout (primary recommendation: two files)

Faithful to the request — ASGS is the base, OSGS the extension that uses it. Put the shared
core in the ASGS file under a clearly fenced section so the "base" file reads as
self-contained.

```
src/solvers/
  linear_solvers.jl     (unchanged except D.4 / B.4)
  nonlinear.jl          (the shared Newton kernel — UNCHANGED structurally; see note below)
  asgs_solver.jl        (NEW name for porous_solver.jl): shared core + ASGS + orchestrator
  osgs_solver.jl        (NEW): OSGS projections + coupled Stage-II solve
  # accelerators.jl     DELETED (Part B.1)
```

`asgs_solver.jl` internal section order:
1. `# ===== Shared solver core =====` — structs, `safe_fe_solve!`, `_record_stage!`,
   `CascadePolicy`/`cascade_step_outcome`, `_pingpong_cascade!`, `check_porous_solver_parameters`.
2. `# ===== Stage I — ASGS algebraic boot (Algorithm B) =====` — `_initialize_asgs_state!`.
3. `# ===== ASGS MMS plateau extension (Algorithm D) =====` — `_run_asgs_mms_extension!`.
4. `# ===== Orchestrator (Algorithm O) =====` — `solve_system`.

`osgs_solver.jl`:
1. `inner_projection_u`, `inner_projection_p`
2. `discrete_l2_projection` (both methods)
3. `solve_osgs_stage!` (the merged OSGS entry point — see A.3)

> **Note on `nonlinear.jl`.** Do **not** split it. It is the *shared* Newton kernel
> (Algorithm A + leaves A.1–A.3 + `SafeNewtonSolver` + `build_iter_solvers`). It belongs to
> neither method exclusively; both stages drive it. Splitting it would invert the dependency the
> author asked for. Leave its structure intact (its comments get the Part D treatment).

### A.3 Move *all* OSGS setup out of the orchestrator into one OSGS entry point

Right now `solve_system`'s `method == "OSGS"` branch (lines 913–968) itself builds the
projection spaces, assembles + Cholesky-factors `M_u`/`M_p`, then calls `_run_osgs_relaxation!`.
That setup is OSGS-specific and should not live in the orchestrator. Fold the whole branch body
into a single function in `osgs_solver.jl`:

```julia
# osgs_solver.jl
function solve_osgs_stage!(success, final_x0, setup::FETopology, formulation::VMSFormulation,
                           config::PorousNSConfig, solver_newton, freeze_cusp, mms_cfg,
                           diag_cache, iter_count_ref)
    # 1. Build unconstrained projection spaces V_proj/Q_proj/U_proj/P_proj
    #    (the V_free/Q_free fallback currently at lines 935–938).
    # 2. Assemble M_u, M_p and Cholesky-factor once (lines 944–955).
    # 3. Run the coupled solve (the current body of _run_osgs_relaxation!, lines 663–788).
    # 4. Return (success, pi_u, pi_p, elapsed).
end
```

Then the orchestrator's OSGS branch collapses to:

```julia
if method == "OSGS"
    (success, pi_u, pi_p, osgs_elapsed) =
        solve_osgs_stage!(success, final_x0, setup, formulation, config,
                          solver_newton, freeze_cusp, mms_cfg, diag_cache, iter_count_ref)
    eval_time += osgs_elapsed
    return success, _mms_plateau(), final_x0, iter_count_ref[], eval_time
end
```

This is the *only* OSGS-aware code left in the ASGS/core file: one `solve_osgs_stage!` call plus
the `method == "OSGS"` guard. That satisfies "as little [OSGS] as possible."

Effect on `_run_osgs_relaxation!`'s signature: it currently takes a long, partly-dead argument
list (see B.2). When you fold it into `solve_osgs_stage!`, derive `X, Y, dΩ, h_cf, …` from
`setup` inside the function instead of threading them, and drop the dead parameters entirely.

### A.4 Julia mechanics — include order and forward references (read carefully)

Update `src/PorousNSSolver.jl` (currently lines 27–32):

```julia
# Solvers
include("solvers/linear_solvers.jl")
include("solvers/nonlinear.jl")
include("solvers/asgs_solver.jl")   # defines the structs + solve_system
include("solvers/osgs_solver.jl")   # defines solve_osgs_stage!
# (delete the accelerators.jl include — Part B.1)
include("metrics.jl")
```

Two distinct reference directions, with **different** evaluation timing — this is the subtle part:

- **`solve_system` (asgs file) → `solve_osgs_stage!` (osgs file).** This reference sits *inside a
  function body*, so it is resolved at **call time**, not at include time. By the time anything
  calls `solve_system`, both files are loaded. ✅ Safe regardless of include order. (This is the
  ordinary way mutually-referential functions span files in one module.)

- **`solve_osgs_stage!`'s signature annotates `setup::FETopology`, `formulation::VMSFormulation`.**
  Type annotations in a method signature are resolved when the `function` is **evaluated during
  include**, *not* at call time. Therefore `FETopology`/`VMSFormulation` must already exist when
  `osgs_solver.jl` is parsed. ⇒ **`asgs_solver.jl` must be included before `osgs_solver.jl`.**
  This is a hard ordering constraint, not a preference.

If you prefer to make the two solver files order-independent, the alternative is to hoist the
three structs into `nonlinear.jl` (or a tiny new `solver_types.jl` included first). The
two-file recommendation above does not require that; just keep ASGS before OSGS.

> **Do not** drop the explicit `::FETopology` / `::VMSFormulation` annotations to dodge the
> ordering — they are doing useful dispatch/clarity work and the ordering constraint is trivial
> to honour.

### A.5 Optional refinement: a third "core" file

If, after the split, the author wants the ASGS file to contain *zero* shared machinery (literal
reading of "without any of the OSGS stuff"), promote section 1 of `asgs_solver.jl` to a
`solver_core.jl` included first. Then: `solver_core.jl` (structs + cascade + `safe_fe_solve!`),
`asgs_solver.jl` (Stage-I + MMS + `solve_system`), `osgs_solver.jl` (OSGS). The orchestrator
still lives with ASGS because Stage I *is* ASGS. This is cleaner separation at the cost of one
extra file and departs slightly from the literal "OSGS uses ASGS" phrasing (it becomes "both use
core"). Recommend the two-file version unless the author wants the stricter separation.

### A.6 Rename cost

`porous_solver.jl` is named in 16 files, but only `PorousNSSolver.jl` `include`s it; the rest are
docstrings/comments/markdown (`docs/…`, `CLAUDE.md`, and **comments** in
`run_test.jl`/`cascade_policy_symmetry_blitz_test.jl`). No test `include`s it directly (tests do
`using PorousNSSolver`). So the functional rename cost is one line in `PorousNSSolver.jl`; the
rest is a doc/comment find-replace `porous_solver.jl → asgs_solver.jl` (plus adding
`osgs_solver.jl` where appropriate). If the author would rather not touch the docs, an acceptable
fallback is to **keep the name `porous_solver.jl`** for the core+ASGS+orchestrator file and only
*add* `osgs_solver.jl`; the split is identical, only the base file's name differs.

---

## Part B — Legacy / dead code to remove

Each item below was verified to have **no live consumer** (greps in the Verification section).

### B.1 The Anderson acceleration subsystem — fully dead

`AndersonAccelerator` is **never instantiated anywhere** in `src/` or `test/`. The matching
config struct is plumbed through validation and schema but read by nothing.

Delete, in order:

1. `src/solvers/accelerators.jl` — the whole file (95 lines).
2. `src/PorousNSSolver.jl` line 30: `include("solvers/accelerators.jl")`.
3. `src/config.jl`:
   - `struct AcceleratorConfig` (lines 33–38).
   - `accelerator::AcceleratorConfig` field in `SolverConfig` (line 100).
   - `StructTypes.StructType(::Type{AcceleratorConfig}) = …` (line 127).
   - the three `@assert sol.accelerator.*` in `validate!` (lines 161–163).
   - the `accelerator` branch in `_check_unknown_keys_hierarchical` (lines 210–212).
4. **JSON side** (not in the repomix snapshot, but the file exists — confirmed via
   `lessons_learned.md`): remove the `"accelerator": { … }` block from
   `config/base_config.json`, the two sweep configs, and the `accelerator` property block in
   `config/porous_ns.schema.json` (brace-match carefully, then re-validate the JSON). Grep the
   repo for `"accelerator"` to find every occurrence.

Safety: `_check_unknown_keys` only `@warn`s on unknown keys (config.jl line 184) — it does not
throw — so a config that still carries an `accelerator` block during a partial rollout will warn,
not crash. This makes the removal order-insensitive.

> The theory doc already says OSGS uses "no Anderson acceleration" (`osgs_algorithm.tex` line
> 1273). Removing the dead subsystem makes code and doc agree.

### B.2 Dead locals and parameters in the OSGS function

Inside `_run_osgs_relaxation!` (to become `solve_osgs_stage!`):

- **Dead locals** (assigned, never read):
  - `osgs_tol = stab_cfg.osgs_tolerance` (line 673) — never used after assignment.
  - `eff_osgs_iters` (lines 684–687) — assigned and incremented, never read.
  - `max_osgs_iters` (line 674) — used only to seed the dead `eff_osgs_iters`; becomes dead once
    `eff_osgs_iters` is removed. (It is also re-derived and used only for a log string at line
    926 in the orchestrator — keep that single use if you keep the log line, otherwise drop.)
- **Dead parameters** (0 uses in the body): `V_free`, `Q_free`, `ftol`, `stagnation_tol`. (The
  two textual hits for `ftol` in the body are a comment and the *keyword name* `ftol=` in a
  `_with_overrides` call — not the parameter.)

**Caveat — read Part C first.** `osgs_tol`/`max_osgs_iters` being dead is *itself* the symptom of
the Part-C divergence. If Part C is resolved by **wiring** `osgs_iterations`/`osgs_tolerance` into
the coupled solve, then `max_osgs_iters` and `osgs_tol` stop being dead — they get *used*. So do
Part C before deleting these two specifically. `eff_osgs_iters`, `V_free`, `Q_free`,
`stagnation_tol` are dead under either resolution and can go regardless.

### B.3 The `osgs_plateau_machine_floor_shortcut` config knob — dead

Declared in `SolverConfig` (config.jl line 96) with a comment pointing at
`_run_osgs_relaxation!`. After the coupled-only leaning that function has no consecutive-pass
plateau loop to short-circuit (the only `require_consecutive_passes` logic that survives is in
the **ASGS** MMS extension, `_run_asgs_mms_extension!` line 587). The knob is read by nothing.

Remove the field from `SolverConfig`, and its JSON-side counterpart in `base_config.json` and the
schema. (No `@assert` references it, so nothing else to touch in `validate!`.)

### B.4 Dead `CholeskyNumericalSetup` methods

`linear_solvers.jl` defines two `numerical_setup!` (in-place refactor) methods for
`CholeskyNumericalSetup` (lines 39–42 and 44–56). The OSGS mass matrices are assembled and
factored exactly once (orchestrator lines 944–955) via `numerical_setup` (no bang) and then
reused read-only through `solve!`. **`numerical_setup!` is never called on a
`CholeskyNumericalSetup`** anywhere in `src/` or `test/`.

Recommendation: delete both `numerical_setup!(::CholeskyNumericalSetup, …)` methods. They are
defensive scaffolding for an in-place re-factorisation path the code does not use; keeping them
implies a re-assembly story (the comment at lines 44–56 even narrates an "OSGS re-assembly
fast-path") that no longer exists. If the author wants to retain the capability for future use,
keep them but rewrite the comment to say "currently unused; provided for in-place refactor if
mass matrices ever become state-dependent."

(Note: `ILUGMRESSolver` and its `numerical_setup!` are also unused inside the repo, but
`ILUGMRESSolver` is **exported** (`PorousNSSolver.jl` line 47) as a public alternative linear
solver — keep it.)

### B.5 Dead `_with_overrides` keywords

`_with_overrides` (nonlinear.jl, signature at lines 109–111) accepts seven keywords. Call-site
audit shows only five are ever passed: `max_iters`, `ftol`, `mode`, `picard_gain_target`,
`stall_window`. **Never passed:** `relative_ftol_per_field` and `stall_min_rel_improvement`.

The `relative_ftol_per_field` override even carries a comment ("covariant OSGS warmup
relaxation") referencing the *removed* OSGS warmup. Remove both unused keywords and the two
inheritance lines that handle them (lines 126–127 for `relative_ftol_per_field`; line 134 for
`stall_min_rel_improvement`). This shrinks the helper to exactly the overrides in use.

(Leave `build_iter_solvers`' richer keyword set alone — the MMS harness, `run_test.jl:943`,
exercises it.)

---

## Part C — `osgs_iterations` / `osgs_tolerance` are silently ignored (decision required)

### C.1 The fact

The coupled OSGS solve runs with the **Newton** budget and the **Newton** ftol, not the OSGS
ones:

- `base_nls = solver_newton.nls` (line 678). `solver_newton` is built by `build_iter_solvers`
  with `newton_max_iters = sol_cfg.newton_iterations` and `newton_ftol = sol_cfg.ftol`
  (run_simulation.jl lines 365–369). So `base_nls.max_iters == newton_iterations` and
  `base_nls.ftol == ftol`.
- `coupled_nls = _with_overrides(base_nls; stall_window=0)` (line 738) overrides **only**
  `stall_window`. ⇒ `coupled_nls.max_iters == newton_iterations`, `coupled_nls.ftol == ftol`.
- Both the non-ping-pong path (line 755, `FESolver(coupled_nls)`) and the ping-pong path
  (line 750, Newton segments use `FESolver(coupled_nls)`) therefore cap at `newton_iterations`
  and converge at `ftol`.
- `stab_cfg.osgs_iterations` and `stab_cfg.osgs_tolerance` are read into the dead locals
  `max_osgs_iters`/`eff_osgs_iters`/`osgs_tol` (B.2) and **never reach the solver.**

Meanwhile `config.jl` (`validate!` lines 168–169) and `check_porous_solver_parameters` (lines
57–62) both *require* `osgs_iterations ≥ 1` and `osgs_tolerance > 0`. The user must supply two
parameters that have **zero effect** on the run.

### C.2 What the documentation says

`theory/osgs_algorithm.tex`, Table `tab:params` (line 287):

> *N_OSGS — `stab.osgs_iterations` — Inner-Newton iteration cap for the single coupled OSGS
> solve (Alg. C) …*

So the *documented contract* is that `osgs_iterations` caps the coupled Newton (and the table's
surrounding text treats `osgs_tolerance` as its convergence target). The code does not honour
this. This is a real divergence, and given the comments around it (the coupled solve "converges
slowly-monotone … linear rate"), it is most likely a **bug**: a linearly-converging coupled solve
typically needs a *larger* iteration budget than the quadratically-converging ASGS Newton boot,
which is exactly why a separate `osgs_iterations` knob exists. Running it at `newton_iterations`
may be silently under-resolving OSGS on stiff cases (and, combined with `OSGS_INNER_POLICY`
accepting soft stalls, returning a not-fully-converged state as "success").

### C.3 The two resolutions

**Option (a) — Honour the doc (recommended).** Wire the OSGS knobs into the coupled solve:

```julia
coupled_nls = _with_overrides(base_nls;
                              stall_window = 0,
                              max_iters    = stab_cfg.osgs_iterations,
                              ftol         = stab_cfg.osgs_tolerance)
```

This makes `max_osgs_iters`/`osgs_tol` live (so B.2 keeps them), restores an independently
tunable OSGS depth, and makes code match `tab:params`. **Numerical-behaviour change:** results on
configs where `osgs_iterations ≠ newton_iterations` or `osgs_tolerance ≠ ftol` will move. Before
committing, run the MMS and Cocquet harnesses and compare convergence-rate tables (this is a
`src/`-level numerical change, exactly the kind the harnesses exist to guard).

> Subtlety to decide alongside (a): should `osgs_tolerance` override the **per-field** gate too?
> Today the per-field relative tolerances (`relative_ftol_per_field`) come from the harness and
> set the *effective* gate; a scalar `ftol` override is only a floor (nonlinear.jl
> `_initialize_effective_thresholds`, lines 315–345). If OSGS is meant to converge to a *looser*
> target than the per-field gate, a scalar override alone will not loosen it. Clarify intent.

**Option (b) — Treat the knobs as vestigial and remove them.** If, post-leaning, the intended
design is genuinely "the coupled solve runs at the Newton budget," then `osgs_iterations` and
`osgs_tolerance` are leftovers from the staggered era (where `osgs_iterations` counted *outer*
fixed-point cycles). Remove both from `StabilizationConfig`, `validate!`,
`check_porous_solver_parameters`, the JSON/schema, and update `tab:params` to state that
Algorithm C uses the Newton budget/ftol. This is the smaller-surface option but discards a useful
tuning lever and is a bigger config/schema/harness edit (every config carries these keys).

**Recommendation:** (a). It is the smaller code change, it makes the doc true, and it gives back
control that the linear convergence rate of the coupled solve genuinely benefits from. But this
is the author's call — it changes numbers. Decide C before doing B.2/Part F.

---

## Part D — Comments and naming

### D.1 The comment cleanup philosophy

The author asked for "maximum elegance and clarity, removing as many leftovers from the process
as possible." Concretely, the solvers carry ~66 lines of process-archaeology comments
(`nonlinear.jl` ~17, `porous_solver.jl` ~26). Apply this rule:

> **A comment should explain the *current* design and the *load-bearing why*. It should not
> narrate the history that produced it.** History lives in git and in `docs/lessons_learned.md`.

A "smell → fix" table for the patterns present:

| Smell (delete or rewrite) | Examples (file:line) | Replace with |
|---|---|---|
| Refactor-brief tags `[P0] [P1] [P4] [B5] [B6] [B10] [M9] [H1] [H7]`, `refactor-brief`, `plan Fix 6 / P-007` | porous_solver 202, 411, 480; nonlinear 143; config 86–99 | nothing, or a plain-English statement of what the code does |
| "bit-identical to legacy", "today's behaviour", "matches the prior positional-ctor default byte-for-byte" | nonlinear 36, 71–72, 148; run_simulation 358–361 | drop — the reader does not know "legacy" |
| Dated leaning notes "(2026-06-08 leaning)", "(2026-06)" | porous_solver 26–27, 656, 708; config 45–46 | drop the date; keep only a present-tense design statement if needed |
| References to **removed** machinery: `staggered`, `freeze_after_k`, OSGS warmup/drift, `_run_osgs_inner_cascade!`, `_compute_state_drift`, `projection_drift` | porous_solver 25, 86, 209, 656, 697–708, 920, 957 | describe the coupled solve as it is now; do not contrast against code that no longer exists |
| "historically contrasted with the now-removed …" asides | porous_solver 363–364 | drop the aside; state the Stage-I rule directly |
| Long defensive essays on the legacy `max_iters_caught` exception path | porous_solver 130–143; osgs_algorithm.tex 900–926 | one sentence: "`SafeNewtonSolver` exits cleanly with `max_iters_stagnation`; the exception branch defends against non-`SafeNewtonSolver` solvers." |

**Keep** the comments that carry real mathematical or numerical reasoning, e.g.:
- the positive-sign convective adjoint and its anti-SUPG consequence (porous_solver/formulation;
  doc lines 538–541),
- why the projection is on the *unconstrained* space (O(1) boundary residual destroys the rate),
- why the merit divergence check uses Φ in `:newton` but ‖b‖∞ in `:picard` (nonlinear 497–510),
- why the stall sensor is disabled for the coupled solve (porous_solver 732–737).

These are load-bearing; only tighten their prose.

### D.2 One comment that is now *false* (fix, do not just trim)

`porous_solver.jl` lines 919–924, inside the OSGS branch of `solve_system`, still says OSGS is
solved "in a staggered, iterative scheme … the global FE residual projection from the previous
iteration is held fixed …". That describes the **removed** staggered method, not the current
single coupled solve. This is actively misleading. Replace with an accurate one-liner: the
coupled solve recomputes `π = Π(R(u))` from the *current* iterate at every residual evaluation
(no lag), with a local frozen-π Jacobian. (When you fold this branch into `solve_osgs_stage!` per
A.3, write the correct comment there.)

### D.3 Rename to match the theory doc (kill naming drift)

The doc's Algorithm C is **`CoupledOSGSSolve`** (`osgs_algorithm.tex` caption, line 1318). The
code still calls it the old name in several places:

- `_run_osgs_relaxation!` (function name) and its banner "Algorithm C, OSGS fractional
  relaxation" (porous_solver 619),
- docstring "Algorithm C (OSGSFractionalRelaxation)" (porous_solver 957),
- `solve_system` docstring "OSGS staggered relaxation" (porous_solver 797).

Rename the function to **`solve_osgs_coupled!`** (or fold into `solve_osgs_stage!` per A.3) and
replace every "fractional relaxation" / "staggered relaxation" label with "coupled solve,"
matching `CoupledOSGSSolve`. Update `docs/solver/algorithm-code-mapping.md` to point at the new
name.

### D.4 Rename the `IterativeSolvers` struct (name collision)

`linear_solvers.jl` line 5 does `using IterativeSolvers` (the package, for `gmres!`), while
`porous_solver.jl` line 115 defines `struct IterativeSolvers`. Both live in module
`PorousNSSolver`. It works today only because the struct shadows the package module binding and
`gmres!` is reached by its unqualified imported name — but it is fragile (any future
`IterativeSolvers.foo` qualified call breaks) and confusing.

Rename the struct to something domain-meaningful, e.g. **`StageSolvers`** (it holds the
`(picard, newton)` `FESolver` pair). Update its two uses: the constructor in `run_simulation.jl`
line 374 and the type in `solve_system`'s signature (line 802). The `IterativeSolvers` package
import is then unambiguous.

### D.5 Fix the misnamed `solution_scale_per_field`

In `nonlinear.jl`, the field `SafeIterationState.solution_scale_per_field` (line 289) and the
local of the same name (line 657) actually hold the **initial residual norm ‖R₀‖**, not a
solution-magnitude scale — the comment even admits "(Name kept for history; it now holds ‖R₀‖.)".
Rename both to `initial_residual_scale_per_field` and delete the parenthetical history note. This
is a pure clarity win with no behavioural effect (it is the encoding-invariance gate; do not
change the math, only the name).

---

## Part E — Other simplifications (smaller, optional)

### E.1 De-duplicate the merit closure

`eval_merit_W(b_vec) = 0.5 * sum(idx -> (b_vec[idx]/w[idx])^2, …)` is defined twice over the same
captured `w`: in `eval_armijo_linesearch_pass!` (nonlinear 433) and in `_safe_solve_inner!`
(nonlinear 663). Lift it to a single small helper, e.g.
`_merit_W(b, w) = 0.5 * sum(i -> (b[i]/w[i])^2, eachindex(b))`, and call it from both sites. Minor,
but removes a copy and a subtle "must keep these two in sync" hazard.

### E.2 (Optional, measure first) per-evaluation projection allocations

In the coupled residual closure (porous_solver 716–723), each residual evaluation does two
`discrete_l2_projection` calls, and the 6-arg `discrete_l2_projection` allocates a `b_vec` and a
solution vector per call (lines 75–77), as does `CholeskySolver.solve!` (line 65, `x .= F \ b`).
With several line-search trials per Newton step this is a steady allocation stream. The existing
comments correctly judge it acceptable (the inner Newton dominates), so treat this as
**optional** and only if profiling says projections matter: preallocate the two projection
load/solution buffers in `solve_osgs_stage!` and pass them into a buffer-taking projection
routine; and replace `x .= F \ b` with a buffer + `ldiv!`-style solve where the factor type
supports it. Do **not** do this speculatively — it complicates the hot path for a cost that is
likely second-order.

### E.3 (Optional) flatten the Stage-I cascade body

`_initialize_asgs_state!`'s non-ping-pong branch (lines 383–498) is a long, deeply-nested
Newton→Picard→Newton with per-state logging. It is *correct* and the logging is informative, but
once the `[B6]`/`[P3a]` tags and the "historically contrasted" asides are stripped (Part D), the
remaining structure is "Newton; if not success → restore, Picard; if diverged → fail; else Newton
again." Consider whether the three near-identical `:ok`/`:nonfinite`/`:exception`/
`:max_iters_caught` ladders can be expressed once via `cascade_step_outcome` (they were
*partially* unified already). This is a readability refactor with zero behavioural change — only
attempt it with the cascade-policy blitz tests (`cascade_policy_symmetry_blitz_test.jl`,
`pingpong_*`, `stall_guard_*`) as a safety net, and keep the verdicts bit-identical.

---

## Part F — `theory/osgs_algorithm.tex` structural changes

The doc is already well-structured (Algorithm A = shared kernel, B = ASGS cascade, C = OSGS
coupled solve, mirroring the proposed file split). Recommended changes are surgical:

1. **Reconcile `tab:params` with Part C.** This is mandatory and follows whatever C decision is
   made: either keep the `N_OSGS = osgs_iterations` row (and ensure the code now honours it,
   Option a) or rewrite the row to say Algorithm C uses the Newton budget/ftol and remove
   `osgs_tolerance` from the table (Option b). The doc and code must not disagree here.
2. **Purge the stale `max_iters_caught` essay.** Lines 900–926 spend a full page on a legacy
   exception path that "is normally never taken with the modern solver stack." Compress to two or
   three sentences (see D.1 table). The companion code comment shrinks the same way.
3. **Add a one-line file map.** In §"Module Hierarchy" (around line 233), note that the code now
   mirrors this tree: Algorithm A → `nonlinear.jl`; Algorithms B + O (shared core + ASGS) →
   `asgs_solver.jl`; Algorithm C → `osgs_solver.jl`. This makes the doc's tree a literal map of
   the source after the split.
4. **Leave the legitimate "staggered" contrasts.** Unlike the code, the doc's "staggered"
   mentions (lines 635, 680, 1265, 1288) correctly describe the *companion article's* method for
   contrast against the implemented slaved/coupled scheme — keep them. Only remove "staggered"
   where it purports to describe *this* code (none of those are in the .tex; they are all in code
   comments, Part D).
5. **JFNK section (§`sec:jfnk`).** This documents a *recovery plan* for the dropped ∂π/∂U
   coupling, not shipped code. Leave it as-is unless the author wants it explicitly marked
   "planned / not implemented" — out of scope for this cleanup.

---

## Verification — how to check each claim and each change

### Re-run the audit greps (paste-ready)

```bash
# B.1 Anderson is dead (expect: only accelerators.jl, config.jl, and docs):
grep -rln "Anderson\|AndersonAccelerator\|AcceleratorConfig\|\.accelerator" . | grep -v docs

# B.3 / osgs_plateau knob is dead (expect: only config.jl):
grep -rln "osgs_plateau_machine_floor_shortcut" . | grep -v docs

# B.4 numerical_setup! never called (expect: only the 3 definitions in linear_solvers.jl):
grep -rn "numerical_setup!" src/ test/

# B.5 _with_overrides kwargs actually passed (expect: max_iters, stall_window, ftol, mode, picard_gain_target only):
grep -rn "_with_overrides(" src/ | grep -v "function _with_overrides"

# C  coupled_nls overrides (expect: only stall_window today):
sed -n '738p' src/solvers/porous_solver.jl

# D.4 name collision (expect: package import + struct def):
grep -rn "using IterativeSolvers\|struct IterativeSolvers" src/

# A  OSGS-only symbols stay inside the OSGS file + one test:
grep -rln "inner_projection_u\|inner_projection_p\|discrete_l2_projection" src/ test/
```

### Behavioural regression gates (run after edits)

- **Pure refactors (Parts A, B.1, B.3–B.5, D, E.1, E.3)** must be behaviour-preserving. The
  blitz suite is the fast guard: `julia test/run_blitz_tests.jl` (the cascade/pingpong/stall
  tests in particular pin the Algorithm-B verdicts). Also run the quick suite
  (`test/run_quick_tests.jl`) — `osgs_orthogonality_quick_test.jl` exercises
  `inner_projection_*`/`discrete_l2_projection` and will catch a broken OSGS move;
  `encoding_invariance_quick_test.jl` guards the per-field gate that D.5 renames.
- **Part C (and Option-b of C) changes numbers.** Run the MMS and Cocquet harnesses
  (`test/extended/ManufacturedSolutions/run_test.jl`,
  `test/extended/CocquetFormMMS/run_test.jl`) and diff the convergence-rate tables against the
  baselines in `docs/mms/convergence-baseline.md`. A *good* result for Option (a) is OSGS rates
  unchanged-or-better with the coupled solve now allowed its full `osgs_iterations` budget; a
  *red flag* is OSGS silently matching ASGS (degeneration).
- **Config/schema edits (B.1, B.3, C-b)** must keep every JSON config valid: re-parse each via
  the strict loader and confirm `_check_unknown_keys` emits no new warnings.

### Suggested order of operations (lowest risk first)

1. **Part C decision** (no code yet) — it gates B.2 and F.1.
2. **Part B removals** (dead code; behaviour-preserving) — smallest, safest, shrinks the surface
   before the move. Run blitz+quick.
3. **Part A split** (mechanical move + `solve_osgs_stage!` extraction + include order) — run the
   full suite; the diff should be a pure relocation plus the orchestrator collapse.
4. **Part D + E.1** (comments, renames, merit helper) — run blitz+quick.
5. **Apply the Part C code change** (if Option a) — run MMS/Cocquet, diff rate tables.
6. **Part F** (sync the doc to the final code) — last, so it documents what actually shipped.
7. Add a single `docs/lessons_learned.md` row recording the split + dead-code purge (this is the
   one place process history *should* live).
