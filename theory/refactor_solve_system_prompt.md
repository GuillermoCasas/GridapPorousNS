# Refactoring `solve_system` in `src/solvers/porous_solver.jl`

This document contains:

1. My understanding of the codebase and `solve_system` structure.
2. The structural argument for refactoring.
3. A super-detailed prompt for your AI assistant to perform the refactor with **zero behavioural change**.
4. Separately documented bugs / cleanup opportunities found during review (deliberately **not** mixed into the refactor).

---

## Part 1 — What `solve_system` actually does

`solve_system` is the code-side embodiment of three named, well-defined algorithms from `theory/osgs_algorithm.tex` (Table~\ref{tab:functions}):

| Algorithm in TeX                              | Where it currently lives in `solve_system`     |
|-----------------------------------------------|-----------------------------------------------|
| Algorithm O — `SimulationOrchestration`       | The whole function body (the outer flow)      |
| Algorithm B — `RobustNonlinearCascade`        | **Inlined twice**: lines ~226–323 (Stage I) and ~546–656 (OSGS inner solve, per outer iter) |
| Algorithm C — `OSGSFractionalRelaxation`      | Lines ~459–815 (the OSGS branch in toto)      |
| Algorithm D — `VerifyMMSPlateau`              | **Inlined twice**: lines ~360–432 (ASGS post-solve) and ~743–797 (per OSGS outer iter after `S_conv` fires) |

The TeX document explicitly anticipates this: it says (line 308) that Algorithm B "appears twice---once for Stage~I (ASGS operators, $\boldsymbol{x}^0_{\mathrm{backup}}$) and once inside the OSGS outer-loop body of Algorithm~\ref{alg:C} (OSGS operators, $\boldsymbol{x}_{\mathrm{prev}}$ at the start of each outer iteration)." The TeX algorithm box for Algorithm B is parameterised on a *restore vector* precisely so that one module can serve both call sites.

So the **code is currently more inlined than the paper**. The paper's modular structure is plainly visible: O calls B and C; C calls B once per outer iteration; the MMS hook D is callable from both branches.

---

## Part 2 — Yes, you should refactor. Here is why.

### Structural argument

The natural decomposition mirrors the paper exactly:

```
solve_system  ←  Algorithm O (Orchestrator)
├─ run_asgs_initialization_cascade!  ←  Algorithm B (with π_h ≡ 0, ASGS operators, restore = x0_backup)
├─ run_asgs_mms_plateau_extension!   ←  the ASGS-branch inline hook of Algorithm D
└─ run_osgs_fractional_relaxation!   ←  Algorithm C
   ├─ run_osgs_inner_cascade!        ←  Algorithm B (with π_h frozen, OSGS operators, restore = x_prev)
   ├─ compute_state_drift            ←  the StateDrift helper from §3.6.1 of osgs_algorithm.tex
   ├─ update_projection_with_anderson!
   ├─ compute_projection_drift
   ├─ decide_osgs_convergence        ←  the Mode_stop logic from §3.6.2
   └─ run_osgs_mms_plateau_step!     ←  the OSGS-branch inline hook of Algorithm D
```

### Why I think it is worth doing

- **Reader's mental model collapses to the paper.** Today the reader has to mentally separate "Stage I cascade" from "OSGS inner cascade" while reading two visually similar 80-line blocks; after the refactor, both are one named function with two call sites whose arguments encode the difference (operators, restore vector, success policy).
- **The "two policies" difference becomes a parameter, not a code clone.** Stage I rejects Newton-1 that hits noise floor (it wants the Picard homotopy to deepen into the quadratic basin); OSGS accepts any non-structural Newton finish (it has its own outer loop to converge through). Today this is hidden in 80 lines of duplicated structure. After the refactor it can be one boolean / strategy argument with a docstring explaining the asymmetry.
- **The MMS hook becomes the leaf the paper says it is.** Today the same plateau-test logic is copy-pasted with minor wiring differences in two places (ASGS post-solve loop, OSGS post-`S_conv` block). After the refactor, a single `verify_mms_plateau_step!` mirrors Algorithm~\ref{alg:D}.
- **Long-term safety.** Several P-00x audit items in `theory/Code Audit Findings.md` (e.g., P-003 about stop-reason propagation in the OSGS inner cascade) live inside the duplicated logic. After the refactor those rules are stated once, in one helper, with one docstring — a much smaller surface to keep aligned with the paper.

### Why this refactor is *low-risk*

- The function is already broken into clearly delimited blocks by big banner comments and `if method == "ASGS" / "OSGS"` walls.
- The `safe_fe_solve!` helper is already a successful, well-documented earlier refactor of the same nature (it absorbed cascade boilerplate). The current refactor extends the same pattern from "one tool call" to "one stage of the cascade."
- The pre-refactor behaviour quirks are already explicitly preserved with comments (look at the multi-line comments about "Preserve pre-refactor behavior:" at lines 237–242, 264–268, 588–591, 611–613). These notes carry over verbatim into the new helpers.
- The diagnostics surface (`diag_cache` keys, return tuple, `println` strings) is touched by tests and `instrumented_solver.jl`; the prompt below explicitly fences it as **unchanged**.

### Recommended structure

I recommend extracting **six** private helpers (all `function` not `do` blocks, all in the same file, all returning explicit named tuples; no struct redesigns). The new `solve_system` becomes ~80 lines of straight-line orchestration. The helpers are:

1. `_initialize_asgs_state!` — Algorithm B for Stage I (lines 208–324).
2. `_run_asgs_mms_extension!` — Algorithm D hook in the ASGS path (lines 343–435).
3. `_run_osgs_relaxation!` — Algorithm C (lines 459–826). It internally calls:
   - `_run_osgs_inner_cascade!` — Algorithm B for the OSGS inner solve (lines 548–656).
   - `_compute_state_drift` — StateDrift helper (lines 660–673).
   - `_update_and_project!` — the projection + Anderson step (lines 675–695).
   - `_decide_osgs_convergence` — Mode_stop logic (lines 711–731).
   - `_handle_osgs_mms_step!` — Algorithm D hook in the OSGS path (lines 739–797).

This is the smallest decomposition that gives each algorithm box from the paper exactly one corresponding Julia function. It also keeps everything `solve_system`-local — no new public API, no changes to `PorousNSSolver.jl` exports, no test changes required.

---

## Part 3 — The prompt for your AI assistant

Copy everything between the horizontal rules below verbatim into your AI assistant. It is written to make a strict behaviour-preserving refactor as boring and mechanical as possible.

---

> # Task
>
> Refactor the function `solve_system` in `src/solvers/porous_solver.jl` so that it mirrors the algorithmic structure of `theory/osgs_algorithm.tex` (Algorithms O, B, C, D) — without changing any observable behaviour.
>
> The current function is ~657 lines (lines 172–828 of `src/solvers/porous_solver.jl`). It contains two visually similar inlined copies of the Newton→Picard→Newton cascade (Algorithm B in the paper) and two inlined copies of the MMS plateau verifier (Algorithm D), plus the OSGS outer fractional-relaxation loop (Algorithm C). The paper explicitly says Algorithm B is reused at two call sites; the helper `safe_fe_solve!` already in the file is a partial earlier refactor along these same lines. This task continues that pattern.
>
> # Hard constraints (read these twice before touching code)
>
> **C1. Zero behavioural change.** The refactor must be a pure code reorganisation. Every externally observable quantity must be byte-identical for any input:
>
> - The return tuple `(success, mms_plateau_success, final_x0, iter_count, eval_time)` must have the same values for any input.
> - Every `diag_cache[...]` key written by the old function must still be written, with the same value, in the same order whenever order matters (e.g., `diag_cache["outer_osgs_diagnostics"]` is `push!`-appended inside the loop — preserve the push order).
> - Every `println(...)` line must be emitted with the **same string** (including exact punctuation, capitalisation, leading whitespace, and the variables interpolated into them) and in the **same order** as the original. If you need to extract a block into a helper, the `println`s move with it; do not rewrite them.
> - The `iter_count` integer must increment by the same amounts at the same logical points (Newton iterations, Picard iterations, `+= 1` on the legacy `max_iters_caught` path in the OSGS inner cascade, `+= 1` on the legacy `Reached max iterations` path inside the ASGS-MMS Newton loop, etc.).
> - The `success` flag must take the same value on the same control paths. In particular: in Stage I, Newton-1 that hits the noise floor but not `ftol` does **not** set success and falls through to Picard (lines 251–255). In the OSGS inner cascade, Newton-1 that finishes any non-structural way (i.e., `stop_reason` not in `{"linesearch_failed", "merit_divergence_escaped", "linear_solve_nan"}`) does **not** mark `newton_failed`. These asymmetries are intentional and must be preserved.
> - The `eval_time = @elapsed begin ... end` measurement must cover the same code regions. Currently there are two `@elapsed` blocks: one wrapping Stage I (lines 226–324), and one wrapping the ASGS-MMS verification loop (lines 360–425), and one wrapping the OSGS outer loop (lines 503–815). Each `@elapsed` is `+=`-accumulated into `eval_time`. Preserve all three regions and the `+=` accumulation pattern exactly. A simple way to do this: keep the `@elapsed begin ... end` blocks in the new `solve_system` and call the new helpers from inside them. Do **not** move the `@elapsed` inside helpers — that would change which Julia-side overhead is timed.
> - `safe_fe_solve!` is **not** modified. Use it as-is.
>
> **C2. No new types, no new exports, no signature changes to existing public functions.** The public API of the file (the exported names from `src/PorousNSSolver.jl` and the call sites in `src/instrumented_solver.jl` and the test files) does not change. All new helpers are file-local — give them leading-underscore names (Julia convention for "internal") and do not export them.
>
> **C3. Do not "improve" anything outside the refactor.** If you spot a bug, a misleading variable name, dead code, a redundant assignment, a missing tolerance check, a place where the code disagrees with the paper, or anything else worth fixing — **stop, do not change it, and write it to a separate file `refactor_followups.md`** at the repo root. The current task is structural only. Bug fixes change behaviour; that requires its own PR with its own test.
>
> Examples of "do not touch":
> - The two `u_h, p_h = final_x0` unpacks at lines 665 and 676 (one of them is redundant). Leave both.
> - The `u_h, p_h = x0` at line 187 (the variables it binds are never used in the function body). Leave it.
> - `prev_x_diff` and `prev_pi_drift` are written at lines 525–526 and 708–709 but never read. Leave them.
> - The `solver_newton_asgs` constructor at line 222 retypes the entire `SafeNewtonSolver` argument list rather than using `SafeNewtonSolver(ls, ::SolverConfig)` from `nonlinear.jl`. Leave the verbose form.
> - Any `println` that uses awkward phrasing ("vigorously", "natively", "structurally aborted", etc.). Leave them — they are matched by tests and human-eyeballed log output.
>
> **C4. Comment preservation.** Every block-comment and inline comment that documents a behavioural invariant must move with the code it documents. In particular:
>
> - The `# Preserve pre-refactor behavior:` comments at lines 237–242, 264–268, 588–591, and 611–613 are load-bearing — they explain why specific control-flow paths take their specific shape. They must end up inside the new helpers, attached to the same lines they currently sit above.
> - The `# P-003 back-pointer:` comment at lines 572–577 (which references `theory/osgs_algorithm.tex` §1.2.4) must move with the structural-failure stop-reason check.
> - The big `# §5.1: scale plateau floors ...` and `# §5.2: rate-aware sanity check ...` comments in the MMS branches (lines 383–390 and 411–415 in the ASGS branch; lines 759–762 and 787 in the OSGS branch) must move with the MMS plateau logic.
> - The big banner comments (`# =====` blocks at 208–213, 446–458, 473–476, 489–494) become docstrings of the corresponding new helpers, **without losing any of their content**. Verbatim copy is preferred over paraphrase.
>
> **C5. Diff discipline.** The PR should be readable. Aim for:
>
> 1. New helper functions added in a logical order (cascade helper before its first use, etc.).
> 2. The new `solve_system` body replacing the old one in a single block.
> 3. No formatting churn elsewhere in the file. Do not reflow lines, reorder imports, or rename existing local variables outside the moved blocks.
> 4. `inner_projection_u`, `inner_projection_p`, `check_porous_solver_parameters`, `discrete_l2_projection` (both methods), `FETopology`, `VMSFormulation`, `IterativeSolvers`, and `safe_fe_solve!` are untouched.
>
> # Target structure
>
> Six new file-local helpers; the new `solve_system` is straight-line orchestration that mirrors Algorithm O.
>
> ## H1. `_initialize_asgs_state!(x0, x0_backup, op_newton_init, op_picard_init, solver_newton_asgs, solver_picard, ftol, diag_cache) → NamedTuple`
>
> Owns the **Stage I** block (current lines 208–324). Implements Algorithm B (Algorithm~\ref{alg:B}) with ASGS operators ($\pi_h \equiv 0$) and `restore = x0_backup`. The construction of `res_fn_init`, `jac_newton_init`, `jac_picard_init`, `op_newton_init`, `op_picard_init`, and `solver_newton_asgs` (current lines 214–224) stays inside `solve_system` and is passed in, because those closures capture `setup`, `formulation`, `phys_cfg`, `freeze_cusp` which already live in `solve_system`'s scope. The `x0_backup` allocation also stays in `solve_system` so the caller can decide its lifetime.
>
> Returns `(; success::Bool, iter_count_delta::Int)`. The helper writes `diag_cache["final_residual_norm"]` exactly where the original does. The helper does **not** time itself; the `@elapsed begin ... end` wrapper around the call stays in `solve_system`.
>
> Internal behavioural rules to preserve verbatim:
>
> - Newton attempt 1: `safe_fe_solve!` with `backup=x0_backup`.
> - On `:nonfinite` / `:exception` / `:max_iters_caught` → set `newton_success = false` and log the pre-refactor messages exactly.
> - On `:ok`: accumulate iterations; only mark `newton_success = true` if `final_res <= ftol`; if `final_res` is between noise floor and `ftol`, fall through to Picard (this is the documented Stage I quadratic-basin guarantee — keep the comment).
> - On fall-through: restore `x0` from `x0_backup`, run Picard (`solver_picard`, no `backup=` argument — the explicit secondary `any(!isfinite, …)` check below is the pre-refactor guard and must be preserved with its comment).
> - On Picard catastrophic non-finite: restore backup, mark `success = false`, return.
> - On Picard non-catastrophic finish without convergence: run Newton attempt 2, with `backup=x0_backup`. The stop-reason classification (`"ftol_reached"`, `"initial_ftol"`, `"stagnation_noise_floor_reached"` → success; everything else → failure) is the Stage I policy and must be preserved exactly.
>
> ## H2. `_run_asgs_mms_extension!(final_x0, op_newton_init, solver_newton, mms_cfg, ftol, diag_cache, iter_count_ref) → Nothing`
>
> Owns the ASGS post-success MMS plateau block (current lines 343–435). Implements the ASGS-branch inline hook of Algorithm D (Algorithm~\ref{alg:D}).
>
> The 1-iteration-cap Newton solver constructed at lines 356–358 (`local_asgs_nls`, `local_fesolver`) is built inside this helper. The `mms_err_hist`, `mms_rc_hist`, `consecutive_passes` locals stay inside. The function writes:
> - `diag_cache["base_convergence_reached"] = true` on entry,
> - `diag_cache["mms_plateau_reached"] = true` on plateau success,
> - `diag_cache["mms_stop_reason"]` per current rules ("nonlinear_failure", "mms_plateau_satisfied", "mms_plateau_at_suboptimal_rate", "mms_budget_exhausted"),
> - `diag_cache["mms_error_history"]`, `diag_cache["mms_relative_change_history"]` at the end.
>
> The `iter_count` is mutated via a `Ref{Int}` (small concession to Julia idiom; the alternative is to return `iter_count_delta` and have the caller add). Pick whichever yields the cleaner diff; document the choice in the docstring.
>
> The `@elapsed` wrapper around the inner `for cycle = 1:max_extra_cycles` loop (current line 360) stays in this helper — it is internal to the MMS extension and is `+=`-accumulated into `eval_time` after the helper returns by returning the elapsed time. **Re-check:** actually the cleanest preservation is to wrap the call from `solve_system` in `eval_time += @elapsed _run_asgs_mms_extension!(...)`. Pick the latter for symmetry with the OSGS site. The `@elapsed` block inside the helper at line 360 is therefore removed and re-instantiated at the call site. This is the only `@elapsed`-region reshuffling allowed by this task; verify the wrapped region is identical.
>
> ## H3. `_run_osgs_relaxation!(final_x0, X, Y, V_free, Q_free, setup, formulation, phys_cfg, stab_cfg, sol_cfg, solver_newton, ftol, stagnation_tol, freeze_cusp, mms_cfg, diag_cache, iter_count_ref) → (success::Bool, pi_u, pi_p)`
>
> Owns the OSGS branch (current lines 459–826). Implements Algorithm C (Algorithm~\ref{alg:C}). This is the largest helper but it is still well under half the size of the current monolith; the structure of Algorithm C in the paper makes the body almost mechanical to map.
>
> Order of operations inside, mirroring the paper:
>
> 1. Setup: extract `U, P = X`, `V, Q = Y`, `V_proj`, `Q_proj`, `U_proj`, `P_proj` (current lines 465–471). Print the corresponding logging line.
> 2. Assemble and factorise the mass matrices `M_u`, `M_p` with `CholeskySolver` (current lines 477–488). Print the corresponding logging line.
> 3. Set the effective outer-iteration cap `eff_osgs_iters` (current lines 495–501).
> 4. Open the `@elapsed begin ... end` block (current line 503), append the elapsed time to `eval_time` (returned from the helper).
> 5. Inside the elapsed block:
>    - Construct optional Anderson accelerators (current lines 504–510).
>    - Initialise `pi_u`, `pi_p` to zero on `U_proj`, `P_proj` (current lines 516–520). Print the corresponding line.
>    - Initialise `diag_cache["inner_osgs_diagnostics"] = []`, `diag_cache["outer_osgs_diagnostics"] = []` (current lines 522–523).
>    - Initialise `prev_pi_drift = 1.0`, `prev_x_diff = 1.0` (current lines 525–526). **Keep these even though they are dead — see C3.**
>    - The outer `for osgs_iter = 1:eff_osgs_iters` loop, calling four sub-helpers per iteration: H4, H5, H6, then H7 inline.
> 6. After the loop, the post-loop MMS budget check (current lines 816–822). Print the corresponding line.
> 7. Write `diag_cache["pi_u"]`, `diag_cache["pi_p"]` (current lines 824–825).
>
> Return `(success, pi_u, pi_p)`. The `eval_time` increment is returned as a separate value `(elapsed::Float64)` so `solve_system` does `eval_time += elapsed_from_osgs`.
>
> ## H4. `_run_osgs_inner_cascade!(final_x0, x_prev, op_newton_osgs, op_picard_osgs, local_fesolver, local_fesolver_picard, diag_cache, iter_count_ref) → (success::Bool, structurally_failed::Bool)`
>
> Owns the per-outer-iteration inner cascade (current lines 548–656). Implements Algorithm B with OSGS operators and `restore = x_prev`.
>
> The `op_newton_osgs`, `op_picard_osgs` and the two `local_fesolver`s (current lines 533–544) are constructed by the caller (`_run_osgs_relaxation!`) because they depend on `pi_u`, `pi_p`, and the iteration-dependent `tau_inner_m` (current line 531), all of which already live in the caller's loop scope.
>
> Internal rules:
>
> - Newton attempt 1: `safe_fe_solve!` with `backup=x_prev`. Push to `diag_cache["inner_osgs_diagnostics"]` exactly as today. Set `newton_failed = true` iff `stop_reason in {"linesearch_failed", "merit_divergence_escaped", "linear_solve_nan"}` on the `:ok` path; set on `:nonfinite`/`:exception`; do **not** set on `:max_iters_caught` (which gets `iter_count += 1` instead). Preserve the P-003 comment.
> - On `newton_failed`: restore from `x_prev`, run Picard (no `backup=`). Preserve the secondary `any(!isfinite, …)` guard. `:exception`/`:ok max_iters_caught` paths follow current rules. On catastrophic non-finite or Picard non-completion, set `structurally_failed = true` and return.
> - On Picard success and re-engagement of Newton: identical rules to Newton-1 for structural-failure stop reasons (`linesearch_failed`/`merit_divergence_escaped`/`linear_solve_nan`); on those, set `structurally_failed = true`.
>
> Return `(success = !structurally_failed, structurally_failed)` so the caller decides whether to `break` the outer loop.
>
> ## H5. `_compute_state_drift(final_x0, x_prev, X, dΩ, osgs_state_drift_scale) → (x_diff::Float64, x_diff_inf::Float64)`
>
> Implements `StateDrift` from §3.6.1 of `osgs_algorithm.tex`. Pure function (no side effects). Current lines 660–673. Returns both the mode-selected `x_diff` and the always-computed `x_diff_inf` (needed by the absolute noise-floor short-circuit at lines 724–731).
>
> ## H6. `_update_and_project!(final_x0, pi_u, pi_p, accel_u, accel_p, form, dΩ, h_cf, f_cf, alpha_cf, g_cf, c_1, c_2, U_proj, V_proj, P_proj, Q_proj, M_u, M_p, num_u_fac, num_p_fac) → (pi_u_next, pi_p_next, pi_u_drift::Float64, pi_p_drift::Float64, R_u_algebraic_norm::Float64)`
>
> Owns the residual evaluation, $L^2$ projection, Anderson mixing, and projection-drift computation (current lines 675–698). Pure-ish — no `diag_cache` writes, no `println` (the `outer_osgs_diagnostics` push and console line happen in the caller). Returns enough values for the caller to assemble the diagnostics tuple.
>
> ## H7. `_decide_osgs_convergence(x_diff, x_diff_inf, pi_u_drift, pi_p_drift, osgs_iter, stab_cfg, osgs_tol, stagnation_tol) → overall_converged::Bool`
>
> Implements §3.6.2's `Mode_stop` logic plus the absolute noise-floor short-circuit (current lines 711–731). Pure function.
>
> The MMS plateau hook (current lines 739–797) is left inline inside `_run_osgs_relaxation!` because it touches `diag_cache`, `success`, the `break` statement on the outer loop, and the first-`S_conv`-event seeding logic, all of which are inseparable from the surrounding loop body. Extracting it into a helper would either require passing back a `break` request as a return value (uglier) or wouldn't save much over leaving it inline (the MMS hook is ~60 lines and well-commented). Leave it inline. If the AI assistant disagrees, it must justify it in writing in `refactor_followups.md` and **not** extract.
>
> # The new `solve_system`
>
> After the refactor, the body of `solve_system` should read like an algorithm box:
>
> ```julia
> function solve_system(setup::FETopology, formulation::VMSFormulation, iter_solvers::IterativeSolvers,
>                       config::PorousNSConfig, x0;
>                       diagnostics_cache=nothing, mms_cfg=nothing)
>
>     # ===== Argument unpacking and parameter validation =====
>     # (current lines 176-207; keep exactly)
>     ...
>
>     # ===== Build init-stage operators and ASGS solver (closures over setup/form/phys_cfg) =====
>     # (current lines 214-224; keep exactly)
>     res_fn_init  = ...
>     jac_newton_init = ...
>     jac_picard_init = ...
>     op_newton_init = FEOperator(res_fn_init, jac_newton_init, X, Y)
>     op_picard_init = FEOperator(res_fn_init, jac_picard_init, X, Y)
>     solver_newton_asgs = FESolver(SafeNewtonSolver(...))
>     x0_backup = copy(get_free_dof_values(x0))
>
>     # ===== Stage I: ASGS algebraic initialisation (Algorithm B) =====
>     eval_time = @elapsed begin
>         init = _initialize_asgs_state!(x0, x0_backup, op_newton_init, op_picard_init,
>                                        solver_newton_asgs, solver_picard, ftol, diag_cache)
>     end
>     success    = init.success
>     iter_count = init.iter_count_delta
>
>     # ===== MMS plateau helper definition (used by both branches in the early-return) =====
>     mms_enabled = !isnothing(mms_cfg) && mms_cfg.enabled
>     _mms_plateau() = mms_enabled ? get(diag_cache, "mms_plateau_reached", false) : nothing
>
>     # ===== Stage I failure: short-circuit return =====
>     if !success
>         diag_cache["pi_u"] = nothing
>         diag_cache["pi_p"] = nothing
>         return false, _mms_plateau(), x0, iter_count, eval_time
>     end
>     final_x0 = x0
>
>     # ===== Stage II: method dispatch =====
>     if method == "ASGS"
>         if mms_enabled
>             println("\n      [!] Commencing ASGS MMS Error Verification Plateau Loop...")
>             iter_count_ref = Ref(iter_count)
>             eval_time += @elapsed _run_asgs_mms_extension!(final_x0, op_newton_init, solver_newton,
>                                                            mms_cfg, ftol, diag_cache, iter_count_ref)
>             iter_count = iter_count_ref[]
>         else
>             println("\n      [+] ASGS Formulation exclusively resolved. Exiting solver module.")
>             diag_cache["base_convergence_reached"] = true
>             diag_cache["mms_stop_reason"] = "base_convergence_only"
>         end
>         diag_cache["pi_u"] = nothing
>         diag_cache["pi_p"] = nothing
>         return success, _mms_plateau(), final_x0, iter_count, eval_time
>     end
>
>     if method == "OSGS"
>         iter_count_ref = Ref(iter_count)
>         osgs_success, pi_u, pi_p, osgs_elapsed =
>             _run_osgs_relaxation!(final_x0, X, Y, V_free, Q_free, setup, formulation, phys_cfg,
>                                   stab_cfg, sol_cfg, solver_newton, ftol, stagnation_tol,
>                                   freeze_cusp, mms_cfg, diag_cache, iter_count_ref)
>         iter_count = iter_count_ref[]
>         eval_time  += osgs_elapsed
>         success     = osgs_success
>         diag_cache["pi_u"] = pi_u
>         diag_cache["pi_p"] = pi_p
>         return success, _mms_plateau(), final_x0, iter_count, eval_time
>     end
> end
> ```
>
> This is illustrative — your final code may vary in argument ordering or `Ref` vs return-delta choice, but the **shape** (Stage I, early-return, ASGS branch, OSGS branch) must read this cleanly.
>
> # Verification protocol (you must run this before declaring done)
>
> **V1. Reading-level check.** Open `theory/osgs_algorithm.tex` side-by-side with the new `solve_system` and confirm: Algorithm O's `\For{m = 1, ..., N_OSGS}` lines 915–956 of the TeX correspond line-for-line to `_run_osgs_relaxation!`'s outer loop; Algorithm B's three-step cascade (lines 571–587 of the TeX) corresponds line-for-line to `_initialize_asgs_state!` (Stage I) and to `_run_osgs_inner_cascade!` (Stage II inner).
>
> **V2. Test suite.**
>
> 1. Run `test/run_blitz_tests.jl` (fast unit tests). Every test that passed before must pass after. No new test failures, no new warnings.
> 2. Run `test/run_quick_tests.jl`. Same expectation.
> 3. Run `julia --project=. test/extended/ManufacturedSolutions/run_test.jl` with `data/small_test_config.json` (the canonical small MMS sweep). The h5/markdown outputs must be byte-identical to a pre-refactor run of the same config. If you cannot compare against a stored pre-refactor baseline, run the same script on the pre-refactor branch first, then on the refactored branch, and `diff` the outputs.
>
> **V3. Diagnostics-cache snapshot test.** Write one new throwaway test (you may delete it after verification — do not commit) that runs `solve_system` once with `method = "OSGS"`, `mms_cfg.enabled = true`, on a 10×10 mesh from `test_config.json`, and serialises the resulting `diag_cache` to JSON. Run it on `main`, then on your refactor branch, and confirm the JSONs are equal modulo non-deterministic timing fields. The keys to compare are at least: `"final_residual_norm"`, `"base_convergence_reached"`, `"mms_plateau_reached"`, `"mms_stop_reason"`, `"mms_error_history"`, `"mms_relative_change_history"`, `"mms_consecutive_passes"`, `"inner_osgs_diagnostics"`, `"outer_osgs_diagnostics"`. Do the same with `method = "ASGS"` and with `mms_cfg = nothing` (no MMS).
>
> **V4. Log-output snapshot.** Capture stdout from one ASGS run and one OSGS run on `small_test_config.json` before and after the refactor. `diff` them; the only allowed differences are line ordering between concurrently-emitted lines (there shouldn't be any here — `solve_system` is single-threaded — so a clean `diff` is required) and timing numbers in the `Eval Time` summary.
>
> **V5. Followups file.** If `refactor_followups.md` exists, attach it to the PR description. If it does not exist, state explicitly in the PR description that no follow-ups were found.
>
> # Out of scope
>
> - Renaming any existing identifier (variables, functions, struct fields, config keys).
> - Adding or removing `println` calls.
> - Changing any tolerance value, default, or convergence criterion.
> - Switching solver algorithms (line search, accelerator, mass-matrix factorisation method).
> - Touching `safe_fe_solve!`, `inner_projection_u`, `inner_projection_p`, `discrete_l2_projection`, `check_porous_solver_parameters`, or any code outside `solve_system` itself.
> - Re-architecting `diag_cache` from a `Dict{String,Any}` to a typed struct (even though this would be a clear improvement). That is a separate, follow-up task.
> - Adding type annotations to the new helpers beyond the bare minimum needed for clarity. Julia infers fine; over-annotating can constrain dispatch in ways that break consumers.
>
> # Definition of done
>
> 1. `solve_system` is under 100 lines and reads as Algorithm O.
> 2. Six new file-local helpers exist, named per the H1–H7 plan, each with a docstring that names the algorithm box it implements (e.g., `"""Implements Algorithm B (RobustNonlinearCascade) for the Stage I ASGS initialisation. See theory/osgs_algorithm.tex §..."""`).
> 3. V1–V5 above all pass.
> 4. `git diff` outside `src/solvers/porous_solver.jl` is empty.
> 5. The new file is at most ~5% longer than the old one (helpers add some docstring overhead but the inlined duplication goes away; net should be neutral to slightly shorter).

---

## Part 4 — Bugs and improvements found during review (separate from refactor)

These are written here, in this same document, but they should **not** be addressed in the refactor PR. Each is a candidate for its own ticket.

### B1. Dead writes: `prev_x_diff` and `prev_pi_drift` (cosmetic)

`src/solvers/porous_solver.jl` lines 525–526 (initialisation) and 708–709 (per-iteration update). The values are never read anywhere — confirmed by `grep -n "prev_x_diff\|prev_pi_drift" src/solvers/porous_solver.jl`. Either remove (preferred) or — if they were meant to feed into a drift-ratio diagnostic that was never added — wire them into `diag_cache["outer_osgs_diagnostics"]` so the diagnostic actually exists. The shape of these variables (one scalar each, "previous iteration" naming) strongly suggests they were planned to support a $d_{\mathrm{state}}^m / d_{\mathrm{state}}^{m-1}$ contraction-ratio diagnostic, which would be a useful convergence-monitoring signal and is mathematically the natural quantity to track for a fixed-point iteration. Recommend: either delete cleanly, or implement the contraction-ratio diagnostic and surface it. Do not leave dead code in a published scientific solver.

### B2. Redundant unpacking: `u_h, p_h = final_x0` at line 665 and 676 (cosmetic)

Inside the OSGS outer loop. The first one (line 665) is inside the `if stab_cfg.osgs_state_drift_scale == "L2_mass"` branch and is used to build `e_u`, `e_p`. The second (line 676) is unconditional. They produce the same binding. Either:

- (preferred) hoist a single `u_h, p_h = final_x0` above the `if`, or
- delete the conditional one if you keep the unconditional one below.

Either is a trivial cleanup; pick whichever yields the cleaner diff with `_compute_state_drift` and `_update_and_project!` after refactoring. (Recommended only after the structural refactor lands.)

### B3. Vestigial unpacking: `u_h, p_h = x0` at line 187 (cosmetic)

Bound but never used. Delete after the refactor.

### B4. `solver_newton_asgs` constructed verbosely (cosmetic)

Line 222 rebuilds a `SafeNewtonSolver` by enumerating every field, when there is already an ergonomic constructor `SafeNewtonSolver(ls, ::SolverConfig; mode=:newton)` in `src/solvers/nonlinear.jl:36`. The verbose form was likely written before the ergonomic constructor was added. After refactor, replace with:

```julia
solver_newton_asgs = FESolver(SafeNewtonSolver(base_nls_global.ls, sol_cfg))
```

This is a 10-line → 1-line cleanup, and is robust against future fields being added to `SolverConfig`/`SafeNewtonSolver` (the verbose form silently drops new fields and uses defaults — a real correctness risk).

The same pattern recurs at line 357 (`local_asgs_nls` for the MMS extension), line 533 (`local_osgs_nls`), and line 536 (`local_osgs_nls_picard`). Each of these overrides one or two specific fields of the base solver. After the ergonomic constructor + a tiny `with(::SafeNewtonSolver; field=value, ...)` helper, these become one-liners. **However** — be careful: today's verbose form is explicit about which fields are inherited and which are overridden. If you add a wither, the override-vs-inherit boundary becomes implicit. I would write a small `_with_overrides(nls; max_iters, ftol, mode)` helper that takes only the fields that are *ever* overridden in `porous_solver.jl` and asserts the rest. This is a separate cleanup PR.

### B5. Asymmetric `max_iters_caught` handling between Stage I and OSGS inner cascade (algorithmic)

This is the subtlest issue I found. The two cascades treat the legacy Gridap exception `"Reached maximum iterations"` differently:

- Stage I (lines 236–242 and 301–304): treats it as a **failure** and logs `"Newton ConvergenceError"`.
- OSGS inner cascade (lines 554–556, 633–634): treats it as a **partial success**, with `iter_count += 1`, no failure mark, no message about an error.

The comments at lines 237–242 and 264–268 explicitly call out that this is "pre-refactor behaviour" being preserved. This *is* asymmetric. Whether it is *correct* asymmetric depends on a finer reading of the algorithm document:

- In Stage I, the iteration is the algorithm's only chance to enter the quadratic basin. Treating "max iterations" as failure forces the Picard homotopy to run, which is the safer choice.
- In OSGS, the outer loop will re-evaluate convergence per iteration anyway, so a single inner Newton hitting iteration cap is not by itself an outer-loop failure.

The asymmetry is therefore *probably* correct, but it is not stated in the paper. `theory/osgs_algorithm.tex` Table~\ref{tab:reasons} says the modern `SafeNewtonSolver` exits cleanly with `stop_reason = "max_iters_stagnation"` on the `:ok` path; the `:max_iters_caught` exception path is purely defensive against legacy or non-`SafeNewtonSolver` solver instances (this is documented in `safe_fe_solve!`'s docstring). So in practice the path is almost never taken with the current solver stack.

Recommend: write a paragraph in `theory/osgs_algorithm.tex` §3.5 stating that the legacy `max_iters_caught` exception is treated as a failure in Stage I (driving fall-through to Picard) and as a single-iteration success in the OSGS inner cascade (the outer loop re-evaluates). If the asymmetry is actually a historical accident, decide which behaviour is correct and unify. Either way: state it in the paper or remove it. Today's situation — silent asymmetry with a "preserve pre-refactor" comment — is the worst of both worlds.

### B6. `solve_system` does not accept Stage I Newton-1 noise-floor convergence (algorithmic, low-priority)

Lines 251–255: after Stage I Newton-1 hits `:ok`, success is only declared if `final_res <= ftol`. If the residual is between `stagnation_noise_floor` and `ftol`, the code does **not** mark success and falls through to Picard, even though `SafeNewtonSolver` itself reports `stop_reason = "stagnation_noise_floor_reached"` (which Newton-2 at lines 313–316 accepts as success). The comment at lines 252–254 explains: "When initializing, if it hits noise floor but not ftol, we still allow Picard to attempt homotopy to strictly guarantee we enter the quadratic basin."

This is defensible, but is again not stated in the paper. Algorithm B in the paper (lines 571–587 of the TeX) does not distinguish "Newton hit `ftol`" from "Newton hit noise floor" — it returns success on the first `ExactNewtonPipeline` call iff `S_N = True`, period. To match the paper exactly, the Stage I Newton-1 should accept any non-structural `:ok` finish (matching how OSGS inner Newton-1 already behaves). The current behaviour adds an extra Picard pass in a regime where it is empirically helpful but not algorithmically necessary.

Recommend: either prove (via MMS sweeps) that the extra Picard pass measurably improves convergence at the noise floor → state in the paper as an addendum to Algorithm B for Stage I specifically; or remove the asymmetry and let Newton-1's noise-floor success short-circuit Picard.

### B7. The MMS plateau check is computed *before* Newton iterates the iterate (algorithmic)

In the ASGS-MMS loop (current lines 360–425), each cycle does: solve one Newton step → compute new errors → compare to previous errors → check plateau. So the plateau check uses the error pair `(E^k, E^{k-1})` where $E^k$ is computed *after* the k-th Newton step.

In the OSGS-MMS branch (current lines 743–797), the plateau check uses `(E^m, E^{m-1})` where $E^m$ is computed *after* outer iteration $m$ has fully converged its inner solve and re-projected. But there is a subtle ordering issue: the first-`S_conv` event "seeds" the history at lines 743–749 (mms_err_hist gets E^m), then a subsequent outer iteration enters the *else* branch at 751 with `E^m` being the *new* (m+1)-th outer iterate's error. The `mms_err_hist[end-1]` access at line 757 picks up the seed. So the plateau test fires on the very first comparison after the seed iteration, with a pair `(E^{m+1}, E^m)` separated by *one full OSGS outer iteration plus inner cascade plus reprojection*, not just one Newton step. This is different from the ASGS site.

Algorithm D in the paper (Algorithm~\ref{alg:D}) is mode-agnostic — it just compares the last two entries of $\mathcal{H}$. The semantic content of "one entry in $\mathcal{H}$" is therefore different in the two branches. This is probably fine — the paper's logic in §5.2 ("plateau" means "no further FE-error reduction") works for either spacing — but it should be stated explicitly. Without that statement, `\varepsilon_{u,L^2}` etc. floors are calibrated against an unstated quantity.

Recommend: add a paragraph in `osgs_algorithm.tex` §5.2 stating that one $\mathcal{H}$-entry corresponds to one Newton step in the ASGS path and one outer-staggered iteration in the OSGS path; users tuning $\tau_{\mathrm{err}}$ should keep this in mind.

### B8. `eval_time = @elapsed begin ... end` does not include all work (cosmetic)

`eval_time` is the user-visible solver wall time. Currently the @elapsed blocks cover Stage I, the ASGS MMS verification loop, and the OSGS outer loop — but **not** the mass-matrix factorisation (current lines 477–488) which can be a non-trivial chunk of OSGS first-iteration cost on large meshes. The factorisation happens inside `eval_time = @elapsed begin ... end` only for OSGS (it sits between lines 459–502 inside the OSGS branch). Wait — re-reading lines 476–488, the factorisation is *before* the `eval_time += @elapsed begin ... end` block at line 503. So it is **excluded** from `eval_time` in the OSGS branch.

This means `eval_time` reported to the user is the time spent in the iterative loop only, not in the OSGS-specific setup. For comparability with the ASGS path (where there is no extra setup), this is *fine* — but it should be stated in the per-result diagnostic, and the user should know that switching from ASGS to OSGS adds setup time invisible to `eval_time`. Recommend: rename `eval_time` → `iterative_eval_time` in the return tuple, or add a new `setup_eval_time` field for the factorisation/setup so the user sees the total cost. (Bigger change — requires updating callers and diagnostic exports.)

### B9. The post-OSGS-loop MMS budget-exhaustion branch sets `success = true` (algorithmic)

Lines 816–822: after the outer OSGS loop exits without satisfying the MMS plateau, if `base_convergence_reached` is true but `mms_plateau_reached` is false, the code sets `success = true` and writes `mms_stop_reason = "mms_budget_exhausted"`. This makes "base converged but MMS plateau not verified" report success.

This is a **policy choice**: do you trust the base OSGS convergence even when the MMS verifier did not get to fire enough plateau samples? The current answer is yes (success = true). Algorithm O in the paper (lines 488–495 of TeX) returns on `OSGSFractionalRelaxation`'s success flag, and `OSGSFractionalRelaxation` (Algorithm C, lines 944–953 of TeX) treats `mms_cfg.enabled` purely as a `continue` veto on the first `S_conv` event. The paper's Algorithm C does not have an explicit `mms_budget_exhausted = success` provision — it implicitly fails (returns False at line 956) if the outer loop runs out.

So the current code's lines 816–822 are an *unstated* generosity to the user: "if base convergence happened but MMS verification ran out of budget, still report success." This is probably the right user-facing behaviour, but **it is not in the paper**. The `mms_plateau_success` return value (the second tuple slot) is set to `false` in this case, so a careful caller can detect it via `(success=true, mms_plateau_success=false)`. This is the P-007 / Fix 6 from the audit. The behaviour is documented in the code comment at lines 326–330. State it in the paper.

### B10. ASGS-MMS loop catches "Reached max iterations" as `iter_count += 1` without logging (algorithmic, very minor)

Lines 368–376: the ASGS-MMS verification loop's `try`/`catch` increments `iter_count += 1` silently on the legacy "Reached maximum iterations" exception, but logs and breaks on any other exception. The Stage I site (which fires the same legacy exception) treats this as `newton_success = false` and logs explicitly. The OSGS inner site (lines 554–556) logs the line "OSGS Newton 1-step interior pass dynamically executed." So the same legacy exception is handled three different ways. This is the same family of issue as B5 and probably has the same correct answer (state in paper or unify).

---

## Quick summary

- **Refactor:** strongly recommended. Six new file-local helpers; new `solve_system` is ~80 lines and reads as Algorithm O. Pure code reorganisation with bit-for-bit behaviour preservation. The detailed prompt in Part 3 is engineered so an AI assistant can execute it mechanically.
- **Don't bundle bug fixes with the refactor.** I found ten distinct items above; each one is its own thinking exercise. Mixing them into a "refactor" PR is what creates regressions.
- **The biggest real issue I found** is not in `solve_system` at all — it is B5 + B6 + B10, the asymmetric treatment of "Reached max iterations" / noise-floor convergence between the two cascades. These are documented as "preserved pre-refactor" but are not stated in the paper. The right fix is paper-side first: state which behaviour is canonical, then unify the code.
- The dead-code (B1, B2, B3) and the verbose-constructor (B4) items are pure cleanup and can be batched.
- B8 and B9 are documentation items, not bugs, but reporting them lets the next reader know they were considered.
