# src/solvers/asgs_solver.jl
#=
    asgs_solver.jl

# Role
The shared solver core + the ASGS stabilization + the orchestrator for the Porous Navier-Stokes system.
It bridges the continuous operators, stabilized forms, and Jacobians (`viscous_operators.jl`) with the
iterative execution (`nonlinear.jl`) needed to reach discrete convergence. The OSGS extension that uses
this core lives in `osgs_solver.jl` (included right after this file).

# Methodological Background
Following the formulation in the companion article, the Galerkin finite element discretization of the generalized Navier-Stokes equations suffers from LBB condition violations for equal-order spaces and instabilities in convection- or reaction-dominated flows. We use VMS to approximate the unresolved sub-grid scales (SGS) ``\widetilde{U}`` in terms of the resolved scales' residual ``\mathcal{R}(U_h)`` and a local element matrix of stabilization parameters ``\boldsymbol{\tau}_K``.

Depending on the chosen space ``\widetilde{\mathcal{X}}`` for the sub-grid scales, two stabilized methods are supported:
1. **ASGS (Algebraic Sub-Grid Scale)** — *this file*: the SGS space is the space of finite element residuals; the projection onto it is the identity ``\widetilde{\Pi} = \mathcal{I}``, so the FE projection ``\boldsymbol{\pi}_h = \boldsymbol{0}``.
2. **OSGS (Orthogonal Sub-Grid Scale)** — *`osgs_solver.jl`*: the SGS space is orthogonal to the FE space (``\mathcal{X}_{h0}^{\perp}``); it computes the ``L^2``-projection ``\boldsymbol{\pi}_h`` of the residual and subtracts it in the sub-scale equation.

# Relations to other files
- `src/formulations/viscous_operators.jl`: translates the abstract operators (e.g. ``\mathcal{L}_{\nu}, \mathcal{L}_c, \mathcal{L}_b``) into Gridap closures; this file delegates `eval_strong_residual`, `build_stabilized_weak_form_jacobian`, etc. to it.
- `src/solvers/nonlinear.jl`: the shared `SafeNewtonSolver` kernel (`solver_picard`, `solver_newton`) that converges the ASGS/OSGS algebraic problems.
- `src/solvers/osgs_solver.jl`: the OSGS L²-projection helpers + the coupled OSGS solve (`solve_osgs_stage!`).

# Algorithm-to-code mapping
See `docs/solver/algorithm-code-mapping.md` for the full table. In short, `solve_system` mirrors
Algorithm O (SimulationOrchestration) of `theory/osgs_algorithm/osgs_algorithm.tex`; it delegates to
`_initialize_asgs_state!` (Stage-I ASGS boot, Algorithm B) and `solve_osgs_stage!` (the coupled OSGS
solve, Algorithm C; in `osgs_solver.jl`) — plus the shared Newton↔Picard cascade (`_pingpong_cascade!`
/ `cascade_step_outcome`). The optional MMS plateau verification (Algorithm D) is decoupled behind the
`SolutionVerifier` seam (`on_asgs_converged!` / `on_osgs_converged!`); the concrete `MMSPlateauVerifier`
lives in `src/solvers/mms_verification.jl`, and production uses the no-op `NoVerification`.
=#
using Gridap
using Gridap.Algebra
using LinearAlgebra

# NOTE: the OSGS L²-projection helpers (`inner_projection_u/p`, `discrete_l2_projection`) and the
# coupled OSGS solve (`solve_osgs_stage!`) live in `osgs_solver.jl`, included right after this file.

"""
    check_porous_solver_parameters(stab_cfg, sol_cfg)

Enforces strict parameter boundaries for the top-level VMS solver configurations,
preventing unbounded numerical cascades resulting from malformed JSON properties.
"""
function check_porous_solver_parameters(stab_cfg::StabilizationConfig, sol_cfg::SolverConfig)
    if stab_cfg.method != "ASGS" && stab_cfg.method != "OSGS"
        throw(ArgumentError("Stabilization method must be strictly 'ASGS' or 'OSGS'. Passed: $(stab_cfg.method)"))
    end
    if sol_cfg.ftol <= 0.0 || sol_cfg.stagnation_noise_floor <= 0.0
        throw(ArgumentError("Solver outer tolerances (ftol, stagnation_noise_floor) must be strictly positive floats."))
    end
end

struct FETopology
    X
    Y
    model
    Ω
    dΩ
    V_free
    Q_free
    h_cf
    f_cf
    alpha_cf
    g_cf
end

struct VMSFormulation
    form
    c_1
    c_2
end

struct StageSolvers
    picard
    newton
end

# ==============================================================================
# Post-convergence verification seam (observer; production default is a no-op)
# ==============================================================================
# The core solver is verification-blind: at each convergence point it invokes a hook on a
# `SolutionVerifier`. Production passes `NoVerification` (multiple dispatch resolves the hooks to
# no-ops, so it costs nothing). The MMS harness passes an `MMSPlateauVerifier`
# (`src/solvers/mms_verification.jl`) that owns the manufactured-solution oracle and the plateau loop.
# The core never names MMS, reads an oracle, or writes an `mms_*` diagnostic key — that is the
# verifier's concern. This keeps the optional Algorithm-D verification out of the main solver body.
abstract type SolutionVerifier end

"""Production default: the solver runs no post-convergence verification."""
struct NoVerification <: SolutionVerifier end

# The core calls ONLY these three; concrete verifiers (e.g. MMSPlateauVerifier) override them.
#   on_asgs_converged! — ASGS plateaus by extra 1-iteration Newton cycles, so it gets `step_once!`.
#   on_osgs_converged! — the OSGS coupled solve is a single solve (no stepping), so its hook is stateless.
on_asgs_converged!(::NoVerification, x, step_once!, diag, iter_count_ref) = nothing
on_osgs_converged!(::NoVerification, x, diag) = nothing
verification_result(::SolutionVerifier) = nothing   # 2nd return element of solve_system; MMS overrides

"""
    safe_fe_solve!(x, fesolver, op; backup=nothing)

Single-attempt FE-solver wrapper that absorbs the boilerplate every cascade
site in `solve_system` repeats: tuple unwrapping of Gridap's `solve!` return
value, non-finite-DOF guard with optional backup restoration, and exception
classification (Gridap's old "Reached maximum iterations" string vs other
exceptions).

Does NOT classify success/failure or do logging — the criteria differ across
the six call sites (Stage I Newton 1 / Picard / Newton 2 and OSGS inner
Newton 1 / Picard / Newton 2 retry), so each caller owns its own decision
logic. This helper exists purely to eliminate the duplicated try/catch +
tuple-unwrap + finite-check boilerplate.

Returns a `NamedTuple` with field `state`:
- `:ok` — finite state; also carries `iterations`, `residual_norm`,
  `initial_residual_norm`, `step_norm`, `stop_reason` from the result cache.
- `:nonfinite` — at least one DOF was non-finite; if `backup !== nothing`,
  the state was restored from `backup` before returning.
- `:max_iters_caught` — `solve!` threw a "Reached max(imum) iterations"
  exception. This is a legacy Gridap contract; `SafeNewtonSolver` uses
  `stop_reason = "max_iters_stagnation"` on the `:ok` path instead. Kept
  for defence against legacy or non-`SafeNewtonSolver` solver instances.
- `:exception` — `solve!` threw any other exception. Field `exc` holds it.

The backup restoration on `:nonfinite` is the only side effect beyond
`solve!`'s in-place mutation of `x`.
"""
function safe_fe_solve!(x, fesolver, op; backup=nothing)
    try
        res = solve!(x, fesolver, op)
        cache = res isa Tuple ? res[2] : res
        nls_cache = cache isa Tuple ? cache[2] : cache
        if any(!isfinite, get_free_dof_values(x))
            if backup !== nothing
                get_free_dof_values(x) .= backup
            end
            return (state = :nonfinite,)
        end
        r = nls_cache.result
        return (state = :ok,
                iterations = r.iterations,
                residual_norm = r.residual_norm,
                initial_residual_norm = r.initial_residual_norm,
                step_norm = r.step_norm,
                stop_reason = r.stop_reason,
                iteration_history = r.iteration_history,
                initial_residual_normalized = r.initial_residual_normalized,
                residual_normalized = r.residual_normalized)
    catch e
        msg = string(e)
        if occursin("Reached maximum iterations", msg) || occursin("Reached max iterations", msg)
            return (state = :max_iters_caught,)
        else
            return (state = :exception, exc = e)
        end
    end
end

# One raw Newton step on `op`: unwraps Gridap's `solve!` return and returns the result cache
# (`.iterations`, `.residual_norm`, …). Deliberately NOT wrapped in a try/catch — a verifier that
# needs to tolerate the legacy "Reached maximum iterations" exception owns that try/catch itself.
# Used by `on_asgs_converged!` for the plateau loop's single-iteration re-solves.
function _solve_one_step!(x, fesolver, op)
    res = solve!(x, fesolver, op)
    cache = res isa Tuple ? res[2] : res
    nls_cache = cache isa Tuple ? cache[2] : cache
    return nls_cache.result
end

# [trajectory] Append one algorithm-stage record to diag_cache["trajectory"], capturing a case's exact
# path through the nonlinear orchestration (Algorithm O → B/C → A) for later analysis. `stage` is a
# documentation-aligned code (e.g. "B:StageI:N1", "C:OSGS[3]:Picard"). `res` is a `safe_fe_solve!`
# NamedTuple; non-:ok early-return states carry only `state`, so the numeric fields fall back to defaults.
function _record_stage!(diag_cache, stage::String, res)
    haskey(diag_cache, "trajectory") || (diag_cache["trajectory"] = Any[])
    push!(diag_cache["trajectory"], (
        stage       = stage,
        state       = String(res.state),
        stop_reason = get(res, :stop_reason, ""),
        iters       = get(res, :iterations, 0),
        res_in      = get(res, :initial_residual_norm, NaN),
        res_out     = get(res, :residual_norm, NaN),
        # [trajectory] normalized residuals (per-field convergence quantity, ≤1 ⟺ converged); the
        # plotter prefers these and falls back to the inf-norms above when they are absent.
        res_in_norm  = get(res, :initial_residual_normalized, NaN),
        res_out_norm = get(res, :residual_normalized, NaN),
        history     = get(res, :iteration_history, NamedTuple[]),
    ))
end

# ==============================================================================
# Shared cascade success-policy (Algorithm B)
# ==============================================================================
# The ASGS Stage-I cascade (`_initialize_asgs_state!`) and the coupled OSGS solve
# (`solve_osgs_stage!`) both reuse this Algorithm-B cascade. Each classifies its own
# stop-reason, which is the *only* place they legitimately differ. `CascadePolicy` +
# `cascade_step_outcome` make that difference one explicit, shared decision instead of
# duplicated control flow, and give the Newton↔Picard ping-pong a single reusable verdict
# primitive.
#
# THREE policy flags are required (not two): Stage-I Newton-2 accepts a noise-floor
# finish but rejects a soft stall (xtol/max-iters/no-progress), whereas the OSGS inner
# accepts both — so `accept_noise_floor` and `accept_soft_stall` must be independent.
#
#                       accept_noise_floor | accept_soft_stall | max_iters_caught_is_failure
#   STAGE_I_POLICY      false                false               true   (N1: only ftol → success; else → Picard)
#   STAGE_I_N2_POLICY   true                 false               true   (N2: ftol+noise_floor → success; soft/struct → fail)
#   OSGS_INNER_POLICY   true                 true                false  (accept ftol/noise/soft; struct → Picard; max_iters_caught → 1-iter)
#
# Verdict vocabulary (caller owns logging, diagnostics, iter accounting, backup restore):
#   :success           — accept this Newton/Picard finish.
#   :one_iter_success  — a legacy `:max_iters_caught` counted as a 1-iteration success (OSGS only).
#   :reject            — not a success; the caller decides (Stage-I N1 → Picard; Stage-I N2 → fail;
#                        OSGS N1 → Picard; OSGS N2 → abort outer loop).
struct CascadePolicy
    accept_noise_floor::Bool
    accept_soft_stall::Bool
    max_iters_caught_is_failure::Bool
end

const STAGE_I_POLICY    = CascadePolicy(false, false, true)
const STAGE_I_N2_POLICY = CascadePolicy(true,  false, true)
const OSGS_INNER_POLICY = CascadePolicy(true,  true,  false)

"""
    cascade_step_outcome(res, policy::CascadePolicy) -> Symbol

Pure verdict for one Algorithm-B cascade segment from a `safe_fe_solve!` result + a
`CascadePolicy`. Returns `:success`, `:one_iter_success`, or `:reject`. The caller keeps
all side effects (logging, diagnostic pushes, iter-count accounting, backup restoration).
"""
function cascade_step_outcome(res, policy::CascadePolicy)
    st = res.state
    if st === :nonfinite || st === :exception
        return :reject
    elseif st === :max_iters_caught
        return policy.max_iters_caught_is_failure ? :reject : :one_iter_success
    else  # :ok — branch on the per-field solver verdict
        sr = res.stop_reason
        if sr == "ftol_reached" || sr == "initial_ftol"
            return :success
        elseif sr == "stagnation_noise_floor_reached"
            return policy.accept_noise_floor ? :success : :reject
        elseif sr == "linesearch_failed" || sr == "merit_divergence_escaped" || sr == "residual_divergence_escaped" || sr == "linear_solve_nan"
            return :reject  # structural failures are never accepted
        else  # xtol_stagnation / max_iters_stagnation / no_progress_stall — a "soft stall"
            return policy.accept_soft_stall ? :success : :reject
        end
    end
end

# ==============================================================================
# _pingpong_cascade!  (adaptive Newton↔Picard scheduling, opt-in)
# ==============================================================================
"""
    _pingpong_cascade!(final_x0, restore_vec, op_newton, op_picard, solver_newton,
                       solver_picard_gain, policy::CascadePolicy, diag_cache, iter_count_ref;
                       stage_prefix::String, max_swaps::Int) -> Symbol

Adaptive Newton↔Picard ping-pong shared by Stage I (ASGS) and the OSGS inner cascade. Runs Newton
until it stalls (a non-success `cascade_step_outcome` under `policy`), then a Picard segment that stops
the moment it has driven ‖R‖∞ down `picard_gain_target` orders (`solver_picard_gain` carries the gain
target → stop_reason "picard_gain_reached") — just enough to re-enter Newton's basin — then back to
Newton, bounded by `max_swaps`. Each segment is tagged honestly via `_record_stage!`
(`<stage_prefix>:PP[swap]:N` / `:P`) — never relabel a Picard step as Newton.

Returns `:success` (a Newton segment was accepted), `:structural_abort` (a non-finite/exception/merit-
divergence segment, OR a Picard segment that fails to reduce ‖R‖ — neither solver can advance, so the
caller fails and the homotopy drops eps_pert; restore done), or `:reject` (swaps exhausted). Safeguards
(merit/line-search/divergence within each Newton segment, monotone-residual within each Picard segment)
are unchanged — this only schedules *between* solves. Inert unless the caller opts in (pingpong_enabled).
"""
function _pingpong_cascade!(final_x0, restore_vec, op_newton, op_picard, solver_newton,
                            solver_picard_gain, policy::CascadePolicy, diag_cache, iter_count_ref;
                            stage_prefix::String, max_swaps::Int)
    # Account a Newton segment's iters + diagnostics (mirrors the cascade bookkeeping) and return its verdict.
    function _account_newton!(res, label)
        _record_stage!(diag_cache, label, res)
        if res.state == :ok
            iter_count_ref[] += res.iterations
            diag_cache["final_residual_norm"] = res.residual_norm
            get!(diag_cache, "initial_residual_norm", res.initial_residual_norm)
        elseif res.state == :max_iters_caught
            iter_count_ref[] += 1
        end
        return cascade_step_outcome(res, policy)
    end

    # Segment 0 — Newton from the entry iterate. (backup=restore_vec ⇒ helper auto-restores on non-finite.)
    res = safe_fe_solve!(final_x0, solver_newton, op_newton; backup=restore_vec)
    outcome = _account_newton!(res, "$(stage_prefix):PP[0]:N")
    if outcome == :success || outcome == :one_iter_success
        return :success
    end

    swap = 0
    while swap < max_swaps
        # Picard segment — globalize until the gain target re-enters Newton's basin.
        res_p = safe_fe_solve!(final_x0, solver_picard_gain, op_picard; backup=restore_vec)
        _record_stage!(diag_cache, "$(stage_prefix):PP[$(swap)]:P", res_p)
        if res_p.state == :ok
            iter_count_ref[] += res_p.iterations
            diag_cache["final_residual_norm"] = res_p.residual_norm
        elseif res_p.state == :max_iters_caught
            iter_count_ref[] += 1
        end
        # [ping-pong] The Picard segment is the globalizer after Newton diverged. If it ALSO fails to make
        # progress — a structural failure (non-finite / exception / merit-divergence), a depleted line
        # search, or simply no reduction in ‖R‖ — then NEITHER solver can advance from this iterate. Deem
        # the attempt failed (restore the entry iterate) so the caller fails and the OUTER homotopy loop
        # drops eps_pert, instead of repeating the identical Newton↔Picard cycle to max_swaps.
        picard_no_progress = res_p.state == :ok &&
            (res_p.stop_reason == "merit_divergence_escaped" ||
             res_p.stop_reason == "linesearch_failed" ||
             res_p.residual_norm >= res_p.initial_residual_norm)
        if res_p.state == :nonfinite || res_p.state == :exception ||
           any(!isfinite, get_free_dof_values(final_x0)) || picard_no_progress
            get_free_dof_values(final_x0) .= restore_vec
            return :structural_abort
        end

        # Newton segment on the Picard-smoothed iterate.
        res_n = safe_fe_solve!(final_x0, solver_newton, op_newton; backup=restore_vec)
        outcome = _account_newton!(res_n, "$(stage_prefix):PP[$(swap+1)]:N")
        if outcome == :success || outcome == :one_iter_success
            return :success
        elseif res_n.state == :nonfinite || res_n.state == :exception
            return :structural_abort  # helper restored; cannot recover
        end
        swap += 1
    end
    return :reject  # swaps exhausted without success → caller falls through to terminal logic
end

# ==============================================================================
# _initialize_asgs_state!  (Algorithm B, Stage I ASGS init)
# ==============================================================================
"""
    _initialize_asgs_state!(x0, x0_backup, op_newton_init, op_picard_init,
                             solver_newton_asgs, solver_picard, ftol,
                             diag_cache, iter_count_ref) → success::Bool

Implements Algorithm B (RobustNonlinearCascade) of `theory/osgs_algorithm/osgs_algorithm.tex`
for the Stage I ASGS initialisation. Owns the algebraic initialisation cascade
(Newton → Picard → Newton). The init-stage operators (`op_*_init`) and
`solver_newton_asgs` are constructed by the caller because they close over
`setup`/`formulation`/`phys_cfg`; the `x0_backup` allocation also lives in the
caller. Mutates `x0`, writes `diag_cache["final_residual_norm"]`, and
accumulates `iter_count_ref`.

Internal policy (Stage I addendum): Newton-1 that hits the noise floor but not
`ftol` does **not** mark success — it falls through to Picard to guarantee
entering the quadratic basin. (This asymmetry was historically contrasted with the now-removed
OSGS inner cascade; the coupled OSGS solve reuses the same shared cascade policy.)
"""
function _initialize_asgs_state!(x0, x0_backup, op_newton_init, op_picard_init,
                                  solver_newton_asgs, solver_picard, ftol,
                                  diag_cache, iter_count_ref;
                                  pingpong_enabled::Bool=false, pingpong_max_swaps::Int=0,
                                  solver_picard_gain=nothing)
    # Opt-in adaptive Newton↔Picard ping-pong replaces the one-way Newton→Picard→Newton cascade.
    # Default OFF ⇒ the existing body below runs unchanged (bit-identical). `solver_picard_gain` is the
    # Picard solver carrying the gain-then-return target (built by the caller when pingpong_enabled);
    # Stage I runs Newton at the full `newton_iterations` budget, so ping-pong is effective here.
    if pingpong_enabled
        v = _pingpong_cascade!(x0, x0_backup, op_newton_init, op_picard_init,
                               solver_newton_asgs, solver_picard_gain, STAGE_I_POLICY,
                               diag_cache, iter_count_ref;
                               stage_prefix="B:StageI", max_swaps=pingpong_max_swaps)
        return v == :success
    end

    success = false
    println("      -> ASGS Initializer: Attempting Exact Newton solver initially...")
    newton_success = false
    res = safe_fe_solve!(x0, solver_newton_asgs, op_newton_init; backup=x0_backup)
    _record_stage!(diag_cache, "B:StageI:N1", res)
    if res.state == :nonfinite
        println("      -> ASGS Initializer: Exact Newton produced non-finite state. Restoring backup, falling through to Picard.")
        newton_success = false
    elseif res.state == :exception
        println("      -> ASGS Initializer: Newton ConvergenceError. Exception: ", res.exc)
        newton_success = false
    elseif res.state == :max_iters_caught
        # "Reached max iterations" at the initializer counts as a non-success (logged as a
        # ConvergenceError); Stage I demands genuine convergence before it may bypass Picard.
        println("      -> ASGS Initializer: Newton ConvergenceError. Exception: Reached maximum iterations")
        newton_success = false
    else  # :ok
        final_res = res.iterations > 0 ? res.residual_norm : 0.0
        diag_cache["final_residual_norm"] = final_res
        get!(diag_cache, "initial_residual_norm", res.initial_residual_norm)
        iter_count_ref[] += res.iterations
        # [stage-I success = the solver's OWN per-field convergence verdict, not an absolute floor]
        # Bypass Picard whenever Newton reached its (per-field, dynamic) ftol target — stop_reason
        # "ftol_reached"/"initial_ftol" — i.e. it is already in the exact-Newton quadratic basin.
        # Keying on `final_res <= ftol` would be wrong: that scalar `ftol` is a machine-precision FLOOR
        # (10·eps in the MMS harness), not the convergence target, so a per-field-converging solve almost
        # never meets it and Picard would fire on every cell. A stagnation / xtol / max-iters / divergence
        # exit still falls through to Picard to secure the quadratic basin before OSGS (B6).
        # STAGE_I_POLICY rejects noise-floor + soft-stall (accept_*=false), so for an :ok finish
        # `cascade_step_outcome == :success` ⟺ stop_reason ∈ {ftol_reached, initial_ftol} (the B6 rule).
        if cascade_step_outcome(res, STAGE_I_POLICY) == :success
            newton_success = true
            println("      -> ASGS Initializer: Exact Newton reached its ftol target (stop=$(res.stop_reason), ‖R‖=$(final_res)) — already in the quadratic basin. Bypassing Picard.")
        else
            println("      -> ASGS Initializer: Exact Newton exited without reaching ftol (stop=$(res.stop_reason), ‖R‖=$(final_res)). Engaging Picard to secure the quadratic basin (B6).")
        end
    end

    if newton_success
        success = true
    else
        println("      -> ASGS Initializer: Exact Newton structurally aborted loop without geometric saturation. Orchestrating Picard Homotopy fallback...")
        get_free_dof_values(x0) .= x0_backup

        # Backup is intentionally NOT passed to the helper here: Picard can go non-finite mid-iteration
        # without throwing, so the explicit `any(!isfinite, …)` check below handles that divergence.
        picard_diverged = false
        res_p = safe_fe_solve!(x0, solver_picard, op_picard_init)
        _record_stage!(diag_cache, "B:StageI:Picard", res_p)
        if res_p.state == :ok
            final_res_picard = res_p.residual_norm
            diag_cache["final_residual_norm"] = final_res_picard
            get!(diag_cache, "initial_residual_norm", res_p.initial_residual_norm)
            iter_count_ref[] += res_p.iterations
            # [picard-divergence early-exit] Picard's role here is to globalise a Newton
            # failure; if Picard *itself* diverges (merit escaped — FINITE but exploding,
            # so the `any(!isfinite)` guard below misses it) the current eps_pert is hopeless.
            # Bail now instead of burning an expensive Newton-2 retry on a diverging iterate,
            # so the caller falls back to a milder eps_pert.
            picard_diverged = (res_p.stop_reason == "merit_divergence_escaped")
            if final_res_picard <= ftol
                println("      -> ASGS Initializer: Picard fully converged! Escaping to evaluation.")
                success = true
            end
        elseif res_p.state == :max_iters_caught
            println("      -> ASGS Initializer: Picard concluded initial smoothing loops.")
        elseif res_p.state == :exception
            println("      -> ASGS Initializer: Picard threw unrecoverable error. Exception: ", res_p.exc)
            get_free_dof_values(x0) .= x0_backup
        end
        # `:nonfinite` is caught by the `any(!isfinite, …)` check below; the helper does not
        # auto-restore the backup here because no `backup=` was passed.

        if any(!isfinite, get_free_dof_values(x0)) || picard_diverged
            println("      -> ASGS Initializer: Picard diverged (merit escaped / non-finite). Restoring backup; skipping Newton retry to fall back to a milder eps_pert.")
            get_free_dof_values(x0) .= x0_backup
            success = false
        elseif !success
            println("      -> ASGS Initializer: Picard smoothing finalized. Re-engaging Exact Newton...")
            res2 = safe_fe_solve!(x0, solver_newton_asgs, op_newton_init; backup=x0_backup)
            _record_stage!(diag_cache, "B:StageI:N2", res2)
            if res2.state == :nonfinite
                println("      -> ASGS Initializer: Newton Homotopy Pass produced non-finite state. Restoring backup, marking failure.")
                success = false
            elseif res2.state == :exception
                println("      -> ASGS Initializer: Newton ConvergenceError on Homotopy Pass. Exception: ", res2.exc)
                success = false
            elseif res2.state == :max_iters_caught
                # "Reached max iterations" on the homotopy pass is treated as a ConvergenceError (failure).
                println("      -> ASGS Initializer: Newton ConvergenceError on Homotopy Pass. Exception: Reached maximum iterations")
                success = false
            else  # :ok
                iter_count_ref[] += res2.iterations
                final_res = res2.residual_norm
                diag_cache["final_residual_norm"] = final_res
                get!(diag_cache, "initial_residual_norm", res2.initial_residual_norm)

                # STAGE_I_N2_POLICY accepts ftol + noise-floor (accept_noise_floor=true) but rejects
                # a soft stall (accept_soft_stall=false) and structural failures — i.e. Newton-2 may exit
                # on the noise floor (unlike Newton-1) but not on xtol/max-iters. The three-way logging is
                # preserved; only the `success` verdict is routed through the shared policy.
                stop_reason = res2.stop_reason
                if cascade_step_outcome(res2, STAGE_I_N2_POLICY) == :success
                    if stop_reason == "ftol_reached" || stop_reason == "initial_ftol"
                        println("      -> ASGS Initializer: Newton Homotopy Pass achieved exact theoretical tolerance ($ftol).")
                    else  # stagnation_noise_floor_reached (the only other :success under STAGE_I_N2_POLICY)
                        println("      -> ASGS Initializer: Newton Homotopy Pass cleanly saturated at numerical noise floor ($final_res). Approving.")
                    end
                    success = true
                else
                    println("      -> ASGS Initializer: Newton Homotopy Pass structurally failed (Reason: $stop_reason). Bounding collapse.")
                    success = false
                end
            end
        end
    end

    return success
end

"""
    solve_system(...)

Algorithm O (SimulationOrchestration) of `theory/osgs_algorithm/osgs_algorithm.tex`. Orchestrates
the nonlinear solver iteration loop over the chosen Variational Multiscale space: Stage-I ASGS
algebraic initialisation → method dispatch (ASGS, or the OSGS coupled solve in `osgs_solver.jl`),
with optional post-convergence verification through the `SolutionVerifier` hooks.

See the module-level "Algorithm-to-code mapping" docstring at the top of this file, and
`docs/solver/algorithm-code-mapping.md`.
"""
function solve_system(setup::FETopology, formulation::VMSFormulation, iter_solvers::StageSolvers,
                      config::PorousNSConfig, x0;
                      diagnostics_cache=nothing, verifier::SolutionVerifier=NoVerification())

    # The orchestrator only needs the trial/test spaces (for the init operators) and the solver pair;
    # the OSGS-specific fields (projection spaces, h/forcing CellFields) are unpacked from `setup` inside
    # `solve_osgs_stage!`, and the init operators take `setup`/`formulation` whole.
    X, Y = setup.X, setup.Y
    solver_picard, solver_newton = iter_solvers.picard, iter_solvers.newton

    phys_cfg = config.physical_properties
    stab_cfg = config.numerical_method.stabilization
    sol_cfg  = config.numerical_method.solver
    diag_cache = isnothing(diagnostics_cache) ? Dict{String, Any}() : diagnostics_cache
    pi_u = nothing
    pi_p = nothing


    local final_x0 = x0
    local success = false
    local eval_time = 0.0
    iter_count_ref = Ref(0)

    check_porous_solver_parameters(stab_cfg, sol_cfg)

    method = stab_cfg.method
    ftol = solver_newton.nls.ftol
    freeze_cusp = sol_cfg.freeze_jacobian_cusp

    # ==============================================================================
    # ALGEBRAIC INITIALIZATION (Stage I)
    # Even for OSGS, we first boot the global PDE with the zero-projection (ASGS) formulation, to map
    # the nonlinear fields into the exact-Newton quadratic basin before the OSGS coupled solve runs.
    # ==============================================================================
    res_fn_init(x, y) = build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing)
    jac_picard_init(x, dx, y) = build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0)
    jac_newton_init(x, dx, y) = sol_cfg.ablation_mode == "picard_only" ?
        build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0) :
        build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=nothing, pi_p=nothing)

    op_picard_init = FEOperator(res_fn_init, jac_picard_init, X, Y)
    op_newton_init = FEOperator(res_fn_init, jac_newton_init, X, Y)

    base_nls_global = solver_newton.nls
    # Verbatim re-wrap of the configured Newton solver (no field overrides) —
    # the explicit FESolver(_with_overrides(base_nls_global)) form expresses
    # "use the configured solver verbatim" without depending on `solver_newton`'s
    # FESolver wrapper identity.
    solver_newton_asgs = FESolver(_with_overrides(base_nls_global))

    x0_backup = copy(get_free_dof_values(x0))

    # Ping-pong scheduling (opt-in; default off ⇒ bit-identical one-way cascade). When enabled, the
    # Stage-I Picard segment uses a gain-then-return target so it stops the moment it re-enters Newton's
    # basin. The gain target is applied here (single point) so build_iter_solvers / call sites need not.
    pingpong_enabled = sol_cfg.pingpong_enabled
    pingpong_max_swaps = sol_cfg.pingpong_max_swaps
    solver_picard_gain = pingpong_enabled ?
        FESolver(_with_overrides(solver_picard.nls; picard_gain_target=sol_cfg.pingpong_picard_gain_orders)) :
        solver_picard

    # Stage I — Algorithm B (RobustNonlinearCascade), ASGS operators, restore=x0_backup.
    eval_time = @elapsed begin
        success = _initialize_asgs_state!(x0, x0_backup, op_newton_init, op_picard_init,
                                          solver_newton_asgs, solver_picard, ftol,
                                          diag_cache, iter_count_ref;
                                          pingpong_enabled=pingpong_enabled,
                                          pingpong_max_swaps=pingpong_max_swaps,
                                          solver_picard_gain=solver_picard_gain)
    end

    # The 2nd return element reports the verifier's outcome (`verification_result`): `nothing` for
    # production (NoVerification), or the MMS plateau verdict for the MMS harness. It is independent
    # of the inner-solver `success` flag, so callers can distinguish "solver converged but
    # verification failed (budget exhausted)" from "solver failed".
    if !success
        diag_cache["pi_u"] = pi_u
        diag_cache["pi_p"] = pi_p
        return false, verification_result(verifier), x0, iter_count_ref[], eval_time
    end

    final_x0 = x0

    if method == "ASGS"
        println("\n      [+] ASGS base convergence reached.")
        # Post-convergence verification hook (Algorithm D). Production (NoVerification) is a no-op;
        # the MMS harness runs the plateau loop. `step_once!` advances the converged state by one
        # Newton iteration on the SAME ASGS operator the boot used — built here because operator
        # construction lives in the core, but the verifier owns the loop.
        local_solver = FESolver(_with_overrides(solver_newton.nls; max_iters=1))
        step_once! = () -> _solve_one_step!(final_x0, local_solver, op_newton_init)
        eval_time += @elapsed on_asgs_converged!(verifier, final_x0, step_once!, diag_cache, iter_count_ref)
        diag_cache["base_convergence_reached"] = true   # core key (set regardless of verification)
        diag_cache["pi_u"] = pi_u
        diag_cache["pi_p"] = pi_p
        return success, verification_result(verifier), final_x0, iter_count_ref[], eval_time
    end

    if method == "OSGS"
        # Algorithm C — the coupled OSGS solve. All OSGS-specific setup (unconstrained projection
        # spaces, factored L² mass matrices) and the solve itself live in `solve_osgs_stage!`
        # (osgs_solver.jl); the orchestrator stays OSGS-agnostic apart from this dispatch. Stage I's
        # `success` (always `true` here — Stage-I failure short-circuit-returned earlier) is threaded
        # through so an OSGS solve that exhausts its budget without an explicit verdict keeps it.
        (success, pi_u, pi_p, osgs_elapsed) = solve_osgs_stage!(
            success, final_x0, setup, formulation, config, solver_newton, verifier,
            diag_cache, iter_count_ref)

        eval_time += osgs_elapsed

        # solve_osgs_stage! already wrote diag_cache["pi_u"] and diag_cache["pi_p"].
        return success, verification_result(verifier), final_x0, iter_count_ref[], eval_time
    end
end
