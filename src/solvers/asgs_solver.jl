# src/solvers/asgs_solver.jl
#=
    asgs_solver.jl

# Role
**ASGS (Algebraic Sub-Grid Scale) — the method-specific Stage-I box only.** The SGS space is the space
of finite element residuals; the projection onto it is the identity ``\widetilde{\Pi} = \mathcal{I}``, so
the FE projection ``\boldsymbol{\pi}_h = \boldsymbol{0}``. This file owns exactly two things that are
unique to ASGS:

1. `_initialize_asgs_state!` — Algorithm B (RobustNonlinearCascade), the Stage-I ASGS algebraic
   initialisation (Newton → Picard → Newton, or the opt-in Newton↔Picard ping-pong). It is also the
   boot phase the OSGS path runs *before* its coupled solve, so the orchestrator always calls it first.
2. `STAGE_I_POLICY` / `STAGE_I_N2_POLICY` — the two `CascadePolicy` *values* that say which solver
   stop-reasons Stage-I Newton-1 and Newton-2 accept. (The `CascadePolicy` *type*, the interpreter
   `cascade_step_outcome`, and the `_pingpong_cascade!` scheduler are the shared machinery and live in
   `solver_core.jl`; OSGS's counterpart value `OSGS_INNER_POLICY` lives in `osgs_solver.jl`.)

Everything ASGS shares with OSGS — the FE containers, the verifier seam, the FE-solve plumbing, the
cascade machinery, and the orchestrator `solve_system` — lives in `solver_core.jl` (included before
this file). The OSGS-specific coupled solve lives in `osgs_solver.jl` (included after this file). Reading
`asgs_solver.jl` and `osgs_solver.jl` side by side shows precisely where the two methods diverge.

# Algorithm-to-code mapping
See `docs/solver/algorithm-code-mapping.md`. `_initialize_asgs_state!` implements Algorithm B; the
orchestrator that calls it (`solve_system`, Algorithm O) and the shared cascade primitives are in
`solver_core.jl`.
=#
using Gridap
using Gridap.Algebra
using LinearAlgebra

# ==============================================================================
# ASGS cascade success-policies (Algorithm B) — what Stage I accepts
# ==============================================================================
# The two `CascadePolicy` rows Stage I uses (the type/interpreter/scheduler are shared, in
# `solver_core.jl`; see the full cross-method truth table there and in
# test/blitz/cascade_policy_symmetry_blitz_test.jl).
#                       accept_noise_floor | accept_soft_stall | max_iters_caught_is_failure
#   STAGE_I_POLICY      false                false               true   (N1: only ftol → success; else → Picard)
#   STAGE_I_N2_POLICY   true                 false               true   (N2: ftol+noise_floor → success; soft/struct → fail)
const STAGE_I_POLICY    = CascadePolicy(false, false, true)
const STAGE_I_N2_POLICY = CascadePolicy(true,  false, true)

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
`ftol` does **not** mark success — it falls through to Picard to secure entry
to the quadratic basin. The coupled OSGS solve reuses the same shared
cascade policy machinery (`solver_core.jl`).
"""
function _initialize_asgs_state!(x0, x0_backup, op_newton_init, op_picard_init,
                                  solver_newton_asgs, solver_picard, ftol,
                                  diag_cache, iter_count_ref;
                                  pingpong_enabled::Bool=false, pingpong_max_swaps::Int=0,
                                  solver_picard_gain=nothing)
    # Two Stage-I schedules are available. When `pingpong_enabled`, an adaptive Newton↔Picard
    # ping-pong runs instead of the default one-way Newton→Picard→Newton cascade in the body below.
    # `solver_picard_gain` is the Picard solver carrying the gain-then-return target (built by the
    # caller for the ping-pong path); Stage I runs Newton at the full `newton_iterations` budget, so
    # ping-pong is effective here.
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
                # on the noise floor (unlike Newton-1) but not on xtol/max-iters. The `success` verdict
                # is decided by the shared policy; the branches below only choose the log message.
                stop_reason = res2.stop_reason
                if cascade_step_outcome(res2, STAGE_I_N2_POLICY) == :success
                    if stop_reason == "ftol_reached" || stop_reason == "initial_ftol"
                        println("      -> ASGS Initializer: Newton Homotopy Pass achieved exact theoretical tolerance ($ftol).")
                    elseif stop_reason == "residual_floor_reached"
                        println("      -> ASGS Initializer: Newton Homotopy Pass converged at the residual floor ($final_res) — momentum scale-free-converged, mass gate envelope collapsed. Approving.")
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
