# src/solvers/mms_verification.jl
#
# [paper-faithful — Algorithm D] Method-of-Manufactured-Solutions (MMS) plateau verification, plugged
# into the solver via the `SolutionVerifier` seam defined in solver_core.jl. The core solver is
# verification-agnostic: after each convergence point it calls the hooks `on_asgs_converged!` /
# `on_osgs_converged!`. Production runs pass `NoVerification` (the hooks are no-ops); the MMS harnesses
# pass an `MMSPlateauVerifier`, which carries the manufactured-solution `oracle` and drives the plateau
# loop. See `theory/osgs_algorithm/osgs_algorithm.tex`, Algorithm D (VerifyMMSPlateau).
#
# What "plateau" means: as Newton iterates past the residual-based convergence point, the
# manufactured-solution error E should stop changing (it has reached the FE discretization floor, not
# zero). Verification confirms the relative change in E flattens out, and optionally that the floor it
# settles at is FE-optimal (consistent with the O(h^{k+1}) accuracy budget).
#
# Results channel: the verifier records its outcome both in its own fields (`plateau_reached`,
# `stop_reason`, `error_history`, `relative_change_history`) and in the `diag["mms_*"]` dictionary keys
# the harnesses read. The verifier's `plateau_reached` is also surfaced as `verification_result`, the
# 2nd element of `solve_system`'s return tuple.

# Holds everything needed to verify an MMS plateau at a single mesh resolution: the error oracle, the
# plateau-loop budget/tolerances, and the discretization scales used to convert dimensionless floors into
# h-dependent effective thresholds. One verifier instance corresponds to one (h, kv) point of an MMS sweep.
Base.@kwdef mutable struct MMSPlateauVerifier{F} <: SolutionVerifier
    oracle::F                              # closure (uh, ph) -> (E_u_L2, E_p_L2, E_u_H1, E_p_H1): the four
                                           # manufactured-solution error norms on the current FE state.
    max_extra_cycles::Int                  # cap on extra Newton cycles spent watching the error sequence.
    require_consecutive_passes::Int        # how many consecutive cycles must satisfy the plateau test before declaring success.
    tau_err::Float64                       # plateau tolerance: relative change in E must fall below this.
    eps_u_l2::Float64                      # dimensionless baseline floor for the velocity L² ratio denominator.
    eps_u_h1::Float64                      # dimensionless baseline floor for the velocity H¹ ratio denominator.
    eps_p_l2::Float64                      # dimensionless baseline floor for the pressure L² ratio denominator.
    h_local::Float64                       # mesh size h at this point; scales the floors and the rate-check budget.
    kv::Int                                # velocity polynomial order k; sets the convergence rates (L²: h^{k+1}, H¹: h^k).
    rate_check_factor::Float64             # slack multiplier on the FE accuracy budget for the optimal-rate sanity check.
    # Outcome fields, populated by the hooks (also mirrored into diag["mms_*"]).
    plateau_reached::Union{Bool,Nothing} = nothing   # verdict; nothing until a hook runs.
    stop_reason::String = ""                          # why the loop stopped (e.g. "mms_plateau_satisfied").
    error_history::Vector = []                        # the E tuple recorded at each cycle.
    relative_change_history::Vector = []              # the per-cycle relative-change ratios (r_u2, r_p2, r_u1).
end

# Surfaces the plateau verdict as the 2nd element of solve_system's return (overrides the
# NoVerification default, which returns nothing).
verification_result(v::MMSPlateauVerifier) = v.plateau_reached

"""
    on_asgs_converged!(v::MMSPlateauVerifier, x, step_once!, diag, iter_count_ref)

ASGS-branch MMS plateau loop (Algorithm D). Called once the ASGS Newton solve has converged on residual.
It advances the converged state by extra single Newton iterations (`step_once!`, a closure supplied by
the solver core that re-solves on the same operator the boot used) and watches the manufactured-solution
error sequence. Each cycle: take one Newton step, re-evaluate the four error norms via the `oracle`,
form the relative change of each against the previous cycle, and test it against `tau_err`. The loop
declares success once the test passes for `require_consecutive_passes` consecutive cycles, or gives up
after `max_extra_cycles`. `iter_count_ref` accumulates the iteration count so the harness reports a
faithful total. Outcomes land both in `diag["mms_*"]` and in the verifier's own fields.
"""
function on_asgs_converged!(v::MMSPlateauVerifier, x, step_once!, diag, iter_count_ref)
    println("\n      [!] Commencing ASGS MMS Error Verification Plateau Loop...")

    diag["base_convergence_reached"] = true
    diag["mms_plateau_reached"] = false
    mms_err_hist = []
    mms_rc_hist = []

    E_u2_0, E_p2_0, E_u1_0, E_p1_0 = v.oracle(x...)
    push!(mms_err_hist, (E_u2_0, E_p2_0, E_u1_0, E_p1_0))

    consecutive_passes = 0

    for cycle in 1:v.max_extra_cycles
        try
            r = step_once!()
            iter_count_ref[] += r.iterations
            diag["final_residual_norm"] = r.residual_norm
        catch e
            if occursin("Reached maximum iterations", string(e)) || occursin("Reached max iterations", string(e))
                # A max-iters ConvergenceError on a verification re-solve is benign here: at the plateau
                # the state is already converged, so we count it as a 1-iteration pass and let the outer
                # loop re-evaluate the error. Only the iteration tally is affected, not the verdict.
                println("      -> ASGS MMS Verification: Newton ConvergenceError (Reached maximum iterations) — counting as a 1-iteration verification pass.")
                iter_count_ref[] += 1
            else
                println("      -> ASGS MMS Verification Newton failed. Aborting extension. Exception: ", e)
                diag["mms_stop_reason"] = "nonlinear_failure"
                break
            end
        end

        E_u2_k, E_p2_k, E_u1_k, E_p1_k = v.oracle(x...)
        push!(mms_err_hist, (E_u2_k, E_p2_k, E_u1_k, E_p1_k))

        E_u2_prev, E_p2_prev, E_u1_prev, E_p1_prev = mms_err_hist[end-1]

        # §5.1: scale the plateau floors by the FE discretization budget so the relative-change test is
        # measured against errors of the right order. The expected convergence rates are L²: h^(kv+1),
        # H¹: h^kv. Each `eps_*` is a dimensionless baseline; the effective floor is baseline × h^(rate).
        eps_u_l2_eff = v.eps_u_l2 * v.h_local^(v.kv + 1)
        eps_u_h1_eff = v.eps_u_h1 * v.h_local^v.kv
        eps_p_l2_eff = v.eps_p_l2 * v.h_local^(v.kv + 1)

        # Relative change of each error norm vs. the previous cycle. The effective floor enters the
        # denominator so that once E reaches the discretization floor the ratio collapses toward 0
        # (a true plateau) rather than blowing up on a vanishing denominator.
        r_u2 = abs(E_u2_k - E_u2_prev) / max(E_u2_k, E_u2_prev, eps_u_l2_eff)
        r_u1 = abs(E_u1_k - E_u1_prev) / max(E_u1_k, E_u1_prev, eps_u_h1_eff)
        r_p2 = abs(E_p2_k - E_p2_prev) / max(E_p2_k, E_p2_prev, eps_p_l2_eff)

        push!(mms_rc_hist, (r_u2, r_p2, r_u1))
        max_r = max(r_u2, r_u1, r_p2)

        println("        * [Verification Cycle $cycle] Plateau max ratio: $max_r (Target: $(v.tau_err))")

        if max_r < v.tau_err
            consecutive_passes += 1
        else
            consecutive_passes = 0
        end

        if consecutive_passes >= v.require_consecutive_passes
            println("        [+] ASGS MMS Plateau formally established natively ($consecutive_passes consecutive < $(v.tau_err)).")
            diag["mms_plateau_reached"] = true
            diag["mms_stop_reason"] = "mms_plateau_satisfied"

            # §5.2: rate-aware sanity check. A plateau only means the error stopped moving — it does not
            # guarantee the error settled at the FE-optimal floor. If the converged L² velocity error
            # exceeds the accuracy budget h^(kv+1) by more than `rate_check_factor`, the iteration
            # plateaued at a non-optimal state. We record this through the stop reason but keep the
            # plateau verdict as success, since a plateau IS established.
            budget = v.h_local^(v.kv + 1)
            if E_u2_k > v.rate_check_factor * budget
                diag["mms_stop_reason"] = "mms_plateau_at_suboptimal_rate"
                println("        [!] Plateau at sub-optimal rate: E_u2=$E_u2_k > $(v.rate_check_factor) × h^(kv+1)=$(v.rate_check_factor * budget). Flagged for inspection.")
            end

            break
        end
    end

    if !get(diag, "mms_plateau_reached", false) && !haskey(diag, "mms_stop_reason")
        diag["mms_stop_reason"] = "mms_budget_exhausted"
    end

    diag["mms_error_history"] = mms_err_hist
    diag["mms_relative_change_history"] = mms_rc_hist

    # Record the outcome on the verifier (also exposed via verification_result and the harness's diag reads).
    v.error_history = mms_err_hist
    v.relative_change_history = mms_rc_hist
    v.plateau_reached = get(diag, "mms_plateau_reached", false)
    v.stop_reason = get(diag, "mms_stop_reason", "")
    return nothing
end

"""
    on_osgs_converged!(v::MMSPlateauVerifier, x, diag)

OSGS-branch MMS verification (Algorithm D, OSGS path). Unlike ASGS, the OSGS coupled solve converges in a
single solve, so there is no plateau loop to run: this hook simply evaluates the four error norms once on
the converged state (the plateau is trivially established) and applies the same sub-optimal-rate flag as
the ASGS path — comparing the L² velocity error against `rate_check_factor` × h^(kv+1).
"""
function on_osgs_converged!(v::MMSPlateauVerifier, x, diag)
    E_u2, E_p2, E_u1, E_p1 = v.oracle(x...)
    diag["mms_plateau_reached"] = true
    diag["mms_stop_reason"] = "coupled_single_solve"
    budget = v.h_local^(v.kv + 1)
    if E_u2 > v.rate_check_factor * budget
        diag["mms_stop_reason"] = "coupled_at_suboptimal_rate"
        println("        [!] Coupled OSGS at sub-optimal rate: E_u2=$E_u2 > $(v.rate_check_factor)·h^(kv+1).")
    end
    v.plateau_reached = true
    v.error_history = [(E_u2, E_p2, E_u1, E_p1)]
    v.stop_reason = get(diag, "mms_stop_reason", "")
    return nothing
end
