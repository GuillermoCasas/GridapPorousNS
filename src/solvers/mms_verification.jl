# src/solvers/mms_verification.jl
#
# [available-option — Algorithm D] Method-of-Manufactured-Solutions plateau verification, implemented
# as a pluggable `SolutionVerifier`. The core solver (asgs_solver.jl/osgs_solver.jl) is verification-blind: it calls
# `on_asgs_converged!` / `on_osgs_converged!` at each convergence point. Production uses `NoVerification`
# (no-op); the MMS harnesses pass an `MMSPlateauVerifier`, which owns the manufactured-solution `oracle`
# and the plateau loop. See `theory/osgs_algorithm/osgs_algorithm.tex`, Algorithm D (VerifyMMSPlateau).
#
# Results channel: the verifier OWNS its outcome in its own fields (`plateau_reached`, `stop_reason`,
# `error_history`, `relative_change_history`) AND mirrors the legacy `diag["mms_*"]` keys, so existing
# harness read-sites (and `verification_result`, the 2nd element of `solve_system`'s return) are
# unchanged. The plateau loop body is a verbatim relocation of the former `_run_asgs_mms_extension!`.

Base.@kwdef mutable struct MMSPlateauVerifier{F} <: SolutionVerifier
    oracle::F                              # (uh, ph) -> (E_u_L2, E_p_L2, E_u_H1, E_p_H1)
    max_extra_cycles::Int
    require_consecutive_passes::Int
    tau_err::Float64
    eps_u_l2::Float64
    eps_u_h1::Float64
    eps_p_l2::Float64
    h_local::Float64
    kv::Int
    rate_check_factor::Float64
    # Results (owned by the verifier; also mirrored into diag["mms_*"]).
    plateau_reached::Union{Bool,Nothing} = nothing
    stop_reason::String = ""
    error_history::Vector = []
    relative_change_history::Vector = []
end

verification_result(v::MMSPlateauVerifier) = v.plateau_reached

"""
    on_asgs_converged!(v::MMSPlateauVerifier, x, step_once!, diag, iter_count_ref)

ASGS-branch MMS plateau loop (Algorithm D). Advances the converged ASGS state by extra single Newton
iterations (`step_once!`, built by the orchestrator on the same operator the boot used) and watches the
manufactured-solution error sequence until the relative change plateaus for `require_consecutive_passes`
consecutive cycles. Verbatim relocation of the former `_run_asgs_mms_extension!` body — the only
substitution is `solve!(...)` → `step_once!()`; the try/catch, `eps_* = baseline·h^rate` scaling, and
plateau arithmetic are unchanged. Mirrors outcomes into `diag["mms_*"]` and into the verifier's fields.
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
                # Legacy max-iters ConvergenceError on a verification re-solve: counted as a 1-iteration
                # pass (the outer plateau loop re-evaluates anyway). Purely cosmetic, no numeric effect.
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

        # §5.1: scale plateau floors by the FE discretization budget.
        # L² convergence rate is h^(kv+1); H¹ rate is h^kv. `eps_*` is a baseline (dimensionless)
        # coefficient; the effective floor is baseline × h^(rate).
        eps_u_l2_eff = v.eps_u_l2 * v.h_local^(v.kv + 1)
        eps_u_h1_eff = v.eps_u_h1 * v.h_local^v.kv
        eps_p_l2_eff = v.eps_p_l2 * v.h_local^(v.kv + 1)

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

            # §5.2: rate-aware sanity check. If the converged L² velocity error exceeds the FE accuracy
            # budget by more than `rate_check_factor`, the iteration plateaued at a non-FE-optimal state.
            # Flag via the stop reason; keep success in the calling context (the plateau IS established).
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

    # Own the results (also exposed via verification_result / the harness's diag reads).
    v.error_history = mms_err_hist
    v.relative_change_history = mms_rc_hist
    v.plateau_reached = get(diag, "mms_plateau_reached", false)
    v.stop_reason = get(diag, "mms_stop_reason", "")
    return nothing
end

"""
    on_osgs_converged!(v::MMSPlateauVerifier, x, diag)

OSGS-branch MMS verification (Algorithm D, OSGS path). The coupled OSGS solve is a single solve, so this
hook is stateless: one oracle evaluation on the converged state plus a sub-optimal-rate flag. Verbatim
relocation of the former OSGS-tail MMS block.
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
