# [must-test] P3a — pins the three CascadePolicy truth tables so a future edit cannot silently
# desymmetrize the ASGS Stage-I vs OSGS-inner cascade (refactor-brief B5/B6/B10). The shared verdict
# primitive `cascade_step_outcome(res, policy)` is what both cascades (and P4's ping-pong) route through;
# this test fixes its decision boundaries on synthetic `safe_fe_solve!` results.
#
# Policy matrix (the `CascadePolicy` type + interpreter live in `src/solvers/solver_core.jl`; the
# values below are defined in their method files — STAGE_I_* in `asgs_solver.jl`, OSGS_INNER in
# `osgs_solver.jl`):
#                       accept_noise_floor | accept_soft_stall | max_iters_caught_is_failure
#   STAGE_I_POLICY      false                false               true
#   STAGE_I_N2_POLICY   true                 false               true
#   OSGS_INNER_POLICY   true                 true                false

using Test
using PorousNSSolver
const _PNS = PorousNSSolver

# Synthetic safe_fe_solve! result: cascade_step_outcome only reads `.state` and `.stop_reason`.
_mkres(state; stop_reason = "") = (state = state, stop_reason = stop_reason)

@testset "CascadePolicy truth tables [P3a / B5,B6,B10]" begin
    SI  = _PNS.STAGE_I_POLICY
    SI2 = _PNS.STAGE_I_N2_POLICY
    OS  = _PNS.OSGS_INNER_POLICY
    out = _PNS.cascade_step_outcome

    # Converged finishes (ftol / initial_ftol): success under every policy.
    for p in (SI, SI2, OS), sr in ("ftol_reached", "initial_ftol")
        @test out(_mkres(:ok; stop_reason = sr), p) == :success
    end

    # Noise-floor finish: Stage-I N1 rejects (B6 — fall to Picard to secure the basin);
    # Stage-I N2 and OSGS-inner accept.
    @test out(_mkres(:ok; stop_reason = "stagnation_noise_floor_reached"), SI)  == :reject
    @test out(_mkres(:ok; stop_reason = "stagnation_noise_floor_reached"), SI2) == :success
    @test out(_mkres(:ok; stop_reason = "stagnation_noise_floor_reached"), OS)  == :success

    # Soft stalls (xtol / max-iters / no-progress): Stage-I N1 and N2 reject; OSGS-inner accepts
    # as outer-loop progress (the key asymmetry — N2 accepts noise-floor but NOT soft stalls).
    for sr in ("xtol_stagnation", "max_iters_stagnation", "no_progress_stall")
        @test out(_mkres(:ok; stop_reason = sr), SI)  == :reject
        @test out(_mkres(:ok; stop_reason = sr), SI2) == :reject
        @test out(_mkres(:ok; stop_reason = sr), OS)  == :success
    end

    # Structural failures: rejected under every policy.
    for p in (SI, SI2, OS), sr in ("linesearch_failed", "merit_divergence_escaped", "linear_solve_failed")
        @test out(_mkres(:ok; stop_reason = sr), p) == :reject
    end

    # Non-:ok states.
    for p in (SI, SI2, OS), st in (:nonfinite, :exception)
        @test out(_mkres(st), p) == :reject
    end
    # Legacy :max_iters_caught — Stage-I treats as failure; OSGS-inner as a 1-iteration success.
    @test out(_mkres(:max_iters_caught), SI)  == :reject
    @test out(_mkres(:max_iters_caught), SI2) == :reject
    @test out(_mkres(:max_iters_caught), OS)  == :one_iter_success
end
