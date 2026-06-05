# [must-test] P4 — Picard gain-then-return stop predicate.
#
# Pins the gain-stop in `_safe_solve_inner!` (src/solvers/nonlinear.jl): in :picard mode with a FINITE
# `picard_gain_target`, the Picard segment stops as soon as ‖R‖∞ ≤ ‖R₀‖∞·10^(-picard_gain_target) — just
# enough to re-enter Newton's basin, so the ping-pong orchestrator hands back to Newton. Mirrored in pure
# Julia (same idiom as nonlinear_blitz_test.jl / stall_guard_blitz_test.jl) so the decision boundary is
# locked: it must be a DIMENSIONLESS relative drop off the frozen initial residual (encoding-invariant),
# inert when the target is Inf (default ⇒ every non-ping-pong Picard solve unchanged), and ignored in
# :newton mode. The swap orchestration itself routes its accept/reject verdict through
# `cascade_step_outcome`, pinned separately by cascade_policy_symmetry_blitz_test.jl.

using Test

"Iteration at which the P4 Picard gain-stop fires (mirror of the nonlinear.jl predicate), or 0 if never."
function _gain_stop_iter(r0::Float64, seq::Vector{Float64}; gain::Float64, mode::Symbol = :picard)
    (mode === :picard && isfinite(gain)) || return 0      # inert in :newton mode or with Inf target
    thresh = r0 * 10.0^(-gain)
    for (i, r) in enumerate(seq)
        r <= thresh && return i
    end
    return 0
end

@testset "Picard gain-then-return stop [P4]" begin
    # gain = 1.5 ⇒ stop once ‖R‖∞ has dropped 1.5 orders below r0 (thresh = r0·10^-1.5 ≈ 0.0316·r0).
    @test _gain_stop_iter(1.0, [0.5, 0.1, 0.03, 0.001]; gain = 1.5) == 3   # 0.03 ≤ 0.0316 first at iter 3
    @test _gain_stop_iter(1.0, [0.5, 0.04];             gain = 1.5) == 0   # 0.04 > 0.0316 ⇒ never fires

    # Dimensionless / scale-covariant: the threshold tracks r0, so a ∝U-rescaled residual stops at the
    # same RELATIVE drop (r0 = 100 ⇒ thresh ≈ 3.16).
    @test _gain_stop_iter(100.0, [50.0, 3.0, 1.0]; gain = 1.5) == 2        # 3.0 ≤ 3.16 first at iter 2

    # Inert when the target is Inf (the default ⇒ non-ping-pong Picard runs to its ftol/cap, unchanged).
    @test _gain_stop_iter(1.0, [1e-9, 1e-12]; gain = Inf) == 0
    # Ignored in :newton mode (the gain stop is Picard-only).
    @test _gain_stop_iter(1.0, [1e-9]; gain = 1.5, mode = :newton) == 0

    # A larger gain target demands a deeper drop before returning to Newton.
    @test _gain_stop_iter(1.0, [0.5, 0.05, 0.001]; gain = 1.0) == 2        # thresh 0.1: 0.5>0.1, 0.05≤0.1 at iter 2
    @test _gain_stop_iter(1.0, [0.1, 1e-2, 1e-3]; gain = 2.5) == 3         # thresh ≈3.2e-3: 1e-3 first ≤ at iter 3
end
