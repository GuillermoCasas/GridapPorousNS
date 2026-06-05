# [must-test] P1 — no-progress stall guard decision rule.
#
# Pins the stall-guard bookkeeping in `_safe_solve_inner!` (src/solvers/nonlinear.jl, the block that
# tracks `best_b_inf` / `last_improve_iter` and bails with stop_reason "no_progress_stall"). Mirrored
# in pure Julia (same idiom as nonlinear_blitz_test.jl, which checks the Armijo logic inline) so the
# decision boundary is locked without standing up a full MultiField NonlinearOperator. The replica
# below reproduces these exact expressions:
#   if r < best
#       if r < best*(1 - min_rel);  last_improve = i;  end
#       best = r
#   end
#   if window > 0 && (i - last_improve) >= window;  BAIL("no_progress_stall");  end
# with `best` seeded to the iter-0 residual `r0` and `last_improve = 0`.

using Test

"Return the 1-based iteration at which the stall guard bails, or 0 if it never does."
function _stall_bail_iter(r0::Float64, seq::Vector{Float64}; window::Int, min_rel::Float64)
    best = r0
    last_improve = 0
    for (i, r) in enumerate(seq)
        if r < best
            if r < best * (1.0 - min_rel)
                last_improve = i
            end
            best = r
        end
        if window > 0 && (i - last_improve) >= window
            return i
        end
    end
    return 0
end

@testset "no-progress stall guard [P1]" begin
    # 1) Flat residual (no improvement at all) bails exactly at i == window.
    @test _stall_bail_iter(1.0, fill(1.0, 10); window = 2, min_rel = 1e-3) == 2
    @test _stall_bail_iter(1.0, fill(1.0, 10); window = 4, min_rel = 1e-3) == 4

    # 2) Disabled when window == 0 (the shipped default): never bails, even on a flat sequence.
    @test _stall_bail_iter(1.0, fill(1.0, 50); window = 0, min_rel = 1e-3) == 0

    # 3) P1 CAVEAT: a single-step plateau followed by a large (quadratic) drop must NOT trip the guard.
    #    i=1 plateaus (improvement below min_rel ⇒ window counter not reset), i=2 drops 6 orders ⇒ resets.
    @test _stall_bail_iter(1.0, [0.9995, 1e-6, 1e-12]; window = 2, min_rel = 1e-3) == 0
    #    And genuine quadratic descent (each step improves by many orders) never trips it.
    @test _stall_bail_iter(1.0, [1e-2, 1e-4, 1e-8, 1e-16]; window = 2, min_rel = 1e-3) == 0

    # 4) Improvements SMALLER than min_rel do not count as progress ⇒ sustained micro-creep still bails.
    #    Each step improves by 5e-4 < min_rel=1e-3, so last_improve never advances ⇒ bail at i == window.
    @test _stall_bail_iter(1.0, [0.9995, 0.9990, 0.9985, 0.9980]; window = 2, min_rel = 1e-3) == 2

    # 5) A real descent that stalls only after several good steps bails `window` iters after the last gain.
    #    Improves through i=3 (last_improve=3), then flat ⇒ bail at i = 3 + window = 5.
    @test _stall_bail_iter(1.0, [0.5, 0.2, 0.05, 0.05, 0.05, 0.05]; window = 2, min_rel = 1e-3) == 5
end
