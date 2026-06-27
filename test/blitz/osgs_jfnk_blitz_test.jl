# ==============================================================================================
# Nature & Intent:
# [JFNK Phase-1] Fast, FE-free unit checks for the Jacobian-Free Newton–Krylov inner solve added to the
# OSGS coupled solve (gate: docs/solver/jfnk-phase0-preconditioner-gate.md). Two concerns:
#   (1) CONFIG STRICTNESS (repo hard rule): the osgs_jfnk_* knobs are required (no silent default),
#       fail-loud on out-of-range values, and the two opt-in OSGS paths (JFNK / Anderson) are mutually
#       exclusive — both enabled is a configuration error, not a precedence to guess.
#   (2) The matrix-free building blocks are correct in isolation: `JFNKMatVec` reproduces J·v for an
#       affine residual to FD precision, and `JFNKPrecond` applies A⁻¹ via a factorization (both ldiv!
#       forms). The on-the-real-OSGS-operator equivalence (capturing ∂π/∂u) is the quick-tier test.
#
# Associated Files / Functions:
# - `src/solvers/linear_solvers.jl` (`JFNKLinearSolver`, `JFNKMatVec`, `JFNKPrecond`)
# - `src/config.jl` (`SolverConfig` osgs_jfnk_* fields; `validate!`)
# ==============================================================================================
using Test
using PorousNSSolver
using LinearAlgebra
using Gridap.Algebra
using JSON3
const _PNS = PorousNSSolver

# A complete config dict = base_config + the eps_val the strict path needs, with a solver override applied.
function _cfg_with_solver_overrides(overrides::Dict)
    base = joinpath(dirname(pathof(_PNS)), "..", "config", "base_config.json")
    d = JSON3.read(read(base, String), Dict{String, Any})
    d["physical_properties"]["eps_val"] = 1e-8
    for (k, v) in overrides
        d["numerical_method"]["solver"][k] = v
    end
    return d
end

@testset "blitz: OSGS JFNK config strictness + matrix-free units" begin

    @testset "osgs_jfnk_* knobs load and carry through" begin
        cfg = _PNS.load_config_from_dict(_cfg_with_solver_overrides(Dict{String,Any}()))
        sc = cfg.numerical_method.solver
        @test sc.osgs_jfnk_enabled == false                  # OFF by default (behavior-preserving)
        @test sc.osgs_jfnk_gmres_rel_tol > 0 && sc.osgs_jfnk_gmres_rel_tol < 1
        @test sc.osgs_jfnk_gmres_maxiter >= 1
        @test sc.osgs_jfnk_gmres_restart >= 1
        @test sc.osgs_jfnk_fd_epsilon > 0
    end

    @testset "validate! rejects both opt-in OSGS paths enabled (mutual exclusion)" begin
        d = _cfg_with_solver_overrides(Dict("osgs_jfnk_enabled" => true,
                                            "osgs_anderson_enabled" => true))
        @test_throws AssertionError _PNS.load_config_from_dict(d)
    end

    @testset "validate! rejects out-of-range JFNK knobs (fail-loud)" begin
        @test_throws AssertionError _PNS.load_config_from_dict(_cfg_with_solver_overrides(Dict("osgs_jfnk_gmres_rel_tol" => 1.5)))
        @test_throws AssertionError _PNS.load_config_from_dict(_cfg_with_solver_overrides(Dict("osgs_jfnk_gmres_rel_tol" => 0.0)))
        @test_throws AssertionError _PNS.load_config_from_dict(_cfg_with_solver_overrides(Dict("osgs_jfnk_gmres_maxiter" => 0)))
        @test_throws AssertionError _PNS.load_config_from_dict(_cfg_with_solver_overrides(Dict("osgs_jfnk_gmres_restart" => 0)))
        @test_throws AssertionError _PNS.load_config_from_dict(_cfg_with_solver_overrides(Dict("osgs_jfnk_fd_epsilon" => 0.0)))
    end

    @testset "JFNK + Anderson each enabled alone validate fine" begin
        @test _PNS.load_config_from_dict(_cfg_with_solver_overrides(Dict("osgs_jfnk_enabled" => true))) isa _PNS.PorousNSConfig
        @test _PNS.load_config_from_dict(_cfg_with_solver_overrides(Dict("osgs_anderson_enabled" => true))) isa _PNS.PorousNSConfig
    end

    @testset "JFNKMatVec reproduces J·v for an affine residual (FD precision)" begin
        # F(v) = A v + c  ⇒  J ≡ A. The directional FD must recover A·v exactly (to FD truncation).
        A = [2.0 1.0 0.0; 0.5 3.0 -0.4; 0.0 -0.2 2.5]
        c = [0.1, -0.2, 0.3]
        F = v -> A * v .+ c
        x0 = [0.5, -0.3, 0.8]
        mv = _PNS.JFNKMatVec(F, x0, F(x0), 1e-7)
        for v in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.3, -0.7, 0.4])
            y = similar(v); mul!(y, mv, v)
            @test norm(y .- A * v) / norm(A * v) < 1e-5
        end
        # zero direction ⇒ zero action (no division by ‖v‖=0)
        yz = ones(3); mul!(yz, mv, zeros(3)); @test all(iszero, yz)
        # the `*` form agrees with mul!
        @test norm(mv * [0.3, -0.7, 0.4] .- A * [0.3, -0.7, 0.4]) / norm(A * [0.3, -0.7, 0.4]) < 1e-5
    end

    @testset "JFNKPrecond applies A⁻¹ via a factorization (both ldiv! forms)" begin
        M = [4.0 1.0 0.0; 1.0 3.0 0.5; 0.0 0.5 2.0]
        ns = numerical_setup(symbolic_setup(LUSolver(), M), M)
        Pre = _PNS.JFNKPrecond(ns, zeros(3))
        r = [1.0, 2.0, -0.5]
        y = similar(r); ldiv!(y, Pre, r)              # 3-arg: y = M⁻¹ r
        @test norm(M * y .- r) / norm(r) < 1e-10
        rc = copy(r); ldiv!(Pre, rc)                  # 2-arg in-place: rc ← M⁻¹ rc
        @test norm(M * rc .- r) / norm(r) < 1e-10
        @test norm(rc .- y) < 1e-12                    # both forms agree
    end
end
