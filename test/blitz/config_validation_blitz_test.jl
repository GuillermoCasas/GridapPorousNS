# ==============================================================================================
# Config validation contract test — the load-time `@assert` invariants added in the 2026-06 audit
# (C.2 / SOLV-04 / MODE-02/03, docs/formulation-audit-2026-06-24.md §C.2): reaction SPSD
# (σ_lin/σ_nonlin/σ_const ≥ 0), porosity/domain bounds (α₀ ∈ (0,1], r₁ < r₂, bounding_box parity),
# velocity-floor positivity (SmoothVelocityFloor stays C¹), and the scale-free convergence-gate
# tolerances (eps_tol_momentum/eps_tol_mass > 0). Each invalid value must fail LOUDLY at load.
#
# `load_config_from_dict` deep-merges the override onto `config/base_config.json`, then deserializes and
# runs `validate!`. base_config intentionally omits `physical_epsilon`, so every override below supplies it so the
# config is COMPLETE (deserialization passes) and the throw comes from `validate!`, not a missing field.
# ==============================================================================================

using Test
using PorousNSSolver

# physical_properties override that, merged onto base_config, yields a complete + valid config.
_phys(extra...) = merge(Dict("physical_epsilon" => 0.0), Dict(extra...))
_load(override) = PorousNSSolver.load_config_from_dict(override)

@testset "config validation: a valid config still loads (positive control)" begin
    cfg = _load(Dict("physical_properties" => _phys()))
    @test cfg isa PorousNSSolver.PorousNSConfig
end

@testset "config validation: reaction coefficients must be SPSD (≥ 0)" begin
    @test_throws Exception _load(Dict("physical_properties" => _phys("sigma_linear" => -1.0)))
    @test_throws Exception _load(Dict("physical_properties" => _phys("sigma_nonlinear" => -1.0)))
    @test_throws Exception _load(Dict("physical_properties" => _phys("sigma_constant" => -1.0)))
end

@testset "config validation: domain / porosity bounds" begin
    @test_throws Exception _load(Dict("physical_properties" => _phys(), "domain" => Dict("alpha_0" => 1.5)))   # α₀ > 1
    @test_throws Exception _load(Dict("physical_properties" => _phys(), "domain" => Dict("alpha_0" => 0.0)))   # α₀ ≤ 0
    @test_throws Exception _load(Dict("physical_properties" => _phys(), "domain" => Dict("r_1" => 0.6, "r_2" => 0.5)))  # r₁ ≥ r₂
end

@testset "config validation: velocity floor must stay C¹ (strictly positive)" begin
    # both base + h floor zero ⇒ SmoothVelocityFloor degenerates to sqrt(u·u) (non-differentiable at 0)
    @test_throws Exception _load(Dict("physical_properties" => _phys("u_base_floor_ref" => 0.0, "h_floor_weight" => 0.0)))
    @test_throws Exception _load(Dict("physical_properties" => _phys("epsilon_floor" => 0.0)))                 # denominator guard
end

@testset "config validation: scale-free convergence-gate tolerances > 0" begin
    @test_throws Exception _load(Dict("physical_properties" => _phys(),
        "numerical_method" => Dict("solver" => Dict("eps_tol_momentum" => 0.0))))
    @test_throws Exception _load(Dict("physical_properties" => _phys(),
        "numerical_method" => Dict("solver" => Dict("eps_tol_mass" => -1.0))))
end

# ----------------------------------------------------------------------------------------------
# Stabilization constant multipliers (added 2026-07-17). c₁ is NOT dimension-independent: the
# coercivity condition eq:conditions_on_num_param (c₁ > 2ξ·C̄_inv²) is element-family dependent, so the
# paper adopts c₁ = 4k⁴ in 2D (§7.1) and c₁ = 16k⁴ for the 3D tetrahedra (§7.2). Before this the 3D
# constant lived ONLY as a harness kwarg defaulting to the 2D value, so `run_simulation` on a tet mesh
# silently used 4k⁴ and no stored result recorded which c₁ produced it.
# ----------------------------------------------------------------------------------------------
@testset "config validation: stabilization c₁/c₂ multipliers must be > 0" begin
    # c₁ ≤ 0 ⇒ τ_{1,NS}⁻¹ = c₁ν/h² + c₂|a|/h can go ≤ 0 ⇒ the subgrid scales invert sign (anti-stabilization).
    @test_throws Exception _load(Dict("physical_properties" => _phys(),
        "numerical_method" => Dict("stabilization" => Dict("c1_multiplier" => 0.0))))
    @test_throws Exception _load(Dict("physical_properties" => _phys(),
        "numerical_method" => Dict("stabilization" => Dict("c1_multiplier" => -1.0))))
    @test_throws Exception _load(Dict("physical_properties" => _phys(),
        "numerical_method" => Dict("stabilization" => Dict("c2_multiplier" => 0.0))))
end

@testset "config validation: stabilization multipliers round-trip (provenance)" begin
    # The whole point is that a stored config states the c₁ that produced the result.
    cfg = _load(Dict("physical_properties" => _phys()))
    @test cfg.numerical_method.stabilization.c1_multiplier == 1.0   # base_config ships the 2D value
    @test cfg.numerical_method.stabilization.c2_multiplier == 1.0
    cfg3d = _load(Dict("physical_properties" => _phys(),
        "numerical_method" => Dict("stabilization" => Dict("c1_multiplier" => 4.0))))
    @test cfg3d.numerical_method.stabilization.c1_multiplier == 4.0  # the paper's 3D value, c₁ = 16k⁴
end

@testset "stabilization multiplier: c₁ enters τ in the DENOMINATOR (raising it WEAKENS stabilization)" begin
    # [known-fragility] The sign of this relation has misled reviewers: c₁ is the coefficient of the
    # viscous eigenvalue of τ_{1,NS}⁻¹, so a LARGER c₁ gives a SMALLER τ₁ and τ₂ — c₁ = 16k⁴ is *weaker*
    # stabilization than 4k⁴, not stronger. Pin it so "c₁×4 over-stabilizes" cannot be asserted silently.
    P = PorousNSSolver
    c1b, c2b = P.get_c1_c2(P.PaperGeneralFormulation, 2)
    @test (c1b, c2b) == (64.0, 8.0)          # 4k⁴, 2k² at k=2
    kin = P.KinematicState(P.VectorValue(1.0, 0.0), P.TensorValue(0.0, 0.0, 0.0, 0.0), 1.0)  # |u| ≠ 0
    med = P.MediumState(1.0, P.VectorValue(0.0, 0.0), 0.05)
    rxn = P.ConstantSigmaLaw(0.0)
    τ1_x1 = P.compute_tau_1(kin, med, 1.0, 1.0 * c1b, c2b, 0.0, rxn)
    τ1_x4 = P.compute_tau_1(kin, med, 1.0, 4.0 * c1b, c2b, 0.0, rxn)
    τ2_x1 = P.compute_tau_2(kin, med, 1.0, 1.0 * c1b, c2b, 0.0)
    τ2_x4 = P.compute_tau_2(kin, med, 1.0, 4.0 * c1b, c2b, 0.0)
    @test τ1_x4 < τ1_x1
    @test τ2_x4 < τ2_x1

    # At |u| = 0 the c₁ in τ₂'s numerator and in τ_{1,NS}⁻¹ cancel EXACTLY: τ₂ = h²/(c₁·α·(h²/(c₁ν))) = ν/α,
    # independent of c₁. A test that probes τ₂ at rest therefore cannot see the multiplier at all.
    kin0 = P.KinematicState(P.VectorValue(0.0, 0.0), P.TensorValue(0.0, 0.0, 0.0, 0.0), 0.0)
    @test P.compute_tau_2(kin0, med, 1.0, c1b, c2b, 0.0) ≈ 1.0            # ν/α
    @test P.compute_tau_2(kin0, med, 1.0, 4.0 * c1b, c2b, 0.0) ≈ 1.0      # unchanged by c₁×4
end

@testset "no ambient-environment knobs in the τ kernel" begin
    # [no-hard-coded-parameters] TAU_VISC_MULT (removed 2026-07-17) was read from ENV inside the
    # quadrature-point hot path, so it was absent from every result's provenance. Guard the regression:
    # τ must depend on nothing but its explicit arguments.
    src = read(joinpath(@__DIR__, "..", "..", "src", "stabilization", "tau.jl"), String)
    body = join([l for l in split(src, '\n') if !startswith(strip(l), "#")], '\n')  # ignore commentary
    @test !occursin("ENV", body)
end
