# ==============================================================================================
# Config validation contract test — the load-time `@assert` invariants added in the 2026-06 audit
# (C.2 / SOLV-04 / MODE-02/03, docs/formulation-audit-2026-06-24.md §C.2): reaction SPSD
# (σ_lin/σ_nonlin/σ_const ≥ 0), porosity/domain bounds (α₀ ∈ (0,1], r₁ < r₂, bounding_box parity),
# velocity-floor positivity (SmoothVelocityFloor stays C¹), and the scale-free convergence-gate
# tolerances (eps_tol_momentum/eps_tol_mass > 0). Each invalid value must fail LOUDLY at load.
#
# `load_config_from_dict` deep-merges the override onto `config/base_config.json`, then deserializes and
# runs `validate!`. base_config intentionally omits `eps_val`, so every override below supplies it so the
# config is COMPLETE (deserialization passes) and the throw comes from `validate!`, not a missing field.
# ==============================================================================================

using Test
using PorousNSSolver

# physical_properties override that, merged onto base_config, yields a complete + valid config.
_phys(extra...) = merge(Dict("eps_val" => 0.0), Dict(extra...))
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
