# src/diagnostics_helpers.jl
#
# Diagnostic / verification helpers used by the test tiers (not part of the solver
# hot path). They re-assemble individual pieces of the VMS weak form in isolation so
# tests can inspect them term by term, plus a few linear-algebra sanity checks
# (finite-difference Jacobian, merit-function descent, conditioning).
#
# These mirror the production assembly in `src/formulations/` and `src/stabilization/`
# but are deliberately standalone: they take an explicit `config`, a measure `dΩ`, and
# the element size `h`, so a test can probe one term without spinning up the full driver.
using Gridap
using Gridap.Algebra
using LinearAlgebra

# Galerkin reaction term  ∫ v·(σ u) dΩ  in isolation.
# `σ = σ(α, u)` is the Darcy-Brinkman-Forchheimer resistance coefficient — the
# (symmetric positive-semidefinite) inverse-permeability tensor, here a scalar
# (eq:DBFResistanceTerm) — built from the configured reaction model.
# `alpha_custom` overrides the porosity field `α` (defaults to the uniform `alpha_0`),
# letting tests feed a manufactured spatially-varying porosity.
function weak_form_g_sigma(u, v, config::PorousNSSolver.PorousNSConfig, dΩ, h, alpha_custom=nothing)
    model = PorousNSSolver.build_reaction_model(config)
    return _weak_form_g_sigma_impl(u, v, config, model, dΩ, h, alpha_custom)
end

function _weak_form_g_sigma_impl(u, v, config::PorousNSSolver.PorousNSConfig, model::M, dΩ, h, alpha_custom) where {M<:PorousNSSolver.AbstractReactionModel}
    α = isnothing(alpha_custom) ? config.domain.alpha_0 : alpha_custom
    σ = Operation((u_v, h_v, a_v) -> model(PorousNSSolver.ThermodynamicState(u_v, h_v, a_v)))(u, h, α)
    return ∫( v ⋅ (σ * u) )dΩ
end

# Reaction part of the VMS stabilization term in isolation:
#   ∫ (-σ v)·(τ₁ R_u) dΩ
# Here `-σ v` is the reaction adjoint operator L*_reac(v) applied to the test
# function, weighting the strong momentum residual `R_u` scaled by the first
# stabilization parameter `τ₁` (eq:Tau1). `X = (u, p)` is the trial state, `f_custom`
# overrides the momentum forcing, `alpha_custom` overrides the porosity `α`.
function weak_form_s_sigma(X, v, config::PorousNSSolver.PorousNSConfig, dΩ, h, f_custom=nothing, alpha_custom=nothing)
    model = PorousNSSolver.build_reaction_model(config)
    return _weak_form_s_sigma_impl(X, v, config, model, dΩ, h, f_custom, alpha_custom)
end

function _weak_form_s_sigma_impl(X, v, config::PorousNSSolver.PorousNSConfig, model::M, dΩ, h, f_custom, alpha_custom) where {M<:PorousNSSolver.AbstractReactionModel}
    u, p = X
    α = isnothing(alpha_custom) ? config.domain.alpha_0 : alpha_custom
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom
    
    # ν: kinematic viscosity; k_deg: velocity polynomial degree.
    # c_1 = 4k⁴, c_2 = 2k² are the equal-order interpolation constants entering τ₁
    # (paper Remark after eq:conditions_on_num_param; see get_c1_c2 in the solver).
    let ν = config.phys.nu,
        k_deg = config.numerical_method.element_spaces.k_velocity,
        c_1 = 4.0 * config.numerical_method.element_spaces.k_velocity^4,
        c_2 = 2.0 * config.numerical_method.element_spaces.k_velocity^2,
        reaction_trait = PorousNSSolver.get_reaction_mode_trait(config.numerical_method.solver.experimental_reaction_mode),
        tau_reg_lim = config.phys.tau_regularization_limit

        σ = Operation((u_v, h_v, a_v) -> model(PorousNSSolver.ThermodynamicState(u_v, h_v, a_v)))(u, h, α)
        τ_1 = Operation((u_v, h_v, a_v) -> PorousNSSolver.compute_tau_1(PorousNSSolver.ThermodynamicState(u_v, h_v, a_v), ν, c_1, c_2, tau_reg_lim, model, reaction_trait))(u, h, α)
        alpha_conv = α

        # Strong residual of the momentum equation R_u (eq:StrongMomentumEquation),
        # term by term: convection α(u·∇)u, pressure gradient α∇p, reaction σu,
        # the viscous divergence ∇·(α ν ∇u), and forcing f.
        # The viscous divergence splits into ν(∇u·∇α) (always present, from the
        # porosity gradient) plus the Laplacian α ν Δu, which only survives when the
        # FE space is rich enough (k_deg > 1) for second derivatives to be nonzero.
        conv_u = ∇(u)' ⋅ u
        div_visc_u = ν * (∇(u)' ⋅ ∇(α))
        if k_deg > 1
            div_visc_u = div_visc_u + α * ν * Δ(u)
        end
        R_u = alpha_conv * conv_u + alpha_conv * ∇(p) + σ * u - div_visc_u - f

        # Reaction adjoint L*_reac(v) = -σ v weighting τ₁ R_u (the reaction piece of
        # the VMS stabilization bilinear form B_S under eq:OSGSProblem).
        return ∫( (- σ * v) ⋅ (τ_1 * R_u) )dΩ
    end
end

# Per-term L² norms of the strong residual, returned as a labelled Dict.
# Splits the strong momentum residual R_u and mass residual R_p into their physical
# contributions and integrates ‖·‖² over the domain, so a test can see WHICH term
# dominates (e.g. is convection or reaction carrying the imbalance). Returns the
# square root of each ∫‖term‖² dΩ, i.e. the L²(Ω) norm of each component.
# `f_custom`/`g_custom` override momentum/mass forcings; `alpha_custom` the porosity.
function evaluate_residual_components(u_h, p_h, config::PorousNSSolver.PorousNSConfig, dΩ, h, f_custom=nothing, alpha_custom=nothing, g_custom=nothing)
    model = PorousNSSolver.build_reaction_model(config)
    return _evaluate_residual_components_impl(u_h, p_h, config, model, dΩ, h, f_custom, alpha_custom, g_custom)
end

function _evaluate_residual_components_impl(u_h, p_h, config::PorousNSSolver.PorousNSConfig, model::M, dΩ, h, f_custom, alpha_custom, g_custom) where {M<:PorousNSSolver.AbstractReactionModel}
    α = isnothing(alpha_custom) ? config.domain.alpha_0 : alpha_custom
    ν = config.phys.nu
    k_deg = config.numerical_method.element_spaces.k_velocity
    f = isnothing(f_custom) ? VectorValue(config.phys.f_x, config.phys.f_y) : f_custom
    eps_val = config.phys.eps_val
    g_mass = isnothing(g_custom) ? 0.0 : g_custom
    let ν = config.phys.nu,
        k_deg = config.numerical_method.element_spaces.k_velocity,
        c_1 = 4.0 * k_deg^4,
        c_2 = 2.0 * k_deg^2
        
        σ = Operation((u_v, h_v, a_v) -> model(PorousNSSolver.ThermodynamicState(u_v, h_v, a_v)))(u_h, h, α)
        alpha_conv = α

    # Momentum residual R_u (eq:StrongMomentumEquation), one term per physical effect.
    conv_u = ∇(u_h)' ⋅ u_h
    term_conv = alpha_conv * conv_u   # convection  α(u·∇)u
    term_pres = alpha_conv * ∇(p_h)   # pressure gradient  α∇p
    term_reac = σ * u_h               # Darcy-Brinkman-Forchheimer reaction  σu

    # Viscous divergence ∇·(α ν ∇u): porosity-gradient part always present; the
    # Laplacian part needs k_deg > 1 to be representable in the FE space.
    div_visc_u = ν * (∇(u_h)' ⋅ ∇(α))
    if k_deg > 1
        div_visc_u = div_visc_u + α * ν * Δ(u_h)
    end
    term_diff = - div_visc_u          # diffusion  -∇·(α ν ∇u)
    term_forc = - f                   # body force  -f

    R_u = term_conv + term_pres + term_reac + term_diff + term_forc

    # Mass residual R_p (eq:StrongMassEquation), with the ε p pressure-stabilizing /
    # regularization term, the weighted divergence ∇·(α u), and the mass source g.
    term_eps_p = eps_val * p_h        # pressure regularization  ε p
    div_alpha_u = α * (∇⋅u_h) + u_h ⋅ ∇(α)
    term_div = div_alpha_u            # weighted divergence  ∇·(α u)
    term_mass_forc = - g_mass         # mass source  -g

    R_p = term_eps_p + term_div + term_mass_forc

    norm_sq(v) = v ⋅ v
    
    n_conv = sqrt(abs(sum(∫(norm_sq(term_conv))dΩ)))
    n_pres = sqrt(abs(sum(∫(norm_sq(term_pres))dΩ)))
    n_reac = sqrt(abs(sum(∫(norm_sq(term_reac))dΩ)))
    n_diff = sqrt(abs(sum(∫(norm_sq(term_diff))dΩ)))
    n_forc = sqrt(abs(sum(∫(norm_sq(term_forc))dΩ)))
    n_Ru   = sqrt(abs(sum(∫(norm_sq(R_u))dΩ)))
    
    n_epsp = sqrt(abs(sum(∫(term_eps_p * term_eps_p)dΩ)))
    n_div  = sqrt(abs(sum(∫(term_div * term_div)dΩ)))
    n_mfor = sqrt(abs(sum(∫(term_mass_forc * term_mass_forc)dΩ)))
    n_Rp   = sqrt(abs(sum(∫(R_p * R_p)dΩ)))
    
    return Dict(
        "mom_conv" => n_conv, "mom_pres" => n_pres, "mom_reac" => n_reac,
        "mom_diff" => n_diff, "mom_forc" => n_forc, "mom_total" => n_Ru,
        "mass_eps" => n_epsp, "mass_div" => n_div, "mass_forc" => n_mfor, "mass_total" => n_Rp
    )
    end
end

# Finite-difference check of an analytic Jacobian. Compares the matrix-vector product
# J·d against the directional derivative of the residual,
#   (R(x + ε d) − R(x)) / ε  →  J·d   as ε → 0,
# so a wrong or incomplete Jacobian (e.g. a dropped ExactNewton ∂τ/∂u term) shows up
# as a non-vanishing relative difference. `op` is the Gridap operator, `U_space` maps
# the DOF vector `x` to an FEFunction, and `d` is the probe direction. Sweeps several
# step sizes ε and returns, per ε, the norms ‖J·d‖, ‖FD‖, ‖FD − J·d‖ and their ratio.
function test_jacobian_fd(op, U_space, x::AbstractVector, d::AbstractVector)
    x_state = FEFunction(U_space, x)
    b0 = allocate_residual(op, x_state)
    residual!(b0, op, x_state)

    A = allocate_jacobian(op, x_state)
    jacobian!(A, op, x_state)

    J_d = A * d

    # Span a few orders of magnitude in ε to expose the round-off / truncation tradeoff:
    # too large ε leaves truncation error, too small ε is swamped by floating-point noise.
    eps_vals = [1e-4, 1e-6, 1e-8]
    x_pert = similar(x)
    b_pert = similar(b0)
    
    results = Dict{Float64, Dict{String, Float64}}()
    for ep in eps_vals
        x_pert .= x .+ ep .* d
        x_pert_state = FEFunction(U_space, x_pert)
        residual!(b_pert, op, x_pert_state)
        
        FD = (b_pert .- b0) ./ ep
        diff = FD .- J_d
        
        n_Jd = norm(J_d)
        n_FD = norm(FD)
        n_diff = norm(diff)
        rel_diff = n_diff / (n_FD + 1e-16)
        
        results[ep] = Dict(
            "norm_Jd" => n_Jd,
            "norm_FD" => n_FD,
            "norm_diff" => n_diff,
            "rel_diff" => rel_diff
        )
    end
    return results
end

function test_merit_descent(b::AbstractVector, A, dx::AbstractVector)
    # Checks that the Newton step is a descent direction for the line-search merit
    # function φ(x) = ½‖R(x)‖², whose directional derivative along d is
    #   φ'(0) = R(x)ᵀ J(x) d = bᵀ A d.
    # Convention (must match the solver): the linear solve `solve!(dx, ns, b)` gives
    # A·dx = b with A = J and b = R, and the update steps as x ← x − α·dx, so the
    # actual search direction is d = −dx. Hence
    #   φ'(0) = bᵀ A (−dx) = −bᵀ(A dx) = −bᵀ b < 0,
    # i.e. a strictly negative slope guarantees the merit function decreases. A
    # non-negative return value signals a broken Jacobian or a sign-flipped convention.
    Adx = A * dx
    phi_prime = dot(b, -Adx)  # direction is -dx
    return phi_prime
end

# Conditioning diagnostic for the Jacobian A. The 2-norm condition number κ₂(A) =
# σ_max/σ_min measures how close the linearized system is to singular (large κ ⇒
# ill-conditioned, e.g. at extreme Re/Da), explaining slow or stalled Newton solves.
# Returns a labelled Dict; the keys differ by branch (see below).
function estimate_jacobian_condition(A)
    n = size(A, 1)
    if n > 5000
        # Above this size a full SVD is too expensive, so fall back to the cheap
        # operator ∞-norm and flag the exact condition number as unavailable (-1.0).
        norm_inf = opnorm(A, Inf)
        return Dict("size" => n, "norm_inf" => norm_inf, "cond_estimate" => -1.0)
    else
        # Small enough to densify and take an exact SVD: singular values S are sorted
        # descending, so κ₂ = S[1]/S[end]. If densification/SVD fails, fall back to
        # the ∞-norm estimate as above.
        try
            Adense = Matrix(A)
            S = svdvals(Adense)
            cnd = S[1] / S[end]
            return Dict("size" => n, "norm_A" => S[1], "norm_inv_A" => 1.0/S[end], "cond" => cnd, "min_sv" => S[end], "max_sv" => S[1])
        catch
            return Dict("size" => n, "norm_inf" => opnorm(A, Inf), "cond_estimate" => -1.0)
        end
    end
end
