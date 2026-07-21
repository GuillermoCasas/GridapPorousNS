# test/extended/CocquetFormMMS/run_test.jl
# ==============================================================================================
# Nature & Intent:
# Method of Manufactured Solutions (MMS) workflow whose GOVERNING OPERATOR is identical to the
# Cocquet benchmark — the symmetric-gradient viscous term and the nonlinear, porosity-dependent
# Forchheimer–Ergun reaction σ(α,u)=a(α)+b(α)|u| — but with a smooth, closed-form exact solution.
# It is a self-contained sibling of test/extended/ManufacturedSolutions and shares the SAME
# generalized manufactured-solution machinery in src/problems/mms_paper.jl: the exact fields
# (u_ex, p_ex) and the forcing oracle `evaluate_exactness_diagnostics`, which dispatches on the
# configured reaction law (constant σ vs. nonlinear Forchheimer-Ergun) and viscous operator. The
# only difference between this test and the standard MMS is the configured formulation — which is
# exactly the point: the same oracle highlights where the two formulations diverge.
#
# Purpose: discriminate whether the Cocquet benchmark's sub-optimal convergence stems from the
# formulation (a code defect) or from the physical problem. If this smooth-solution MMS converges
# optimally, the formulation is sound and the benchmark's behaviour is physical/pre-asymptotic.
#
# Associated Files / Functions:
# - `src/formulations/continuous_problem.jl`
# - `src/solvers/nonlinear.jl` (`solve_system`)
# - `src/problems/mms_paper.jl` (exact fields + reaction-aware forcing oracle)
# ==============================================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using PorousNSSolver
using Gridap
using Gridap.Algebra
using JSON3
using DelimitedFiles
using HDF5
using LineSearches
using Printf
using Random
using LinearAlgebra

# [harness-frame] Re/Da iteration-budget knobs (relocated out of production SolverConfig — audit §A.1/F1).
@isdefined(read_mms_dynamic_budget) || include(joinpath(@__DIR__, "..", "harness_dynamic_budget.jl"))

# ==============================================================================
# Helper Constructors
# ==============================================================================

function _build_local_mesh(domain_cfg, mesh_cfg, L::Float64=1.0)
    # The JSON `bounding_box` is the L=1 baseline; the physical domain extends as `L .* bounding_box`,
    # so the manufactured shape sin(πx/L) and the domain scale together (the dimensionless problem is
    # preserved; only the conditioning changes). Mirrors the regular ManufacturedSolutions harness.
    domain = Tuple(collect(L .* domain_cfg.bounding_box))
    partition = Tuple(mesh_cfg.partition)
    if mesh_cfg.element_type == "TRI"
        model = CartesianDiscreteModel(domain, partition; isperiodic=Tuple(fill(false, length(partition))), map=identity)
        model = simplexify(model)
    else
        model = CartesianDiscreteModel(domain, partition)
    end
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "inlet", [7])
    add_tag_from_tags!(labels, "outlet", [8])
    add_tag_from_tags!(labels, "walls", [1, 2, 3, 4, 5, 6])
    return model
end

# Creates the domain porosity field structure parameterized by alpha bounds and geometry.
# `r_1, r_2` are L=1 baselines; the porosity bump lives at L*r_1 < r < L*r_2 so its relative footprint
# on the L-scaled domain is unchanged (mirrors the regular ManufacturedSolutions harness).
function build_porosity_field(config, alpha_0, alpha_infty, L::Float64=1.0)
    PorousNSSolver.SmoothRadialPorosity(Float64(alpha_0), Float64(alpha_infty), L * config.domain.r_1, L * config.domain.r_2)
end

# Forchheimer-minmax encoding: pick (L, U) minimising the dynamic range of {U, ν, σ} for the NONLINEAR
# Ergun reaction σ(U) = a(α₀) + b(α₀)·U (which GROWS with the velocity scale U and FLOORS at a(α₀)).
# Standard minmax (Constant_Sigma) balances ν=σ=1/U; the correction for the nonlinear reaction picks U
# solving 1/U = a + b·U (so |ln U| = |ln σ|) and sets ν = √(U·σ) (geometric mean ⇒ |ln ν| ≤ max(|ln U|,|ln σ|)),
# giving L = ν·Re/U. This lifts ν to O(1), avoiding the ν=U·L/Re → 0 τ-saturation that floors ε_M at high
# Re (validated: unit U=L=1 floors ε_M~1e-3 at Re=1e5/α=0.1; this encoding reaches the 1e-9 gate). α_0 is
# the CORE porosity (strongest drag). "unit" reproduces the legacy L=1/U=1 behaviour bit-identically.
function compute_L_and_U(strategy, Re, sigma_linear, sigma_nonlinear, alpha_0)
    if strategy == "unit"
        return (1.0, 1.0)
    elseif strategy != "forchheimer_minmax"
        error("Unknown encoding_strategy \"$strategy\". Valid: \"forchheimer_minmax\", \"unit\".")
    end
    # Damköhler number = dimensionless reaction-term size (article.tex eq:DimensionlessParameters,
    # Da = σL²/(α_∞ν); the reaction term is `Da·u*` in eq:DimensionlessMomentumEquation). With a,b now
    # scaling with the encoding (see build_mms_formulation) Da is Re-INDEPENDENT: for σ(U)=a+b|u| at |u*|~1,
    # Da(α) = σ_lin·((1-α)/α)² + σ_nl·((1-α)/α). α_∞ = 1 in this harness.
    ratio = (1.0 - Float64(alpha_0)) / Float64(alpha_0)
    Da_eff = Float64(sigma_linear) * ratio^2 + Float64(sigma_nonlinear) * ratio
    if Da_eff <= 0.0
        return (1.0, 1.0)                       # pure Navier–Stokes (α₀→1, no drag): no reaction to balance
    end
    # minmax encoding (centered_encoding.tex / regular ManufacturedSolutions harness, with Da→Da_eff):
    # L = √(α_∞·Da), U = (Re²/(α_∞·Da))^{1/4}  ⇒  ν = σ = (α_∞·Da/Re²)^{1/4}, minimal {U,ν,σ} dynamic range.
    L = sqrt(Da_eff)
    U = (Float64(Re)^2 / Da_eff)^(0.25)
    return (L, U)
end

# Sets up the continuous VMS formulation for the Cocquet-form Manufactured Solution.
# Mirrors the canonical build_formulation (src/run_simulation.jl): constant σ keeps the
# reaction-trim projection; the nonlinear Forchheimer law must project the full residual
# (the constant-σ trim is rejected by sanitize_projection_policy for nonlinear laws).
function build_mms_formulation(config, Da, Re, U_amp, L, alpha_infty)
    # Velocity regularization. h_floor_weight is fixed to 0 so the speed floor is
    # mesh-independent — required for the forcing oracle to stay exact under Forchheimer
    # (see evaluate_exactness_diagnostics in src/problems/mms_paper.jl).
    reg = PorousNSSolver.SmoothVelocityFloor(
        config.physical_properties.u_base_floor_ref,
        0.0,
        config.physical_properties.epsilon_floor,
        config.physical_properties.velocity_magnitude_derivative_floor
    )

    # Derive kinematic viscosity from the Reynolds sweep input.
    nu_calculated = U_amp * L / Float64(Re)
    # [covariance] physical_epsilon is DIMENSIONAL: in the continuity penalty `physical_epsilon·p`, [physical_epsilon] = (U/L)/P_c, so
    # it MUST scale with the (L,U) encoding. The config value is the DIMENSIONLESS penalty ε̂; eps_calculated
    # is derived per cell so ε̂ = eps_calculated·P_c·L/U is encoding-invariant. A FIXED physical_epsilon breaks
    # scale-covariance (worst in the pressure / OSGS projection). Mirrors the regular ManufacturedSolutions
    # harness; P_c per get_characteristic_scales: P_c = (1+Re+Da)·U·ν/L.
    P_c_cell = (1.0 + Float64(Re) + Float64(Da)) * U_amp * nu_calculated / L
    eps_calculated = config.physical_properties.physical_epsilon * (U_amp / L) / P_c_cell

    if config.physical_properties.reaction_model == "Constant_Sigma"
        sigma_c = Float64(Da) * alpha_infty * nu_calculated / (L^2)
        rxn = PorousNSSolver.ConstantSigmaLaw(sigma_c)
        # Mirror src/run_simulation.jl:57 — the constant-σ reaction trim (paper §4.4, removes σu from
        # the stabilization/projection) is applied ONLY under experimental_reaction_mode=="standard";
        # otherwise keep the FULL residual so the reaction stays IN the stabilization. This lets us A/B,
        # at matched uniform σ, whether reaction-in-the-stabilization is what folds the low-α cells.
        proj = config.numerical_method.solver.experimental_reaction_mode == "standard" ?
            PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma() :
            PorousNSSolver.ProjectFullResidual()
    else
        # [encoding-covariant reaction] σ_linear/σ_nonlinear are the DIMENSIONLESS Darcy/Forchheimer
        # numbers (cf. article.tex eq:DimensionlessParameters Da = σL²/(α_∞ν), and the Ergun a(α)=(150/Re)·…
        # whose 1/Re=ν factor keeps Da Re-independent). The dimensional drags must therefore scale with the
        # (L,U) encoding: a(α)=σ_lin·((1-α)/α)²·(ν/L²), b(α)=σ_nl·((1-α)/α)·(ν/(U·L²)), so the Damköhler
        # number Da(α)=σ_lin·((1-α)/α)²+σ_nl·((1-α)/α) (the reaction-term size, eq:DimensionlessMomentumEquation
        # `Da·u*`) is Re-INDEPENDENT — exactly as in article.tex. Passing them FIXED (the old code) made the
        # effective Da ∝ Re (~4e6 at Re=1e5), spuriously reaction-dominating and breaking the encoding.
        a_scale = nu_calculated / (L^2)
        b_scale = nu_calculated / (U_amp * L^2)
        rxn = PorousNSSolver.ForchheimerErgunLaw(
            config.physical_properties.sigma_linear   * a_scale,
            config.physical_properties.sigma_nonlinear * b_scale,
        )
        proj = PorousNSSolver.ProjectFullResidual()
    end

    # Bound the exact viscous analytical formulation directly targeting the governing operator from the numerical schema definitions
    visc_type = config.numerical_method.viscous_operator_type
    if visc_type == "DeviatoricSymmetric"
        visc_op = PorousNSSolver.DeviatoricSymmetricViscosity()
    elseif visc_type == "SymmetricGradient"
        visc_op = PorousNSSolver.SymmetricGradientViscosity()
    else
        visc_op = PorousNSSolver.LaplacianPseudoTractionViscosity()
    end
    
    PorousNSSolver.PaperGeneralFormulation(visc_op, rxn, proj, reg, nu_calculated, eps_calculated)
end

# The dimensionless error functional `calculate_normalized_errors` is shared with the 2D MMS harness
# and the interpolation-reference harness via `test/extended/mms_error_norms.jl` (consolidated
# 2026-07-17; the two copies were verified functionally identical before merging). One definition
# keeps this harness's rows, the 2D rows, and the interpolation reference printed in the paper
# tables on the same measuring stick.
include(joinpath(@__DIR__, "..", "mms_error_norms.jl"))

# Declare typed parametric functor evaluating interpolation seamlessly avoiding JIT closures globally at the top level
struct PerturbationFunc{F1, F2} <: Function
    u_base::F1
    h_func::F2
    scale::Float64
end
(f::PerturbationFunc)(x) = f.u_base(x) + f.scale * f.h_func(x)

struct MMSSetup
    u_final
    p_final
    h_raw_func
    u_ex_L2
    norm_h
    U_c
    P_c
    L
end

struct PerturbationConfig
    eps_pert_base::Float64
    max_n_pert::Int
end

# [paper-faithful] Pure-Galerkin solve (mult_mom = mult_mass = 0) — the literal Cocquet
# paper discrete formulation for inf-sup-stable Taylor-Hood. Mirrors the design of
# test/extended/CocquetExperiment/galerkin_driver.jl: reuses the same weak-form builders
# the VMS path uses, with the stabilization multipliers zeroed. No τ-stabilization, no
# orthogonal-projection stage. Returns the same tuple shape solve_system does for
# `(success, final_x0, iter_count, eval_time)` so the caller does not branch downstream.
function execute_solver_galerkin_inline!(
    setup::PorousNSSolver.FETopology, formulation::PorousNSSolver.VMSFormulation,
    iter_solvers::PorousNSSolver.StageSolvers, config::PorousNSConfig, x0
)
    phys = config.physical_properties
    freeze_cusp = config.numerical_method.solver.freeze_jacobian_cusp

    res_fn(x, y)        = PorousNSSolver.build_stabilized_weak_form_residual(x, y, setup, formulation, phys;
                                pi_u=nothing, pi_p=nothing, mult_mom=0.0, mult_mass=0.0)
    jac_picard(x, dx, y) = PorousNSSolver.build_picard_jacobian(x, dx, y, setup, formulation, phys;
                                pi_u=nothing, pi_p=nothing, mult_mom=0.0, mult_mass=0.0)
    jac_newton(x, dx, y) = PorousNSSolver.build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys,
                                freeze_cusp, PorousNSSolver.ExactNewtonMode();
                                pi_u=nothing, pi_p=nothing, mult_mom=0.0, mult_mass=0.0)

    op_picard = FEOperator(res_fn, jac_picard, setup.X, setup.Y)
    op_newton = FEOperator(res_fn, jac_newton, setup.X, setup.Y)

    iter_count = 0
    success = false
    final_residual_norm = NaN
    x0_backup = copy(get_free_dof_values(x0))

    eval_time = @elapsed begin
        # Exact-Newton attempt first; Picard globalization fallback; final Newton polish.
        res_n = PorousNSSolver.safe_fe_solve!(x0, iter_solvers.newton, op_newton; backup=x0_backup)
        final_residual_norm = res_n.residual_norm
        if res_n.state == :ok
            iter_count = res_n.iterations
            success = true
        else
            println("      [Galerkin] Newton state $(res_n.state); falling back to Picard...")
            get_free_dof_values(x0) .= x0_backup
            res_p = PorousNSSolver.safe_fe_solve!(x0, iter_solvers.picard, op_picard; backup=x0_backup)
            iter_count += res_p.iterations
            final_residual_norm = res_p.residual_norm
            if res_p.state == :ok
                res_n2 = PorousNSSolver.safe_fe_solve!(x0, iter_solvers.newton, op_newton; backup=x0_backup)
                final_residual_norm = res_n2.residual_norm
                if res_n2.state == :ok
                    iter_count += res_n2.iterations
                    success = true
                end
            end
        end
    end
    return success, x0, iter_count, eval_time, final_residual_norm
end

# Algorithm E: Outer Homotopy Parameter Scaling
function execute_outer_homotopy_perturbation_loop!(
    setup::PorousNSSolver.FETopology, formulation::PorousNSSolver.VMSFormulation,
    iter_solvers::PorousNSSolver.StageSolvers, config::PorousNSConfig,
    method::String, dynamic_ftol::Float64, mms_setup::MMSSetup, pert_cfg::PerturbationConfig,
    mms_verification_enabled::Bool, mms_tau_err, mms_eps_u_l2, mms_eps_u_h1, mms_eps_p_l2,
    mms_max_extra_cycles, mms_require_consecutive_passes,
    mms_rate_check_factor,   # §5.2: factor above FE budget that flags a plateau as
                             # sub-optimal-rate (still success, just flagged).
    n::Int, kv::Int   # §5.1: mesh partition count and velocity polynomial order
                      # for h-scaled plateau floors.
)
    success = false
    mms_plateau_success = nothing   # set per attempt; mirrors solver-side flag
    successful_eps = -1.0
    eval_time = 0.0
    iter_count_attempt = 0
    final_x0 = nothing
    final_residual_attempt = NaN
    osgs_short_circuited = false   # [OSGS-leak guard] did the accepted attempt's OSGS stage never advance off entry?

    for attempt in 0:(pert_cfg.max_n_pert + 1)
        # [design-intent] Same hard → easy ordering as `ManufacturedSolutions/run_test.jl`:
        # default to the largest perturbation; only fall back to milder ones if Newton fails.
        # `eval_eps[n]` records the largest eps_pert that worked, giving a robustness fingerprint
        # per cell. See that file (around line 161) for the full rationale.
        eps_p = attempt <= pert_cfg.max_n_pert ? pert_cfg.eps_pert_base / (10.0^attempt) : 0.0
        
        u_0_func = PerturbationFunc(mms_setup.u_final, mms_setup.h_raw_func, eps_p * (mms_setup.u_ex_L2 / mms_setup.norm_h))
        x0 = interpolate_everywhere([u_0_func, mms_setup.p_final], setup.X)
        
        println("\n    ==================================================")
        println("    [Attempt $(attempt+1)/$(pert_cfg.max_n_pert + 2)] Homotopy Perturbation Scale: eps_pert = $eps_p")
        println("    [!] Delegating orchestration to the VMS solver (`solve_system`, src/solvers/solver_core.jl) (Mode: $method)")
        println("    ==================================================")
        
        local_diagnostics_cache = Dict{String, Any}()
        verifier = mms_verification_enabled ?
            PorousNSSolver.MMSPlateauVerifier(
                oracle = (uh, ph) -> calculate_normalized_errors(uh, ph, mms_setup.u_final, mms_setup.p_final, mms_setup.U_c, mms_setup.P_c, mms_setup.L, setup.dΩ),
                max_extra_cycles = mms_max_extra_cycles,
                require_consecutive_passes = mms_require_consecutive_passes,
                tau_err = mms_tau_err,
                eps_u_l2 = mms_eps_u_l2, eps_u_h1 = mms_eps_u_h1, eps_p_l2 = mms_eps_p_l2,
                h_local = 1.0 / n,   # §5.1: mesh size for h-scaling plateau floors
                kv = kv,             # §5.1: velocity polynomial order for the scaling exponent
                rate_check_factor = mms_rate_check_factor,   # §5.2: sub-optimal-rate flag threshold
            ) :
            PorousNSSolver.NoVerification()

        if uppercase(method) == "GALERKIN"
            # [paper-faithful] Cocquet-paper pure-Galerkin path (mult_mom = mult_mass = 0).
            # Bypasses solve_system's ASGS/OSGS machinery entirely. No plateau loop —
            # Galerkin doesn't have τ-stabilization noise to track, and the error vs.
            # the exact MMS solution is computed once-and-for-all after convergence.
            sys_success, sys_final_x0, sys_iter_count, sys_eval_time, gal_final_residual =
                execute_solver_galerkin_inline!(setup, formulation, iter_solvers, config, x0)
            sys_mms_plateau_success = nothing
            # Populate the diagnostics cache the same way solve_system does, so the analyzer's
            # true-root gate (`is_true_root` reads `eval_residuals`) sees a finite residual.
            local_diagnostics_cache["final_residual_norm"] = gal_final_residual
        else
            sys_success, sys_mms_plateau_success, sys_final_x0, sys_iter_count, sys_eval_time = PorousNSSolver.solve_system(
                setup, formulation, iter_solvers,
                config, x0;
                diagnostics_cache=local_diagnostics_cache, verifier=verifier
            )
        end
        
        if haskey(local_diagnostics_cache, "mms_stop_reason")
            println("    -> MMS Plateau Reason: ", local_diagnostics_cache["mms_stop_reason"])
            println("    -> Base Convergence Reached: ", get(local_diagnostics_cache, "base_convergence_reached", false))
            println("    -> MMS Plateau Reached: ", get(local_diagnostics_cache, "mms_plateau_reached", false))
        end
        
        if sys_success
            println("\n      [✅] Full non-linear algebraic system converged gracefully mathematically! Escaping constraint loop.")
            success = true
            mms_plateau_success = sys_mms_plateau_success
            # Fix 6: report MMS-verification status separately from solver convergence.
            # `sys_success` reflects only the inner solver; `sys_mms_plateau_success`
            # reflects whether the MMS plateau check was formally established. Both must
            # be true (or plateau-check disabled) for the cell to count as fully verified.
            if mms_plateau_success === false
                println("      [⚠] Solver converged but MMS plateau was NOT formally verified (budget exhausted).")
            end
            successful_eps = eps_p
            final_x0 = sys_final_x0
            eval_time = sys_eval_time
            iter_count_attempt = sys_iter_count
            final_residual_attempt = get(local_diagnostics_cache, "final_residual_norm", NaN)
            # [OSGS-leak guard] the OSGS coupled stage surfaces this when it reported success without advancing
            # off its entry iterate (a 0-iteration initial_ftol short-circuit). With the ASGS boot ON that entry
            # is the ASGS root, so the "OSGS" state is byte-identical to ASGS — not a genuine OSGS datum.
            osgs_short_circuited = get(local_diagnostics_cache, "osgs_short_circuited_on_entry", false)
            break
        else
            println("\n      [❌] Outer loop execution completely stalled structurally above convergence tolerance (`$(dynamic_ftol)`) or system fully diverged.")
            final_residual_attempt = get(local_diagnostics_cache, "final_residual_norm", NaN)
        end
    end

    if !success
         println("    [WARNING] Completely failed to find root basin. Returning NaN.")
    end

    return success, mms_plateau_success, successful_eps, final_x0, eval_time, iter_count_attempt, final_residual_attempt, osgs_short_circuited
end

function run_mms(config_file="test_config.json")
    config_path = joinpath(@__DIR__, "data", config_file)
    test_dict = JSON3.read(read(config_path, String), Dict{String, Any})
    budget = read_mms_dynamic_budget(test_dict)   # [harness-frame] Re/Da iteration-budget knobs (audit §A.1/F1)

    as_list(x) = x isa Vector ? x : [x]
    
    Re_list = as_list(test_dict["physical_properties"]["Re"])
    Da_list = as_list(test_dict["physical_properties"]["Da"])
    alpha_list = as_list(test_dict["domain"]["alpha_0"])
    
    
    nm_dict = test_dict["numerical_method"]
    # [harness-frame, audit R2/I07] Porosity-interpolation ablation: 0 (default) => alpha is evaluated
    # ANALYTICALLY at the quadrature points, as in the paper. p>0 => the FORMULATION uses a degree-p
    # Lagrangian FE interpolant of alpha, while the oracle keeps the analytic alpha, so the run isolates
    # the MODEL error introduced by interpolating the porosity (the forcing stays exact).
    porosity_interp_order = Int(get(nm_dict, "porosity_interpolation_order", 0))
    elem_dict = nm_dict["element_spaces"]
    mesh_dict = nm_dict["mesh"]
    stab_dict = get(nm_dict, "stabilization", Dict("method" => ["ASGS", "OSGS"]))
    
    kv_list = as_list(elem_dict["k_velocity"])
    kp_list = as_list(elem_dict["k_pressure"])
    etype_list = as_list(mesh_dict["element_type"])
    methods = as_list(get(stab_dict, "method", ["ASGS", "OSGS"]))
    
    equal_order_only = get(test_dict, "equal_order_only", false)
    if haskey(nm_dict, "element_spaces") && haskey(elem_dict, "equal_order_only")
        equal_order_only = elem_dict["equal_order_only"]
    end
    
    mms_verification_enabled = get(test_dict, "mms_verification_enabled", false)
    mms_tau_err = Float64(get(test_dict, "mms_tau_err", 1e-4))
    mms_eps_u_l2 = Float64(get(test_dict, "mms_eps_u_l2", 1e-12))
    mms_eps_u_h1 = Float64(get(test_dict, "mms_eps_u_h1", 1e-12))
    mms_eps_p_l2 = Float64(get(test_dict, "mms_eps_p_l2", 1e-12))
    mms_max_extra_cycles = Int(get(test_dict, "mms_max_extra_cycles", 5))
    mms_require_consecutive_passes = Int(get(test_dict, "mms_require_consecutive_passes", 2))
    # §5.2: tolerance multiplier above the FE budget at which the established
    # plateau is flagged as "sub-optimal rate" (still success). Default 100.0:
    # allow up to 2 orders of magnitude above the discretization scale before
    # flagging — anything larger is the iteration plateauing at a non-FE-optimal
    # state rather than a genuine MMS convergence.
    mms_rate_check_factor = Float64(get(test_dict, "mms_rate_check_factor", 100.0))
    
    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    h5_filename = get(test_dict, "h5_filename", "convergence_data.h5")
    h5_path = joinpath(results_dir, h5_filename)
    
    erase_past = get(test_dict, "erase_past_results", false)
    h5_mode = erase_past ? "w" : "cw"
    
    h5f = h5open(h5_path, h5_mode)
    max_idx = 0
    existing_signatures = Dict()
    # [RESUME] per-(cell, method, mesh) results already on disk, so a re-run (erase_past_results=false)
    # REUSES them instead of recomputing — turning an interrupted sweep into a true resume. Keyed by
    # (Re, Da, alpha_0, kv, kp, etype, method) → Dict(n => saved metric values). The final h5 is identical
    # to an uninterrupted run (reused cells carry their original values; only the missing meshes are solved).
    resume_results = Dict()
    if !erase_past
        for gname in keys(h5f)
            parts = split(gname, "_")
            if length(parts) >= 3 && parts[1] == "config"
                idx = parse(Int, parts[2])
                max_idx = max(max_idx, idx)
                g = h5f[gname]
                att = attributes(g)
                try
                    r = Float64(read(att["Re"]))
                    d = Float64(read(att["Da"]))
                    a = Float64(read(att["alpha_0"]))
                    kv_v = Int(read(att["k_velocity"]))
                    kp_v = Int(read(att["k_pressure"]))
                    et = String(read(att["element_type"]))
                    sig_base = (r, d, a, kv_v, kp_v, et)
                    existing_signatures[sig_base] = idx
                    # load the per-mesh curve for reuse on resume
                    meth = String(parts[3])
                    hs_r  = read(g["h"]); eul = read(g["err_u_l2"]); epl = read(g["err_p_l2"])
                    euh   = read(g["err_u_h1"]); eph = read(g["err_p_h1"]); et_r = read(g["eval_times"])
                    ei_r  = read(g["eval_iters"]); ee_r = read(g["eval_eps"]); er_r = read(g["eval_residuals"])
                    fd_r  = read(g["fold"]); na_r = read(g["osgs_no_advance"])
                    per_n = Dict{Int,Dict{String,Any}}()
                    for i in 1:length(hs_r)
                        per_n[Int(round(1.0/hs_r[i]))] = Dict{String,Any}(
                            "err_u_l2"=>eul[i], "err_p_l2"=>epl[i], "err_u_h1"=>euh[i], "err_p_h1"=>eph[i],
                            "eval_times"=>et_r[i], "eval_iters"=>ei_r[i], "eval_eps"=>ee_r[i],
                            "eval_residuals"=>er_r[i], "fold"=>fd_r[i], "osgs_no_advance"=>na_r[i])
                    end
                    resume_results[(r, d, a, kv_v, kp_v, et, meth)] = per_n
                catch
                end
            end
        end
    end
    close(h5f)
    
    conv_parts = mesh_dict["convergence_partitions"]
    
    total_runs = sum(1 for etype in etype_list, kv in kv_list, kp in kp_list, alpha_0 in alpha_list, Da in Da_list, Re in Re_list if !(equal_order_only && kv != kp)) * length(methods)
    run_idx = 0
    
    # Pre-allocate cache for collecting metrics over partitions dynamically
    results_cache = Dict()
    
    for etype in etype_list
        for kv in kv_list
            for kp in kp_list
                if equal_order_only && kv != kp
                    continue
                end
                
                # Setup dictionary trackers for all physics configurations natively inside this geometry
                for alpha_0 in alpha_list, Da in Da_list, Re in Re_list, method in methods
                    k_id = (etype, kv, kp, alpha_0, Da, Re, method)
                    results_cache[k_id] = Dict(
                        "hs" => Float64[],
                        "err_u_l2" => Float64[],
                        "err_p_l2" => Float64[],
                        "err_u_h1" => Float64[],
                        "err_p_h1" => Float64[],
                        "eval_times" => Float64[],
                        "eval_iters" => Int[],
                        "eval_eps" => Float64[],
                        "eval_residuals" => Float64[],
                        "mms_plateau_success" => Union{Bool,Nothing}[],
                        "overall_verification_success" => Bool[],
                        "fold" => Bool[],   # true ⇒ recorded at the achievable ε_M floor (gate not reached), not NaN
                        "osgs_no_advance" => Bool[]   # true ⇒ OSGS "succeeded" but never advanced off the ASGS
                                                      # Stage-I boot state (initial_ftol short-circuit): a leaked
                                                      # ASGS datum, recorded as NaN (not a genuine OSGS solve)
                    )
                end
                
                for n in conv_parts
                    println("\n========================================")
                    println("BUILDING MESH N = $n for etype = $etype, k_v = $kv, k_p = $kp")
                    println("========================================")
                    
                    # ==============================================================================
                    # MESH & SPATIAL DISCRETIZATION
                    # Evaluated EXACTLY ONCE per resolution (n) to bypass intensive julia compilations
                    # ==============================================================================
                    # Thread the formulation-defining fields from the JSON so this test selects the
                    # Cocquet formulation (SymmetricGradient viscous + Forchheimer reaction). The two
                    # formulations differ in BOTH the reaction term and the viscous term; both are
                    # carried here and dispatched in build_mms_formulation and the forcing oracle.
                    pp_in = get(test_dict, "physical_properties", Dict())
                    config_dict = Dict(
                        "physical_properties" => Dict(
                            "nu" => 1.0, "physical_epsilon" => 1e-8,
                            "reaction_model" => get(pp_in, "reaction_model", "Constant_Sigma"),
                            "sigma_constant" => get(pp_in, "sigma_constant", 1.0),
                            "sigma_linear" => get(pp_in, "sigma_linear", 0.0),
                            "sigma_nonlinear" => get(pp_in, "sigma_nonlinear", 0.0),
                        ),
                        "domain" => Dict(
                            "alpha_0" => 0.4,
                            "bounding_box" => test_dict["domain"]["bounding_box"],
                            "r_1" => test_dict["domain"]["r_1"],
                            "r_2" => test_dict["domain"]["r_2"]
                        ),
                        "numerical_method" => Dict(
                            "viscous_operator_type" => get(nm_dict, "viscous_operator_type", "DeviatoricSymmetric"),
                            "element_spaces" => Dict("k_velocity" => Int(kv), "k_pressure" => Int(kp)),
                            "mesh" => Dict("element_type" => String(etype), "partition" => [n, n]),
                            # [element size] carry the test config's stabilization.element_size when set
                            # (volume|shortest_edge|average_edge|diameter); otherwise base_config supplies
                            # the default (shortest_edge) via the deep-merge in load_config_from_dict.
                            "stabilization" => haskey(stab_dict, "element_size") ?
                                Dict("method" => "ASGS", "element_size" => stab_dict["element_size"]) :
                                Dict("method" => "ASGS"),
                            "solver" => get(get(test_dict, "numerical_method", Dict()), "solver", Dict())
                        )
                    )
                    
                    # Dynamically instantiate hierarchy overrides ensuring JSON parser rigor
                    config = PorousNSSolver.load_config_from_dict(config_dict)

                    # [graceful fold recording] A config with a RELAXED scale-free momentum gate, used ONLY as a
                    # retry for cells that fail the tight gate — the high-Re×low-α "fold" corner the paper itself
                    # skips, where ε_M floors above 1e-9 yet the solution is a usable (pre-asymptotic O(h^k)) one.
                    # The solver then accepts the achievable floor and returns the real solution instead of NaN.
                    # Cells that reach the tight gate never enter this retry, so they are BIT-UNAFFECTED.
                    eps_tol_momentum_fold = Float64(get(test_dict, "eps_tol_momentum_fold", 1e-2))
                    config_fold_dict = deepcopy(config_dict)
                    config_fold_dict["numerical_method"]["solver"] = merge(config_fold_dict["numerical_method"]["solver"], Dict("eps_tol_momentum" => eps_tol_momentum_fold))
                    config_fold = PorousNSSolver.load_config_from_dict(config_fold_dict)
                    
                    # [encoding] The mesh, FE spaces, quadrature, perturbation field and porosity are all
                    # L-scaled and therefore built PER CELL below (L depends on α and Re via the encoding),
                    # not once per N. Only the L-independent controls (c_1/c_2, solver tolerances) stay here.

                    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, kv)
                    # [diagnostic] C1_MULT env-var hook (default 1.0 = paper c₁=4k⁴, byte-identical). Mirrors
                    # smoke3d.jl's c1_mult: scale the coercivity constants to probe whether the low-α/high-Re
                    # fold is a c₁ under-stabilization (as the 3D-P2 Kuhn case was). NOT a committed feature.
                    _c1m = parse(Float64, get(ENV, "C1_MULT", "1.0"))
                    c_1 *= _c1m; c_2 *= _c1m
                    _c1m == 1.0 || (@info "CocquetFormMMS: c₁/c₂ scaled by C1_MULT" C1_MULT=_c1m)
                    tau_reg_lim = config.physical_properties.tau_regularization_limit
                    solver_newton_it = config.numerical_method.solver.newton_iterations
                    solver_picard_it = config.numerical_method.solver.picard_iterations
                    max_inc = config.numerical_method.solver.max_increases
                    xtol = config.numerical_method.solver.xtol
                    stagnation_tol = config.numerical_method.solver.stagnation_noise_floor
                    ftol = config.numerical_method.solver.ftol
                    ls_alpha_min = config.numerical_method.solver.linesearch_alpha_min
                    freeze_cusp = config.numerical_method.solver.freeze_jacobian_cusp

                    # [encoding] (L,U) per cell via the Forchheimer-minmax (default) so ν=U·L/Re stays O(1)
                    # and the high-Re τ-saturation that floored ε_M is avoided. Set "unit" in the JSON to
                    # reproduce the legacy L=1/U=1 behaviour. Coefficients drive the σ(U)=a+b·U balance.
                    encoding_strategy = get(test_dict, "encoding_strategy", "forchheimer_minmax")
                    sigma_lin_enc = config.physical_properties.sigma_linear
                    sigma_nl_enc  = config.physical_properties.sigma_nonlinear

                    # ==============================================================================
                    # PHYSICS PARAMETER SWEEP
                    # Iterate fluid configurations using precompiled outer topological operators
                    # ==============================================================================
                    for alpha_0 in alpha_list
                        alpha_infty = 1.0
                        for Da in Da_list
                            for Re in Re_list
                                # [encoding] per-cell (L, U): minmax with the Forchheimer correction so the
                                # kinematic viscosity ν = U·L/Re stays O(1) and the high-Re τ-saturation that
                                # floored ε_M is avoided. L scales the mesh, porosity bump and perturbation,
                                # so all mesh-dependent objects are (re)built here, per cell.
                                (L, U_amp) = compute_L_and_U(encoding_strategy, Re, sigma_lin_enc, sigma_nl_enc, alpha_0)

                                model = _build_local_mesh(config.domain, config.numerical_method.mesh, L)
                                labels = get_face_labeling(model)
                                add_tag_from_tags!(labels, "all_boundaries", [1,2,3,4,5,6,7,8])

                                # L-scaled homotopy perturbation field (bump × oscillation on the L-scaled domain)
                                xmin, xmax = L*config.domain.bounding_box[1], L*config.domain.bounding_box[2]
                                ymin, ymax = L*config.domain.bounding_box[3], L*config.domain.bounding_box[4]
                                B_fn(x) = (x[1]-xmin)^2 * (xmax-x[1])^2 * (x[2]-ymin)^2 * (ymax-x[2])^2
                                h_raw_func(x) = B_fn(x) * VectorValue(sin(3π*x[1]/L)*cos(2π*x[2]/L), -cos(3π*x[1]/L)*sin(2π*x[2]/L))

                                refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
                                refe_p = ReferenceFE(lagrangian, Float64, kp)
                                V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
                                Q = TestFESpace(model, refe_p, conformity=:H1)
                                V_free = TestFESpace(model, refe_u, conformity=:H1)
                                Q_free = TestFESpace(model, refe_p, conformity=:H1)

                                quad_rxn_law = config.physical_properties.reaction_model == "Constant_Sigma" ?
                                    PorousNSSolver.ConstantSigmaLaw(0.0) : PorousNSSolver.ForchheimerErgunLaw(0.0, 0.0)
                                degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, kv, quad_rxn_law)
                                Ω = Triangulation(model)
                                dΩ = Measure(Ω, degree + 4)

                                # [element size] h(K) via the configured convention (StabilizationConfig.element_size).
                                # Default "shortest_edge" (Codina min edge) ≡ the old √(2·A)/√A grid-spacing on the
                                # structured simplex/square mesh, so this is byte-identical there; "volume" reproduces
                                # the legacy formula exactly; "diameter"/"average_edge" rescale τ. See src/geometry.jl.
                                esize_conv = PorousNSSolver.element_size_convention(config.numerical_method.stabilization.element_size)
                                h_cf = PorousNSSolver.element_size_field(Ω, model, esize_conv)
                                h_pert_cf = CellField(h_raw_func, Ω)
                                norm_h = sqrt(abs(sum(∫( h_pert_cf ⋅ h_pert_cf )dΩ)))
                                norm_h > 0.0 || error("Perturbation field norm must be strictly positive.")
                                Y = MultiFieldFESpace([V, Q])

                                alpha_field = build_porosity_field(config, alpha_0, alpha_infty, L)
                                # [audit R2/I07] Formulation porosity: analytic (default) or a degree-p FE
                                # interpolant. The oracle below (PaperMMS) always keeps the analytic alpha_field,
                                # so a p>0 run measures ONLY the model error from interpolating alpha (exact forcing).
                                alpha_cf = if porosity_interp_order > 0
                                    V_alpha = TestFESpace(model, ReferenceFE(lagrangian, Float64, porosity_interp_order); conformity=:H1)
                                    interpolate(x -> PorousNSSolver.alpha(alpha_field, x), V_alpha)
                                else
                                    CellField(x -> PorousNSSolver.alpha(alpha_field, x), Ω)
                                end

                                form = build_mms_formulation(config, Da, Re, U_amp, L, alpha_infty)
                                
                                # Exact execution of full manufactured solutions analytical expressions
                                mms = PorousNSSolver.PaperMMS(form, U_amp, alpha_field; L=L, alpha_infty=alpha_infty)
                                U_c, P_c = PorousNSSolver.get_characteristic_scales(mms)
                                
                                u_final = PorousNSSolver.get_u_ex(mms)
                                p_final = PorousNSSolver.get_p_ex(mms)
                                
                                U = TrialFESpace(V, u_final)  
                                P = TrialFESpace(Q, p_final)
                                X = MultiFieldFESpace([U, P])
                                
                                # Evaluate symbolic PDE operators mapping directly onto the exact physical boundary source terms
                                f_cf, g_cf = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg_lim)
                                
                                ar_c1 = config.numerical_method.solver.armijo_c1
                                div_fac = config.numerical_method.solver.divergence_merit_factor
                                
                                # Adaptively tighten nonlinear tolerance proportionally to expected spatial scaling O(h^{k+1})
                                h_scale = 1.0 / n
                                spatial_err_est = h_scale^(kv + 1)
                                # Target an algebraic residual strictly bounded physically scaling by spatial discretization tolerances
                                c_ceil = budget.ftol_ceiling
                                c_sf = budget.ftol_spatial_safety_factor
                                dynamic_ftol = max(config.numerical_method.solver.ftol, min(c_ceil, c_sf * spatial_err_est))

                                # Extrapolate dynamic noise scaling bounded against topological condition limits
                                condition_scaling = Float64(n)^2 * max(1.0, Float64(Re))
                                n_base = config.numerical_method.solver.condition_noise_floor_baseline
                                n_min = config.numerical_method.solver.condition_noise_floor_absolute_min
                                n_sf = config.numerical_method.solver.condition_noise_floor_safety_factor
                                dynamic_noise_floor = min(config.numerical_method.solver.stagnation_noise_floor, max(n_min, n_base * condition_scaling))
                                # Strictly preserve safety margin above FTOL
                                dynamic_noise_floor = max(dynamic_noise_floor, dynamic_ftol * n_sf)
                                max_ls_iters = config.numerical_method.solver.max_linesearch_iterations
                                ls_contract = config.numerical_method.solver.linesearch_contraction_factor
                                
                                # Extract dynamic algebraic complexity scaling for Picard limits
                                local_picard_it = solver_picard_it
                                if Re >= budget.picard_re_threshold
                                    # Convection-dominated fine boundaries fundamentally demand increased smoothing allocations
                                    local_picard_it = max(local_picard_it, budget.picard_re_iterations)
                                end
                                if Da >= budget.picard_da_threshold
                                    # Massive reaction-dominated geometries natively force boundary constraints requiring homogenization
                                    local_picard_it = max(local_picard_it, budget.picard_da_iterations)
                                end

                                # Newton budget bumped in high-Re convection-dominated regimes — the Armijo line
                                # search has to backtrack many times near the iso-spikes the SmoothVelocityFloor /
                                # τ-regularisation introduce along the search line, so the iteration budget needs
                                # to span both the smoothing-out phase and the quadratic-convergence tail. See
                                # test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md for the
                                # Re=1e6, Da=1, α=0.05 case that motivates this. The schema default leaves
                                # well-resolved (low-Re) regimes bit-identical.
                                local_newton_it = solver_newton_it
                                if Re >= budget.newton_re_threshold
                                    local_newton_it = max(local_newton_it, budget.newton_re_iterations)
                                end

                                nls_picard = PorousNSSolver.SafeNewtonSolver(LUSolver(), local_picard_it, max_inc, xtol, dynamic_ftol, ls_alpha_min, ar_c1, div_fac, dynamic_noise_floor, max_ls_iters, ls_contract; mode=:picard)
                                nls_newton = PorousNSSolver.SafeNewtonSolver(LUSolver(), local_newton_it, max_inc, xtol, dynamic_ftol, ls_alpha_min, ar_c1, div_fac, dynamic_noise_floor, max_ls_iters, ls_contract)
                                
                                solver_picard = FESolver(nls_picard)
                                solver_newton = FESolver(nls_newton)
                                
                                # Interpolate initial mathematical guess state perfectly atop exact solution roots
                                x0_exact = interpolate_everywhere([u_final, p_final], X)
                                
                                for method in methods
                                    run_idx += 1
                                    println("\n--- Progress $(run_idx) / $(total_runs * length(conv_parts)) | Re=$(Re), Da=$(Da), α=$(alpha_0), method=$(method), N=$(n) ---")

                                    # [RESUME] if this (cell, method, mesh) already has a result on disk, reuse it and
                                    # skip the (expensive) solve. results_cache still accumulates it in mesh order, so a
                                    # later COMPUTED mesh for the same cell rewrites a complete group; a fully-reused cell
                                    # is never rewritten and keeps its on-disk group untouched.
                                    sig_r = (Float64(Re), Float64(Da), Float64(alpha_0), Int(kv), Int(kp), String(etype), String(method))
                                    if haskey(resume_results, sig_r) && haskey(resume_results[sig_r], Int(n))
                                        sv = resume_results[sig_r][Int(n)]
                                        rc = results_cache[(etype, kv, kp, alpha_0, Da, Re, method)]
                                        push!(rc["hs"], 1.0 / n)
                                        push!(rc["err_u_l2"], sv["err_u_l2"]);     push!(rc["err_p_l2"], sv["err_p_l2"])
                                        push!(rc["err_u_h1"], sv["err_u_h1"]);     push!(rc["err_p_h1"], sv["err_p_h1"])
                                        push!(rc["eval_times"], sv["eval_times"]); push!(rc["eval_iters"], sv["eval_iters"])
                                        push!(rc["eval_eps"], sv["eval_eps"]);     push!(rc["eval_residuals"], sv["eval_residuals"])
                                        push!(rc["mms_plateau_success"], nothing); push!(rc["overall_verification_success"], true)
                                        push!(rc["fold"], sv["fold"]);             push!(rc["osgs_no_advance"], sv["osgs_no_advance"])
                                        println("  [RESUME] N=$n reused from disk — skipping solve")
                                        continue
                                    end

                                    # [fix: ASGS≠OSGS] The per-N config_dict hardcodes stabilization.method="ASGS"; without
                                    # this override every OSGS cell silently ran ASGS (solve_system reads the method from the
                                    # config, not the loop variable). Rebuild config + its fold variant with the LOOP's method.
                                    # Galerkin bypasses solve_system (execute_solver_galerkin_inline!), so its config method is
                                    # irrelevant — keep "ASGS" (load_config_from_dict need not accept "Galerkin").
                                    cell_stab_method = uppercase(method) == "GALERKIN" ? "ASGS" : method
                                    config_dict_m = deepcopy(config_dict)
                                    config_dict_m["numerical_method"]["stabilization"]["method"] = cell_stab_method
                                    config_m = PorousNSSolver.load_config_from_dict(config_dict_m)
                                    config_fold_dict_m = deepcopy(config_dict_m)
                                    config_fold_dict_m["numerical_method"]["solver"] = merge(config_fold_dict_m["numerical_method"]["solver"], Dict("eps_tol_momentum" => eps_tol_momentum_fold))
                                    config_fold_m = PorousNSSolver.load_config_from_dict(config_fold_dict_m)

                                    eps_pert_base = Float64(get(test_dict, "epsilon_pert", [0.1])[1])
                                    max_n_pert = Int(get(test_dict, "max_n_pert", 5))
                                    
                                    u_h_exact, p_h_exact = x0_exact
                                    u_ex_L2 =  sqrt(abs(sum(∫(u_h_exact ⋅ u_h_exact)dΩ)))
                                    
                                    setup = PorousNSSolver.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
                                    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)
                                    iter_solvers = PorousNSSolver.StageSolvers(solver_picard, solver_newton)
                                    mms_setup = MMSSetup(u_final, p_final, h_raw_func, u_ex_L2, norm_h, U_c, P_c, L)
                                    pert_cfg = PerturbationConfig(eps_pert_base, max_n_pert)
                                    
                                    # ==============================================================================
                                    # NONLINEAR CONVERGENCE LOOP
                                    # Incrementally shrink initial numerical perturbation forcing iterative validation
                                    # ==============================================================================
                                    success, mms_plateau_success, successful_eps, final_x0, eval_time, iter_count_attempt, final_residual_attempt, osgs_short_circuited = execute_outer_homotopy_perturbation_loop!(
                                        setup, formulation, iter_solvers, config_m, method, dynamic_ftol,
                                        mms_setup, pert_cfg,
                                        mms_verification_enabled, mms_tau_err, mms_eps_u_l2, mms_eps_u_h1, mms_eps_p_l2,
                                        mms_max_extra_cycles, mms_require_consecutive_passes,
                                        mms_rate_check_factor,
                                        n, kv
                                    )

                                    # [graceful fold recording] If the tight gate was not reached, retry ONCE with
                                    # the relaxed gate so the solver accepts the achievable ε_M floor and returns the
                                    # real (pre-asymptotic O(h^k)) solution — flagged 'fold' — instead of NaN. Mirrors
                                    # the regular ManufacturedSolutions harness recording its high-Re fold cells.
                                    # [OSGS-leak guard] OSGS can report success while never advancing off the ASGS
                                    # Stage-I boot state (the nonlinear.jl initial_ftol short-circuit). The recorded
                                    # state is then byte-identical to ASGS — NOT a genuine OSGS datum for an
                                    # ASGS-vs-OSGS study. Record it honestly as NaN (flagged osgs_no_advance) and do
                                    # NOT fold-retry (the boot would short-circuit again). See docs/lessons_learned.md.
                                    osgs_no_advance = osgs_short_circuited && uppercase(method) != "GALERKIN"
                                    if osgs_no_advance
                                        println("  [OSGS-leak guard] OSGS reported success but did NOT advance off the ASGS boot state (0 OSGS iterations) — recording NaN, not the leaked ASGS state.")
                                        success = false
                                    end

                                    is_fold = false
                                    if !success && !osgs_no_advance
                                        println("  [fold] Tight ε_M gate not reached — retrying with relaxed gate $(eps_tol_momentum_fold) to record the achievable-floor solution (high-Re×low-α corner)...")
                                        succ_f, _, x0_f, it_f, t_f = PorousNSSolver.solve_system(setup, formulation, iter_solvers, config_fold_m, x0_exact)
                                        if succ_f && x0_f !== nothing && all(isfinite, get_free_dof_values(x0_f))
                                            success = true; is_fold = true; final_x0 = x0_f; successful_eps = 0.0
                                            eval_time += t_f; iter_count_attempt += it_f
                                            println("  [fold] Recorded at the achievable floor — flagged 'fold' (NOT gate-converged).")
                                        else
                                            println("  [fold] Relaxed-gate retry also failed — genuine non-convergence, recording NaN.")
                                        end
                                    end

                                    # Fix 6: overall verification combines solver convergence and (when MMS
                                    # is enabled) the MMS plateau status. `nothing` plateau means MMS was
                                    # disabled, which counts as verified by default.
                                    overall_verification_success = success && (isnothing(mms_plateau_success) || mms_plateau_success)
                                    if success && mms_plateau_success === false
                                        println("  [⚠] Cell flagged: solver_success=true but mms_plateau_success=false (budget exhausted).")
                                    end

                                    if success && final_x0 !== nothing
                                        u_h, p_h = final_x0
                                        el2_u, el2_p, eh1_u, eh1_p = calculate_normalized_errors(u_h, p_h, u_final, p_final, U_c, P_c, L, dΩ)
                                        
                                        vtk_dir = joinpath("results", "vtk")
                                        if !isdir(vtk_dir)
                                            mkpath(vtk_dir)
                                        end
                                        u_ex_cf = interpolate_everywhere(u_final, U)
                                        p_ex_cf = interpolate_everywhere(p_final, P)
                                        filename = joinpath(vtk_dir, "mms_$(method)_Re$(Float64(Re))_Da$(Float64(Da))_N$(n).vtu")
                                        writevtk(Ω, filename, cellfields=["uh"=>u_h, "ph"=>p_h, "uex"=>u_ex_cf, "pex"=>p_ex_cf, "alpha"=>alpha_cf, "err_u"=>u_h - u_ex_cf, "err_p"=>p_h - p_ex_cf])
                                    else
                                        el2_u, el2_p, eh1_u, eh1_p = NaN, NaN, NaN, NaN
                                    end
                                    
                                    k_id = (etype, kv, kp, alpha_0, Da, Re, method)
                                    push!(results_cache[k_id]["hs"], 1.0 / n)
                                    push!(results_cache[k_id]["err_u_l2"], el2_u)
                                    push!(results_cache[k_id]["err_p_l2"], el2_p)
                                    push!(results_cache[k_id]["err_u_h1"], eh1_u)
                                    push!(results_cache[k_id]["err_p_h1"], eh1_p)
                                    push!(results_cache[k_id]["eval_times"], eval_time)
                                    push!(results_cache[k_id]["eval_iters"], iter_count_attempt)
                                    push!(results_cache[k_id]["eval_eps"], successful_eps)
                                    push!(results_cache[k_id]["eval_residuals"], final_residual_attempt)
                                    push!(results_cache[k_id]["mms_plateau_success"], mms_plateau_success)
                                    push!(results_cache[k_id]["overall_verification_success"], overall_verification_success)
                                    push!(results_cache[k_id]["fold"], is_fold)
                                    push!(results_cache[k_id]["osgs_no_advance"], osgs_no_advance)
                                    
                                    println("  -> L2 u/p: ", round(el2_u, sigdigits=4), " / ", round(el2_p, sigdigits=4), " | H1 u/p: ", round(eh1_u, sigdigits=4), " / ", round(eh1_p, sigdigits=4), is_fold ? "  [FOLD — recorded at floor, gate not reached]" : (success ? "  [converged]" : "  [NaN]"))
                                    # =========================================================================================
                                    # EXPORT CURRENT CURVE TO HDF5 DYNAMICALLY
                                    # =========================================================================================
                                    sig_base = (Float64(Re), Float64(Da), Float64(alpha_0), Int(kv), Int(kp), String(etype))
                                    if haskey(existing_signatures, sig_base)
                                        c_idx = existing_signatures[sig_base]
                                    else
                                        max_idx += 1
                                        c_idx = max_idx
                                        existing_signatures[sig_base] = c_idx
                                    end
                                    
                                    h5open(h5_path, "r+") do h5f_out
                                        res = results_cache[k_id]
                                        grp_name = "config_$(c_idx)_$(method)"
                                        
                                        if haskey(h5f_out, grp_name)
                                            delete_object(h5f_out, grp_name)
                                        end
                                        
                                        g = create_group(h5f_out, grp_name)
                                        g["h"] = res["hs"]
                                        g["err_u_l2"] = res["err_u_l2"]
                                        g["err_p_l2"] = res["err_p_l2"]
                                        g["err_u_h1"] = res["err_u_h1"]
                                        g["err_p_h1"] = res["err_p_h1"]
                                        g["eval_times"] = res["eval_times"]
                                        g["eval_iters"] = res["eval_iters"]
                                        g["eval_eps"] = res["eval_eps"]
                                        g["eval_residuals"] = res["eval_residuals"]
                                        g["fold"] = res["fold"]
                                        g["osgs_no_advance"] = res["osgs_no_advance"]

                                        attributes(g)["total_time_s"] = sum(res["eval_times"])
                                        attributes(g)["total_iters"] = sum(res["eval_iters"])
                                        attributes(g)["Re"] = Float64(Re)
                                        attributes(g)["Da"] = Float64(Da)
                                        attributes(g)["alpha_0"] = Float64(alpha_0)
                                        attributes(g)["physical_epsilon"] = 0.0
                                        attributes(g)["numerical_epsilon_coeff"] = 0.0
                                        attributes(g)["k_velocity"] = Int(kv)
                                        attributes(g)["k_pressure"] = Int(kp)
                                        attributes(g)["element_type"] = String(etype)
                                        attributes(g)["method"] = String(method)
                                        attributes(g)["config_file"] = String(config_file)
                                        attributes(g)["target_ftol"] = Float64(dynamic_ftol)
                                        attributes(g)["osgs_tolerance"] = Float64(dynamic_ftol)
                                        
                                        compute_slope(x, y) = sum((x .- sum(x)/length(x)) .* (y .- sum(y)/length(y))) / (sum((x .- sum(x)/length(x)).^2) + 1e-15)
                                        log_h = log.(res["hs"])
                                        attributes(g)["rate_u_l2"] = compute_slope(log_h, log.(res["err_u_l2"]))
                                        attributes(g)["rate_u_h1"] = compute_slope(log_h, log.(res["err_u_h1"]))
                                        attributes(g)["rate_p_l2"] = compute_slope(log_h, log.(res["err_p_l2"]))
                                        attributes(g)["rate_p_h1"] = compute_slope(log_h, log.(res["err_p_h1"]))
                                        
                                        println("\n    [💾] Appended $(grp_name) accurately to HDF5 file layout. Available for plotting!")
                                    end
                                    
                                    # Force garbage collection to prevent C-pointer memory leaks from UMFPACK LU factorizations across the N sweeps
                                    GC.gc()
                                    
                                end # end method
                            end # end Re
                        end # end Da
                    end # end alpha_0
                end # end n
            end # end kp
        end # end kv
    end # end etype
end # end function

if abspath(PROGRAM_FILE) == @__FILE__
    config_file = length(ARGS) > 0 ? ARGS[1] : "test_config.json"
    run_mms(config_file)
end
