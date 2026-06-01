# test/long/ManufacturedSolutions/run_test.jl
# ==============================================================================================
# Nature & Intent:
# The definitive Method of Manufactured Solutions (MMS) workflow. Constructs an exact artificial 
# polynomial state $(u_{ex}, p_{ex})$ and evaluates its continuous symbolic derivatives to force the 
# solver's RHS. By running the full Picard/Newton non-linear steps, it validates the continuous ExactNewton 
# exactness and global system numerical stabilization properties. Extensively sweeps ($Re, Da, h$) configurations
# to map asymptotic convergence bounds mapping to the error definitions $O(h^{k+1})$.
#
# Mathematical Formulation Alignment:
# Completely aligns with Equation 1018-1039 of `article.tex`. Uses parameter normalizations ($U_c, P_c$)
# strictly matching continuum characteristic analysis, forbidding arbitrary rescaling limits.
#
# Associated Files / Functions:
# - `src/formulations/continuous_problem.jl`
# - `src/solvers/nonlinear.jl` (`solve_system`)
# - MMS analytical fields mapping (`u_ex`, `p_ex`)
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

# ==============================================================================
# Helper Constructors
# ==============================================================================

function _build_local_mesh(domain_cfg, mesh_cfg, L::Float64=1.0)
    # The JSON `bounding_box` is interpreted as the L=1 baseline; the actual physical
    # domain extends as `L .* bounding_box`. This keeps `Re = U·L/ν` and `Da = α·σ·L²/ν`
    # as TRUE dimensionless numbers (the manufactured shape `sin(πx/L)` and the physical
    # domain scale together, preserving the dimensionless analysis under L-rescaling).
    bbox = collect(L .* domain_cfg.bounding_box)
    domain = Tuple(bbox)
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
# The radii `r_1, r_2` are L=1 baselines that scale with the characteristic length: the
# actual porosity-bump transition lives at `L*r_1 < r < L*r_2`, keeping the bump's
# *relative* footprint on the L-scaled domain unchanged.
function build_porosity_field(config, alpha_0, alpha_infty, L::Float64=1.0)
    PorousNSSolver.SmoothRadialPorosity(
        Float64(alpha_0), Float64(alpha_infty),
        L * config.domain.r_1, L * config.domain.r_2
    )
end

# [encoding] Compute the characteristic-length `L` and velocity scale `U_amp` for one
# (Re, Da, α_∞) cell, per the chosen encoding strategy. See theory/centered_encoding.tex
# Section 6 for the derivation of the four formulas.
#
#   "centered"  — historical default: `L=1`, `U=Re/√(α_∞·Da)`. Centers ν·σ=1.
#   "balanced"  — `L=√(α_∞·Da)`, `U=Re/√(α_∞·Da)`. Forces ν=σ; U unchanged from centered.
#   "minmax"    — `L=√(α_∞·Da)`, `U=(Re²/(α_∞·Da))^(1/4)`. Minimises max(|log U|,|log ν|,|log σ|);
#                 at the C7 corner reduces the worst dimensional excursion from 9 to 4.5 orders.
#   "unit"      — `L=1`, `U=1`. The dimensional-default baseline; kept for diagnostic probes.
function compute_L_and_U(strategy::String, Re::Float64, Da::Float64, alpha_infty::Float64)
    if strategy == "centered"
        return (1.0, Float64(Re) / sqrt(alpha_infty * Float64(Da)))
    elseif strategy == "balanced"
        L = sqrt(alpha_infty * Float64(Da))
        return (L, Float64(Re) / sqrt(alpha_infty * Float64(Da)))
    elseif strategy == "minmax"
        L = sqrt(alpha_infty * Float64(Da))
        U_amp = (Float64(Re)^2 / (alpha_infty * Float64(Da)))^(1.0/4.0)
        return (L, U_amp)
    elseif strategy == "unit"
        return (1.0, 1.0)
    else
        error("Unknown encoding_strategy: \"$strategy\". Valid: \"centered\", \"balanced\", \"minmax\", \"unit\".")
    end
end

# Sets up the specific continuous VMS mathematical behavior for the Manufactured Solution
function build_mms_formulation(config, Da, Re, U_amp, L, alpha_infty)
    # Define exact residual projections enforcing strictly positive stabilization operators
    proj = PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma()
    
    # Establish velocity regularization parameters strictly from configuration
    reg = PorousNSSolver.SmoothVelocityFloor(
        config.physical_properties.u_base_floor_ref, 
        0.0, 
        config.physical_properties.epsilon_floor
    )
    
    # Derive kinematic viscosity and exact constant sigma for the reaction operator 
    # to perfectly represent Darcy-scale bounds parameterized by Reynolds / Darcy inputs.
    nu_calculated = U_amp * L / Float64(Re)
    eps_calculated = config.physical_properties.eps_val
    
    sigma_c = Float64(Da) * alpha_infty * nu_calculated / (L^2)
    rxn = PorousNSSolver.ConstantSigmaLaw(sigma_c)
    
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

# Error evaluator returning genuinely DIMENSIONLESS norms (encoding-invariant under L-rescaling).
#
# Derivation (writing the dimensional error as e(x) = U_c · ê(x/L), with x̂ = x/L the
# dimensionless coordinate and Ω the L-scaled physical domain):
#
#   ‖e‖²_{L²(Ω)}     = ∫_Ω |e|² dx = U_c² · ∫_Ω |ê(x/L)|² dx
#                    = U_c² · L^d · ‖ê‖²_{L²(Ω̂)}
#   ⇒  ‖e‖_{L²(Ω)}   = U_c · √(|Ω|) · ‖ê‖_{L²(Ω̂)}
#
#   ‖∇e‖²_{L²(Ω)}    = ∫_Ω (U_c/L)² |∇_x̂ ê|² dx = (U_c/L)² · L^d · ‖∇ê‖²_{L²(Ω̂)}
#   ⇒  ‖∇e‖_{L²(Ω)}  = U_c · L^{d/2 - 1} · ‖∇ê‖_{L²(Ω̂)}   (independent of L in 2D where d=2)
#
# So the dimensionless quantities are:
#
#     el2_u_dimless   = ‖e_u‖_{L²(Ω)}  / (U_c · √(|Ω|))   = ‖ê_u‖_{L²(Ω̂)}
#     eh1_u_dimless   = ‖∇e_u‖_{L²(Ω)} / U_c              = ‖∇ê_u‖_{L²(Ω̂)}        (2D)
#
# For L=1 with a unit-area baseline bounding box `[-0.5, 0.5]²` this collapses to the
# legacy form `el2_u = ‖e‖/U_c`, `eh1_u = ‖∇e‖/(U_c/L)` — so existing centered-encoding
# K=1 sweeps reproduce bit-identically. For L≠1 (balanced / minmax encodings) the new form
# removes the L-inflation that the legacy normalisation introduced.
function calculate_normalized_errors(u_h, p_h, u_final, p_final, U_c, P_c, L, dΩ)
    e_u = u_final - u_h
    e_p = p_final - p_h

    # Domain measure |Ω| (this is the *physical* area of the L-scaled domain).
    area = sum(∫(1.0)dΩ)
    sqrt_area = sqrt(abs(area))     # √(|Ω|) = L · √(|Ω̂|); for L=1 with [-0.5, 0.5]² this is 1.0

    # 1. Velocity L² error, dimensionless via ‖e_u‖/(U_c · √(|Ω|)).
    el2_u = sqrt(sum(∫(e_u ⋅ e_u)dΩ)) / (U_c * sqrt_area)

    # Pressure null-space alignment (centre the error so the gauge-mode is not penalised).
    mean_e_p = sum(∫(e_p)dΩ) / area
    e_p_centered = e_p - mean_e_p

    # 2. Pressure L² error, dimensionless via ‖e_p‖/(P_c · √(|Ω|)).
    el2_p = sqrt(sum(∫(e_p_centered * e_p_centered)dΩ)) / (P_c * sqrt_area)

    # 3. Semi-H¹ errors. In 2D the L from integration cancels the 1/L from the gradient, so
    # the dimensionless H¹ semi-norm is simply ‖∇e‖/U_c (no L factor in the divisor).
    eh1_semi_u = sqrt(sum(∫(∇(e_u) ⊙ ∇(e_u))dΩ)) / U_c
    eh1_semi_p = sqrt(sum(∫(∇(e_p) ⋅ ∇(e_p))dΩ)) / P_c

    return el2_u, el2_p, eh1_semi_u, eh1_semi_p
end

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

# Algorithm E: Outer Homotopy Parameter Scaling
function execute_outer_homotopy_perturbation_loop!(
    setup::PorousNSSolver.FETopology, formulation::PorousNSSolver.VMSFormulation,
    iter_solvers::PorousNSSolver.IterativeSolvers, config::PorousNSConfig,
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
    # [relative-residual gate] Iter-0 residual ‖R(x₀)‖ of the converged attempt,
    # populated by solve_system into `diagnostics_cache["initial_residual_norm"]`.
    # Provides the natural scale ‖f‖ for the gate `‖R_final‖/‖R_initial‖ < tol`
    # so Re=10⁶ cells stop being false-flagged on raw residual magnitude.
    initial_residual_attempt = NaN
    
    for attempt in 0:(pert_cfg.max_n_pert + 1)
        # [design-intent] Iteration order is **hard → easy**: start with the largest perturbation
        # (eps_pert_base, default 1.0) and only fall back to milder perturbations / eps=0 (clean
        # u_ex initial guess) if Newton fails. Break-at-first-success means the harness records
        # in `eval_eps[n]` the LARGEST perturbation the solver could absorb at this mesh; that
        # column is an explicit map of where each cell needed help. A solver that only works when
        # primed with u_ex is not practical — this test exposes that.
        #
        # NOTE: at extreme-parameter cells (e.g. config_18 Re=Da=1e6) the eps=1.0 attempt may
        # land Newton in the basin's noise-floor region rather than at the true discrete minimum
        # (probe `diagnostics/velocity_centering_probe.jl` measured a ~2× tighter L²(u) reachable
        # via eps=0). That noise-floor convergence is the SOLVER'S honest behaviour from a generic
        # initial guess — it shows up as sub-optimal slope and is the diagnostic signal we want,
        # not noise to be optimised away.
        eps_p = attempt <= pert_cfg.max_n_pert ? pert_cfg.eps_pert_base / (10.0^attempt) : 0.0
        
        u_0_func = PerturbationFunc(mms_setup.u_final, mms_setup.h_raw_func, eps_p * (mms_setup.u_ex_L2 / mms_setup.norm_h))
        x0 = interpolate_everywhere([u_0_func, mms_setup.p_final], setup.X)
        
        println("\n    ==================================================")
        println("    [Attempt $(attempt+1)/$(pert_cfg.max_n_pert + 2)] Homotopy Perturbation Scale: eps_pert = $eps_p")
        println("    [!] Delegating orchestration to PDE assembly module via `src/solvers/porous_solver.jl` (Mode: $method)")
        println("    ==================================================")
        
        local_stab_cfg = PorousNSSolver.StabilizationConfig(
            method=method,
            osgs_iterations=config.numerical_method.stabilization.osgs_iterations,
            osgs_inner_newton_iters=config.numerical_method.stabilization.osgs_inner_newton_iters,
            osgs_tolerance=dynamic_ftol,
            osgs_stopping_mode=config.numerical_method.stabilization.osgs_stopping_mode,
            osgs_projection_tolerance=dynamic_ftol,
            osgs_state_drift_scale=config.numerical_method.stabilization.osgs_state_drift_scale,
            osgs_warmup_iterations=config.numerical_method.stabilization.osgs_warmup_iterations,
            osgs_warmup_tolerance=config.numerical_method.stabilization.osgs_warmup_tolerance
        )
        
        local_diagnostics_cache = Dict{String, Any}()
        mms_cfg = nothing
        if mms_verification_enabled
            mms_cfg = (
                enabled = true, tau_err = mms_tau_err, eps_u_l2 = mms_eps_u_l2, eps_u_h1 = mms_eps_u_h1,
                eps_p_l2 = mms_eps_p_l2, max_extra_cycles = mms_max_extra_cycles,
                require_consecutive_passes = mms_require_consecutive_passes,
                h_local = 1.0 / n,   # §5.1: mesh size for h-scaling plateau floors
                kv = kv,             # §5.1: velocity polynomial order for the scaling exponent
                rate_check_factor = mms_rate_check_factor,   # §5.2: sub-optimal-rate flag threshold
                oracle = (uh, ph) -> calculate_normalized_errors(uh, ph, mms_setup.u_final, mms_setup.p_final, mms_setup.U_c, mms_setup.P_c, mms_setup.L, setup.dΩ)
            )
        end
        
        sys_success, sys_mms_plateau_success, sys_final_x0, sys_iter_count, sys_eval_time = PorousNSSolver.solve_system(
            setup, formulation, iter_solvers,
            config, x0;
            diagnostics_cache=local_diagnostics_cache, mms_cfg=mms_cfg
        )
        
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
            initial_residual_attempt = get(local_diagnostics_cache, "initial_residual_norm", NaN)
            break
        else
            println("\n      [❌] Outer loop execution completely stalled structurally above convergence tolerance (`$(local_stab_cfg.osgs_tolerance)`) or system fully diverged.")
            final_residual_attempt = get(local_diagnostics_cache, "final_residual_norm", NaN)
            initial_residual_attempt = get(local_diagnostics_cache, "initial_residual_norm", NaN)
        end
    end

    if !success
         println("    [WARNING] Completely failed to find root basin. Returning NaN.")
    end

    return success, mms_plateau_success, successful_eps, final_x0, eval_time, iter_count_attempt, final_residual_attempt, initial_residual_attempt
end

function run_mms(config_file="test_config.json")
    config_path = joinpath(@__DIR__, "data", config_file)
    test_dict = JSON3.read(read(config_path, String), Dict{String, Any})
    
    as_list(x) = x isa Vector ? x : [x]
    
    Re_list = as_list(test_dict["physical_properties"]["Re"])
    Da_list = as_list(test_dict["physical_properties"]["Da"])
    alpha_list = as_list(test_dict["domain"]["alpha_0"])

    # [paper-grid orchestration] Optional opt-in skip list for individual physics cells
    # within the (Re × Da × α₀) cross-product. Format: a list of [Re, Da, alpha_0] triples.
    # Used by the paper-grid Round-1 configs to defer the (Re=1e6, α₀=0.05) fold corner to a
    # dedicated Round-2 batch. Default empty → no cells skipped, existing behavior unchanged.
    skip_cells_set = Set{Tuple{Float64,Float64,Float64}}()
    for triple in get(test_dict, "skip_cells", Any[])
        @assert length(triple) == 3 "skip_cells entries must be [Re, Da, alpha_0]; got $(triple)"
        push!(skip_cells_set, (Float64(triple[1]), Float64(triple[2]), Float64(triple[3])))
    end

    nm_dict = test_dict["numerical_method"]
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

    # [encoding] Strategy for choosing the dimensional `(L, U_amp)` per cell. See
    # theory/centered_encoding.tex Section 6 for the four formulas. Default `"centered"`
    # preserves the pre-2026-06 behaviour (L=1, centered U_amp) bit-identically.
    encoding_strategy = String(get(test_dict, "encoding_strategy", "centered"))

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    h5_filename = get(test_dict, "h5_filename", "convergence_data.h5")
    h5_path = joinpath(results_dir, h5_filename)
    
    erase_past = get(test_dict, "erase_past_results", false)
    h5_mode = erase_past ? "w" : "cw"
    
    h5f = h5open(h5_path, h5_mode)
    max_idx = 0
    existing_signatures = Dict()
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
                catch
                end
            end
        end
    end
    close(h5f)
    
    conv_parts = mesh_dict["convergence_partitions"]
    
    total_runs = sum(1 for etype in etype_list, kv in kv_list, kp in kp_list, alpha_0 in alpha_list, Da in Da_list, Re in Re_list if !(equal_order_only && kv != kp) && !((Float64(Re), Float64(Da), Float64(alpha_0)) in skip_cells_set)) * length(methods)
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
                        "eval_initial_residuals" => Float64[],
                        "mms_plateau_success" => Union{Bool,Nothing}[],
                        "overall_verification_success" => Bool[]
                    )
                end

                # [resume-aware] Preload completed cells from existing HDF5 so the inner
                # solve loop can skip cells whose (Re, Da, α, method) at the current N is
                # already on disk. Only loads groups matching the current (etype, kv, kp).
                if !erase_past && isfile(h5_path)
                    h5open(h5_path, "r") do h5f_in
                        for gname in keys(h5f_in)
                            parts = split(gname, "_")
                            if length(parts) >= 3 && parts[1] == "config"
                                grp = h5f_in[gname]
                                att = attributes(grp)
                                try
                                    r = Float64(read(att["Re"]))
                                    d = Float64(read(att["Da"]))
                                    a = Float64(read(att["alpha_0"]))
                                    kv_v = Int(read(att["k_velocity"]))
                                    kp_v = Int(read(att["k_pressure"]))
                                    et = String(read(att["element_type"]))
                                    m = String(read(att["method"]))
                                    if et == etype && kv_v == kv && kp_v == kp
                                        k_id_load = (etype, kv, kp, a, d, r, m)
                                        if haskey(results_cache, k_id_load)
                                            results_cache[k_id_load]["hs"]              = Vector{Float64}(read(grp["h"]))
                                            results_cache[k_id_load]["err_u_l2"]        = Vector{Float64}(read(grp["err_u_l2"]))
                                            results_cache[k_id_load]["err_p_l2"]        = Vector{Float64}(read(grp["err_p_l2"]))
                                            results_cache[k_id_load]["err_u_h1"]        = Vector{Float64}(read(grp["err_u_h1"]))
                                            results_cache[k_id_load]["err_p_h1"]        = Vector{Float64}(read(grp["err_p_h1"]))
                                            results_cache[k_id_load]["eval_times"]      = Vector{Float64}(read(grp["eval_times"]))
                                            results_cache[k_id_load]["eval_iters"]      = Vector{Int}(read(grp["eval_iters"]))
                                            results_cache[k_id_load]["eval_eps"]        = Vector{Float64}(read(grp["eval_eps"]))
                                            results_cache[k_id_load]["eval_residuals"]  = Vector{Float64}(read(grp["eval_residuals"]))
                                            # Backward-compatible: pre-existing HDF5s lack `eval_initial_residuals`.
                                            # Fill with NaN of matching length so the column remains valid in
                                            # the cache and the analyzer's relative gate gracefully falls back
                                            # to the absolute gate for legacy rows.
                                            if haskey(grp, "eval_initial_residuals")
                                                results_cache[k_id_load]["eval_initial_residuals"] = Vector{Float64}(read(grp["eval_initial_residuals"]))
                                            else
                                                results_cache[k_id_load]["eval_initial_residuals"] = fill(NaN, length(results_cache[k_id_load]["hs"]))
                                            end
                                            if haskey(grp, "overall_verification_success")
                                                ovs = read(grp["overall_verification_success"])
                                                results_cache[k_id_load]["overall_verification_success"] = Bool.(ovs .== 1)
                                            end
                                            if haskey(grp, "mms_plateau_success")
                                                mps = read(grp["mms_plateau_success"])
                                                results_cache[k_id_load]["mms_plateau_success"] = Union{Bool,Nothing}[x == -1 ? nothing : (x == 1) for x in mps]
                                            end
                                        end
                                    end
                                catch
                                    # unreadable group; skip
                                end
                            end
                        end
                    end
                    n_preloaded = sum(length(results_cache[k]["hs"]) for k in keys(results_cache) if k[1] == etype && k[2] == kv && k[3] == kp; init=0)
                    println("[resume] Preloaded $(n_preloaded) (cell, N) data points from $(h5_path) for etype=$(etype), k_v=$(kv), k_p=$(kp).")
                end

                for n in conv_parts
                    println("\n========================================")
                    println("BEGIN N = $n for etype = $etype, k_v = $kv, k_p = $kp (encoding=$encoding_strategy)")
                    println("========================================")

                    # ==============================================================================
                    # PER-N CONSTANTS
                    # The L-independent setup: reference FE (depends on kv/kp), quadrature degree
                    # (depends on formulation type + kv), stabilization constants c_1, c_2.
                    # Everything that depends on L (mesh, FE spaces over the mesh, perturbation
                    # polynomial, porosity radii) is rebuilt inside the per-cell loop below because
                    # L is now per-cell via the encoding strategy.
                    # ==============================================================================
                    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, kv)
                    refe_p = ReferenceFE(lagrangian, Float64, kp)
                    # Coordinate quadrature degree evaluation recursively mapped back to formulation definitions.
                    # The MMS runner always uses ConstantSigmaLaw (see build_mms_formulation), whose
                    # min_quadrature_degree default is 0, so the result matches the legacy base rule.
                    degree = PorousNSSolver.get_quadrature_degree(PorousNSSolver.PaperGeneralFormulation, kv, PorousNSSolver.ConstantSigmaLaw(0.0))
                    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, kv)

                    # ==============================================================================
                    # PHYSICS PARAMETER SWEEP
                    # Per-cell: pick L and U_amp from the encoding strategy, then build the L-coupled
                    # mesh / FE spaces / quadrature / perturbation / porosity / formulation / solver.
                    # ==============================================================================
                    for alpha_0 in alpha_list
                        alpha_infty = 1.0
                        for Da in Da_list
                            for Re in Re_list
                                if (Float64(Re), Float64(Da), Float64(alpha_0)) in skip_cells_set
                                    println("[skip_cells] Re=$(Re), Da=$(Da), α=$(alpha_0) — skipped by config")
                                    continue
                                end

                                # [encoding] Pick the per-cell characteristic length and velocity scale
                                # from the configured strategy. Default "centered" reproduces the
                                # pre-2026-06 behaviour (L=1, U=Re/√(α_∞·Da)) bit-identically.
                                # See theory/centered_encoding.tex Section 6 for the four formulas.
                                (L_cell, U_amp) = compute_L_and_U(encoding_strategy, Float64(Re), Float64(Da), alpha_infty)
                                println("  [cell setup] Re=$(Re), Da=$(Da), α=$(alpha_0), strategy=$encoding_strategy → L=$(L_cell), U_amp=$(U_amp)")

                                # Build per-cell config. The bounding_box and r_1, r_2 carried here are
                                # the L=1 BASELINE values from JSON; _build_local_mesh and
                                # build_porosity_field apply the L_cell scaling internally.
                                config_dict = Dict(
                                    "physical_properties" => Dict("nu" => 1.0, "eps_val" => 1e-8, "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0),
                                    "domain" => Dict(
                                        "alpha_0" => 0.4,
                                        "bounding_box" => test_dict["domain"]["bounding_box"],
                                        "r_1" => test_dict["domain"]["r_1"],
                                        "r_2" => test_dict["domain"]["r_2"]
                                    ),
                                    "numerical_method" => Dict(
                                        "element_spaces" => Dict("k_velocity" => Int(kv), "k_pressure" => Int(kp)),
                                        "mesh" => Dict("element_type" => String(etype), "partition" => [n, n]),
                                        "stabilization" => Dict(
                                            "method" => "ASGS",
                                            "osgs_inner_newton_iters" => get(get(get(test_dict, "numerical_method", Dict()), "stabilization", Dict()), "osgs_inner_newton_iters", 3)
                                        ),
                                        "solver" => get(get(test_dict, "numerical_method", Dict()), "solver", Dict())
                                    )
                                )
                                config = PorousNSSolver.load_config_from_dict(config_dict)

                                # Build L-scaled mesh (bounding_box · L_cell) and FE spaces over it.
                                model = _build_local_mesh(config.domain, config.numerical_method.mesh, L_cell)
                                labels = get_face_labeling(model)
                                add_tag_from_tags!(labels, "all_boundaries", [1,2,3,4,5,6,7,8])

                                V = TestFESpace(model, refe_u, conformity=:H1, labels=labels, dirichlet_tags=["all_boundaries"])
                                Q = TestFESpace(model, refe_p, conformity=:H1)
                                V_free = TestFESpace(model, refe_u, conformity=:H1)
                                Q_free = TestFESpace(model, refe_p, conformity=:H1)
                                Ω = Triangulation(model)
                                dΩ = Measure(Ω, degree + 4)

                                # Element-wise h field (scales as L_cell·area_baseline because the
                                # cell measure already inherits the L_cell-scaled domain).
                                # Note: Re_h = U_amp·h/ν = (U_amp·L_cell/ν)·(h/L_cell) = Re/N — L_cell
                                # cancels, so τ_NS stabilisation stays at the same dimensionless scale
                                # across encodings.
                                if etype == "TRI"
                                    h_array = lazy_map(v -> sqrt(2.0 * abs(v)), get_cell_measure(Ω))
                                else
                                    h_array = lazy_map(v -> sqrt(abs(v)), get_cell_measure(Ω))
                                end
                                h_cf = CellField(collect(h_array), Ω)

                                # Homotopy perturbation field, L-rescaled. B_fn is normalised by L^8
                                # so its magnitude stays O(1) on the L-scaled domain (raw
                                # (x-xmin)²·(xmax-x)²·… would grow as L^8); h_raw_func's polynomial
                                # frequencies pick up the same 1/L_cell factor as u_ex so the
                                # perturbation oscillates the same number of periods regardless of
                                # L_cell.
                                xmin_phys = L_cell * config.domain.bounding_box[1]
                                xmax_phys = L_cell * config.domain.bounding_box[2]
                                ymin_phys = L_cell * config.domain.bounding_box[3]
                                ymax_phys = L_cell * config.domain.bounding_box[4]
                                L_for_pert = L_cell
                                B_fn(x) = (x[1] - xmin_phys)^2 * (xmax_phys - x[1])^2 * (x[2] - ymin_phys)^2 * (ymax_phys - x[2])^2 / L_for_pert^8
                                k_pert = pi / L_for_pert
                                h_raw_func(x) = B_fn(x) * VectorValue(sin(3*k_pert*x[1])*cos(2*k_pert*x[2]), -cos(3*k_pert*x[1])*sin(2*k_pert*x[2]))

                                h_pert_cf = CellField(h_raw_func, Ω)
                                norm_h = sqrt(abs(sum(∫( h_pert_cf ⋅ h_pert_cf )dΩ)))
                                if norm_h <= 0.0
                                    error("Perturbation field norm must be strictly positive (when perturbing).")
                                end

                                Y = MultiFieldFESpace([V, Q])

                                # Solver constants from per-cell config (independent of L_cell; read
                                # here to keep the cell loop self-contained after the restructure).
                                tau_reg_lim = config.physical_properties.tau_regularization_limit
                                solver_newton_it = config.numerical_method.solver.newton_iterations
                                solver_picard_it = config.numerical_method.solver.picard_iterations
                                max_inc = config.numerical_method.solver.max_increases
                                xtol = config.numerical_method.solver.xtol
                                stagnation_tol = config.numerical_method.solver.stagnation_noise_floor
                                ftol = config.numerical_method.solver.ftol
                                ls_alpha_min = config.numerical_method.solver.linesearch_alpha_min
                                freeze_cusp = config.numerical_method.solver.freeze_jacobian_cusp

                                # Porosity field with L-scaled radii (JSON r_1, r_2 are L=1 baselines).
                                alpha_field = build_porosity_field(config, alpha_0, alpha_infty, L_cell)
                                alpha_cf = CellField(x -> PorousNSSolver.alpha(alpha_field, x), Ω)

                                # Build formulation and MMS using the encoding-chosen (L_cell, U_amp).
                                form = build_mms_formulation(config, Da, Re, U_amp, L_cell, alpha_infty)

                                # Exact execution of full manufactured solutions analytical expressions
                                mms = PorousNSSolver.Paper2DMMS(form, U_amp, alpha_field; L=L_cell, alpha_infty=alpha_infty)
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
                                c_ceil = config.numerical_method.solver.dynamic_ftol_ceiling
                                c_sf = config.numerical_method.solver.dynamic_ftol_spatial_safety_factor
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

                                # Per-field RELATIVE tolerances (Option C, intrinsic-scale variant):
                                # pass dimensionless targets per field. The solver derives the per-field
                                # absolute thresholds at each iteration from ‖x_current[block_k]‖_∞ (with
                                # cross-field Bernoulli fallback when zero). No problem-specific external
                                # scale (U_c, P_c) is plumbed through — the proxy comes from the iterate
                                # itself. The shared target `c_sf · h^(k+1)` is the dimensionless "1% of
                                # FE discretization error" goal; each field gets its own absolute threshold
                                # via solver-side `relative_target * ‖x[block]‖` (floored at the scalar
                                # ftol so the solver never tries below the user's absolute floor).
                                rel_target_per_field = [c_sf * spatial_err_est, c_sf * spatial_err_est]
                                # The noise-floor relative target mirrors `dynamic_noise_floor / dynamic_ftol`
                                # in the legacy formula. With the per-iter intrinsic-scale machinery, the
                                # solver applies the same `‖x[block]‖` scaling — so what we pass here is the
                                # *relative* ratio between noise-floor and ftol, multiplied by c_sf·h^(k+1).
                                # In the centered encoding this naturally scales with the solution magnitude.
                                rel_noise_floor_per_field = [n_sf * c_sf * spatial_err_est, n_sf * c_sf * spatial_err_est]
                                
                                # Extract dynamic algebraic complexity scaling for Picard limits
                                local_picard_it = solver_picard_it
                                if Re >= config.numerical_method.solver.dynamic_picard_re_threshold
                                    # Convection-dominated fine boundaries fundamentally demand increased smoothing allocations
                                    local_picard_it = max(local_picard_it, config.numerical_method.solver.dynamic_picard_re_iterations)
                                end
                                if Da >= config.numerical_method.solver.dynamic_picard_da_threshold
                                    # Massive reaction-dominated geometries natively force boundary constraints requiring homogenization
                                    local_picard_it = max(local_picard_it, config.numerical_method.solver.dynamic_picard_da_iterations)
                                end

                                # Newton budget bumped in high-Re convection-dominated regimes — the Armijo line
                                # search has to backtrack many times near the iso-spikes the SmoothVelocityFloor /
                                # τ-regularisation introduce along the search line, so the iteration budget needs
                                # to span both the smoothing-out phase and the quadratic-convergence tail. See
                                # test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md for the
                                # Re=1e6, Da=1, α=0.05 case that motivates this. The schema default leaves
                                # well-resolved (low-Re) regimes bit-identical.
                                local_newton_it = solver_newton_it
                                if Re >= config.numerical_method.solver.dynamic_newton_re_threshold
                                    local_newton_it = max(local_newton_it, config.numerical_method.solver.dynamic_newton_re_iterations)
                                end

                                # [honest-exit] gate: a noise-floor stop counts as converged only if
                                # ‖R‖∞ ≤ k_nf · effective_ftol. Read from config (base default 1e30 = disabled;
                                # the Phase-1 sweep configs set 10.0 to reject high-Re fold stalls).
                                k_nf = config.numerical_method.solver.noise_floor_success_max_ftol_multiple
                                # Pass a true machine-precision floor as the SafeNewtonSolver scalar ftol —
                                # NOT `dynamic_ftol`, and NOT the user's `config.solver.ftol`. The dynamic
                                # spatial scaling (c_sf · h^{k+1}) is already encoded in `rel_target_per_field`,
                                # and the per-field rule recovers absolute tolerances through ε_k · ‖x[block_k]‖.
                                # The scalar floor's only role is to prevent the per-field rule from descending
                                # below floating-point round-off. Setting it at `10 · eps(Float64)` keeps it
                                # effectively inert for any regime in which `ε_k · ‖x[block_k]‖` exceeds
                                # machine precision (i.e., for all physically meaningful cells), while leaving
                                # one order of safety against κ(A)-amplified assembly round-off.
                                static_ftol_floor = 10 * eps(Float64)
                                nls_picard = PorousNSSolver.SafeNewtonSolver(LUSolver(), local_picard_it, max_inc, xtol, static_ftol_floor, ls_alpha_min, ar_c1, div_fac, dynamic_noise_floor, max_ls_iters, ls_contract; mode=:picard, noise_floor_success_max_ftol_multiple=k_nf,
                                                                              relative_ftol_per_field=rel_target_per_field, relative_stagnation_noise_floor_per_field=rel_noise_floor_per_field)
                                nls_newton = PorousNSSolver.SafeNewtonSolver(LUSolver(), local_newton_it, max_inc, xtol, static_ftol_floor, ls_alpha_min, ar_c1, div_fac, dynamic_noise_floor, max_ls_iters, ls_contract; noise_floor_success_max_ftol_multiple=k_nf,
                                                                              relative_ftol_per_field=rel_target_per_field, relative_stagnation_noise_floor_per_field=rel_noise_floor_per_field)
                                
                                solver_picard = FESolver(nls_picard)
                                solver_newton = FESolver(nls_newton)
                                
                                # Interpolate initial mathematical guess state perfectly atop exact solution roots
                                x0_exact = interpolate_everywhere([u_final, p_final], X)
                                
                                for method in methods
                                    run_idx += 1
                                    println("\n--- Progress $(run_idx) / $(total_runs * length(conv_parts)) | Re=$(Re), Da=$(Da), α=$(alpha_0), method=$(method), N=$(n) ---")

                                    # [resume-aware] If this (cell, N) is already in HDF5 AS A VALID RESULT,
                                    # skip the solve. If the existing entry is NaN (= prior solve failed),
                                    # remove the stale entry and re-solve.
                                    let k_id_skip = (etype, kv, kp, alpha_0, Da, Re, method),
                                        target_h = 1.0 / n
                                        rc = results_cache[k_id_skip]
                                        hs_done = rc["hs"]
                                        idx = findfirst(h -> abs(h - target_h) < 1e-12, hs_done)
                                        if idx !== nothing
                                            eu_existing = rc["err_u_l2"][idx]
                                            if !isnan(eu_existing)
                                                println("    [resume-skip] Already in HDF5 (target h=$(target_h)); skipping solve.")
                                                continue
                                            else
                                                println("    [resume-retry] Prior result at h=$(target_h) was NaN; dropping stale entry and re-solving.")
                                                # Drop the stale entry so the appended new result lands in the right slot.
                                                for key in ("hs", "err_u_l2", "err_p_l2", "err_u_h1", "err_p_h1",
                                                            "eval_times", "eval_iters", "eval_eps", "eval_residuals",
                                                            "eval_initial_residuals",
                                                            "overall_verification_success", "mms_plateau_success")
                                                    if length(rc[key]) >= idx
                                                        deleteat!(rc[key], idx)
                                                    end
                                                end
                                            end
                                        end
                                    end

                                    eps_pert_base = Float64(get(test_dict, "epsilon_pert", [0.1])[1])
                                    max_n_pert = Int(get(test_dict, "max_n_pert", 5))

                                    # [osgs-dispatch-fix 2026-05-26] The outer `config_dict` hardcodes
                                    # "method" => "ASGS" (see line ~370). That makes solve_system always
                                    # run the ASGS path regardless of the outer `method` loop variable —
                                    # so OSGS rows in earlier HDF5s are mislabelled ASGS solves.
                                    # Fix: rebuild a per-method PorousNSConfig with the right stabilization
                                    # method AND propagate the test JSON's OSGS-specific overrides
                                    # (osgs_iterations / osgs_tolerance / etc., which would otherwise be
                                    # silently inherited from base_config.json defaults).
                                    method_stab_dict = Dict{String,Any}(
                                        "method" => String(method),
                                        "osgs_iterations"          => get(stab_dict, "osgs_iterations", get(get(get(test_dict, "numerical_method", Dict()), "stabilization", Dict()), "osgs_iterations", 3)),
                                        "osgs_inner_newton_iters"  => get(stab_dict, "osgs_inner_newton_iters", get(get(get(test_dict, "numerical_method", Dict()), "stabilization", Dict()), "osgs_inner_newton_iters", 3)),
                                        "osgs_tolerance"           => get(stab_dict, "osgs_tolerance", get(get(get(test_dict, "numerical_method", Dict()), "stabilization", Dict()), "osgs_tolerance", 1e-5)),
                                        "osgs_projection_tolerance" => get(stab_dict, "osgs_projection_tolerance", get(get(get(test_dict, "numerical_method", Dict()), "stabilization", Dict()), "osgs_projection_tolerance", 1e-5)),
                                        "osgs_stopping_mode"       => get(stab_dict, "osgs_stopping_mode", get(get(get(test_dict, "numerical_method", Dict()), "stabilization", Dict()), "osgs_stopping_mode", "state_drift")),
                                        "osgs_state_drift_scale"   => get(stab_dict, "osgs_state_drift_scale", get(get(get(test_dict, "numerical_method", Dict()), "stabilization", Dict()), "osgs_state_drift_scale", "Linf")),
                                        "osgs_warmup_iterations"   => get(stab_dict, "osgs_warmup_iterations", get(get(get(test_dict, "numerical_method", Dict()), "stabilization", Dict()), "osgs_warmup_iterations", 2)),
                                        "osgs_warmup_tolerance"    => get(stab_dict, "osgs_warmup_tolerance", get(get(get(test_dict, "numerical_method", Dict()), "stabilization", Dict()), "osgs_warmup_tolerance", 1e-3)),
                                    )
                                    method_config_dict = deepcopy(config_dict)
                                    method_config_dict["numerical_method"]["stabilization"] = method_stab_dict
                                    method_config = PorousNSSolver.load_config_from_dict(method_config_dict)

                                    u_h_exact, p_h_exact = x0_exact
                                    u_ex_L2 =  sqrt(abs(sum(∫(u_h_exact ⋅ u_h_exact)dΩ)))

                                    setup = PorousNSSolver.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
                                    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)
                                    iter_solvers = PorousNSSolver.IterativeSolvers(solver_picard, solver_newton)
                                    mms_setup = MMSSetup(u_final, p_final, h_raw_func, u_ex_L2, norm_h, U_c, P_c, L_cell)
                                    pert_cfg = PerturbationConfig(eps_pert_base, max_n_pert)
                                    
                                    # ==============================================================================
                                    # NONLINEAR CONVERGENCE LOOP
                                    # Incrementally shrink initial numerical perturbation forcing iterative validation
                                    # ==============================================================================
                                    success, mms_plateau_success, successful_eps, final_x0, eval_time, iter_count_attempt, final_residual_attempt, initial_residual_attempt = execute_outer_homotopy_perturbation_loop!(
                                        setup, formulation, iter_solvers, method_config, method, dynamic_ftol,
                                        mms_setup, pert_cfg,
                                        mms_verification_enabled, mms_tau_err, mms_eps_u_l2, mms_eps_u_h1, mms_eps_p_l2,
                                        mms_max_extra_cycles, mms_require_consecutive_passes,
                                        mms_rate_check_factor,
                                        n, kv
                                    )

                                    # Fix 6: overall verification combines solver convergence and (when MMS
                                    # is enabled) the MMS plateau status. `nothing` plateau means MMS was
                                    # disabled, which counts as verified by default.
                                    overall_verification_success = success && (isnothing(mms_plateau_success) || mms_plateau_success)
                                    if success && mms_plateau_success === false
                                        println("  [⚠] Cell flagged: solver_success=true but mms_plateau_success=false (budget exhausted).")
                                    end

                                    if success && final_x0 !== nothing
                                        u_h, p_h = final_x0
                                        el2_u, el2_p, eh1_u, eh1_p = calculate_normalized_errors(u_h, p_h, u_final, p_final, U_c, P_c, L_cell, dΩ)
                                        
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
                                    push!(results_cache[k_id]["eval_initial_residuals"], initial_residual_attempt)
                                    push!(results_cache[k_id]["mms_plateau_success"], mms_plateau_success)
                                    push!(results_cache[k_id]["overall_verification_success"], overall_verification_success)
                                    
                                    println("  -> L2 u/p: ", round(el2_u, sigdigits=4), " / ", round(el2_p, sigdigits=4), " | H1 u/p: ", round(eh1_u, sigdigits=4), " / ", round(eh1_p, sigdigits=4))
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
                                        g["eval_initial_residuals"] = res["eval_initial_residuals"]
                                        # Per-mesh convergence status for the detection step (HDF5 can't store
                                        # Union{Bool,Nothing}, so encode: -1=nothing/MMS-disabled, 0=false, 1=true).
                                        g["overall_verification_success"] = Int8.(res["overall_verification_success"])
                                        g["mms_plateau_success"] = Int8[x === nothing ? Int8(-1) : (x ? Int8(1) : Int8(0)) for x in res["mms_plateau_success"]]

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
