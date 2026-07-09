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
using SHA                              # content-addressed HDF5 group keys (sha1)
using FileWatching.Pidfile: mkpidlock  # cross-process write lock for one shared results DB
# NOTE: SHA / FileWatching are stdlibs loaded via the default LOAD_PATH fallback, matching how
# this harness already uses Printf / Random / DelimitedFiles (none are listed in Project.toml).

# [harness-frame] Re/Da iteration-budget knobs (relocated out of production SolverConfig — audit §A.1/F1).
@isdefined(read_mms_dynamic_budget) || include(joinpath(@__DIR__, "..", "harness_dynamic_budget.jl"))

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
# (Re, Da, α_∞) cell, per the chosen encoding strategy. See theory/centered_encoding/centered_encoding.tex
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

# ==============================================================================
# Content-addressed keys + CLI cell selection (shared-DB / concurrent-launch support)
# ==============================================================================

# [content-address] Launch-independent signature string for one PHYSICS+DISCRETIZATION cell
# (Re, Da, α₀, k_v, k_p, element_type) — deliberately NOT the stabilization method, which is
# carried in the group-name suffix instead, so a cell's ASGS and OSGS groups share one tag.
# The `%.12e` float formatting is load-bearing: it canonicalises every representation of the same
# IEEE double (1e6, 1.0e6, 1000000.0) to one string, so two concurrent launches compute
# byte-identical signatures — and therefore identical content tags — for the same cell.
function _cell_signature(Re, Da, alpha_0, kv, kp, etype)
    return @sprintf("Re=%.12e|Da=%.12e|a0=%.12e|kv=%d|kp=%d|et=%s",
                    Float64(Re), Float64(Da), Float64(alpha_0), Int(kv), Int(kp), String(etype))
end

# Short, filesystem/HDF5-safe content tag. 12 hex chars of SHA1 — collision probability is
# negligible for the O(10²)-cell grids this harness sweeps.
_content_tag(sig::AbstractString) = bytes2hex(sha1(sig))[1:12]

# Canonical numeric string, reused for both the content tag and `--filter` matching, so that
# e.g. `--filter Re=1e6` matches a config value written as `1000000.0`.
_canon_num(x) = @sprintf("%.12e", Float64(x))

# [cli] Does one cell (etype,kv,kp,alpha_0,Da,Re,method) pass the `--filter` selection?
# AND across distinct keys, OR within a key's repeated values; an absent key is unconstrained.
function _cell_passes_filter(filt::Dict{Symbol,Vector{String}}, etype, kv, kp, alpha_0, Da, Re, method)
    num_ok(key, have) = !haskey(filt, key) || any(w -> _canon_num(parse(Float64, w)) == _canon_num(have), filt[key])
    str_ok(key, have) = !haskey(filt, key) || any(w -> w == String(have), filt[key])
    return num_ok(:Re, Re) && num_ok(:Da, Da) && num_ok(:alpha0, alpha_0) &&
           num_ok(:kv, kv) && num_ok(:kp, kp) &&
           str_ok(:etype, etype) && str_ok(:method, method)
end

# [cli] Parse `--filter key=val,key=val …` and `--shard k/N` from the flag args (the config-path
# positional has already been consumed). Fails loud on any malformed / unknown input — no silent
# defaults, per the repo's hard rules.
function _parse_cli_args(args)
    filt = Dict{Symbol,Vector{String}}()
    shard = nothing
    h5_name = nothing
    max_n = nothing
    valid_keys = (:Re, :Da, :alpha0, :kv, :kp, :etype, :method)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--filter"
            i += 1
            i <= length(args) || error("--filter requires an argument, e.g. --filter Re=1e6,etype=QUAD")
            for kvpair in split(args[i], ",")
                isempty(strip(kvpair)) && continue
                kvparts = split(kvpair, "="; limit=2)
                length(kvparts) == 2 || error("--filter entry \"$(kvpair)\" must be key=value.")
                key = Symbol(strip(kvparts[1]))
                key in valid_keys || error("Unknown --filter key \"$(strip(kvparts[1]))\". Valid: Re, Da, alpha0, kv, kp, etype, method.")
                push!(get!(filt, key, String[]), String(strip(kvparts[2])))
            end
        elseif a == "--shard"
            i += 1
            i <= length(args) || error("--shard requires k/N, e.g. --shard 2/4")
            m = match(r"^(\d+)/(\d+)$", strip(args[i]))
            m === nothing && error("--shard must be k/N (e.g. 2/4); got \"$(args[i])\".")
            k = parse(Int, m.captures[1]); Nsh = parse(Int, m.captures[2])
            (Nsh >= 1 && 1 <= k <= Nsh) || error("--shard k/N requires 1 ≤ k ≤ N; got $(k)/$(Nsh).")
            shard = (k, Nsh)
        elseif a == "--h5"
            i += 1
            i <= length(args) || error("--h5 requires a filename, e.g. --h5 quad_k2.h5")
            h5_name = String(strip(args[i]))
        elseif a == "--max-N"
            i += 1
            i <= length(args) || error("--max-N requires a positive integer, e.g. --max-N 40")
            mn = tryparse(Int, strip(args[i]))
            (mn !== nothing && mn > 0) || error("--max-N must be a positive integer; got \"$(args[i])\".")
            max_n = mn
        else
            error("Unrecognized argument \"$(a)\". Expected --filter, --shard, --h5, or --max-N.")
        end
        i += 1
    end
    return (filter=filt, shard=shard, h5=h5_name, max_n=max_n)
end

# Sets up the specific continuous VMS mathematical behavior for the Manufactured Solution
function build_mms_formulation(config, Da, Re, U_amp, L, alpha_infty)
    # [audit 2026-06] Projection policy is config-driven (mirrors src/run_simulation.jl:42-45): trim the
    # reaction from the orthogonal projection ONLY when experimental_reaction_mode=="standard" (constant
    # sigma); otherwise project the FULL residual. Lets the harness A/B trim vs full-residual OSGS — the
    # decisive test of whether (I-Pi)(sigma*u_h)=0 actually holds for the implemented V_free projection.
    proj = config.numerical_method.solver.experimental_reaction_mode == "standard" ?
           PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma() :
           PorousNSSolver.ProjectFullResidual()
    
    # Establish velocity regularization parameters strictly from configuration
    reg = PorousNSSolver.SmoothVelocityFloor(
        config.physical_properties.u_base_floor_ref,
        0.0,
        config.physical_properties.epsilon_floor,
        config.physical_properties.velocity_magnitude_derivative_floor
    )
    
    # Derive kinematic viscosity and exact constant sigma for the reaction operator 
    # to perfectly represent Darcy-scale bounds parameterized by Reynolds / Darcy inputs.
    nu_calculated = U_amp * L / Float64(Re)
    # [covariance 2026-06-02] physical_epsilon is DIMENSIONAL: in the continuity penalty `physical_epsilon·p`,
    # [physical_epsilon] = (U/L)/P_c, so it MUST scale with the (L,U) encoding — exactly like ν=U·L/Re and
    # σ=Da·α∞·ν/L². The config value is treated as the DIMENSIONLESS penalty ε̂; physical_epsilon is derived
    # per-cell so ε̂ = physical_epsilon·P_c·L/U is encoding-invariant. P_c per get_characteristic_scales:
    # P_c = (1+Re+Da)·U·ν/L. A FIXED physical_epsilon (the previous 1e-8) makes ε̂ encoding-dependent and breaks
    # scale-covariance (worst in the pressure / OSGS projection). See encoding_invariance_test.jl.
    P_c_cell = (1.0 + Float64(Re) + Float64(Da)) * U_amp * nu_calculated / L
    eps_calculated = config.physical_properties.physical_epsilon * (U_amp / L) / P_c_cell

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
    # [relative-residual gate] Iter-0 residual ‖R(x₀)‖ of the converged attempt,
    # populated by solve_system into `diagnostics_cache["initial_residual_norm"]`.
    # Provides the natural scale ‖f‖ for the gate `‖R_final‖/‖R_initial‖ < tol`
    # so Re=10⁶ cells stop being false-flagged on raw residual magnitude.
    initial_residual_attempt = NaN
    # [trajectory] per-attempt nonlinear trajectory (eps_pert → algorithm stages → per-iteration dots).
    attempt_traces = Any[]

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
        println("    [!] Delegating orchestration to the VMS solver (`solve_system`, src/solvers/solver_core.jl) (Mode: $method)")
        println("    ==================================================")
        flush(stdout)  # [observability] surface the current eps_pert attempt live (log is block-buffered)
        
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

        sys_success, sys_mms_plateau_success, sys_final_x0, sys_iter_count, sys_eval_time = PorousNSSolver.solve_system(
            setup, formulation, iter_solvers,
            config, x0;
            diagnostics_cache=local_diagnostics_cache, verifier=verifier
        )

        # [trajectory] capture this attempt's stage-by-stage path (built by the orchestrator into
        # diag_cache["trajectory"]) before the next attempt overwrites the local cache. For OSGS we
        # ALSO capture the Algorithm-C outer-loop sequence: per-outer-iteration drift diagnostics,
        # the base-convergence boundary (outer iters after it are MMS plateau-verification re-solves),
        # the per-plateau-cycle MMS relative-change ratios, and the MMS stop reason.
        push!(attempt_traces, (eps_pert = eps_p, success = sys_success,
                               stages = get(local_diagnostics_cache, "trajectory", Any[]),
                               osgs_outer = get(local_diagnostics_cache, "outer_osgs_diagnostics", Any[]),
                               mms_relchange = get(local_diagnostics_cache, "mms_relative_change_history", Any[]),
                               base_conv_outer_iter = get(local_diagnostics_cache, "base_convergence_outer_iter", -1),
                               mms_stop_reason = get(local_diagnostics_cache, "mms_stop_reason", "")))

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
            println("\n      [❌] Outer loop execution completely stalled structurally above convergence tolerance (`$(dynamic_ftol)`) or system fully diverged.")
            final_residual_attempt = get(local_diagnostics_cache, "final_residual_norm", NaN)
            initial_residual_attempt = get(local_diagnostics_cache, "initial_residual_norm", NaN)
        end
    end

    if !success
         println("    [WARNING] Completely failed to find root basin. Returning NaN.")
    end

    return success, mms_plateau_success, successful_eps, final_x0, eval_time, iter_count_attempt, final_residual_attempt, initial_residual_attempt, attempt_traces
end

function run_mms(config_file="test_config.json"; cli_filter=Dict{Symbol,Vector{String}}(), cli_shard=nothing,
                 cli_h5=nothing, cli_max_n=nothing)
    config_path = joinpath(@__DIR__, "data", config_file)
    test_dict = JSON3.read(read(config_path, String), Dict{String, Any})
    budget = read_mms_dynamic_budget(test_dict)   # [harness-frame] Re/Da iteration-budget knobs (audit §A.1/F1)

    # [trajectory diagnostic] When true, attach the scale-free convergence probe to the inner solvers so
    # the per-iteration ε_M (momentum) and ε_C (mass) normalized norms are recorded in each trace's
    # iteration history (for the trajectory plots). OFF by default — the probe re-assembles the force
    # envelope every inner iteration, too costly for a full production sweep; enable it only on the small
    # cell selections you actually want to plot. It is a pure read-only observer (does not change the solve).
    trace_conv_norms = Bool(get(test_dict, "trace_convergence_norms", false))

    # [concurrency] Our cross-process write safety is the mkpidlock sidecar (single-writer critical
    # sections, below). Disable libhdf5's OWN file lock so it neither rejects concurrent opens of
    # the shared DB nor deadlocks against our flock. Mirrors analyze_results.py:robust_open_h5.
    ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

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
    # theory/centered_encoding/centered_encoding.tex Section 6 for the formulas. Default `"minmax"` minimises
    # max(|log U|, |log ν|, |log σ|), keeping the dimensional scale spread tame at extreme
    # (Re, Da). The legacy "centered" encoding's U_amp blows up to ~1e9 at Re=1e6, which
    # starves the nonlinear solves and produced the OSGS C20/C24 residual stall. The other
    # strategies ("centered" / "balanced" / "unit") remain available for back-comparison.
    encoding_strategy = String(get(test_dict, "encoding_strategy", "minmax"))

    # [stall guard / P1] No-progress bail for the nonlinear kernel (Algorithm A): abort an attempt once the
    # best residual fails to improve by `solver_stall_min_rel_improvement` over `solver_stall_window`
    # iterations, so a doomed eps_pert (e.g. a high-Re 1.0 guess that grinds the full Newton budget with
    # merit stuck) falls back to a milder perturbation fast instead of exhausting max_iters.
    # The test-JSON keys are a harness convenience. Their default (0 / 0.0 = DISABLED) is exactly the
    # canonical `SolverConfig.newton_stall_window` / `newton_stall_min_rel_improvement` value shipped in
    # config/base_config.json — kept as an explicit literal here because base_config.json is not directly
    # loadable into a full PorousNSConfig (it intentionally omits physical_epsilon; see docs/known_issues.md). The
    # PRODUCTION path sources these from SolverConfig directly through the shared builder (P6); this is the
    # one harness-side place that mirrors that default.
    solver_stall_window = Int(get(test_dict, "solver_stall_window", 0))
    solver_stall_min_rel_improvement = Float64(get(test_dict, "solver_stall_min_rel_improvement", 0.0))

    # [output] ParaView/VTK field export is ON by default so every cell can be inspected visually.
    # Per-cell outputs are organized per (kv, etype) next to the convergence plots:
    # results/k<kv>/<etype>/{vtk,traces}/ (and results/debug_results/... for ad-hoc/debug runs whose
    # `h5_filename` lives under debug_results/). A full sweep can write many GBs of .vtu; set
    # "write_vtk": false in the config to skip VTK for large sweeps.
    write_vtk = Bool(get(test_dict, "write_vtk", true))

    results_dir = joinpath(@__DIR__, "results")
    mkpath(results_dir)
    # [layout 2026-06-27] Results live in per-(kv,etype) folders as a generically-named DB:
    #   results/k<kv>/<etype>/results.h5   (debug runs: results/debug_results/k<kv>/<etype>/results.h5)
    # co-located with that cell's convergence PNGs, traces/, and vtk/. The DB no longer carries a
    # study-specific filename; instead the FULL config JSON(s) that produced the data are embedded INSIDE
    # the DB (group "configs/<config-file>"), and every result group points to its config via the
    # `config_file` attribute. Usually one config per DB, but a merged DB (e.g. a surgical re-run of a few
    # cells under a second config) can hold several. `h5_filename` is retained ONLY to flag a debug run
    # (its "debug_results/" prefix routes the whole tree under debug_results/); the DB name is always
    # `results.h5`. The per-(kv,etype) `h5_path`/`h5_lock_path` are (re)computed at the top of the
    # etype/kv/kp loop below via `_results_h5_path`/`_ensure_h5!`.
    h5_filename = get(test_dict, "h5_filename", "convergence_data.h5")
    if cli_h5 !== nothing
        h5_filename = cli_h5   # CLI override: route this run under debug_results/ (per-study isolation)
    end
    _debug_run = startswith(h5_filename, "debug_results/")
    # Per-cell output base for the whole tree (DB + VTK + traces): keep debug-run artifacts out of results/.
    _cell_out_base = _debug_run ? joinpath(results_dir, "debug_results") : results_dir
    # Run identity stamped into each trace so plot_trajectory.py groups a run's plots under plots/<run>/.
    # Keyed on the CONFIG basename (the DB name is the generic "results"), e.g. "phase1_quad_k1".
    run_name = splitext(basename(config_file))[1]
    # Raw config text + basename, embedded into each per-(kv,etype) DB this run writes (provenance).
    config_basename = basename(config_file)
    config_raw = read(config_path, String)

    erase_past = get(test_dict, "erase_past_results", false)

    # [concurrency] "erase then run concurrently" is contradictory: one shard would truncate the
    # file another shard is filling. Refuse it loudly.
    if erase_past && cli_shard !== nothing
        error("erase_past_results=true is incompatible with --shard (a shard would wipe another shard's results). Set erase_past_results=false for sharded runs.")
    end

    # DB path + first-touch initializer. The DB is created/truncated and the config JSON embedded EXACTLY
    # ONCE per distinct path per run (tracked by `_initialized_h5`). `erase_past` ⇒ "w" (fresh); otherwise
    # "cw" (create-if-needed, keep existing for resume/merge — the config is added/refreshed).
    #   • OFFICIAL run (default) → the clean per-(kv,etype) generic DB: results/k<kv>/<etype>/results.h5.
    #   • DEBUG/STUDY run (h5_filename under "debug_results/") → keep the EXPLICIT single-DB name for
    #     per-study isolation (e.g. A/B harness tests writing _pp_off.h5 vs _pp_on.h5 then reading them
    #     back); these go under results/debug_results/ and are not the canonical results.
    _results_h5_path(kvv, ets) = _debug_run ?
        joinpath(results_dir, h5_filename) :
        joinpath(_cell_out_base, "k$(kvv)", String(ets), "results.h5")
    _initialized_h5 = Set{String}()
    function _ensure_h5!(p)
        (p in _initialized_h5) && return
        push!(_initialized_h5, p)
        mkpath(dirname(p))
        mkpidlock(p * ".lock"; wait=true) do
            h5open(p, erase_past ? "w" : "cw") do io
                cfgs = haskey(io, "configs") ? io["configs"] : create_group(io, "configs")
                haskey(cfgs, config_basename) && delete_object(cfgs, config_basename)
                cfgs[config_basename] = config_raw   # full config JSON embedded for provenance
            end
        end
    end

    conv_parts = mesh_dict["convergence_partitions"]
    if cli_max_n !== nothing
        conv_parts = filter(n -> n <= cli_max_n, conv_parts)   # CLI override: cap the N-ladder for quick gates
        isempty(conv_parts) && error("--max-N $(cli_max_n) excludes every mesh in convergence_partitions=$(mesh_dict["convergence_partitions"]).")
    end

    # [cell-select] Materialise the full Cartesian cell list ONCE, in the SAME nesting order the
    # solve loops below visit cells, then apply (a) the config's equal_order_only + skip_cells
    # filters, (b) the CLI --filter, (c) deterministic --shard partitioning. This single chokepoint
    # is what lets a user run any sub-combination of factors into one shared DB, and keeps shards
    # disjoint. IMPORTANT: keep this enumeration order in lockstep with the loops below — a reorder
    # there must be mirrored here, or two shards would disagree on the partition.
    all_cells = Tuple[]
    for etype in etype_list, kv in kv_list, kp in kp_list,
        alpha_0 in alpha_list, Da in Da_list, Re in Re_list, method in methods
        (equal_order_only && kv != kp) && continue
        ((Float64(Re), Float64(Da), Float64(alpha_0)) in skip_cells_set) && continue
        push!(all_cells, (etype, kv, kp, alpha_0, Da, Re, method))
    end

    # [display-index] Deterministic, globally-unique config index per physics cell, from the cell's
    # position in the FULL grid — identical across all shards, so config_<idx> never repeats or drifts
    # between concurrent launches (replaces the old per-process counter). Method is excluded so a
    # cell's ASGS and OSGS groups share one index.
    global_cell_index = Dict{Tuple,Int}()
    let next_idx = 0
        for (etype, kv, kp, alpha_0, Da, Re, method) in all_cells
            sb = (Float64(Re), Float64(Da), Float64(alpha_0), Int(kv), Int(kp), String(etype))
            if !haskey(global_cell_index, sb)
                next_idx += 1
                global_cell_index[sb] = next_idx
            end
        end
    end

    selected = filter(c -> _cell_passes_filter(cli_filter, c...), all_cells)
    if cli_shard !== nothing
        kshard, nshard = cli_shard
        selected = selected[kshard:nshard:end]   # round-robin: disjoint, exhaustive, order-stable
    end
    isempty(selected) && error("No cells selected after --filter/--shard — check the filter keys/values against the config grid.")
    selected_set = Set(selected)
    total_runs = length(selected)
    println("[cell-select] $(length(selected)) of $(length(all_cells)) cells selected" *
            (cli_shard === nothing ? "" : " (shard $(cli_shard[1])/$(cli_shard[2]))") *
            (isempty(cli_filter) ? "" : " (filter $(cli_filter))") * ".")

    # [clean-slate] `erase_past_results=true` already truncates the H5 DB ("w" mode above). Extend it to a
    # TRUE clean slate by also clearing the regenerated per-(kv,etype) artifacts — traces/, plots/, vtk/, and
    # the convergence_*.png — for exactly the (kv,etype) combos this run will write. Without this, trajectory
    # plots / convergence PNGs / traces from a PREVIOUS run linger and get mixed with the new ones (the
    # provenance trap). Scoped to the run's own combos (never other studies); incompatible with --shard
    # (erase_past already errors out for shards above). Best-effort — a clean failure must not abort the run.
    if erase_past
        cleared = Set{Tuple{Int,String}}()
        for s in selected
            et = String(s[1]); kvc = Int(s[2])
            (kvc, et) in cleared && continue
            push!(cleared, (kvc, et))
            cell_dir = joinpath(_cell_out_base, "k$(kvc)", et)
            isdir(cell_dir) || continue
            try
                for sub in ("traces", "plots", "vtk")
                    rm(joinpath(cell_dir, sub); force=true, recursive=true)
                end
                for f in readdir(cell_dir; join=true)
                    endswith(f, ".png") && rm(f; force=true)
                end
            catch e
                @warn "erase_past_results artifact clean failed (non-fatal)" cell_dir exception=e
            end
        end
        println("[erase_past_results] Clean slate: cleared traces/plots/vtk + convergence PNGs for $(length(cleared)) (kv,etype) combo(s).")
    end
    run_idx = 0
    
    # Pre-allocate cache for collecting metrics over partitions dynamically
    results_cache = Dict()
    
    for etype in etype_list
        for kv in kv_list
            for kp in kp_list
                if equal_order_only && kv != kp
                    continue
                end

                # [layout] This (kv,etype) writes to its own results/k<kv>/<etype>/results.h5; create +
                # embed the config on first touch. The resume-preload and per-cell write below use these.
                h5_path = _results_h5_path(kv, etype)
                h5_lock_path = h5_path * ".lock"
                _ensure_h5!(h5_path)

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
                    mkpidlock(h5_lock_path; wait=true) do
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
                    end  # mkpidlock (resume-preload read)
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

                                # [cell-select] Skip the entire per-cell setup if no stabilization
                                # method for this physics cell survives the --filter/--shard selection.
                                if !any(m -> (etype, kv, kp, alpha_0, Da, Re, m) in selected_set, methods)
                                    continue
                                end

                                # [encoding] Pick the per-cell characteristic length and velocity scale
                                # from the configured strategy. Default "centered" reproduces the
                                # pre-2026-06 behaviour (L=1, U=Re/√(α_∞·Da)) bit-identically.
                                # See theory/centered_encoding/centered_encoding.tex Section 6 for the four formulas.
                                (L_cell, U_amp) = compute_L_and_U(encoding_strategy, Float64(Re), Float64(Da), alpha_infty)
                                println("  [cell setup] Re=$(Re), Da=$(Da), α=$(alpha_0), strategy=$encoding_strategy → L=$(L_cell), U_amp=$(U_amp)")

                                # Build per-cell config. The bounding_box and r_1, r_2 carried here are
                                # the L=1 BASELINE values from JSON; _build_local_mesh and
                                # build_porosity_field apply the L_cell scaling internally.
                                config_dict = Dict(
                                    "physical_properties" => Dict("nu" => 1.0, "physical_epsilon" => 1e-8, "reaction_model" => "Constant_Sigma", "sigma_constant" => 1.0),
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
                                            "method" => "ASGS"
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
                                mms = PorousNSSolver.PaperMMS(form, U_amp, alpha_field; L=L_cell, alpha_infty=alpha_infty)
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

                                # [scale-free gate] The per-field RE-ANCHORED relative-ftol targets were removed:
                                # convergence is now decided by the scale-free ε_M/ε_C criterion injected by
                                # solve_system. `dynamic_ftol` (the absolute mesh-scaled O(h^{kv+1}) threshold) is
                                # passed as the scalar ftol below — it sets the f_norm trace denominator and the
                                # conv_probe===nothing fallback floor, but no longer re-anchors the gate per segment.

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
                                # docs/mms/convergence-2d.md for the Re=1e6, Da=1, α=0.05 corner that
                                # motivates this. The schema default leaves
                                # well-resolved (low-Re) regimes bit-identical.
                                local_newton_it = solver_newton_it
                                if Re >= budget.newton_re_threshold
                                    local_newton_it = max(local_newton_it, budget.newton_re_iterations)
                                end

                                # [honest-exit] gate: a noise-floor stop counts as converged only if
                                # ‖R‖∞ ≤ k_nf · effective_ftol. Read from config (base default 1e30 = disabled;
                                # the Phase-1 sweep configs set 10.0 to reject high-Re fold stalls).
                                k_nf = config.numerical_method.solver.noise_floor_success_max_ftol_multiple
                                # [P0 shared builder] Harness builds the rich (picard, newton) pair: one shared
                                # machine-precision floor for both ftols, dynamic noise floor, per-field relative
                                # tolerances, the honest-exit gate, dynamic-widened iteration budgets, and the stall
                                # guard. All solver scalars (max_inc, xtol, linesearch params, armijo_c1, div_fac)
                                # are read from `config.numerical_method.solver` inside the builder — identical to the
                                # locals used here. The builder dispatches mode=:picard / :newton internally.
                                _iter_solvers = PorousNSSolver.build_iter_solvers(
                                    config.numerical_method.solver, LUSolver();
                                    newton_max_iters = local_newton_it,
                                    picard_max_iters = local_picard_it,
                                    newton_ftol = dynamic_ftol,
                                    picard_ftol = dynamic_ftol,
                                    stagnation_noise_floor = dynamic_noise_floor,
                                    noise_floor_success_max_ftol_multiple = k_nf,
                                    stall_window = solver_stall_window,
                                    stall_min_rel_improvement = solver_stall_min_rel_improvement)

                                solver_picard = _iter_solvers.picard
                                solver_newton = _iter_solvers.newton
                                
                                # Interpolate initial mathematical guess state perfectly atop exact solution roots
                                x0_exact = interpolate_everywhere([u_final, p_final], X)
                                
                                for method in methods
                                    # [cell-select] Honour --filter/--shard at the single per-method chokepoint.
                                    if !((etype, kv, kp, alpha_0, Da, Re, method) in selected_set)
                                        continue
                                    end
                                    run_idx += 1
                                    println("\n--- Progress $(run_idx) / $(total_runs * length(conv_parts)) | Re=$(Re), Da=$(Da), α=$(alpha_0), method=$(method), N=$(n) ---")
                                    flush(stdout)  # [observability] stdout is block-buffered to the log file; flush per cell so the log tracks real progress (not hours behind)

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
                                    # "method" => "ASGS", so solve_system would always run the ASGS path
                                    # regardless of the outer `method` loop variable. Rebuild a per-method
                                    # stabilization dict with the correct method before loading.
                                    method_stab_dict = Dict{String,Any}(
                                        "method" => String(method),
                                    )
                                    method_config_dict = deepcopy(config_dict)
                                    method_config_dict["numerical_method"]["stabilization"] = method_stab_dict
                                    method_config = PorousNSSolver.load_config_from_dict(method_config_dict)

                                    u_h_exact, p_h_exact = x0_exact
                                    u_ex_L2 =  sqrt(abs(sum(∫(u_h_exact ⋅ u_h_exact)dΩ)))

                                    setup = PorousNSSolver.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
                                    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)
                                    iter_solvers = PorousNSSolver.StageSolvers(solver_picard, solver_newton)
                                    # [convergence gate] The scale-free convergence evaluator (ε_M, ε_C) is now the
                                    # AUTHORITATIVE success gate and is injected by `solve_system` itself (built from
                                    # setup/formulation + the config tolerances), so it is always active and its ε_M/ε_C
                                    # are always recorded in the trace history — no per-harness attach is needed here.
                                    mms_setup = MMSSetup(u_final, p_final, h_raw_func, u_ex_L2, norm_h, U_c, P_c, L_cell)
                                    pert_cfg = PerturbationConfig(eps_pert_base, max_n_pert)
                                    
                                    # ==============================================================================
                                    # NONLINEAR CONVERGENCE LOOP
                                    # Incrementally shrink initial numerical perturbation forcing iterative validation
                                    # ==============================================================================
                                    success, mms_plateau_success, successful_eps, final_x0, eval_time, iter_count_attempt, final_residual_attempt, initial_residual_attempt, cell_attempt_traces = execute_outer_homotopy_perturbation_loop!(
                                        setup, formulation, iter_solvers, method_config, method, dynamic_ftol,
                                        mms_setup, pert_cfg,
                                        mms_verification_enabled, mms_tau_err, mms_eps_u_l2, mms_eps_u_h1, mms_eps_p_l2,
                                        mms_max_extra_cycles, mms_require_consecutive_passes,
                                        mms_rate_check_factor,
                                        n, kv
                                    )

                                    # [trajectory] Persist this run's full nonlinear trajectory (all eps_pert
                                    # attempts → algorithm stages → per-iteration residual dots) as a JSON sidecar,
                                    # one per (cell, method, N), for plot_trajectory.py / post-hoc analysis. Best
                                    # effort — a trace-write failure must never abort the sweep.
                                    try
                                        traces_dir = joinpath(_cell_out_base, "k$(Int(kv))", String(etype), "traces")
                                        isdir(traces_dir) || mkpath(traces_dir)
                                        trace_name = @sprintf("traj_Re%.0e_Da%.0e_a%.2f_kv%d_kp%d_%s_%s_N%d.json",
                                            Float64(Re), Float64(Da), Float64(alpha_0), Int(kv), Int(kp),
                                            String(etype), String(method), Int(n))
                                        trace_obj = (
                                            run = run_name,
                                            cell = (Re=Float64(Re), Da=Float64(Da), alpha_0=Float64(alpha_0),
                                                    kv=Int(kv), kp=Int(kp), etype=String(etype),
                                                    method=String(method), encoding=encoding_strategy),
                                            N = Int(n), success = success, successful_eps = successful_eps,
                                            final_residual = final_residual_attempt, initial_residual = initial_residual_attempt,
                                            # Scale-free convergence thresholds the gate actually compares ε_M/ε_C
                                            # against (eps_tol_momentum/eps_tol_mass). Emitted so the trajectory
                                            # plotter can draw the true tol_M/tol_C lines instead of a generic ε=1.
                                            tol_M = config.numerical_method.solver.eps_tol_momentum,
                                            tol_C = config.numerical_method.solver.eps_tol_mass,
                                            attempts = cell_attempt_traces,
                                        )
                                        open(joinpath(traces_dir, trace_name), "w") do io
                                            # allow_inf: stage records default missing residual fields to NaN
                                            # (e.g. the freeze_after_k path lacks normalized residuals). Without
                                            # this, JSON3 throws "NaN not allowed", the best-effort catch swallows
                                            # it, and the trace lands EMPTY (0 bytes) → the plotter skips it. NaN/Inf
                                            # tokens are read fine by Python's json.load; the plotter skips NaN points.
                                            JSON3.write(io, trace_obj; allow_inf=true)
                                        end
                                    catch e
                                        @warn "Trajectory trace write failed (non-fatal)" exception=e
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
                                        el2_u, el2_p, eh1_u, eh1_p = calculate_normalized_errors(u_h, p_h, u_final, p_final, U_c, P_c, L_cell, dΩ)

                                        if write_vtk
                                            vtk_dir = joinpath(_cell_out_base, "k$(Int(kv))", String(etype), "vtk")
                                            isdir(vtk_dir) || mkpath(vtk_dir)
                                            u_ex_cf = interpolate_everywhere(u_final, U)
                                            p_ex_cf = interpolate_everywhere(p_final, P)
                                            filename = joinpath(vtk_dir, "mms_$(method)_Re$(Float64(Re))_Da$(Float64(Da))_a$(Float64(alpha_0))_N$(n).vtu")
                                            writevtk(Ω, filename, cellfields=["uh"=>u_h, "ph"=>p_h, "uex"=>u_ex_cf, "pex"=>p_ex_cf, "alpha"=>alpha_cf, "err_u"=>u_h - u_ex_cf, "err_p"=>p_h - p_ex_cf])
                                        end
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
                                    c_idx = global_cell_index[sig_base]   # deterministic, shard-independent
                                    
                                    cell_sig = _cell_signature(Re, Da, alpha_0, kv, kp, etype)
                                    cell_tag = _content_tag(cell_sig)

                                    mkpidlock(h5_lock_path; wait=true) do
                                    h5open(h5_path, "r+") do h5f_out
                                        res = results_cache[k_id]
                                        # [content-address] config_<idx>_<tag>_<method>: <tag> hashes the physics
                                        # cell (Re,Da,α,kv,kp,etype) identically across concurrent launches, so two
                                        # processes never clobber distinct cells, and a cell's ASGS/OSGS groups share
                                        # <tag>. <idx> stays human-readable but is no longer the uniqueness guarantee.
                                        grp_name = "config_$(c_idx)_$(cell_tag)_$(method)"

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
                                        attributes(g)["cell_signature"] = String(cell_sig)
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
                                    end  # mkpidlock (per-cell write critical section)

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
    # ARGS[1] is the config filename (optional; defaults to test_config.json). Anything starting
    # with -- is a flag parsed by _parse_cli_args. Examples:
    #   julia run_test.jl phase1_quad_k1.json
    #   julia run_test.jl phase1_quad_k1.json --filter Re=1e6,etype=QUAD,kv=1
    #   julia run_test.jl phase1_quad_k1.json --shard 2/4         # one shard of a concurrent sweep
    flag_args = ARGS
    config_file = "test_config.json"
    if length(ARGS) > 0 && !startswith(ARGS[1], "--")
        config_file = ARGS[1]
        flag_args = ARGS[2:end]
    end
    cli = _parse_cli_args(flag_args)
    run_mms(config_file; cli_filter=cli.filter, cli_shard=cli.shard, cli_h5=cli.h5, cli_max_n=cli.max_n)
end
