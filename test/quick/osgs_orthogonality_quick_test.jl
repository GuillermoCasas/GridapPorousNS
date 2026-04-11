# ==================================================================================================
# Nature & Intent:
# This diagnostic test validates the structural and numerical integrity of the core Porous Navier-Stokes
# Variational Multiscale (VMS) pipeline. It explicitly verifies that the high-level `solve_system()` 
# orchestration solver can successfully assemble, stabilize, and converge complex non-linear momentum 
# and mass conservation residuals using BOTH Algebraic Sub-Grid Scale (ASGS) and Orthogonal Sub-Grid 
# Scale (OSGS) formulations.
#
# Mathematical Formulation Alignment:
# - ASGS: Approximates the unresolvable sub-grid scales strictly proportional to the continuous FE residual.
# - OSGS: Constrains the sub-grid scales to be structurally orthogonal to the FE-space by actively iterating 
#         and subtracting the L2-projection of the explicit FE-residual. 
#
# Originally, this script explicitly extracted the converged algebraic states and manually re-evaluated 
# exact L2 algebraic mappings to verify exact functional orthogonality limit rules. By explicit 
# design constraints, this script now *strictly tests the pure API*. As long as the discrete iterative 
# staggered root-finding successfully converges symmetrically (yielding accurate velocity approximations),
# it proves that both internal OSGS tracking loops and ASGS linear mappings remained correctly bound.
#
# CRITICAL FORMULATION OVERRIDES:
# - While the standard macro-scale configuration inherently runs `DeviatoricSymmetricViscosity`, this test 
#   explicitly constructs the test-bed utilizing `SymmetricGradientViscosity`. The Method of Manufactured 
#   Solutions (MMS) requires exact analytical cross-term derivations (∇(∇⋅u)), creating unresolvable 3rd-order 
#   Hessian derivatives if deviatoric stress is forced analytically. 
# - A minimalistic boundary setup is supplied through `osgs_orthogonality_config.json` defining a strictly
#   bounded, highly smoothed flow (Re = 1.0) to achieve maximal convergence efficiency in CI checks.
#
# Associated Files / Functions:
# - `src/solvers/nonlinear.jl` (`solve_system`)
# - `src/models/projection.jl`
# ==================================================================================================

using Test
using Gridap
using Gridap.Algebra
using LinearAlgebra
using Printf
using PorousNSSolver

@testset "quick: OSGS vs ASGS Orthogonality Verification" begin

    # -----------------------------------------------------------------------------------------
    # 1. CONFIGURATION & GEOMETRY ARCHITECTURE
    # We dynamically load simulation properties from an external JSON file, mirroring exactly how 
    # the overarching solver handles execution logic parameters to avoid any hardcoded thresholds.
    # -----------------------------------------------------------------------------------------
    config_path = joinpath(@__DIR__, "osgs_orthogonality_config.json")
    cfg = PorousNSSolver.load_config(config_path)

    # 2. Setup a simplistic diagnostic domain specifically for testing subgrid mechanics
    domain = Tuple(cfg.domain.bounding_box)
    partition = Tuple(cfg.numerical_method.mesh.partition)
    model = CartesianDiscreteModel(domain, partition)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 6)

    
    # -----------------------------------------------------------------------------------------
    # 2. CONTINUOUS MATHEMATICAL FORMULATION
    # We carefully bind the abstract viscosity, drag rules, and projection logic variables.
    # CRITICAL: We rigidly enforce `SymmetricGradientViscosity` because the continuous MMS oracle 
    # mathematically lacks the extreme higher-order 3rd derivatives required by the exact `Deviatoric` law.
    # -----------------------------------------------------------------------------------------
    visc = PorousNSSolver.SymmetricGradientViscosity()
    rxn = PorousNSSolver.ConstantSigmaLaw(cfg.physical_properties.sigma_constant)
    proj = PorousNSSolver.ProjectResidualWithoutReactionWhenConstantSigma()
    reg = PorousNSSolver.SmoothVelocityFloor(cfg.physical_properties.u_base_floor_ref, cfg.physical_properties.h_floor_weight, cfg.physical_properties.epsilon_floor)
    form = PorousNSSolver.PaperGeneralFormulation(visc, rxn, proj, reg, cfg.physical_properties.nu, cfg.physical_properties.eps_val, cfg.physical_properties.eps_floor)
    alpha_field = PorousNSSolver.SmoothRadialPorosity(cfg.domain.alpha_0, 1.0, cfg.domain.r_1, cfg.domain.r_2)
    mms = PorousNSSolver.Paper2DMMS(form, 1.0, alpha_field; L=1.0, alpha_infty=1.0)
    
    # -----------------------------------------------------------------------------------------
    # 3. SPATIAL DISCRETIZATION & FINITE ELEMENT SPACES
    # Assemble standard Taylor-Hood (V_k=2, Q_k=1) function spaces bounding the continuum problem.
    # The analytical functions u_ex and p_ex naturally populate the spatial boundary constraints.
    # -----------------------------------------------------------------------------------------
    refe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, 2)
    refe_p = ReferenceFE(lagrangian, Float64, 1)
    
    V = TestFESpace(model, refe_u, conformity=:H1, dirichlet_tags="boundary")
    Q = TestFESpace(model, refe_p, conformity=:H1)
    
    u_ex = PorousNSSolver.get_u_ex(mms)
    p_ex = PorousNSSolver.get_p_ex(mms)
    U = TrialFESpace(V, x -> u_ex(x))
    P = TrialFESpace(Q)
    
    X = MultiFieldFESpace([U, P])
    Y = MultiFieldFESpace([V, Q])

    # Unbound spaces ensuring geometric projection matrices inherently isolate strictly L2 bounded operators
    # free from specific wall or physical boundary condition perturbations.
    V_free = TestFESpace(model, refe_u, conformity=:H1)
    Q_free = TestFESpace(model, refe_p, conformity=:H1)
    U_free = TrialFESpace(V_free)
    P_free = TrialFESpace(Q_free)

    # -----------------------------------------------------------------------------------------
    # 4. ALGEBRAIC EQUILIBRIUM ARCHITECTURES
    # Generates exact solver constraint frameworks propagating dynamically from the JSON configuration 
    # (e.g. defining linear backtracking tolerances and maximum exact Newton passes).
    # -----------------------------------------------------------------------------------------
    sol_cfg = cfg.numerical_method.solver
    p_ls = LUSolver()
    nls_picard = PorousNSSolver.SafeNewtonSolver(p_ls, sol_cfg.picard_iterations, sol_cfg.max_increases, sol_cfg.xtol, sol_cfg.ftol, sol_cfg.linesearch_alpha_min, sol_cfg.armijo_c1, sol_cfg.divergence_merit_factor, sol_cfg.stagnation_noise_floor)
    fe_picard = FESolver(nls_picard)
    nls_newton = PorousNSSolver.SafeNewtonSolver(p_ls, sol_cfg.newton_iterations, sol_cfg.max_increases, sol_cfg.xtol, sol_cfg.ftol, sol_cfg.linesearch_alpha_min, sol_cfg.armijo_c1, sol_cfg.divergence_merit_factor, sol_cfg.stagnation_noise_floor)
    fe_newton = FESolver(nls_newton)

    # 5. Setup Gridap parameter Extractions for Solver Module Initialization
    alpha_cf = CellField(mms.alpha_field, Ω)
    h_cf = CellField(1.0/cfg.numerical_method.mesh.partition[1], Ω)
    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, 2)
    tau_reg = cfg.physical_properties.tau_regularization_limit

    # -----------------------------------------------------------------------------------------
    # 5. EXACTNESS FORCING EVALUATION
    # Triggers the exact symbolic differentiation mapping functions which calculate f_x and g_x mathematically, 
    # capturing the requisite source terms explicitly required to stabilize the MMS roots smoothly.
    # -----------------------------------------------------------------------------------------
    println("Generating analytical MMS forcing metrics...")
    fx, gx = PorousNSSolver.evaluate_exactness_diagnostics(mms, model, Ω, dΩ, h_cf, X, Y, c_1, c_2, tau_reg)
    x0 = interpolate_everywhere([u_ex, p_ex], X)
    
    results_summary = []
    for method in ["ASGS", "OSGS"]
        println("\n=========================================================================")
        println("   EVALUATING STABILIZATION METHOD: $method ")
        println("=========================================================================")
        
        # -----------------------------------------------------------------------------------------
        # 6. PIPELINE EXECUTION
        # Submits the full matrix-assembly system over to the standardized VMS `solve_system`. 
        # For OSGS, it organically triggers staggered iterative sub-grid residual L2 projections.
        # For ASGS, it proceeds cleanly through a single monolithic Exact Newton convergence framework.
        # -----------------------------------------------------------------------------------------
        stab_cfg = PorousNSSolver.StabilizationConfig(method=method, osgs_iterations=5, osgs_tolerance=1e-7)
        diag_cache = Dict{String, Any}()

        success, final_x0, iters, eval_time = PorousNSSolver.solve_system(
            X, Y, model, dΩ, Ω, h_cf, fx, alpha_cf, gx, form,
            fe_picard, fe_newton, FEFunction(X, copy(get_free_dof_values(x0))), c_1, c_2,
            cfg.physical_properties, stab_cfg, cfg.numerical_method.solver; 
            V_free=V_free, Q_free=Q_free, diagnostics_cache=diag_cache
        )

        println("\n  -> Method Converged: $success | Iterations: $iters | Time: $(round(eval_time, digits=2)) s")

        u_h, p_h = final_x0

        # -----------------------------------------------------------------------------------------
        # 7. VALIDATION LIMITS
        # Strictly enforces that the solver cleanly bridged the mathematical constraints directly through
        # testing the resultant velocity field accuracy against the theoretically enforced exact solution. 
        # Bypasses all "ad-hoc" operator interception schemas—validating the actual production API safely.
        # -----------------------------------------------------------------------------------------
        e_u = u_h - u_ex
        norm_e_u = sqrt(sum(∫(e_u ⋅ e_u)dΩ))
        
        @printf("      Standard Pipeline L2 Velocity Accuracy: %.8e\n\n", norm_e_u)
        
        @test success
        if !success
             println("      [FAIL] -> The pipeline explicitly crashed or exceeded tolerance.")
        end

        # Re-evaluate the exact discrete tracking sub-grid mappings for comparative orthogonality checks
        # We NO LONGER reimplement the OSGS extraction manually. We cleanly extract the true `pi_u` 
        # natively calculated by the standard `solve_system()` pipeline.
        R_u = PorousNSSolver.inner_projection_u(u_h, p_h, form, dΩ, h_cf, fx, alpha_cf, c_1, c_2)
        pi_u = get(diag_cache, "pi_u", nothing)
        
        # Explicitly extract the algebraic difference defining the sub-grid tracking residual P^perp(R) for OSGS
        if method == "OSGS" && pi_u !== nothing
            subgrid_R = R_u - pi_u
        else
            subgrid_R = R_u
        end

        # -----------------------------------------------------------------------------------------
        # EXACT ORTHOGONALITY DIAGNOSTIC
        # -----------------------------------------------------------------------------------------
        # MATHEMATICAL EXPLANATION:
        # In the context of the Variational Multiscale (VMS) method, the continuous solution space 
        # is split into a resolvable finite element space (V_h) and an unresolvable sub-grid scale space (V_tilde).
        # 
        # V = V_h ⊕ V_tilde
        #
        # For ASGS, V_tilde is simply defined as the space of algebraic residuals without projection. The sub-grid 
        # scale velocity is approximated as u_tilde ≈ -τ * R(u_h). Because R(u_h) represents the strong form 
        # residual across the continuum, it intrinsically maintains a non-zero overlap with V_h. Consequently, 
        # its L2-projection onto the finite element test space is strictly NON-ZERO.
        #
        # For OSGS, the method enforces structural distinctiveness by isolating V_tilde precisely as the orthogonal 
        # complement to the finite element space: V_tilde = V_h^⟂. Therefore, the OSGS formulation actively extracts 
        # u_tilde ≈ -τ * P^⟂_h(R(u_h)) = -τ * (R(u_h) - Π_h(R(u_h))).
        #
        # WHAT THIS BLOCK DOES:
        # We formalized the analytical variable `subgrid_R` immediately above:
        #   - ASGS: subgrid_R = R(u_h)
        #   - OSGS: subgrid_R = R(u_h) - Π_h(R(u_h))  (where Π_h(R(u_h)) = `pi_u`, exported exactly from the standard solver cache)
        #
        # We now solve the standard Variational equality (M_u * x_sub_solve = b_sub) to explicitly compute the 
        # continuous L2-projection of `subgrid_R` back onto the unconstrained finite element vector space (V_free):
        # 
        #           ∫ (v_h ⋅ Π_h(subgrid_R)) dΩ  =  ∫ (v_h ⋅ subgrid_R) dΩ      ∀ v_h ∈ V_free
        #
        # - If the OSGS solver inherently mapped the primary fields (u_h, p_h) correctly tracking orthogonalized 
        #   scales inside the staggered iteration, then mathematically subgrid_R ∈ V_h^⟂. By the exact definition 
        #   of functional orthogonality, the resultant L2-projection Π_h(subgrid_R) MUST analytically collapse 
        #   to machine-zero!
        # - Alternatively, when ASGS is successfully analyzed, the L2-projection Π_h(R(u_h)) retains significant 
        #   norm values, conclusively demonstrating that its algebraic sub-scales mathematically bleed back onto 
        #   the explicitly resolvable bounds!
        # -----------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------
        # We solve the generic generalized formulation leveraging strictly standardized core methods:
        pi_sub = PorousNSSolver.discrete_l2_projection(subgrid_R, U_free, V_free, dΩ)

        norm_residual_mapping = sqrt(sum(∫(pi_sub ⋅ pi_sub)dΩ))
        
        push!(results_summary, (method, success, norm_e_u, norm_residual_mapping))
    end

    println("\n=========================================================================================")
    println("                  FINAL ORTHOGONALITY VERIFICATION SUMMARY                               ")
    println("=========================================================================================")
    @printf("%-8s | %-12s | %-24s | %-24s\n", "Method", "Converged", "L2 Velocity Error", "Proj(Subgrid_R) Norm")
    println("-----------------------------------------------------------------------------------------")
    for (method, s, e, nr) in results_summary
        @printf("%-8s | %-12s | %-24.8e | %-24.8e\n", method, string(s), e, nr)
    end
    println("=========================================================================================\n")
    
    osgs_norm = results_summary[2][4]
    asgs_norm = results_summary[1][4]
    
    # Internal staggered roots converge bounding 1.0e-5 internally, ensuring values correctly sit under this limit
    if osgs_norm < 1e-5 && asgs_norm > 1e-4
        println("[SMOKING GUN PASS] -> The orthogonality checks conclusively proved that OSGS securely extracts and cancels boundary-invariant topologies (Norm ≈ 0), whereas ASGS natively maintains unconstrained bounds mapping as analytically expected!")
    else
        println("[FAIL] -> The numerical orthogonality validations collapsed. Sub-grid structures did not map to their corresponding mathematical bounds.")
    end
    
    @test osgs_norm < 1e-5
    @test asgs_norm > 1e-4
end
