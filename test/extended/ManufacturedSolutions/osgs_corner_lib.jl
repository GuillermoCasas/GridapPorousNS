# test/extended/ManufacturedSolutions/osgs_corner_lib.jl
# ==============================================================================================
# [diagnostic-tool] Standalone OSGS coupled solve for the stiff MMS corner, reused by
# run_corner_osgs.jl. Mirrors `src/solvers/osgs_solver.jl::solve_osgs_stage!` (Algorithm C) EXACTLY
# — same L²-projection of the strong residual onto the unconstrained V_free/Q_free, same
# π = Π(R(u)) recomputed every nonlinear iteration, same local frozen-π ExactNewton Jacobian — but
# driven standalone (tight true-root tolerances, no orchestrator/verifier), warm-started from a
# converged ASGS root. The fold has cleared by N≈512, so warm-starting from the nearby ASGS root
# the frozen-π (Picard-type) coupled solve reaches the OSGS root in a handful of iterations.
# Depends on `include("run_continuation.jl")` first (build_cell, calc_normalized_errors, etc.).
# ==============================================================================================
using Gridap
using Gridap.Algebra
using Gridap.FESpaces

# One OSGS coupled solve on a prebuilt cell, warm-started from free-dof vector `x0_dofs`.
# Returns (reached, rnorm, iters, dofs, norms) with norms = (l2u,l2p,h1u,h1p).
#
# Tolerances mirror PRODUCTION, not a tight true-root: the frozen-π coupled solve converges
# slowly-linearly, and at this corner the production solver itself stops at the dynamic NOISE FLOOR
# (≈1e-5 at N=512/Re=1e6) — where the *error* is already converged (the err-converges-before-residual
# fact behind docs/.../vms-highre-stall). So we stop at `ftol`/`noise_floor`, not 1e-8, which reaches
# the OSGS root error in a handful of iterations instead of grinding for an hour.
function osgs_solve(cell, x0_dofs; max_iters=12, ftol=1e-6, noise_floor=1e-5, verbose=true)
    setup = cell.setup
    formA = cell.formulation            # VMSFormulation
    form = formA.form; c_1 = formA.c_1; c_2 = formA.c_2
    X, Y = setup.X, setup.Y
    dΩ = setup.dΩ
    h_cf, f_cf, alpha_cf, g_cf = setup.h_cf, setup.f_cf, setup.alpha_cf, setup.g_cf
    phys_cfg = cell.phys_cfg
    freeze_cusp = cell.freeze_cusp

    # Unconstrained projection spaces (projecting on the Dirichlet space breaks O(h^{k+1}) — see osgs_solver.jl).
    V, Q = Y
    V_proj = setup.V_free !== nothing ? setup.V_free : V
    Q_proj = setup.Q_free !== nothing ? setup.Q_free : Q
    U_proj = TrialFESpace(V_proj)
    P_proj = TrialFESpace(Q_proj)

    # SPD L² mass matrices, Cholesky-factored once, reused by every per-evaluation projection.
    M_u = assemble_matrix((u,v) -> ∫(v ⋅ u)dΩ, U_proj, V_proj)
    M_p = assemble_matrix((p,q) -> ∫(q * p)dΩ, P_proj, Q_proj)
    num_u_fac = numerical_setup(symbolic_setup(PorousNSSolver.CholeskySolver(), M_u), M_u)
    num_p_fac = numerical_setup(symbolic_setup(PorousNSSolver.CholeskySolver(), M_p), M_p)

    # Residual: recompute π = Π(R(u)) from the CURRENT iterate every evaluation (no lag).
    res_fn = (x, y) -> begin
        u_x, p_x = x
        R_u_cf = PorousNSSolver.inner_projection_u(u_x, p_x, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
        R_p_cf = PorousNSSolver.inner_projection_p(u_x, p_x, form, dΩ, h_cf, alpha_cf, g_cf)
        pi_u_x = PorousNSSolver.discrete_l2_projection(R_u_cf, U_proj, V_proj, dΩ, M_u, num_u_fac)
        pi_p_x = PorousNSSolver.discrete_l2_projection(R_p_cf, P_proj, Q_proj, dΩ, M_p, num_p_fac)
        PorousNSSolver.build_stabilized_weak_form_residual(x, y, setup, formA, phys_cfg; pi_u=pi_u_x, pi_p=pi_p_x)
    end
    # Jacobian: local frozen-π ExactNewton form (the π VALUE is irrelevant to the tangent — a zero
    # π just selects the OSGS branch).
    _zu = allocate_in_domain(M_u); fill!(_zu, 0.0); _pi_u0 = FEFunction(U_proj, _zu)
    _zp = allocate_in_domain(M_p); fill!(_zp, 0.0); _pi_p0 = FEFunction(P_proj, _zp)
    jac_fn = (x, dx, y) -> PorousNSSolver.build_stabilized_weak_form_jacobian(
        x, dx, y, setup, formA, phys_cfg, freeze_cusp, PorousNSSolver.ExactNewtonMode(); pi_u=_pi_u0, pi_p=_pi_p0)

    op = FEOperator(res_fn, jac_fn, X, Y)
    x = FEFunction(X, copy(x0_dofs))
    # Production-like tolerances; stall sensor OFF (OSGS converges slowly-monotone) — matches solve_osgs_stage!.
    nls = PorousNSSolver.SafeNewtonSolver(LUSolver(), max_iters, 10^6, 1e-14, ftol, 1e-8,
                                          1e-4, 1e10, noise_floor, 60, 0.5; mode=:newton, stall_window=0)
    solver = FESolver(nls)
    do_solve = () -> begin
        res = solve!(x, solver, op)
        cache = res isa Tuple ? res[2] : res
        nls_cache = cache isa Tuple ? cache[2] : cache
        hasproperty(nls_cache, :result) ? nls_cache.result : nothing
    end
    r = verbose ? do_solve() : redirect_stdout(do_solve, devnull)
    uh, ph = x
    l2u, l2p, h1u, h1p = calc_normalized_errors(uh, ph, cell.u_final, cell.p_final,
                                                cell.U_c, cell.P_c, cell.L, dΩ)
    final_R = r === nothing ? NaN : r.residual_norm
    reached = isfinite(final_R) && final_R <= 1e-7
    return (reached=reached, rnorm=final_R, iters=(r === nothing ? -1 : r.iterations),
            dofs=copy(get_free_dof_values(x)), l2u=l2u, l2p=l2p, h1u=h1u, h1p=h1p)
end
