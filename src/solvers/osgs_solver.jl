# src/solvers/osgs_solver.jl
#
# OSGS (Orthogonal Sub-Grid Scale) — the coupled Stage-II solve (Algorithm C of
# `theory/osgs_algorithm/osgs_algorithm.tex`). The SGS space is taken strictly orthogonal to the FE
# space (X_{h0}^⊥): we compute the L²-projection π_h = Π(R(U_h)) of the strong residual and subtract it
# in the sub-scale equation. ASGS is the special case π_h = 0 (identity projection) and lives in
# asgs_solver.jl.
#
# This file owns the OSGS-specific pieces: the L²-projection helpers, the coupled solve
# `solve_osgs_stage!`, and the OSGS_INNER_POLICY cascade-policy value (the ASGS counterpart STAGE_I_*
# values live in asgs_solver.jl). It depends on the shared solver core in solver_core.jl, which defines
# the structs FETopology/VMSFormulation (used in `solve_osgs_stage!`'s signature), plus safe_fe_solve!,
# _record_stage!, cascade_step_outcome, _pingpong_cascade!, the SolutionVerifier hooks, and the
# CascadePolicy type. `_with_overrides` lives in nonlinear.jl; CholeskySolver in linear_solvers.jl.
# Because the dependencies are resolved at parse time, this file is included after solver_core.jl and
# asgs_solver.jl (see the include order in PorousNSSolver.jl).
using Gridap
using Gridap.Algebra
using LinearAlgebra

# ==============================================================================
# OSGS cascade success-policy (Algorithm B) — what the coupled solve accepts
# ==============================================================================
# OSGS's single `CascadePolicy` row (the type/interpreter/scheduler are shared, in solver_core.jl; see
# the full cross-method truth table there and in test/blitz/cascade_policy_symmetry_blitz_test.jl).
#                       accept_noise_floor | accept_soft_stall | max_iters_caught_is_failure
#   OSGS_INNER_POLICY   true                 true                false  (accept ftol/noise/soft; struct → Picard; max_iters_caught → 1-iter)
const OSGS_INNER_POLICY = CascadePolicy(true,  true,  false)

# Build the CellField that gets L²-projected to form π_u, the momentum sub-scale. It is the strong
# momentum residual R(U_h) (eval_strong_residual_u) filtered through the projection policy: the policy
# decides which terms enter the orthogonal projection (e.g. the constant-σ reaction trim of paper §4.4
# drops the reaction term, whose L² projection onto the FE space is exactly zero). σ is the reaction
# tensor σ(α,u) evaluated pointwise via SigOp, passed so the policy can apply that trim.
function inner_projection_u(u, p, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
    R_u = eval_strong_residual_u(form, u, p, h_cf, alpha_cf, f_cf, c_1, c_2)
    sig_op = SigOp(form.reaction_law, form.regularization, form.ν, c_1, c_2)
    σ = Operation(sig_op)(u, ∇(u), alpha_cf, ∇(alpha_cf), h_cf)
    return apply_projectable_residual_u(form.projection_policy, R_u, σ, u)
end

# Pressure-equation analogue: the CellField that gets L²-projected to form π_p, the mass sub-scale. It is
# the strong mass residual R_p = physical_epsilon·p + α(∇·u) + u·∇α − g (eval_strong_residual_p), again filtered
# through the projection policy; physical_epsilon is the pressure stabilization floor consulted by the policy.
function inner_projection_p(u, p, form, dΩ, h_cf, alpha_cf, g_cf)
    R_p = eval_strong_residual_p(form, u, p, alpha_cf, g_cf)
    return apply_projectable_residual_p(form.projection_policy, R_p, form.physical_epsilon, p)
end

"""
    discrete_l2_projection(field, U_proj, V_proj, dΩ, M_mat, num_fac)

Compute the discrete L²-projection of a CellField `field` onto the FE space pair (`U_proj` trial,
`V_proj` test): solve `M x = b` where `M` is the L² mass matrix and `bᵢ = ∫(vᵢ ⋅ field)dΩ`, then wrap
the solution coefficients in an `FEFunction`. This is the workhorse called once per Newton evaluation to
form π_u and π_p, so it takes the pre-assembled mass matrix `M_mat` and its factorization `num_fac` as
arguments — reusing them avoids re-assembling/re-factoring inside the solver's inner loop.
"""
function discrete_l2_projection(field, U_proj, V_proj, dΩ, M_mat, num_fac)
    b_vec = assemble_vector(v -> ∫(v ⋅ field)dΩ, V_proj)
    x_solve = allocate_in_domain(M_mat)
    solve!(x_solve, num_fac, b_vec)
    return FEFunction(U_proj, x_solve)
end

"""
    discrete_l2_projection(field, U_proj, V_proj, dΩ)

Convenience wrapper that assembles and Cholesky-factors the L² mass matrix on the fly before delegating
to the five-argument method. Use it for one-off diagnostics; inside the solver's inner loop prefer the
five-argument form with a pre-factored mass matrix, since assembling/factoring per call is wasteful there.
"""
function discrete_l2_projection(field, U_proj, V_proj, dΩ)
    p_ls = CholeskySolver()  # SPD mass matrix — see `CholeskySolver` in `linear_solvers.jl`.
    M_mat = assemble_matrix((u,v) -> ∫(u ⋅ v)dΩ, U_proj, V_proj)
    num_fac = numerical_setup(symbolic_setup(p_ls, M_mat), M_mat)
    return discrete_l2_projection(field, U_proj, V_proj, dΩ, M_mat, num_fac)
end

# ==============================================================================
# Anderson-accelerated staggered OSGS outer loop  (opt-in: sol_cfg.osgs_anderson_enabled)
# ==============================================================================
# When `osgs_anderson_enabled` is set, the OSGS stage runs the STAGGERED fixed point of the OSGS problem
# instead of the default single recompute-π-each-eval inexact Newton. At each outer step k it freezes the
# projection π_k = Π(R(u_k,p_k)), solves the FROZEN-π nonlinear system (exact tangent — π constant ⇒ the
# usual ∂π/∂u drop is moot) to obtain the plain fixed-point map output g_k = G(x_k), then Anderson-mixes the
# (u,p) free-DOF vector: x_{k+1} = update!(acc, x_k, g_k). The bare staggered π-iteration is only linearly
# convergent; Anderson extrapolates it. The least-squares fit is L²-weighted by the block (u,p) mass matrix
# so the mix respects the function-space norm, not raw DOF magnitudes. (src/solvers/accelerators.jl; NONL-03.)
# OFF by default — when disabled, solve_osgs_stage! runs the existing path unchanged (bit-identical).
function _osgs_anderson_outer!(final_x0, setup::FETopology, formulation::VMSFormulation, config::PorousNSConfig,
                               base_nls, U_proj, V_proj, P_proj, Q_proj, M_u, num_u_fac, M_p, num_p_fac,
                               diag_cache, iter_count_ref, initial_success; p_prev=nothing)
    X, Y = setup.X, setup.Y
    dΩ = setup.dΩ
    h_cf, f_cf, alpha_cf, g_cf = setup.h_cf, setup.f_cf, setup.alpha_cf, setup.g_cf
    form, c_1, c_2 = formulation.form, formulation.c_1, formulation.c_2
    phys_cfg = config.physical_properties
    sol_cfg  = config.numerical_method.solver
    freeze_cusp = sol_cfg.freeze_jacobian_cusp

    println("      [+] OSGS ANDERSON mode: staggered outer fixed-point (freeze π → solve → re-project) with Anderson mixing of (u,p).")

    # L²-weight Anderson by the block (u,p) mass matrix on the (Dirichlet-constrained) state space, so the
    # least-squares fit measures the residual in the function-space norm rather than by raw DOF magnitudes.
    M_state = assemble_matrix((du, dv) -> ∫(du[1] ⋅ dv[1] + du[2] * dv[2])dΩ, X, Y)
    acc = AndersonAccelerator(sol_cfg.osgs_anderson_depth, sol_cfg.osgs_anderson_relaxation,
                              sol_cfg.osgs_anderson_safety_factor, M_state)

    inner_solver = FESolver(_with_overrides(base_nls; stall_window=0))
    drifts = Float64[]
    converged = false

    for outer in 1:sol_cfg.osgs_anderson_max_outer
        u_k, p_k = final_x0
        x_k = copy(get_free_dof_values(final_x0))

        # Freeze π at the current iterate (projected on the UNCONSTRAINED spaces, as everywhere in OSGS).
        pi_u_k = discrete_l2_projection(inner_projection_u(u_k, p_k, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2), U_proj, V_proj, dΩ, M_u, num_u_fac)
        pi_p_k = discrete_l2_projection(inner_projection_p(u_k, p_k, form, dΩ, h_cf, alpha_cf, g_cf), P_proj, Q_proj, dΩ, M_p, num_p_fac)

        # g_k = G(x_k): solve the frozen-π nonlinear system (π held fixed in BOTH residual and Jacobian).
        res_frozen = (x, y) -> build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=pi_u_k, pi_p=pi_p_k, p_prev=p_prev)
        jac_frozen = (x, dx, y) -> build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=pi_u_k, pi_p=pi_p_k)
        op_frozen = FEOperator(res_frozen, jac_frozen, X, Y)
        res_in = safe_fe_solve!(final_x0, inner_solver, op_frozen; backup=copy(x_k))
        _record_stage!(diag_cache, "C:OSGS:Anderson[$outer]", res_in)
        if res_in.state == :ok
            iter_count_ref[] += res_in.iterations
        elseif res_in.state == :max_iters_caught
            iter_count_ref[] += 1
        else
            break   # inner frozen-π solve structurally failed → stop, keep the last accepted iterate
        end
        g_k = copy(get_free_dof_values(final_x0))

        # Anderson extrapolation x_{k+1} = update!(acc, x_k, g_k); write it back into final_x0.
        x_next = update!(acc, x_k, g_k)
        copyto!(get_free_dof_values(final_x0), x_next)

        # Outer convergence: relative ℓ∞ drift of the (u,p) state across the π update.
        drift = norm(x_next .- x_k, Inf) / max(1.0, norm(x_next, Inf))
        push!(drifts, drift)
        if drift < sol_cfg.xtol
            converged = true
            break
        end
    end

    diag_cache["outer_anderson_drift"] = drifts
    diag_cache["outer_anderson_iters"] = length(drifts)
    return initial_success && converged
end

# ==============================================================================
# JFNK-accelerated OSGS coupled solve  (opt-in: sol_cfg.osgs_jfnk_enabled)
# ==============================================================================
# When `osgs_jfnk_enabled` is set, the OSGS coupled Newton solves the FULL tangent matrix-free, recovering
# the dense ∂π/∂u coupling the frozen-π tangent drops (which caps the linear rate and DIVERGES on
# stiff/convective cells — see docs/solver/jfnk-phase0-preconditioner-gate.md). The recovery is the JFNK
# inner solve of `theory/osgs_algorithm/osgs_algorithm.tex` §sec:jfnk: GMRES on `J_full·dx = F(U)` whose
# mat-vec is the directional finite difference of the SAME re-projecting residual `res_fn_coupled` (so the
# FD samples ∂π/∂u for free), left-preconditioned by the assembled+factored frozen-π Jacobian.
#
# Structurally this is "change exactly ONE thing": we hand the EXISTING SafeNewtonSolver a drop-in
# `JFNKLinearSolver` (linear_solvers.jl) in place of the direct LU/ILU solver. The outer Newton loop,
# Armijo/merit line search, divergence/stall guards, per-field gate, trajectory, and the [C.1] honesty
# contract are inherited unchanged — no re-implemented safeguards. The mat-vec needs the current iterate
# x_k as its FD base point; we thread it in through a Ref written by a thin wrapper around
# `jac_fn_coupled` (Gridap calls `jacobian!` at the iterate immediately before the inner solve, so the Ref
# holds x_k when `JFNKLinearSolver.solve!` reads it; the residual `b` it receives is exactly F(x_k)).
#
# [C.1] If the inner GMRES does not reach its forcing tolerance, `JFNKLinearSolver.solve!` raises
# `GMRESNotConvergedError`; `eval_linear_system_resolution!` catches it, rolls back, and this routine then
# FALLS BACK to the standard frozen-π coupled solve from the best iterate — never accepting a
# non-converged inner solve, never doing worse than the current solver. OFF by default ⇒ this routine is
# not entered and `solve_osgs_stage!` runs the existing path bit-identically.
function _osgs_jfnk_solve!(final_x0, setup::FETopology, formulation::VMSFormulation, config::PorousNSConfig,
                           base_nls, res_fn_coupled, jac_fn_coupled, jac_precond_fn, diag_cache, iter_count_ref, initial_success)
    X, Y = setup.X, setup.Y
    sol_cfg = config.numerical_method.solver

    # `jac_precond_fn` is the assembled+factored left preconditioner (the frozen-π tangent, optionally c₁-inflated
    # via osgs_jfnk_precond_c1_mult); the matrix-free mat-vec still differences the PHYSICAL-c₁ residual, so the
    # solved system (hence the root) is unchanged — only the preconditioner conditioning improves.
    println("      [+] OSGS JFNK mode: matrix-free full-tangent (∂π/∂u recovered) inner GMRES, " *
            (sol_cfg.osgs_jfnk_precond_c1_mult == 1.0 ? "frozen-π preconditioner." :
             "c₁×$(sol_cfg.osgs_jfnk_precond_c1_mult)-inflated frozen-π preconditioner."))

    # The FD base point x_k, written by the jacobian wrapper below and read by JFNKLinearSolver.solve!.
    xref = Ref{Vector{Float64}}(copy(get_free_dof_values(final_x0)))

    # F as a plain vector of a free-DOF vector: the SAME coupled residual (recomputes π ⇒ captures ∂π/∂u).
    Fvec = (vec) -> assemble_vector(y -> res_fn_coupled(FEFunction(X, vec), y), Y)

    # Frozen-π ExactNewton tangent (the preconditioner) PLUS a side effect: record the iterate the mat-vec
    # differences around. Recording here (not in the residual) is correct because Gridap evaluates the
    # jacobian at the iterate right before the linear solve, while the residual is also evaluated at
    # line-search trial points.
    jac_rec = (x, dx, y) -> begin
        xref[] = copy(get_free_dof_values(x))
        jac_precond_fn(x, dx, y)
    end

    jfnk_ls = JFNKLinearSolver(base_nls.ls, Fvec, xref,
                               sol_cfg.osgs_jfnk_gmres_rel_tol, sol_cfg.osgs_jfnk_gmres_maxiter,
                               sol_cfg.osgs_jfnk_gmres_restart, sol_cfg.osgs_jfnk_fd_epsilon)
    # stall_window=0 for the same reason as the direct coupled solve (slow-monotone ≠ stall).
    jfnk_nls = _with_overrides(base_nls; ls=jfnk_ls, stall_window=0)
    op_jfnk = FEOperator(res_fn_coupled, jac_rec, X, Y)

    x_backup = copy(get_free_dof_values(final_x0))
    res_j = safe_fe_solve!(final_x0, FESolver(jfnk_nls), op_jfnk; backup=x_backup)
    _record_stage!(diag_cache, "C:OSGS:JFNK", res_j)
    if res_j.state == :ok
        iter_count_ref[] += res_j.iterations
        diag_cache["final_residual_norm"] = res_j.residual_norm
        get!(diag_cache, "initial_residual_norm", res_j.initial_residual_norm)
    elseif res_j.state == :max_iters_caught
        iter_count_ref[] += 1
    end
    outcome_j = cascade_step_outcome(res_j, OSGS_INNER_POLICY)
    success = outcome_j == :success || outcome_j == :one_iter_success

    if !success
        # [C.1 fallback] JFNK did not converge structurally → run the standard frozen-π coupled solve from
        # the best iterate. This guarantees JFNK never does worse than the current solver.
        println("      [!] OSGS JFNK did not converge (state=$(res_j.state)); falling back to the frozen-π coupled solve.")
        coupled_nls = _with_overrides(base_nls; stall_window=0)
        op_coupled = FEOperator(res_fn_coupled, jac_fn_coupled, X, Y)
        res_c = safe_fe_solve!(final_x0, FESolver(coupled_nls), op_coupled; backup=copy(get_free_dof_values(final_x0)))
        _record_stage!(diag_cache, "C:OSGS:JFNK-fallback", res_c)
        if res_c.state == :ok
            iter_count_ref[] += res_c.iterations
            diag_cache["final_residual_norm"] = res_c.residual_norm
            get!(diag_cache, "initial_residual_norm", res_c.initial_residual_norm)
        elseif res_c.state == :max_iters_caught
            iter_count_ref[] += 1
        end
        outcome_c = cascade_step_outcome(res_c, OSGS_INNER_POLICY)
        success = outcome_c == :success || outcome_c == :one_iter_success
    end

    return initial_success && success
end

# ==============================================================================
# solve_osgs_stage!  (Algorithm C — the coupled OSGS solve)
# ==============================================================================
"""
    solve_osgs_stage!(success, final_x0, setup::FETopology, formulation::VMSFormulation,
                      config::PorousNSConfig, solver_newton, verifier, diag_cache, iter_count_ref)
        → (success::Bool, pi_u, pi_p, elapsed::Float64)

The single OSGS entry point the orchestrator dispatches to. Builds the unconstrained projection spaces
and the Cholesky-factored L² mass matrices, then runs the coupled solve: ONE Newton solve whose residual
recomputes `π = Π(R(u))` from the current iterate at every evaluation. The Jacobian holds π frozen,
giving a local sparse tangent rather than the prohibitive monolithic ∂π/∂u; this makes the coupling
Picard-type, converging linearly to the discrete OSGS fixed point `R̃ = R − Π(R)`. The shared
Newton↔Picard cascade is the fallback when `pingpong_enabled`; the stall sensor is disabled for this
solve because it converges slowly-monotone (a slow step is not a stall here).

`success` carries Stage I's flag forward: it must survive a budget-exhausted, verification-disabled exit,
so it is reassigned only by an explicit structural-failure / converged verdict below. After convergence,
`on_osgs_converged!(verifier, …)` runs the optional verification (NoVerification ⇒ no-op).

Returns the elapsed wall time of the timed coupled solve only; the mass-matrix assembly/factorization is
treated as setup and runs outside the `@elapsed` region.
"""
function solve_osgs_stage!(success, final_x0, setup::FETopology, formulation::VMSFormulation,
                           config::PorousNSConfig, solver_newton, verifier, diag_cache, iter_count_ref;
                           p_prev=nothing)
    initial_success = success

    # Local unpacking (cheap pointer aliases for Gridap performance)
    X, Y = setup.X, setup.Y
    V_free, Q_free = setup.V_free, setup.Q_free
    dΩ = setup.dΩ
    h_cf, f_cf, alpha_cf, g_cf = setup.h_cf, setup.f_cf, setup.alpha_cf, setup.g_cf
    form, c_1, c_2 = formulation.form, formulation.c_1, formulation.c_2
    phys_cfg = config.physical_properties
    sol_cfg  = config.numerical_method.solver
    freeze_cusp = sol_cfg.freeze_jacobian_cusp
    base_nls = solver_newton.nls

    println("\n      [+] Commencing Orthogonal Subgrid Scale (OSGS) coupled solve...")

    # Gridap L² projections inherit the boundary conditions of the FESpace they map onto. Project on the
    # UNCONSTRAINED V_free/Q_free: projecting on the Dirichlet-constrained space injects an O(1) boundary
    # residual that destroys the O(h^{k+1}) MMS convergence rate.
    U, P = X
    V, Q = Y
    V_proj = V_free !== nothing ? V_free : V
    Q_proj = Q_free !== nothing ? Q_free : Q
    U_proj = TrialFESpace(V_proj)
    P_proj = TrialFESpace(Q_proj)

    # L² mass matrices are SPD and functionally invariant across the solve, so assemble + Cholesky-factor
    # them exactly once (CHOLMOD; reused by every per-evaluation projection). This is setup, not solve, so
    # it lives outside the timed region.
    M_u = assemble_matrix((u,v) -> ∫(v ⋅ u)dΩ, U_proj, V_proj)
    M_p = assemble_matrix((p,q) -> ∫(q * p)dΩ, P_proj, Q_proj)
    num_u_fac = numerical_setup(symbolic_setup(CholeskySolver(), M_u), M_u)
    num_p_fac = numerical_setup(symbolic_setup(CholeskySolver(), M_p), M_p)

    pi_u = nothing
    pi_p = nothing

    coupled_elapsed = @elapsed begin
        println("      [+] OSGS COUPLED mode: single Newton solve; π = Π(R(u)) recomputed each nonlinear iteration (local frozen-π Jacobian; non-monolithic).")
        diag_cache["inner_osgs_diagnostics"] = []
        diag_cache["outer_osgs_diagnostics"] = []

        # ── OSGS coupled tangent: the EXACT frozen-π Newton Jacobian [A.3] ────────────────────────────
        # This is an INEXACT Newton on the fixed point F(U)=0 where F embeds π(U)=Π(R(U)), recomputed from
        # the CURRENT iterate at every residual evaluation (no lag). The full Jacobian would be
        #   dF/dU = ∂F/∂U|_{π frozen}  −  ∫ L*(V)·τ·Π(dR·dU) dΩ      (the 2nd term is ∂F/∂π · dπ/dU),
        # but Π(dR·dU) is a GLOBAL L²-projection (a dense M⁻¹ coupling of every DOF), so assembling it would
        # destroy the sparse local tangent and the direct/ILU solve. We deliberately drop ∂π/∂u (→ a
        # linear/superlinear rate; the matrix-free recovery is the JFNK section of the algorithm note) and
        # keep the EXACT frozen-π tangent ∂F/∂U|_{π frozen} — the most accurate Jacobian available without
        # going matrix-free. [A.3] Correctness of THAT tangent requires the live π: it enters the
        # ExactNewton product-rule terms dτ_1·(R−π) and dL*·(R−π) (continuous_problem.jl). The previous code
        # passed a literal ZERO π there, silently using (R) instead of (R−π) — an extra inexactness beyond
        # the intended ∂π/∂u drop. We now pass the live π (matching the Anderson path), so the ONLY remaining
        # approximation is the sparsity-preserving ∂π/∂u drop. The converged solution is unchanged
        # (R−π → 0 at the fixed point ⇒ the dτ/dL* terms vanish there); only the Newton path/rate moves.
        #
        # π is computed ONCE per iterate and shared by the residual and BOTH Jacobians (all are evaluated at
        # the same U within a Newton step). The cache is keyed on the (u,p) free-DOF vector and recomputes on
        # any mismatch, so correctness never depends on the residual/Jacobian call order — only efficiency
        # does (the common path is a hit, so the exact tangent costs no extra projection vs the old code).
        _pi_dofs = Ref{Union{Nothing, Vector{Float64}}}(nothing)
        _pi_u_ref = Ref{Any}(nothing); _pi_p_ref = Ref{Any}(nothing)
        live_pi! = (x) -> begin
            dofs = get_free_dof_values(x)
            if _pi_dofs[] === nothing || _pi_dofs[] != dofs
                u_x, p_x = x
                _pi_u_ref[] = discrete_l2_projection(inner_projection_u(u_x, p_x, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2), U_proj, V_proj, dΩ, M_u, num_u_fac)
                _pi_p_ref[] = discrete_l2_projection(inner_projection_p(u_x, p_x, form, dΩ, h_cf, alpha_cf, g_cf), P_proj, Q_proj, dΩ, M_p, num_p_fac)
                _pi_dofs[] = copy(dofs)
            end
            return _pi_u_ref[], _pi_p_ref[]
        end

        res_fn_coupled = (x, y) -> begin
            pi_u_x, pi_p_x = live_pi!(x)
            build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=pi_u_x, pi_p=pi_p_x, p_prev=p_prev)
        end
        # [A.3] EXACT frozen-π Newton tangent: pass the SAME live π the residual used (was a zero placeholder).
        jac_fn_coupled = (x, dx, y) -> begin
            pi_u_x, pi_p_x = live_pi!(x)
            build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=pi_u_x, pi_p=pi_p_x)
        end
        # [JFNK precond-c₁] The frozen-π tangent is a WEAK preconditioner for the coupled ∂π/∂u system in 3D-P2:
        # ρ(J_frozen⁻¹·∂π/∂u) ≈ 1178 ≫ 1, so inner GMRES can't converge. Inflating the PRECONDITIONER's c₁
        # (only) shrinks the subscale/∂π/∂u relative to the preconditioner — ρ falls to ≈3.8 at c₁×4 (a U-shaped
        # optimum) — WITHOUT touching the residual F, so the converged root is the physical-c₁ solution. The
        # preconditioner reuses the physical-c₁ live π (its effect on the tangent is 2nd-order near the root, ∝ R−π).
        # mult=1.0 ⇒ jac_precond_fn === jac_fn_coupled (byte-identical; the whole feature is off).
        jac_precond_fn = sol_cfg.osgs_jfnk_precond_c1_mult == 1.0 ? jac_fn_coupled :
            let vmsform_pc = VMSFormulation(form, c_1 * sol_cfg.osgs_jfnk_precond_c1_mult,
                                                  c_2 * sol_cfg.osgs_jfnk_precond_c1_mult)
                (x, dx, y) -> begin
                    pi_u_x, pi_p_x = live_pi!(x)
                    build_stabilized_weak_form_jacobian(x, dx, y, setup, vmsform_pc, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=pi_u_x, pi_p=pi_p_x)
                end
            end

        # [honest-recording guard] Capture the entry iterate so we can tell whether the OSGS coupled stage
        # actually ADVANCED off it. The `initial_ftol` short-circuit (nonlinear.jl:687) returns the entry
        # UNCHANGED in zero iterations: when the entry is the ASGS Stage-I boot state (the default, non-boot-skip
        # path) such a non-advancing "success" is byte-identical to ASGS — a misleading OSGS datum in an
        # ASGS-vs-OSGS study (the documented "OSGS silently reports the ASGS state" leak). We surface it so a
        # comparison harness records it honestly (NaN/fold) instead of as a genuine OSGS solve. ‖Δx‖ is EXACTLY
        # zero on a true short-circuit, so the test is an exact DOF-vector inequality — no tolerance needed.
        x_entry_dofs = copy(get_free_dof_values(final_x0))
        iters_before_osgs = iter_count_ref[]

        if sol_cfg.osgs_jfnk_enabled
            # Opt-in JFNK path (off by default ⇒ the existing path below runs, bit-identical). Recovers the
            # dropped ∂π/∂u via a matrix-free full-tangent inner GMRES preconditioned by the frozen-π
            # Jacobian; falls back to the frozen-π coupled solve on inner non-convergence ([C.1]).
            # Mutually exclusive with osgs_anderson_enabled (enforced by validate!).
            success = _osgs_jfnk_solve!(final_x0, setup, formulation, config, base_nls,
                                        res_fn_coupled, jac_fn_coupled, jac_precond_fn, diag_cache, iter_count_ref, initial_success)
        elseif sol_cfg.osgs_anderson_enabled
            # Opt-in staggered Anderson path (off by default ⇒ the existing path below runs, bit-identical).
            success = _osgs_anderson_outer!(final_x0, setup, formulation, config, base_nls,
                                            U_proj, V_proj, P_proj, Q_proj, M_u, num_u_fac, M_p, num_p_fac,
                                            diag_cache, iter_count_ref, initial_success; p_prev=p_prev)
        else
        op_coupled = FEOperator(res_fn_coupled, jac_fn_coupled, X, Y)
        x_backup = copy(get_free_dof_values(final_x0))
        # [known-fragility] The coupled inexact-Newton converges SLOWLY-MONOTONE (the dropped ∂π/∂U gives a
        # linear rate), so the stall sensor MUST be disabled here (stall_window=0): a slow-but-real step is
        # not a stall. If left on, the high-Da/fine-mesh coupled solve trips the stall test and bails after a
        # couple of steps with U still near the ASGS state — i.e. OSGS silently degenerates into ASGS while
        # reporting under the OSGS label. Genuine failures (line-search failure, divergence, max-iters) are
        # still caught. The Stage-I boot keeps its stall sensor (cold-start oscillation → Picard).
        coupled_nls = _with_overrides(base_nls; stall_window=0)
        if sol_cfg.pingpong_enabled
            # [coupled cascade] Give the coupled solve the SAME Newton↔Picard fallback as the Stage-I boot:
            # if Newton's line search fails or it diverges → hand to Picard (frozen-advection Oseen
            # linearisation, wider convergence basin) → back to Newton once Picard has cleared
            # `picard_gain_orders`. Well-conditioned cells run Newton straight to ftol with Picard inert;
            # hard cells (steep under-resolved porosity layers, where Newton's line search fails) lean on the
            # Picard fallback. The Picard tangent ZEROES the dτ/dL* product-rule terms, so π does not enter it
            # at all (byte-identical whether π is the live value or a zero placeholder — unlike the ExactNewton
            # tangent above, where [A.3] it does). We still route it through the shared `live_pi!` cache for
            # uniformity: the cache hit means no extra projection, and there is no zero-π special case to drift.
            jac_picard_coupled = (x, dx, y) -> begin
                pi_u_x, pi_p_x = live_pi!(x)
                build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=pi_u_x, pi_p=pi_p_x, mult_mom=1.0, mult_mass=1.0)
            end
            op_picard_coupled  = FEOperator(res_fn_coupled, jac_picard_coupled, X, Y)
            solver_picard_gain_c = FESolver(_with_overrides(coupled_nls; ftol=sol_cfg.picard_ftol, mode=:picard, picard_gain_target=sol_cfg.pingpong_picard_gain_orders))
            v_c = _pingpong_cascade!(final_x0, x_backup, op_coupled, op_picard_coupled,
                                     FESolver(coupled_nls), solver_picard_gain_c, OSGS_INNER_POLICY,
                                     diag_cache, iter_count_ref; stage_prefix="C:OSGS", max_swaps=sol_cfg.pingpong_max_swaps)
            success = initial_success && (v_c == :success)
        else
            res_c = safe_fe_solve!(final_x0, FESolver(coupled_nls), op_coupled; backup=x_backup)
            _record_stage!(diag_cache, "C:OSGS:Coupled", res_c)
            if res_c.state == :ok
                iter_count_ref[] += res_c.iterations
                diag_cache["final_residual_norm"] = res_c.residual_norm
                get!(diag_cache, "initial_residual_norm", res_c.initial_residual_norm)
            elseif res_c.state == :max_iters_caught
                iter_count_ref[] += 1
            end
            outcome_c = cascade_step_outcome(res_c, OSGS_INNER_POLICY)
            success = initial_success && (outcome_c == :success || outcome_c == :one_iter_success)
        end
        end  # if sol_cfg.osgs_jfnk_enabled / elseif osgs_anderson_enabled / else (default coupled)

        # [honest-recording guard] Did the OSGS coupled stage advance off its entry iterate? (Path-agnostic:
        # holds for the default coupled, ping-pong cascade, JFNK and Anderson routes alike.)
        osgs_advanced = get_free_dof_values(final_x0) != x_entry_dofs
        diag_cache["osgs_stage_iters"] = iter_count_ref[] - iters_before_osgs
        diag_cache["osgs_advanced_off_entry"] = osgs_advanced
        # The leak signature: OSGS reported success yet never moved off its entry. With the ASGS boot ON
        # (osgs_skip_asgs_boot=false) that entry IS the ASGS root, so the recorded OSGS state == ASGS — a datum
        # a comparison harness must NOT treat as a genuine OSGS solve. (With boot-skip the entry is the initial
        # guess, not an ASGS root, but a non-advancing success is still not a real OSGS solve.)
        diag_cache["osgs_short_circuited_on_entry"] = (success && !osgs_advanced)

        # Final self-consistent projection (for π export / diagnostics).
        u_h, p_h = final_x0
        pi_u = discrete_l2_projection(inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2), U_proj, V_proj, dΩ, M_u, num_u_fac)
        pi_p = discrete_l2_projection(inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf), P_proj, Q_proj, dΩ, M_p, num_p_fac)

        diag_cache["base_convergence_reached"] = true   # core key (set regardless of verification)
        diag_cache["base_convergence_outer_iter"] = 1
        on_osgs_converged!(verifier, final_x0, diag_cache)   # NoVerification ⇒ no-op; MMS ⇒ oracle eval + rate flag
    end
    diag_cache["pi_u"] = pi_u
    diag_cache["pi_p"] = pi_p
    return success, pi_u, pi_p, coupled_elapsed
end
