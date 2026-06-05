# src/solvers/porous_solver.jl
#=
    porous_solver.jl

# Role
This module acts as the overall Variational Multiscale (VMS) orchestrator for the Porous Navier-Stokes system. It bridges the pure algebraic continuous operators, variational stabilized forms, and Jacobians (defined in `viscous_operators.jl`) with the iterative numerical execution and topological logic needed to achieve discrete convergence (defined in `nonlinear.jl`).

# Methodological Background
Following the formulation in the companion article, the Galerkin finite element discretization of the generalized Navier-Stokes equations suffers from LBB condition violations for equal-order spaces and instabilities in convection- or reaction-dominated flows. We use VMS to approximate the unresolved sub-grid scales (SGS) ``\widetilde{U}`` in terms of the resolved scales' residual ``\mathcal{R}(U_h)`` and a local element matrix of stabilization parameters ``\boldsymbol{\tau}_K``.

Depending on the chosen space ``\widetilde{\mathcal{X}}`` for the sub-grid scales, two different stabilized methods are implemented here:
1. **ASGS (Algebraic Sub-Grid Scale)**: The SGS space is taken as the space of finite element residuals. The projection operator onto the SGS space is the identity ``\widetilde{\Pi} = \mathcal{I}``, rendering the FE projection ``\boldsymbol{\pi}_h = \boldsymbol{0}``.
2. **OSGS (Orthogonal Sub-Grid Scale)**: The SGS space is strictly orthogonal to the finite element space (``\mathcal{X}_{h0}^{\perp}``). This requires actively computing the ``L^2``-projection of the residual ``\boldsymbol{\pi}_h`` iteratively and subtracting it from the strong residual in the sub-scale equation.

# Relations to other files
- `src/formulations/viscous_operators.jl`: Contains the translation of the abstract operators (e.g. ``\mathcal{L}_{\nu}, \mathcal{L}_c, \mathcal{L}_b``) into Gridap closures. `porous_solver.jl` delegates the assembly of `eval_strong_residual`, `build_stabilized_weak_form_jacobian`, etc., to this file.
- `src/solvers/nonlinear.jl`: Defines the exact solver execution schemes (`solver_picard`, `solver_newton`) required to converge the coupled nonlinear ASGS/OSGS algebraic problems.

# Algorithm-to-code mapping
See `docs/solver/algorithm-code-mapping.md` for the full table. In short, `solve_system` mirrors
Algorithm O (SimulationOrchestration) of `theory/osgs_algorithm.tex`; it delegates to seven
file-local helpers (`_initialize_asgs_state!`, `_run_asgs_mms_extension!`,
`_run_osgs_inner_cascade!`, `_compute_state_drift`, `_update_and_project!`,
`_decide_osgs_convergence`, `_run_osgs_relaxation!`) that map 1:1 to Algorithms B, C, D and
the §3.6 helpers of the paper.
=#
using Gridap
using Gridap.Algebra
using LinearAlgebra

function inner_projection_u(u, p, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
    # Define L2 projection for momentum residual natively as a CellField
    R_u = eval_strong_residual_u(form, u, p, h_cf, alpha_cf, f_cf, c_1, c_2)
    sig_op = SigOp(form.reaction_law, form.regularization, form.ν, c_1, c_2)
    σ = Operation(sig_op)(u, ∇(u), alpha_cf, ∇(alpha_cf), h_cf)
    return apply_projectable_residual_u(form.projection_policy, R_u, σ, u)
end

function inner_projection_p(u, p, form, dΩ, h_cf, alpha_cf, g_cf)
    # Define L2 projection for mass residual natively as a CellField
    R_p = eval_strong_residual_p(form, u, p, alpha_cf, g_cf)
    return apply_projectable_residual_p(form.projection_policy, R_p, form.eps_val, p)
end

"""
    check_porous_solver_parameters(method, ftol, stagnation_tol, max_osgs_iters, osgs_tol)

Enforces strict parameter boundaries for the top-level VMS solver configurations,
preventing unbounded numerical cascades resulting from malformed JSON properties.
"""
function check_porous_solver_parameters(stab_cfg::StabilizationConfig, sol_cfg::SolverConfig)
    if stab_cfg.method != "ASGS" && stab_cfg.method != "OSGS"
        throw(ArgumentError("Stabilization method must be strictly 'ASGS' or 'OSGS'. Passed: $(stab_cfg.method)"))
    end
    if sol_cfg.ftol <= 0.0 || sol_cfg.stagnation_noise_floor <= 0.0 || stab_cfg.osgs_tolerance <= 0.0
        throw(ArgumentError("Solver outer tolerances (ftol, stagnation_noise_floor, osgs_tol) must be strictly positive floats."))
    end
    if stab_cfg.osgs_iterations < 1
        throw(ArgumentError("OSGS maximum iterations must strictly be integers >= 1."))
    end
end


"""
    discrete_l2_projection(field, U_proj, V_proj, dΩ, M_mat, num_fac)

Generic high-efficiency continuous mapping tool. Evaluates the explicit discrete L2 analytical
projection of a given analytical function (`field`) specifically onto a constructed target algebraic
vector domain (`U_proj`, `V_proj`). Requires the pre-allocated topological mass matrix bounds
(`M_mat`, `num_fac`) strictly to prevent invariant compilation allocations dynamically inside deep solver iterators.
"""
function discrete_l2_projection(field, U_proj, V_proj, dΩ, M_mat, num_fac)
    b_vec = assemble_vector(v -> ∫(v ⋅ field)dΩ, V_proj)
    x_solve = allocate_in_domain(M_mat)
    solve!(x_solve, num_fac, b_vec)
    return FEFunction(U_proj, x_solve)
end

"""
    discrete_l2_projection(field, U_proj, V_proj, dΩ)

Convenience wrapper for single-pass diagnostics. Dynamically derives the underlying finite
element mass matrices mapped to an explicit `LUSolver()`. Highly accurate but algebraically
unsuitable for deep staggered execution loops.
"""
function discrete_l2_projection(field, U_proj, V_proj, dΩ)
    p_ls = CholeskySolver()  # SPD mass matrix — see `CholeskySolver` in `linear_solvers.jl`.
    M_mat = assemble_matrix((u,v) -> ∫(u ⋅ v)dΩ, U_proj, V_proj)
    num_fac = numerical_setup(symbolic_setup(p_ls, M_mat), M_mat)
    return discrete_l2_projection(field, U_proj, V_proj, dΩ, M_mat, num_fac)
end

struct FETopology
    X
    Y
    model
    Ω
    dΩ
    V_free
    Q_free
    h_cf
    f_cf
    alpha_cf
    g_cf
end

struct VMSFormulation
    form
    c_1
    c_2
end

struct IterativeSolvers
    picard
    newton
end

"""
    safe_fe_solve!(x, fesolver, op; backup=nothing)

Single-attempt FE-solver wrapper that absorbs the boilerplate every cascade
site in `solve_system` repeats: tuple unwrapping of Gridap's `solve!` return
value, non-finite-DOF guard with optional backup restoration, and exception
classification (Gridap's old "Reached maximum iterations" string vs other
exceptions).

Does NOT classify success/failure or do logging — the criteria differ across
the six call sites (Stage I Newton 1 / Picard / Newton 2 and OSGS inner
Newton 1 / Picard / Newton 2 retry), so each caller owns its own decision
logic. This helper exists purely to eliminate the duplicated try/catch +
tuple-unwrap + finite-check boilerplate.

Returns a `NamedTuple` with field `state`:
- `:ok` — finite state; also carries `iterations`, `residual_norm`,
  `initial_residual_norm`, `step_norm`, `stop_reason` from the result cache.
- `:nonfinite` — at least one DOF was non-finite; if `backup !== nothing`,
  the state was restored from `backup` before returning.
- `:max_iters_caught` — `solve!` threw a "Reached max(imum) iterations"
  exception. This is a legacy Gridap contract; `SafeNewtonSolver` uses
  `stop_reason = "max_iters_stagnation"` on the `:ok` path instead. Kept
  for defence against legacy or non-`SafeNewtonSolver` solver instances.
- `:exception` — `solve!` threw any other exception. Field `exc` holds it.

The backup restoration on `:nonfinite` is the only side effect beyond
`solve!`'s in-place mutation of `x`.
"""
function safe_fe_solve!(x, fesolver, op; backup=nothing)
    try
        res = solve!(x, fesolver, op)
        cache = res isa Tuple ? res[2] : res
        nls_cache = cache isa Tuple ? cache[2] : cache
        if any(!isfinite, get_free_dof_values(x))
            if backup !== nothing
                get_free_dof_values(x) .= backup
            end
            return (state = :nonfinite,)
        end
        r = nls_cache.result
        return (state = :ok,
                iterations = r.iterations,
                residual_norm = r.residual_norm,
                initial_residual_norm = r.initial_residual_norm,
                step_norm = r.step_norm,
                stop_reason = r.stop_reason,
                iteration_history = r.iteration_history,
                initial_residual_normalized = r.initial_residual_normalized,
                residual_normalized = r.residual_normalized)
    catch e
        msg = string(e)
        if occursin("Reached maximum iterations", msg) || occursin("Reached max iterations", msg)
            return (state = :max_iters_caught,)
        else
            return (state = :exception, exc = e)
        end
    end
end

# [trajectory] Append one algorithm-stage record to diag_cache["trajectory"], capturing a case's exact
# path through the nonlinear orchestration (Algorithm O → B/C → A) for later analysis. `stage` is a
# documentation-aligned code (e.g. "B:StageI:N1", "C:OSGS[3]:Picard"). `res` is a `safe_fe_solve!`
# NamedTuple; non-:ok early-return states carry only `state`, so the numeric fields fall back to defaults.
function _record_stage!(diag_cache, stage::String, res)
    haskey(diag_cache, "trajectory") || (diag_cache["trajectory"] = Any[])
    push!(diag_cache["trajectory"], (
        stage       = stage,
        state       = String(res.state),
        stop_reason = get(res, :stop_reason, ""),
        iters       = get(res, :iterations, 0),
        res_in      = get(res, :initial_residual_norm, NaN),
        res_out     = get(res, :residual_norm, NaN),
        # [trajectory] normalized residuals (per-field convergence quantity, ≤1 ⟺ converged); the
        # plotter prefers these and falls back to the inf-norms above when they are absent.
        res_in_norm  = get(res, :initial_residual_normalized, NaN),
        res_out_norm = get(res, :residual_normalized, NaN),
        history     = get(res, :iteration_history, NamedTuple[]),
    ))
end

# ==============================================================================
# H4 — _compute_state_drift  (pure, §3.6.1 StateDrift)
# ==============================================================================
"""
    _compute_state_drift(final_x0, x_prev, X, dΩ, osgs_state_drift_scale)
        → (x_diff::Float64, x_diff_inf::Float64)

Implements `StateDrift` from §3.6.1 of `theory/osgs_algorithm.tex`. Pure function.
Returns the mode-selected `x_diff` (L²-mass-weighted or ℓ∞ DOF norm, per
`osgs_state_drift_scale`) and the always-computed `x_diff_inf` (needed by the
absolute noise-floor short-circuit in `_decide_osgs_convergence`).
"""
function _compute_state_drift(final_x0, x_prev, X, dΩ, osgs_state_drift_scale)
    x_diff_inf = norm(get_free_dof_values(final_x0) - x_prev, Inf)

    # Split separation L2 mapping norms native evaluation
    if osgs_state_drift_scale == "L2_mass"
        u_prev, p_prev = FEFunction(X, x_prev)
        u_h, p_h = final_x0
        e_u = u_h - u_prev
        e_p = p_h - p_prev
        x_diff_u_l2 = sqrt(abs(sum(∫(e_u ⋅ e_u)dΩ)))
        x_diff_p_l2 = sqrt(abs(sum(∫(e_p * e_p)dΩ)))
        x_diff = max(x_diff_u_l2, x_diff_p_l2)
    else
        x_diff = x_diff_inf
    end

    return x_diff, x_diff_inf
end

# ==============================================================================
# H5 — _update_and_project!  (pure-ish, residual projection + Anderson)
# ==============================================================================
"""
    _update_and_project!(final_x0, pi_u, pi_p, accel_u, accel_p, form, dΩ,
                         h_cf, f_cf, alpha_cf, g_cf, c_1, c_2,
                         U_proj, V_proj, P_proj, Q_proj,
                         M_u, M_p, num_u_fac, num_p_fac)
        → (pi_u_next, pi_p_next, pi_u_drift, pi_p_drift, R_u_algebraic_norm)

Owns the residual evaluation, L² projection, optional Anderson mixing, and
projection-drift computation per OSGS outer iteration. No `diag_cache` writes,
no `println` — the caller assembles diagnostics from the returned tuple.
The Anderson accelerators are stateful (their internal history is mutated when
non-`nothing`); otherwise the function has no side effects on its inputs.
"""
function _update_and_project!(final_x0, pi_u, pi_p, accel_u, accel_p, form, dΩ,
                              h_cf, f_cf, alpha_cf, g_cf, c_1, c_2,
                              U_proj, V_proj, P_proj, Q_proj,
                              M_u, M_p, num_u_fac, num_p_fac)
    # Pre-map the sub-grid scales natively before bounding loop state
    u_h, p_h = final_x0
    R_u = inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
    R_p = inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf)

    pi_u_next = discrete_l2_projection(R_u, U_proj, V_proj, dΩ, M_u, num_u_fac)
    pi_p_next = discrete_l2_projection(R_p, P_proj, Q_proj, dΩ, M_p, num_p_fac)

    if !isnothing(accel_u) && !isnothing(accel_p)
        pi_u_mixed = update!(accel_u, get_free_dof_values(pi_u), get_free_dof_values(pi_u_next))
        pi_p_mixed = update!(accel_p, get_free_dof_values(pi_p), get_free_dof_values(pi_p_next))

        get_free_dof_values(pi_u_next) .= pi_u_mixed
        get_free_dof_values(pi_p_next) .= pi_p_mixed
    end

    dpi_u_vec = get_free_dof_values(pi_u_next) - get_free_dof_values(pi_u)
    dpi_p_vec = get_free_dof_values(pi_p_next) - get_free_dof_values(pi_p)

    pi_u_drift = sqrt(abs(dot(dpi_u_vec, M_u * dpi_u_vec)))
    pi_p_drift = sqrt(abs(dot(dpi_p_vec, M_p * dpi_p_vec)))

    b_u_R = assemble_vector(v -> ∫(v ⋅ R_u)dΩ, V_proj)
    R_u_algebraic_norm = norm(b_u_R, 2)

    return pi_u_next, pi_p_next, pi_u_drift, pi_p_drift, R_u_algebraic_norm
end

# ==============================================================================
# H6 — _decide_osgs_convergence  (pure, §3.6.2 Mode_stop)
# ==============================================================================
"""
    _decide_osgs_convergence(x_diff, x_diff_inf, pi_u_drift, pi_p_drift,
                              osgs_iter, stab_cfg, osgs_tol, stagnation_tol)
        → overall_converged::Bool

Implements §3.6.2 `Mode_stop` logic from `theory/osgs_algorithm.tex`, plus the
absolute ℓ∞ noise-floor short-circuit. Pure function.

The ℓ∞ short-circuit is gated to `state_drift` mode only: in `projection_drift` and `both`
modes the projection may legitimately still be moving while the primary state appears stagnant,
so a raw ℓ∞ short-circuit could accept a non-fixed-point on `(U_h, π_h)`.
"""
function _decide_osgs_convergence(x_diff, x_diff_inf, pi_u_drift, pi_p_drift,
                                   osgs_iter, stab_cfg, osgs_tol, stagnation_tol)
    # Geometrically bind orthogonality tolerance to the exact numerical limitations
    dynamic_osgs_tol = max(osgs_tol, stagnation_tol)
    state_converged = x_diff <= dynamic_osgs_tol
    proj_converged = max(pi_u_drift, pi_p_drift) <= max(stab_cfg.osgs_projection_tolerance, stagnation_tol)

    if stab_cfg.osgs_stopping_mode == "state_drift"
        overall_converged = state_converged
    elseif stab_cfg.osgs_stopping_mode == "projection_drift"
        overall_converged = proj_converged
    else
        overall_converged = state_converged && proj_converged
    end

    if x_diff_inf <= stagnation_tol && osgs_iter >= 2
        if stab_cfg.osgs_stopping_mode == "state_drift"
            # Stagnation noise floor on ℓ∞ acts as a safety net only when state drift
            # is the binding criterion. In "projection_drift" and "both" modes, the
            # projection may still be drifting legitimately — don't short-circuit.
            overall_converged = true
        end
    end

    return overall_converged
end

# ==============================================================================
# Shared cascade success-policy (Algorithm B; refactor-brief B5/B6/B10)
# ==============================================================================
# The ASGS Stage-I cascade (`_initialize_asgs_state!`) and the OSGS inner cascade
# (`_run_osgs_inner_cascade!`) are both Algorithm B. Historically they each inlined
# their own stop-reason classification, which is the *only* place they legitimately
# differ. `CascadePolicy` + `cascade_step_outcome` make that difference one explicit,
# shared decision instead of duplicated control flow, and give P4's Newton↔Picard
# ping-pong a single reusable verdict primitive.
#
# THREE policy flags are required (not two): Stage-I Newton-2 accepts a noise-floor
# finish but rejects a soft stall (xtol/max-iters/no-progress), whereas the OSGS inner
# accepts both — so `accept_noise_floor` and `accept_soft_stall` must be independent.
#
#                       accept_noise_floor | accept_soft_stall | max_iters_caught_is_failure
#   STAGE_I_POLICY      false                false               true   (N1: only ftol → success; else → Picard)
#   STAGE_I_N2_POLICY   true                 false               true   (N2: ftol+noise_floor → success; soft/struct → fail)
#   OSGS_INNER_POLICY   true                 true                false  (accept ftol/noise/soft; struct → Picard; max_iters_caught → 1-iter)
#
# Verdict vocabulary (caller owns logging, diagnostics, iter accounting, backup restore):
#   :success           — accept this Newton/Picard finish.
#   :one_iter_success  — a legacy `:max_iters_caught` counted as a 1-iteration success (OSGS only).
#   :reject            — not a success; the caller decides (Stage-I N1 → Picard; Stage-I N2 → fail;
#                        OSGS N1 → Picard; OSGS N2 → abort outer loop).
struct CascadePolicy
    accept_noise_floor::Bool
    accept_soft_stall::Bool
    max_iters_caught_is_failure::Bool
end

const STAGE_I_POLICY    = CascadePolicy(false, false, true)
const STAGE_I_N2_POLICY = CascadePolicy(true,  false, true)
const OSGS_INNER_POLICY = CascadePolicy(true,  true,  false)

"""
    cascade_step_outcome(res, policy::CascadePolicy) -> Symbol

Pure verdict for one Algorithm-B cascade segment from a `safe_fe_solve!` result + a
`CascadePolicy`. Returns `:success`, `:one_iter_success`, or `:reject`. The caller keeps
all side effects (logging, diagnostic pushes, iter-count accounting, backup restoration).
"""
function cascade_step_outcome(res, policy::CascadePolicy)
    st = res.state
    if st === :nonfinite || st === :exception
        return :reject
    elseif st === :max_iters_caught
        return policy.max_iters_caught_is_failure ? :reject : :one_iter_success
    else  # :ok — branch on the per-field solver verdict
        sr = res.stop_reason
        if sr == "ftol_reached" || sr == "initial_ftol"
            return :success
        elseif sr == "stagnation_noise_floor_reached"
            return policy.accept_noise_floor ? :success : :reject
        elseif sr == "linesearch_failed" || sr == "merit_divergence_escaped" || sr == "residual_divergence_escaped" || sr == "linear_solve_nan"
            return :reject  # structural failures are never accepted
        else  # xtol_stagnation / max_iters_stagnation / no_progress_stall — a "soft stall"
            return policy.accept_soft_stall ? :success : :reject
        end
    end
end

# ==============================================================================
# P4 — _pingpong_cascade!  (adaptive Newton↔Picard scheduling, opt-in)
# ==============================================================================
"""
    _pingpong_cascade!(final_x0, restore_vec, op_newton, op_picard, solver_newton,
                       solver_picard_gain, policy::CascadePolicy, diag_cache, iter_count_ref;
                       stage_prefix::String, max_swaps::Int) -> Symbol

Adaptive Newton↔Picard ping-pong shared by Stage I (ASGS) and the OSGS inner cascade. Runs Newton
until it stalls (a non-success `cascade_step_outcome` under `policy`), then a Picard segment that stops
the moment it has driven ‖R‖∞ down `picard_gain_target` orders (`solver_picard_gain` carries the gain
target → stop_reason "picard_gain_reached") — just enough to re-enter Newton's basin — then back to
Newton, bounded by `max_swaps`. Each segment is tagged honestly via `_record_stage!`
(`<stage_prefix>:PP[swap]:N` / `:P`) — never relabel a Picard step as Newton.

Returns `:success` (a Newton segment was accepted), `:structural_abort` (a non-finite/exception/merit-
divergence segment — restore done), or `:reject` (swaps exhausted without success). Safeguards
(merit/line-search/divergence within each Newton segment, monotone-residual within each Picard segment)
are unchanged — this only schedules *between* solves. Inert unless the caller opts in (pingpong_enabled).
"""
function _pingpong_cascade!(final_x0, restore_vec, op_newton, op_picard, solver_newton,
                            solver_picard_gain, policy::CascadePolicy, diag_cache, iter_count_ref;
                            stage_prefix::String, max_swaps::Int)
    # Account a Newton segment's iters + diagnostics (mirrors the cascade bookkeeping) and return its verdict.
    function _account_newton!(res, label)
        _record_stage!(diag_cache, label, res)
        if res.state == :ok
            iter_count_ref[] += res.iterations
            diag_cache["final_residual_norm"] = res.residual_norm
            get!(diag_cache, "initial_residual_norm", res.initial_residual_norm)
        elseif res.state == :max_iters_caught
            iter_count_ref[] += 1
        end
        return cascade_step_outcome(res, policy)
    end

    # Segment 0 — Newton from the entry iterate. (backup=restore_vec ⇒ helper auto-restores on non-finite.)
    res = safe_fe_solve!(final_x0, solver_newton, op_newton; backup=restore_vec)
    outcome = _account_newton!(res, "$(stage_prefix):PP[0]:N")
    if outcome == :success || outcome == :one_iter_success
        return :success
    end

    swap = 0
    while swap < max_swaps
        # Picard segment — globalize until the gain target re-enters Newton's basin.
        res_p = safe_fe_solve!(final_x0, solver_picard_gain, op_picard; backup=restore_vec)
        _record_stage!(diag_cache, "$(stage_prefix):PP[$(swap)]:P", res_p)
        if res_p.state == :ok
            iter_count_ref[] += res_p.iterations
            diag_cache["final_residual_norm"] = res_p.residual_norm
        elseif res_p.state == :max_iters_caught
            iter_count_ref[] += 1
        end
        # Picard structural failure (non-finite / exception / merit escaped / finite-but-exploding) → abort.
        if res_p.state == :nonfinite || res_p.state == :exception ||
           (res_p.state == :ok && res_p.stop_reason == "merit_divergence_escaped") ||
           any(!isfinite, get_free_dof_values(final_x0))
            get_free_dof_values(final_x0) .= restore_vec
            return :structural_abort
        end

        # Newton segment on the Picard-smoothed iterate.
        res_n = safe_fe_solve!(final_x0, solver_newton, op_newton; backup=restore_vec)
        outcome = _account_newton!(res_n, "$(stage_prefix):PP[$(swap+1)]:N")
        if outcome == :success || outcome == :one_iter_success
            return :success
        elseif res_n.state == :nonfinite || res_n.state == :exception
            return :structural_abort  # helper restored; cannot recover
        end
        swap += 1
    end
    return :reject  # swaps exhausted without success → caller falls through to terminal logic
end

# ==============================================================================
# H3 — _run_osgs_inner_cascade!  (Algorithm B, OSGS operators)
# ==============================================================================
"""
    _run_osgs_inner_cascade!(final_x0, x_prev, op_newton_osgs, op_picard_osgs,
                              local_fesolver, local_fesolver_picard,
                              diag_cache, iter_count_ref)
        → (success::Bool, structurally_failed::Bool)

Implements Algorithm B (RobustNonlinearCascade) of `theory/osgs_algorithm.tex`
with OSGS operators (π_h frozen) and `restore = x_prev`. Owns the
per-outer-iteration Newton→Picard→Newton cascade inside Algorithm C's outer
loop.

Internal policy: any non-structural Newton finish is accepted; only
`linesearch_failed`, `merit_divergence_escaped`, `linear_solve_nan` mark inner
failure (stagnation / max-iters finishes are progress — the outer OSGS loop
re-evaluates anyway). The classification site is below.

Returns `(success, structurally_failed)`. `structurally_failed=true` signals the
caller to `break` the outer OSGS loop.
"""
function _run_osgs_inner_cascade!(final_x0, x_prev, op_newton_osgs, op_picard_osgs,
                                   local_fesolver, local_fesolver_picard,
                                   diag_cache, iter_count_ref; outer_iter::Int=0,
                                   pingpong_enabled::Bool=false, pingpong_max_swaps::Int=0)
    # [P4] Opt-in adaptive Newton↔Picard ping-pong replaces the one-way N1→Picard→N2 cascade. Default
    # OFF ⇒ the existing body below runs unchanged (bit-identical). `local_fesolver_picard` carries the
    # gain target (set in _run_osgs_relaxation! when pingpong_enabled). Note: ping-pong only does real
    # work for OSGS when `osgs_inner_newton_iters > 1` — with a 1-step inner Newton the first segment
    # already exits as an accepted soft-stall, so the loop returns immediately (same numerics, see plan).
    if pingpong_enabled
        v = _pingpong_cascade!(final_x0, x_prev, op_newton_osgs, op_picard_osgs,
                               local_fesolver, local_fesolver_picard, OSGS_INNER_POLICY,
                               diag_cache, iter_count_ref;
                               stage_prefix="C:OSGS[$outer_iter]", max_swaps=pingpong_max_swaps)
        return (v == :success, v != :success)
    end

    # --- Attempt 1: OSGS Inner Newton ---
    newton_failed = false
    res_n1 = safe_fe_solve!(final_x0, local_fesolver, op_newton_osgs; backup=x_prev)
    _record_stage!(diag_cache, "C:OSGS[$outer_iter]:N1", res_n1)
    # [P3a] OSGS_INNER_POLICY: accept noise-floor + soft-stall finishes; only the three structural
    # reasons reject; legacy :max_iters_caught is a 1-iteration success. (See policy matrix above.)
    outcome_n1 = cascade_step_outcome(res_n1, OSGS_INNER_POLICY)
    if res_n1.state == :nonfinite
        println("        -> OSGS Inner Newton produced non-finite state. Restoring previous iterate, engaging Picard fallback.")
        newton_failed = true
    elseif res_n1.state == :exception
        println("        -> OSGS Newton threw unrecoverable exception: ", res_n1.exc, ". Attempting Picard fallback.")
        newton_failed = true
    elseif outcome_n1 == :one_iter_success   # legacy :max_iters_caught
        println("        -> OSGS Newton 1-step interior pass dynamically executed.")
        iter_count_ref[] += 1
    else  # :ok — :success or :reject (the policy decides; only structural reasons → :reject)
        iter_count_ref[] += res_n1.iterations

        push!(diag_cache["inner_osgs_diagnostics"], (
            iterations = res_n1.iterations,
            initial_res = res_n1.initial_residual_norm,
            final_res = res_n1.residual_norm,
            step_norm = res_n1.step_norm,
            stop_reason = res_n1.stop_reason
        ))
        diag_cache["final_residual_norm"] = res_n1.residual_norm

        # Only structural-failure stop reasons mark the inner Newton as failed. xtol_stagnation /
        # max_iters_stagnation (IterationCap) are accepted as outer-loop progress by design
        # (OSGS_INNER_POLICY.accept_soft_stall=true) — see `theory/osgs_algorithm.tex` §1.2.4.
        # Adding IterationCap to the reject set would make OSGS reject converged-but-not-tight
        # inner iterates — revise the algorithm doc (and the policy) first.
        if outcome_n1 == :reject
            newton_failed = true
        end
    end

    # --- Attempt 2 (only if Newton failed): Picard smoother, then Attempt 3: Newton retry ---
    if newton_failed
        println("        -> OSGS Inner Newton failed. Restoring previous iterate and engaging Picard smoother...")
        get_free_dof_values(final_x0) .= x_prev
        picard_completed = false
        # No backup passed to the helper: the explicit `any(!isfinite, …)` check below catches
        # catastrophic divergence in both the OK and the exception path.
        res_p = safe_fe_solve!(final_x0, local_fesolver_picard, op_picard_osgs)
        _record_stage!(diag_cache, "C:OSGS[$outer_iter]:Picard", res_p)
        if res_p.state == :ok
            iter_count_ref[] += res_p.iterations
            push!(diag_cache["inner_osgs_diagnostics"], (
                iterations = res_p.iterations,
                initial_res = res_p.initial_residual_norm,
                final_res = res_p.residual_norm,
                step_norm = res_p.step_norm,
                stop_reason = "picard:" * res_p.stop_reason
            ))
            diag_cache["final_residual_norm"] = res_p.residual_norm
            picard_completed = true
            # [picard-divergence early-exit] If Picard itself diverges (merit escaped — FINITE
            # but exploding, so the `any(!isfinite)` guard below misses it), the iterate / eps_pert
            # is hopeless; abort the OSGS nested sequence now rather than burning a Newton-2 retry
            # on a diverging state. Mirrors the ASGS-init early-exit.
            if res_p.stop_reason == "merit_divergence_escaped"
                println("        -> OSGS Picard diverged (merit escaped). Restoring previous iterate and aborting OSGS nested sequence.")
                get_free_dof_values(final_x0) .= x_prev
                return (false, true)
            end
        elseif res_p.state == :max_iters_caught
            println("        -> OSGS Picard concluded smoothing loops.")
            picard_completed = true
        elseif res_p.state == :exception
            println("        -> OSGS Picard threw unrecoverable exception: ", res_p.exc, ". Restoring backup and aborting OSGS.")
            get_free_dof_values(final_x0) .= x_prev
        end
        # `:nonfinite` falls through to the explicit check below (helper did
        # not auto-restore because `backup=` was not passed).

        if any(!isfinite, get_free_dof_values(final_x0))
            println("        -> OSGS Picard catastrophically diverged. Restoring and aborting OSGS.")
            get_free_dof_values(final_x0) .= x_prev
            return (false, true)
        end

        if !picard_completed
            return (false, true)
        end

        # --- Attempt 3: re-engage Newton on the Picard-smoothed iterate ---
        println("        -> OSGS Picard smoothing finalized. Re-engaging Newton...")
        res_n2 = safe_fe_solve!(final_x0, local_fesolver, op_newton_osgs; backup=x_prev)
        _record_stage!(diag_cache, "C:OSGS[$outer_iter]:N2", res_n2)
        outcome_n2 = cascade_step_outcome(res_n2, OSGS_INNER_POLICY)
        if res_n2.state == :nonfinite
            println("        -> OSGS Inner Newton retry produced non-finite state. Restoring previous iterate, aborting OSGS nested sequence.")
            return (false, true)
        elseif res_n2.state == :exception
            println("        -> OSGS Newton retry threw unrecoverable exception: ", res_n2.exc, ". Aborting OSGS loop.")
            return (false, true)
        elseif outcome_n2 == :one_iter_success   # legacy :max_iters_caught
            iter_count_ref[] += 1
        else  # :ok — :success or :reject
            iter_count_ref[] += res_n2.iterations
            push!(diag_cache["inner_osgs_diagnostics"], (
                iterations = res_n2.iterations,
                initial_res = res_n2.initial_residual_norm,
                final_res = res_n2.residual_norm,
                step_norm = res_n2.step_norm,
                stop_reason = "retry:" * res_n2.stop_reason
            ))
            diag_cache["final_residual_norm"] = res_n2.residual_norm

            if outcome_n2 == :reject
                println("        -> OSGS Inner Newton retry also failed (Reason: $(res_n2.stop_reason)). Aborting OSGS nested sequence.")
                return (false, true)
            end
        end
    end

    return (true, false)
end

# ==============================================================================
# H1 — _initialize_asgs_state!  (Algorithm B, Stage I ASGS init)
# ==============================================================================
"""
    _initialize_asgs_state!(x0, x0_backup, op_newton_init, op_picard_init,
                             solver_newton_asgs, solver_picard, ftol,
                             diag_cache, iter_count_ref) → success::Bool

Implements Algorithm B (RobustNonlinearCascade) of `theory/osgs_algorithm.tex`
for the Stage I ASGS initialisation. Owns the algebraic initialisation cascade
(Newton → Picard → Newton). The init-stage operators (`op_*_init`) and
`solver_newton_asgs` are constructed by the caller because they close over
`setup`/`formulation`/`phys_cfg`; the `x0_backup` allocation also lives in the
caller. Mutates `x0`, writes `diag_cache["final_residual_norm"]`, and
accumulates `iter_count_ref`.

Internal policy (Stage I addendum): Newton-1 that hits the noise floor but not
`ftol` does **not** mark success — it falls through to Picard to guarantee
entering the quadratic basin. This asymmetry vs `_run_osgs_inner_cascade!` is
intentional (see B6 in `theory/refactor_solve_system_prompt.md`).
"""
function _initialize_asgs_state!(x0, x0_backup, op_newton_init, op_picard_init,
                                  solver_newton_asgs, solver_picard, ftol,
                                  diag_cache, iter_count_ref;
                                  pingpong_enabled::Bool=false, pingpong_max_swaps::Int=0,
                                  solver_picard_gain=nothing)
    # [P4] Opt-in adaptive Newton↔Picard ping-pong replaces the one-way Newton→Picard→Newton cascade.
    # Default OFF ⇒ the existing body below runs unchanged (bit-identical). `solver_picard_gain` is the
    # Picard solver carrying the gain-then-return target (built by the caller when pingpong_enabled);
    # Stage I runs Newton at the full `newton_iterations` budget, so ping-pong is effective here.
    if pingpong_enabled
        v = _pingpong_cascade!(x0, x0_backup, op_newton_init, op_picard_init,
                               solver_newton_asgs, solver_picard_gain, STAGE_I_POLICY,
                               diag_cache, iter_count_ref;
                               stage_prefix="B:StageI", max_swaps=pingpong_max_swaps)
        return v == :success
    end

    success = false
    println("      -> ASGS Initializer: Attempting Exact Newton solver initially...")
    newton_success = false
    res = safe_fe_solve!(x0, solver_newton_asgs, op_newton_init; backup=x0_backup)
    _record_stage!(diag_cache, "B:StageI:N1", res)
    if res.state == :nonfinite
        println("      -> ASGS Initializer: Exact Newton produced non-finite state. Restoring backup, falling through to Picard.")
        newton_success = false
    elseif res.state == :exception
        println("      -> ASGS Initializer: Newton ConvergenceError. Exception: ", res.exc)
        newton_success = false
    elseif res.state == :max_iters_caught
        # "Reached max iterations" at the initializer counts as a non-success (logged as a
        # ConvergenceError); Stage I demands genuine convergence before it may bypass Picard.
        println("      -> ASGS Initializer: Newton ConvergenceError. Exception: Reached maximum iterations")
        newton_success = false
    else  # :ok
        final_res = res.iterations > 0 ? res.residual_norm : 0.0
        diag_cache["final_residual_norm"] = final_res
        get!(diag_cache, "initial_residual_norm", res.initial_residual_norm)
        iter_count_ref[] += res.iterations
        # [stage-I success = the solver's OWN per-field convergence verdict, not an absolute floor]
        # Bypass Picard whenever Newton reached its (per-field, dynamic) ftol target — stop_reason
        # "ftol_reached"/"initial_ftol" — i.e. it is already in the exact-Newton quadratic basin.
        # Keying on `final_res <= ftol` would be wrong: that scalar `ftol` is a machine-precision FLOOR
        # (10·eps in the MMS harness), not the convergence target, so a per-field-converging solve almost
        # never meets it and Picard would fire on every cell. A stagnation / xtol / max-iters / divergence
        # exit still falls through to Picard to secure the quadratic basin before OSGS (B6).
        # [P3a] STAGE_I_POLICY rejects noise-floor + soft-stall (accept_*=false), so for an :ok finish
        # `cascade_step_outcome == :success` ⟺ stop_reason ∈ {ftol_reached, initial_ftol} (the B6 rule).
        if cascade_step_outcome(res, STAGE_I_POLICY) == :success
            newton_success = true
            println("      -> ASGS Initializer: Exact Newton reached its ftol target (stop=$(res.stop_reason), ‖R‖=$(final_res)) — already in the quadratic basin. Bypassing Picard.")
        else
            println("      -> ASGS Initializer: Exact Newton exited without reaching ftol (stop=$(res.stop_reason), ‖R‖=$(final_res)). Engaging Picard to secure the quadratic basin (B6).")
        end
    end

    if newton_success
        success = true
    else
        println("      -> ASGS Initializer: Exact Newton structurally aborted loop without geometric saturation. Orchestrating Picard Homotopy fallback...")
        get_free_dof_values(x0) .= x0_backup

        # Backup is intentionally NOT passed to the helper here: Picard can go non-finite mid-iteration
        # without throwing, so the explicit `any(!isfinite, …)` check below handles that divergence.
        picard_diverged = false
        res_p = safe_fe_solve!(x0, solver_picard, op_picard_init)
        _record_stage!(diag_cache, "B:StageI:Picard", res_p)
        if res_p.state == :ok
            final_res_picard = res_p.residual_norm
            diag_cache["final_residual_norm"] = final_res_picard
            get!(diag_cache, "initial_residual_norm", res_p.initial_residual_norm)
            iter_count_ref[] += res_p.iterations
            # [picard-divergence early-exit] Picard's role here is to globalise a Newton
            # failure; if Picard *itself* diverges (merit escaped — FINITE but exploding,
            # so the `any(!isfinite)` guard below misses it) the current eps_pert is hopeless.
            # Bail now instead of burning an expensive Newton-2 retry on a diverging iterate,
            # so the caller falls back to a milder eps_pert.
            picard_diverged = (res_p.stop_reason == "merit_divergence_escaped")
            if final_res_picard <= ftol
                println("      -> ASGS Initializer: Picard fully converged! Escaping to evaluation.")
                success = true
            end
        elseif res_p.state == :max_iters_caught
            println("      -> ASGS Initializer: Picard concluded initial smoothing loops.")
        elseif res_p.state == :exception
            println("      -> ASGS Initializer: Picard threw unrecoverable error. Exception: ", res_p.exc)
            get_free_dof_values(x0) .= x0_backup
        end
        # `:nonfinite` is caught by the `any(!isfinite, …)` check below; the helper does not
        # auto-restore the backup here because no `backup=` was passed.

        if any(!isfinite, get_free_dof_values(x0)) || picard_diverged
            println("      -> ASGS Initializer: Picard diverged (merit escaped / non-finite). Restoring backup; skipping Newton retry to fall back to a milder eps_pert.")
            get_free_dof_values(x0) .= x0_backup
            success = false
        elseif !success
            println("      -> ASGS Initializer: Picard smoothing finalized. Re-engaging Exact Newton...")
            res2 = safe_fe_solve!(x0, solver_newton_asgs, op_newton_init; backup=x0_backup)
            _record_stage!(diag_cache, "B:StageI:N2", res2)
            if res2.state == :nonfinite
                println("      -> ASGS Initializer: Newton Homotopy Pass produced non-finite state. Restoring backup, marking failure.")
                success = false
            elseif res2.state == :exception
                println("      -> ASGS Initializer: Newton ConvergenceError on Homotopy Pass. Exception: ", res2.exc)
                success = false
            elseif res2.state == :max_iters_caught
                # "Reached max iterations" on the homotopy pass is treated as a ConvergenceError (failure).
                println("      -> ASGS Initializer: Newton ConvergenceError on Homotopy Pass. Exception: Reached maximum iterations")
                success = false
            else  # :ok
                iter_count_ref[] += res2.iterations
                final_res = res2.residual_norm
                diag_cache["final_residual_norm"] = final_res
                get!(diag_cache, "initial_residual_norm", res2.initial_residual_norm)

                # [P3a] STAGE_I_N2_POLICY accepts ftol + noise-floor (accept_noise_floor=true) but rejects
                # a soft stall (accept_soft_stall=false) and structural failures — i.e. Newton-2 may exit
                # on the noise floor (unlike Newton-1) but not on xtol/max-iters. The three-way logging is
                # preserved; only the `success` verdict is routed through the shared policy.
                stop_reason = res2.stop_reason
                if cascade_step_outcome(res2, STAGE_I_N2_POLICY) == :success
                    if stop_reason == "ftol_reached" || stop_reason == "initial_ftol"
                        println("      -> ASGS Initializer: Newton Homotopy Pass achieved exact theoretical tolerance ($ftol).")
                    else  # stagnation_noise_floor_reached (the only other :success under STAGE_I_N2_POLICY)
                        println("      -> ASGS Initializer: Newton Homotopy Pass cleanly saturated at numerical noise floor ($final_res). Approving.")
                    end
                    success = true
                else
                    println("      -> ASGS Initializer: Newton Homotopy Pass structurally failed (Reason: $stop_reason). Bounding collapse.")
                    success = false
                end
            end
        end
    end

    return success
end

# ==============================================================================
# H2 — _run_asgs_mms_extension!  (Algorithm D, ASGS branch)
# ==============================================================================
"""
    _run_asgs_mms_extension!(final_x0, op_newton_init, solver_newton, mms_cfg,
                              ftol, diag_cache, iter_count_ref) → elapsed::Float64

Implements Algorithm D (VerifyMMSPlateau) of `theory/osgs_algorithm.tex` for the
ASGS-branch hook. Returns the elapsed wall time of the timed cycle loop only
(matching the original `@elapsed begin ... end` region around the cycle loop);
the caller adds this to its `eval_time`.

Writes to `diag_cache`: `["base_convergence_reached"]`, `["mms_plateau_reached"]`,
`["mms_stop_reason"]`, `["mms_error_history"]`, `["mms_relative_change_history"]`,
and `["final_residual_norm"]` (per cycle).
"""
function _run_asgs_mms_extension!(final_x0, op_newton_init, solver_newton, mms_cfg,
                                   ftol, diag_cache, iter_count_ref)
    println("\n      [!] Commencing ASGS MMS Error Verification Plateau Loop...")

    diag_cache["base_convergence_reached"] = true
    diag_cache["mms_plateau_reached"] = false
    mms_err_hist = []
    mms_rc_hist = []

    E_u2_0, E_p2_0, E_u1_0, E_p1_0 = mms_cfg.oracle(final_x0...)
    push!(mms_err_hist, (E_u2_0, E_p2_0, E_u1_0, E_p1_0))

    consecutive_passes = 0

    base_nls = solver_newton.nls
    local_asgs_nls = _with_overrides(base_nls; max_iters=1)
    local_fesolver = FESolver(local_asgs_nls)

    elapsed = @elapsed begin
        for cycle in 1:mms_cfg.max_extra_cycles
            try
                res_solve = solve!(final_x0, local_fesolver, op_newton_init)
                cache_solve = res_solve isa Tuple ? res_solve[2] : res_solve
                nls_cache = cache_solve isa Tuple ? cache_solve[2] : cache_solve
                iter_count_ref[] += nls_cache.result.iterations
                diag_cache["final_residual_norm"] = nls_cache.result.residual_norm
            catch e
                if occursin("Reached maximum iterations", string(e)) || occursin("Reached max iterations", string(e))
                    # [B10] Legacy max-iters ConvergenceError on a verification re-solve: counted as a
                    # 1-iteration pass (the outer plateau loop re-evaluates anyway). Logged for parity with
                    # the other cascade sites' max_iters handling — purely cosmetic, no numeric effect.
                    println("      -> ASGS MMS Verification: Newton ConvergenceError (Reached maximum iterations) — counting as a 1-iteration verification pass.")
                    iter_count_ref[] += 1
                else
                    println("      -> ASGS MMS Verification Newton failed. Aborting extension. Exception: ", e)
                    diag_cache["mms_stop_reason"] = "nonlinear_failure"
                    break
                end
            end

            E_u2_k, E_p2_k, E_u1_k, E_p1_k = mms_cfg.oracle(final_x0...)
            push!(mms_err_hist, (E_u2_k, E_p2_k, E_u1_k, E_p1_k))

            E_u2_prev, E_p2_prev, E_u1_prev, E_p1_prev = mms_err_hist[end-1]

            # §5.1: scale plateau floors by the FE discretization budget.
            # L² convergence rate is h^(kv+1); H¹ rate is h^kv. The config field
            # `mms_cfg.eps_*` is now a *baseline* (dimensionless) coefficient;
            # the effective floor is baseline × h^(rate).
            eps_u_l2_eff = mms_cfg.eps_u_l2 * mms_cfg.h_local^(mms_cfg.kv + 1)
            eps_u_h1_eff = mms_cfg.eps_u_h1 * mms_cfg.h_local^mms_cfg.kv
            eps_p_l2_eff = mms_cfg.eps_p_l2 * mms_cfg.h_local^(mms_cfg.kv + 1)

            r_u2 = abs(E_u2_k - E_u2_prev) / max(E_u2_k, E_u2_prev, eps_u_l2_eff)
            r_u1 = abs(E_u1_k - E_u1_prev) / max(E_u1_k, E_u1_prev, eps_u_h1_eff)
            r_p2 = abs(E_p2_k - E_p2_prev) / max(E_p2_k, E_p2_prev, eps_p_l2_eff)

            push!(mms_rc_hist, (r_u2, r_p2, r_u1))
            max_r = max(r_u2, r_u1, r_p2)

            println("        * [Verification Cycle $cycle] Plateau max ratio: $max_r (Target: $(mms_cfg.tau_err))")

            if max_r < mms_cfg.tau_err
                consecutive_passes += 1
            else
                consecutive_passes = 0
            end

            if consecutive_passes >= mms_cfg.require_consecutive_passes
                println("        [+] ASGS MMS Plateau formally established natively ($consecutive_passes consecutive < $(mms_cfg.tau_err)).")
                diag_cache["mms_plateau_reached"] = true
                diag_cache["mms_stop_reason"] = "mms_plateau_satisfied"

                # §5.2: rate-aware sanity check. If the converged L² velocity error
                # exceeds the FE accuracy budget by more than `rate_check_factor`,
                # the iteration plateaued at a non-FE-optimal state. Flag via the
                # stop reason; keep success in the calling context (the plateau IS
                # established, just at a sub-optimal level).
                budget = mms_cfg.h_local^(mms_cfg.kv + 1)
                if E_u2_k > mms_cfg.rate_check_factor * budget
                    diag_cache["mms_stop_reason"] = "mms_plateau_at_suboptimal_rate"
                    println("        [!] Plateau at sub-optimal rate: E_u2=$E_u2_k > $(mms_cfg.rate_check_factor) × h^(kv+1)=$(mms_cfg.rate_check_factor * budget). Flagged for inspection.")
                end

                break
            end
        end
    end

    if !get(diag_cache, "mms_plateau_reached", false) && !haskey(diag_cache, "mms_stop_reason")
        diag_cache["mms_stop_reason"] = "mms_budget_exhausted"
    end

    diag_cache["mms_error_history"] = mms_err_hist
    diag_cache["mms_relative_change_history"] = mms_rc_hist

    return elapsed
end

# ==============================================================================
# H7 — _run_osgs_relaxation!  (Algorithm C, OSGS fractional relaxation)
# ==============================================================================
"""
    _run_osgs_relaxation!(initial_success, final_x0, X, Y, V_free, Q_free,
                          setup, formulation, phys_cfg, stab_cfg, sol_cfg,
                          solver_newton, ftol, stagnation_tol, freeze_cusp, mms_cfg,
                          M_u, M_p, num_u_fac, num_p_fac,
                          U_proj, V_proj, P_proj, Q_proj,
                          diag_cache, iter_count_ref)
        → (success::Bool, pi_u, pi_p, elapsed::Float64)

Implements Algorithm C (OSGSFractionalRelaxation) of `theory/osgs_algorithm.tex`
in its entirety, including the terminal `mms_budget_exhausted` branch (M9
deviation: the post-loop MMS budget check lives inside this helper, not in
`solve_system`, so Algorithm C's terminal logic stays in one place).

`initial_success` carries Stage I's success flag into this helper's function-level `success`.
The contract: `success` is `true` after a successful Stage I and is flipped only by an explicit
structural-failure / overall_converged / mms_budget_exhausted branch inside the OSGS loop. So if the
outer loop exhausts its budget without convergence *and* MMS is disabled, the carried `true` is
preserved. Reinitialising to `false` here would silently change the `(success, …)` semantics for that
case.

Calls per outer iteration: `_run_osgs_inner_cascade!` (H3, Algorithm B),
`_compute_state_drift` (H4, §3.6.1), `_update_and_project!` (H5), and
`_decide_osgs_convergence` (H6, §3.6.2). The OSGS-branch MMS plateau hook
(Algorithm D for the OSGS path) is inlined inside the outer loop here, because
extracting it would require returning a `break` signal through a function
boundary.

The factorized mass-matrix handles (`M_u`, `M_p`, `num_u_fac`, `num_p_fac`) and
projection spaces (`U_proj`, `V_proj`, `P_proj`, `Q_proj`) are constructed by
the caller and live outside the timed region, so `eval_time` measures only the
outer loop.

Returns the elapsed wall time of the timed outer loop only; the post-loop
budget-check and final `diag_cache["pi_u"]/["pi_p"]` writes execute *after* the
`@elapsed` block.

Before the staggered loop, this helper dispatches on `stab_cfg.osgs_projection_coupling` and may
early-return from a dedicated branch (each with the same `(success, pi_u, pi_p, elapsed)` contract):
  • `"coupled"` — one Newton solve whose residual recomputes `π = Π(R(u))` each evaluation (frozen-π
    local Jacobian; Picard-type, linearly convergent to the same fixed point).
  • `"freeze_after_k"` — exactly `osgs_freeze_after_k` fully-converged staggered updates then stop (the
    last is the quadratic frozen finish); a velocity-H¹ relative-drift gate (`osgs_freeze_drift_gate`)
    restores the Stage-I ASGS state where the subscale coupling is destabilising. See `efficiency-ideas.md` Idea 5.
The default `"staggered"` mode falls through to the Algorithm-C outer loop documented above.
"""
function _run_osgs_relaxation!(initial_success, final_x0, X, Y, V_free, Q_free,
                                setup, formulation, phys_cfg, stab_cfg, sol_cfg,
                                solver_newton, ftol, stagnation_tol, freeze_cusp, mms_cfg,
                                M_u, M_p, num_u_fac, num_p_fac,
                                U_proj, V_proj, P_proj, Q_proj,
                                diag_cache, iter_count_ref)
    # Local unpacking (cheap pointer aliases for Gridap performance)
    form, c_1, c_2 = formulation.form, formulation.c_1, formulation.c_2
    dΩ = setup.dΩ
    h_cf, f_cf, alpha_cf, g_cf = setup.h_cf, setup.f_cf, setup.alpha_cf, setup.g_cf
    osgs_tol = stab_cfg.osgs_tolerance
    max_osgs_iters = stab_cfg.osgs_iterations

    # Override the Exact Newton solver dynamically for OSGS to perform only a few monolithic iterations
    # per sub-scale projection (e.g., 3 inner steps) rather than unconditionally clamping to 1.
    base_nls = solver_newton.nls
    # Inner tolerancing is determined dynamically in loop

    # OSGS outer-iteration budget is exactly the configured `stab_cfg.osgs_iterations`.
    # When MMS plateau verification is active, the explicit `mms_cfg.max_extra_cycles`
    # extension is added on top so the verifier has its own clearly-named budget.
    eff_osgs_iters = max_osgs_iters
    if !isnothing(mms_cfg) && mms_cfg.enabled
        eff_osgs_iters += mms_cfg.max_extra_cycles
    end

    # Carry Stage I's success forward (see docstring: it must survive a budget-exhausted, MMS-disabled exit).
    success = initial_success
    pi_u = nothing
    pi_p = nothing

    # ==============================================================================
    # [projection-coupling == "coupled"]  Single coupled Newton solve.
    # ------------------------------------------------------------------------------
    # Instead of the staggered outer loop (freeze π → inner Newton → update π), run ONE Newton solve at
    # the full Newton budget whose RESIDUAL recomputes π = Π(R(u)) from the current iterate at every
    # evaluation. This removes the staggering lag (the frozen-π residual) that makes the staggered map
    # contract only linearly. The Jacobian stays the LOCAL frozen-π form (sparse — NOT the prohibitive
    # monolithic ∂π/∂u), so this is a Picard-type coupling, not a monolithic Newton. The per-evaluation
    # projection is the cheap Cholesky-cached mass solve (`discrete_l2_projection`). Converges to the SAME
    # OSGS fixed point as the staggered scheme (R̃ = R − Π(R) orthogonal), so errors match; the aim is to
    # reach it in ~ASGS iteration counts. See docs/solver/efficiency-ideas.md.
    # ==============================================================================
    if stab_cfg.osgs_projection_coupling == "coupled"
        coupled_elapsed = @elapsed begin
            println("      [+] OSGS COUPLED mode: single Newton solve; π = Π(R(u)) recomputed each nonlinear iteration (local frozen-π Jacobian; non-monolithic).")
            diag_cache["inner_osgs_diagnostics"] = []
            diag_cache["outer_osgs_diagnostics"] = []

            # Residual: recompute π from the CURRENT iterate every evaluation (no staggering lag).
            res_fn_coupled = (x, y) -> begin
                u_x, p_x = x
                R_u_cf = inner_projection_u(u_x, p_x, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2)
                R_p_cf = inner_projection_p(u_x, p_x, form, dΩ, h_cf, alpha_cf, g_cf)
                pi_u_x = discrete_l2_projection(R_u_cf, U_proj, V_proj, dΩ, M_u, num_u_fac)
                pi_p_x = discrete_l2_projection(R_p_cf, P_proj, Q_proj, dΩ, M_p, num_p_fac)
                build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=pi_u_x, pi_p=pi_p_x)
            end
            # Jacobian: local frozen-π form. The π VALUE is irrelevant to the Jacobian (ProjectFullResidual
            # ignores it; the reaction-trim uses σ/∂σ); a zero π just selects the OSGS (is_osgs=true) branch.
            _zu = allocate_in_domain(M_u); fill!(_zu, 0.0); _pi_u0 = FEFunction(U_proj, _zu)
            _zp = allocate_in_domain(M_p); fill!(_zp, 0.0); _pi_p0 = FEFunction(P_proj, _zp)
            jac_fn_coupled = (x, dx, y) -> build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=_pi_u0, pi_p=_pi_p0)

            op_coupled = FEOperator(res_fn_coupled, jac_fn_coupled, X, Y)
            x_backup = copy(get_free_dof_values(final_x0))
            res_c = safe_fe_solve!(final_x0, FESolver(base_nls), op_coupled; backup=x_backup)
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

            # Final self-consistent projection (for π export / diagnostics).
            u_h, p_h = final_x0
            pi_u = discrete_l2_projection(inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2), U_proj, V_proj, dΩ, M_u, num_u_fac)
            pi_p = discrete_l2_projection(inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf), P_proj, Q_proj, dΩ, M_p, num_p_fac)

            diag_cache["base_convergence_reached"] = true
            diag_cache["base_convergence_outer_iter"] = 1
            if !isnothing(mms_cfg) && mms_cfg.enabled
                E_u2, E_p2, E_u1, E_p1 = mms_cfg.oracle(final_x0...)
                diag_cache["mms_plateau_reached"] = true
                diag_cache["mms_stop_reason"] = "coupled_single_solve"
                budget = mms_cfg.h_local^(mms_cfg.kv + 1)
                if E_u2 > mms_cfg.rate_check_factor * budget
                    diag_cache["mms_stop_reason"] = "coupled_at_suboptimal_rate"
                    println("        [!] Coupled OSGS at sub-optimal rate: E_u2=$E_u2 > $(mms_cfg.rate_check_factor)·h^(kv+1).")
                end
            end
        end
        diag_cache["pi_u"] = pi_u
        diag_cache["pi_p"] = pi_p
        return success, pi_u, pi_p, coupled_elapsed
    end

    # ==============================================================================
    # [projection-coupling == "freeze_after_k"]  Coupled warm-up, then freeze π and finish.
    # ------------------------------------------------------------------------------
    # WARM-UP — `osgs_freeze_after_k` (=k) single-step staggered iterations: each takes ONE Newton step
    # against the current frozen π, then re-projects π = Π(R(u)). So the projection is refreshed at EVERY
    # nonlinear iteration for the first k iterations (the lagged map; ~linearly contracting). Because π is
    # frozen WITHIN each single step, every warm-up step is itself a valid Newton step — the `D=-2Φ`
    # line-search certificate holds — so no special line-search handling is needed; π only changes BETWEEN
    # steps.
    # FREEZE + FINISH — freeze π = π_k (the projection after the k warm-up steps) and run a full Newton solve
    # to convergence. With π constant the Jacobian is the EXACT tangent of R(·;π_k) (no ∂π/∂u term), so the
    # finish converges QUADRATICALLY to U*(π_k).
    # The result U*(π_k) is optimal-RATE for any fixed k — the paper's stability/convergence theory rests on
    # the π-INDEPENDENT bilinear form B_S, so ASGS (π=0) is already optimal-order and orthogonality only
    # shrinks the error CONSTANT. k slides the constant from ASGS (k=0) toward OSGS (k→∞); k=2-3 captures
    # ~all of the OSGS constant gain in ~ASGS iteration counts.
    # The projection is ALWAYS applied — there is NO ASGS fallback. In the reaction-dominated corner the OSGS
    # *fixed point itself* is sub-optimal, so those cells honestly show OSGS's (poor) behaviour; that is a
    # formulation defect (docs/mms/convergence-status.md caveat #5), NOT something this iterator masks.
    # See docs/solver/efficiency-ideas.md Idea 5.
    # ==============================================================================
    if stab_cfg.osgs_projection_coupling == "freeze_after_k"
        freeze_elapsed = @elapsed begin
            k_freeze = stab_cfg.osgs_freeze_after_k
            println("      [+] OSGS FREEZE-AFTER-K mode: $k_freeze projection-updating warm-up step(s) (π refreshed each nonlinear iteration), then freeze π and a quadratic frozen-π finish.")
            diag_cache["inner_osgs_diagnostics"] = []
            diag_cache["outer_osgs_diagnostics"] = []

            # π^0 = 0 (first warm-up step re-derives π from the ASGS state — same bootstrap as the staggered loop).
            _z_u = allocate_in_domain(M_u); fill!(_z_u, 0.0); pi_u = FEFunction(U_proj, _z_u)
            _z_p = allocate_in_domain(M_p); fill!(_z_p, 0.0); pi_p = FEFunction(P_proj, _z_p)

            _pp_gain = sol_cfg.pingpong_enabled ? sol_cfg.pingpong_picard_gain_orders : Inf
            # WARM-UP solver: ONE Newton step per iteration (max_iters=1) ⇒ one projection update per nonlinear
            # iteration. FINISH solver: the full Newton budget (the quadratic frozen-π solve).
            warm_solver        = FESolver(_with_overrides(base_nls; max_iters=1))
            warm_solver_picard = FESolver(_with_overrides(base_nls; max_iters=1, ftol=sol_cfg.picard_handoff_ftol, mode=:picard, picard_gain_target=_pp_gain))
            finish_solver        = FESolver(base_nls)
            finish_solver_picard = FESolver(_with_overrides(base_nls; ftol=sol_cfg.picard_handoff_ftol, mode=:picard, picard_gain_target=_pp_gain))

            warmup_cut_short = false
            for m in 1:k_freeze
                println("        [Freeze warm-up $m/$k_freeze]  (project π every nonlinear iteration)")
                res_fn_fz = (x, y) -> build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=pi_u, pi_p=pi_p)
                jac_newton_fz = (x, dx, y) -> build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=pi_u, pi_p=pi_p)
                jac_picard_fz = (x, dx, y) -> build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=pi_u, pi_p=pi_p, mult_mom=1.0, mult_mass=1.0)
                op_newton_fz = FEOperator(res_fn_fz, jac_newton_fz, X, Y)
                op_picard_fz = FEOperator(res_fn_fz, jac_picard_fz, X, Y)

                x_prev = copy(get_free_dof_values(final_x0))
                (_, structurally_failed) = _run_osgs_inner_cascade!(
                    final_x0, x_prev, op_newton_fz, op_picard_fz,
                    warm_solver, warm_solver_picard,
                    diag_cache, iter_count_ref; outer_iter=m,
                    pingpong_enabled=sol_cfg.pingpong_enabled, pingpong_max_swaps=sol_cfg.pingpong_max_swaps)

                if structurally_failed
                    println("        [!] Freeze warm-up $m structurally failed — stopping warm-up early; finishing with the current frozen π.")
                    warmup_cut_short = true
                    break
                end

                (x_diff, x_diff_inf) = _compute_state_drift(final_x0, x_prev, X, dΩ, stab_cfg.osgs_state_drift_scale)
                (pi_u_next, pi_p_next, pi_u_drift, pi_p_drift, R_u_algebraic_norm) = _update_and_project!(
                    final_x0, pi_u, pi_p, nothing, nothing, form, dΩ,
                    h_cf, f_cf, alpha_cf, g_cf, c_1, c_2,
                    U_proj, V_proj, P_proj, Q_proj,
                    M_u, M_p, num_u_fac, num_p_fac)

                _fz_E = (!isnothing(mms_cfg) && mms_cfg.enabled) ? mms_cfg.oracle(final_x0...) : (NaN, NaN, NaN, NaN)
                push!(diag_cache["outer_osgs_diagnostics"], (
                    x_diff_inf = x_diff_inf, x_diff_resolved = x_diff,
                    pi_u_drift = pi_u_drift, pi_p_drift = pi_p_drift, R_u_norm = R_u_algebraic_norm,
                    err_u_l2 = _fz_E[1], err_p_l2 = _fz_E[2], err_u_h1 = _fz_E[3], err_p_h1 = _fz_E[4],
                    pi_u_l2 = sqrt(abs(sum(∫(pi_u_next ⋅ pi_u_next)dΩ))),
                    pi_p_l2 = sqrt(abs(sum(∫(pi_p_next * pi_p_next)dΩ))),
                ))
                println("        * warm-up drift ‖ΔU‖_$(stab_cfg.osgs_state_drift_scale)=$x_diff | π drift u=$pi_u_drift p=$pi_p_drift")

                pi_u = pi_u_next
                pi_p = pi_p_next
            end

            # FREEZE + FINISH: π is now frozen at π_k; run the full Newton solve to convergence (quadratic,
            # since π is constant ⇒ exact tangent ⇒ :newton line-search valid).
            println("        [Freeze finish] π frozen after warm-up; quadratic frozen-π Newton to convergence.")
            res_fn_fin = (x, y) -> build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=pi_u, pi_p=pi_p)
            jac_newton_fin = (x, dx, y) -> build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=pi_u, pi_p=pi_p)
            jac_picard_fin = (x, dx, y) -> build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=pi_u, pi_p=pi_p, mult_mom=1.0, mult_mass=1.0)
            op_newton_fin = FEOperator(res_fn_fin, jac_newton_fin, X, Y)
            op_picard_fin = FEOperator(res_fn_fin, jac_picard_fin, X, Y)

            x_prev_fin = copy(get_free_dof_values(final_x0))
            (fin_success, _fin_sf) = _run_osgs_inner_cascade!(
                final_x0, x_prev_fin, op_newton_fin, op_picard_fin,
                finish_solver, finish_solver_picard,
                diag_cache, iter_count_ref; outer_iter=k_freeze + 1,
                pingpong_enabled=sol_cfg.pingpong_enabled, pingpong_max_swaps=sol_cfg.pingpong_max_swaps)

            # Final self-consistent projection (for π export / diagnostics) + the finish's outer-trace entry.
            u_h, p_h = final_x0
            pi_u = discrete_l2_projection(inner_projection_u(u_h, p_h, form, dΩ, h_cf, f_cf, alpha_cf, c_1, c_2), U_proj, V_proj, dΩ, M_u, num_u_fac)
            pi_p = discrete_l2_projection(inner_projection_p(u_h, p_h, form, dΩ, h_cf, alpha_cf, g_cf), P_proj, Q_proj, dΩ, M_p, num_p_fac)
            (x_diff_f, x_diff_inf_f) = _compute_state_drift(final_x0, x_prev_fin, X, dΩ, stab_cfg.osgs_state_drift_scale)
            _fin_E = (!isnothing(mms_cfg) && mms_cfg.enabled) ? mms_cfg.oracle(final_x0...) : (NaN, NaN, NaN, NaN)
            push!(diag_cache["outer_osgs_diagnostics"], (
                x_diff_inf = x_diff_inf_f, x_diff_resolved = x_diff_f,
                pi_u_drift = 0.0, pi_p_drift = 0.0, R_u_norm = NaN,
                err_u_l2 = _fin_E[1], err_p_l2 = _fin_E[2], err_u_h1 = _fin_E[3], err_p_h1 = _fin_E[4],
                pi_u_l2 = sqrt(abs(sum(∫(pi_u ⋅ pi_u)dΩ))),
                pi_p_l2 = sqrt(abs(sum(∫(pi_p * pi_p)dΩ))),
            ))

            success = initial_success && fin_success
            diag_cache["base_convergence_reached"] = true
            diag_cache["base_convergence_outer_iter"] = k_freeze + 1
            diag_cache["mms_plateau_reached"] = true
            diag_cache["mms_stop_reason"] = warmup_cut_short ? "freeze_after_k_warmup_cut_short" : "freeze_after_k"
            if !isnothing(mms_cfg) && mms_cfg.enabled
                E_u2, E_p2, E_u1, E_p1 = mms_cfg.oracle(final_x0...)
                budget = mms_cfg.h_local^(mms_cfg.kv + 1)
                if E_u2 > mms_cfg.rate_check_factor * budget
                    diag_cache["mms_stop_reason"] = "freeze_after_k_at_suboptimal_rate"
                    println("        [!] Freeze-after-k at sub-optimal rate: E_u2=$E_u2 > $(mms_cfg.rate_check_factor)·h^(kv+1) — the reaction-dominated OSGS fixed-point defect (convergence-status.md caveat #5), shown honestly (no ASGS fallback).")
                end
            end
        end
        diag_cache["pi_u"] = pi_u
        diag_cache["pi_p"] = pi_p
        return success, pi_u, pi_p, freeze_elapsed
    end

    elapsed = @elapsed begin
        local accel_u = nothing
        local accel_p = nothing
        if sol_cfg.accelerator.type == "Anderson"
            println("      [+] Initializing Blocked Anderson Acceleration (m = $(sol_cfg.accelerator.m), damping = $(sol_cfg.accelerator.relaxation_factor), safety = $(sol_cfg.accelerator.safety_factor))")
            accel_u = AndersonAccelerator(sol_cfg.accelerator.m, sol_cfg.accelerator.relaxation_factor, sol_cfg.accelerator.safety_factor, M_u)
            accel_p = AndersonAccelerator(sol_cfg.accelerator.m, sol_cfg.accelerator.relaxation_factor, sol_cfg.accelerator.safety_factor, M_p)
        end

        # Initialize the orthogonal projection π_h^0 = 0. The first OSGS outer
        # iteration then re-solves U_h against π_h = 0 — which IS the ASGS problem,
        # already converged — and produces the first non-trivial projection naturally.
        # Avoids the off-distribution bootstrap that the warmup phase had to unwind.
        println("        * Initializing orthogonal projection π_h^0 = 0 (first iter re-derives from ASGS state)...")
        x_u = allocate_in_domain(M_u); fill!(x_u, 0.0)
        x_p = allocate_in_domain(M_p); fill!(x_p, 0.0)
        pi_u = FEFunction(U_proj, x_u)
        pi_p = FEFunction(P_proj, x_p)

        diag_cache["inner_osgs_diagnostics"] = []
        diag_cache["outer_osgs_diagnostics"] = []

        for osgs_iter in 1:eff_osgs_iters
            println("        [OSGS Iter $osgs_iter]")

            # [code-actual] Scale-covariant OSGS warmup. During the first `osgs_warmup_iterations` outer
            # iters the inner solve is loosened via the per-field RELATIVE target: `osgs_warmup_tolerance`
            # is a DIMENSIONLESS relative-reduction target `‖R_k‖ ≤ osgs_warmup_tolerance·‖R₀_k‖`, looser
            # than the normal per-field target and encoding-invariant. (An absolute warmup ftol would be
            # dimensional against a residual ∝U/∝P_c and break OSGS covariance — guarded by
            # encoding_invariance_test.jl.) Outside warmup the normal per-field target applies.
            warmup_rel = (osgs_iter <= stab_cfg.osgs_warmup_iterations && base_nls.relative_ftol_per_field !== nothing) ?
                fill(stab_cfg.osgs_warmup_tolerance, length(base_nls.relative_ftol_per_field)) : nothing

            local_osgs_nls = _with_overrides(base_nls; max_iters=stab_cfg.osgs_inner_newton_iters, relative_ftol_per_field=warmup_rel)
            local_fesolver = FESolver(local_osgs_nls)

            # [P4] When ping-pong is enabled the inner Picard carries the gain-then-return target so it
            # stops the moment it re-enters Newton's basin; Inf (default) ⇒ legacy Picard (run to handoff/cap).
            _pp_gain = sol_cfg.pingpong_enabled ? sol_cfg.pingpong_picard_gain_orders : Inf
            local_osgs_nls_picard = _with_overrides(base_nls; max_iters=stab_cfg.osgs_inner_newton_iters, ftol=sol_cfg.picard_handoff_ftol, mode=:picard, relative_ftol_per_field=warmup_rel, picard_gain_target=_pp_gain)
            local_fesolver_picard = FESolver(local_osgs_nls_picard)

            res_fn_osgs(x, y) = build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=pi_u, pi_p=pi_p)
            jac_newton_osgs(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=pi_u, pi_p=pi_p)
            jac_picard_osgs(x, dx, y) = build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=pi_u, pi_p=pi_p, mult_mom=1.0, mult_mass=1.0)

            op_newton_osgs = FEOperator(res_fn_osgs, jac_newton_osgs, X, Y)
            op_picard_osgs = FEOperator(res_fn_osgs, jac_picard_osgs, X, Y)

            x_prev = copy(get_free_dof_values(final_x0))

            (inner_success, structurally_failed) = _run_osgs_inner_cascade!(
                final_x0, x_prev, op_newton_osgs, op_picard_osgs,
                local_fesolver, local_fesolver_picard,
                diag_cache, iter_count_ref; outer_iter=osgs_iter,
                pingpong_enabled=sol_cfg.pingpong_enabled, pingpong_max_swaps=sol_cfg.pingpong_max_swaps)

            if structurally_failed
                success = false
                break
            end

            (x_diff, x_diff_inf) = _compute_state_drift(final_x0, x_prev, X, dΩ, stab_cfg.osgs_state_drift_scale)

            (pi_u_next, pi_p_next, pi_u_drift, pi_p_drift, R_u_algebraic_norm) = _update_and_project!(
                final_x0, pi_u, pi_p, accel_u, accel_p, form, dΩ,
                h_cf, f_cf, alpha_cf, g_cf, c_1, c_2,
                U_proj, V_proj, P_proj, Q_proj,
                M_u, M_p, num_u_fac, num_p_fac)

            # [trace-enrichment] Capture, per outer iteration, the MMS error (the accuracy trajectory — shows
            # where the error plateaus, independent of the drift gate that never fires) and the projection L2
            # norms ‖π_u‖/‖π_p‖ (to see whether the projection itself limit-cycles or only U_h does). The MMS
            # error is the quantity we ultimately verify; recording it every outer iter (not only after the
            # never-reached drift-base-convergence) makes the OSGS staggered loop's convergence vs limit-cycle
            # behaviour fully diagnosable from the structured trace alone — no log-scraping. Only when the
            # oracle is available (MMS verification runs); a few extra integrals per outer iter.
            _osgs_E = (!isnothing(mms_cfg) && mms_cfg.enabled) ? mms_cfg.oracle(final_x0...) : (NaN, NaN, NaN, NaN)
            _pi_u_l2 = sqrt(abs(sum(∫(pi_u_next ⋅ pi_u_next)dΩ)))
            _pi_p_l2 = sqrt(abs(sum(∫(pi_p_next * pi_p_next)dΩ)))
            push!(diag_cache["outer_osgs_diagnostics"], (
                x_diff_inf = x_diff_inf,
                x_diff_resolved = x_diff,
                pi_u_drift = pi_u_drift,
                pi_p_drift = pi_p_drift,
                R_u_norm = R_u_algebraic_norm,
                err_u_l2 = _osgs_E[1],
                err_p_l2 = _osgs_E[2],
                err_u_h1 = _osgs_E[3],
                err_p_h1 = _osgs_E[4],
                pi_u_l2 = _pi_u_l2,
                pi_p_l2 = _pi_p_l2,
            ))

            overall_converged = _decide_osgs_convergence(
                x_diff, x_diff_inf, pi_u_drift, pi_p_drift,
                osgs_iter, stab_cfg, osgs_tol, stagnation_tol)

            println("        * Comparing limits against previous relaxation step: \n          x_diff_inf = $x_diff_inf | x_diff_mode_$(stab_cfg.osgs_state_drift_scale) = $x_diff \n          pi_u_drift = $pi_u_drift | pi_p_drift = $pi_p_drift \n          overall_converged = $overall_converged")

            if overall_converged
                pi_u = pi_u_next
                pi_p = pi_p_next

                if !get(diag_cache, "base_convergence_reached", false)
                    println("        [+] OSGS Base Staggered Fixed-Point nominally mathematically converged natively!")
                    diag_cache["base_convergence_reached"] = true
                    # [trajectory] outer-iteration index at which the OSGS staggered loop first
                    # base-converged; outer iters AFTER this are MMS plateau-verification re-solves.
                    diag_cache["base_convergence_outer_iter"] = osgs_iter

                    if !isnothing(mms_cfg) && mms_cfg.enabled
                        diag_cache["mms_error_history"] = []
                        diag_cache["mms_relative_change_history"] = []
                        diag_cache["mms_consecutive_passes"] = 0

                        E_u2_0, E_p2_0, E_u1_0, E_p1_0 = mms_cfg.oracle(final_x0...)
                        push!(diag_cache["mms_error_history"], (E_u2_0, E_p2_0, E_u1_0, E_p1_0))
                    end
                else
                    if !isnothing(mms_cfg) && mms_cfg.enabled
                        E_u2_k, E_p2_k, E_u1_k, E_p1_k = mms_cfg.oracle(final_x0...)
                        mms_err_hist = diag_cache["mms_error_history"]
                        push!(mms_err_hist, (E_u2_k, E_p2_k, E_u1_k, E_p1_k))

                        E_u2_prev, E_p2_prev, E_u1_prev, E_p1_prev = mms_err_hist[end-1]

                        # §5.1: h-scaled plateau floors (see ASGS site for full comment).
                        eps_u_l2_eff = mms_cfg.eps_u_l2 * mms_cfg.h_local^(mms_cfg.kv + 1)
                        eps_u_h1_eff = mms_cfg.eps_u_h1 * mms_cfg.h_local^mms_cfg.kv
                        eps_p_l2_eff = mms_cfg.eps_p_l2 * mms_cfg.h_local^(mms_cfg.kv + 1)

                        r_u2 = abs(E_u2_k - E_u2_prev) / max(E_u2_k, E_u2_prev, eps_u_l2_eff)
                        r_u1 = abs(E_u1_k - E_u1_prev) / max(E_u1_k, E_u1_prev, eps_u_h1_eff)
                        r_p2 = abs(E_p2_k - E_p2_prev) / max(E_p2_k, E_p2_prev, eps_p_l2_eff)

                        mms_rc_hist = diag_cache["mms_relative_change_history"]
                        push!(mms_rc_hist, (r_u2, r_p2, r_u1))
                        max_r = max(r_u2, r_u1, r_p2)

                        println("        * [Verification Cycle] Plateau max ratio: $max_r (Target: $(mms_cfg.tau_err))")

                        pass_count = diag_cache["mms_consecutive_passes"]
                        if max_r < mms_cfg.tau_err
                            pass_count += 1
                        else
                            pass_count = 0
                        end
                        diag_cache["mms_consecutive_passes"] = pass_count

                        # [P2] Machine-floor plateau short-circuit. Once the state drift has reached the
                        # TIGHT machine floor (x_diff ≤ osgs_tol, NOT the looser max(osgs_tol, stagnation_tol)),
                        # U_h is at a fixed point, so the MMS error (a deterministic function of U_h) provably
                        # cannot change on the next cycle — re-confirming the plateau `require_consecutive_passes`
                        # times is wasted work. Accept a single passing check instead. This does NOT loosen any
                        # tolerance (max_r < tau_err must still hold this cycle); it recognizes a provably-static
                        # quantity. Guard rails: opt-in (default off); state_drift mode only (in
                        # projection_drift/both the projection may still be moving); tight osgs_tol only (a
                        # stagnation_tol-level x_diff may be stalled, not converged). Hard cells (e.g. Re=1,Da=1e6)
                        # never reach x_diff ≤ osgs_tol, so they are untouched.
                        eff_required_passes = mms_cfg.require_consecutive_passes
                        if sol_cfg.osgs_plateau_machine_floor_shortcut &&
                           stab_cfg.osgs_stopping_mode == "state_drift" && x_diff <= osgs_tol
                            eff_required_passes = 1
                            if pass_count >= 1 && pass_count < mms_cfg.require_consecutive_passes
                                println("        [P2] Machine-floor plateau short-circuit: state frozen (x_diff=$x_diff ≤ osgs_tol=$osgs_tol); accepting plateau after $pass_count pass(es) instead of $(mms_cfg.require_consecutive_passes).")
                            end
                        end

                        if pass_count >= eff_required_passes
                            println("        [+] OSGS MMS Plateau formally established natively ($pass_count consecutive < $(mms_cfg.tau_err)).")
                            diag_cache["mms_plateau_reached"] = true
                            diag_cache["mms_stop_reason"] = "mms_plateau_satisfied"

                            # §5.2: rate-aware sanity check (see ASGS site for full comment).
                            budget = mms_cfg.h_local^(mms_cfg.kv + 1)
                            if E_u2_k > mms_cfg.rate_check_factor * budget
                                diag_cache["mms_stop_reason"] = "mms_plateau_at_suboptimal_rate"
                                println("        [!] Plateau at sub-optimal rate: E_u2=$E_u2_k > $(mms_cfg.rate_check_factor) × h^(kv+1)=$(mms_cfg.rate_check_factor * budget). Flagged for inspection.")
                            end

                            success = true
                            break
                        end
                    end
                end

                if isnothing(mms_cfg) || !mms_cfg.enabled
                    println("        [+] OSGS Staggered Fixed-Point loop successfully converged structurally within mathematical boundaries!")
                    diag_cache["mms_stop_reason"] = "base_convergence_only"
                    success = true
                    break
                end
            else
                if !isnothing(mms_cfg) && mms_cfg.enabled
                    diag_cache["mms_consecutive_passes"] = 0
                end
            end

            pi_u = pi_u_next
            pi_p = pi_p_next
        end
    end

    # Post-loop MMS budget exhaustion check (M9 deviation: kept inside H7 rather than in
    # solve_system, so Algorithm C's terminal logic stays in one place).
    # Runs outside @elapsed, so it is excluded from `eval_time`.
    if !isnothing(mms_cfg) && mms_cfg.enabled
        if get(diag_cache, "base_convergence_reached", false) && !get(diag_cache, "mms_plateau_reached", false)
            println("        [!] OSGS hit maximum extended cycle boundaries without formal plateau verification. Assuming base convergence.")
            diag_cache["mms_stop_reason"] = "mms_budget_exhausted"
            success = true
        end
    end

    diag_cache["pi_u"] = pi_u
    diag_cache["pi_p"] = pi_p

    return success, pi_u, pi_p, elapsed
end

"""
    solve_system(...)

Algorithm O (SimulationOrchestration) of `theory/osgs_algorithm.tex`. Orchestrates
the nonlinear solver iteration loops over a defined Variational Multiscale space:
Stage I ASGS algebraic initialisation → method dispatch (ASGS w/ optional MMS
plateau extension, or OSGS staggered relaxation).

Delegates to seven file-local helpers — see the module-level "Algorithm-to-code
mapping" docstring at the top of this file, and `docs/solver/algorithm-code-mapping.md`.
"""
function solve_system(setup::FETopology, formulation::VMSFormulation, iter_solvers::IterativeSolvers,
                      config::PorousNSConfig, x0;
                      diagnostics_cache=nothing, mms_cfg=nothing)

    X, Y, model, dΩ, Ω = setup.X, setup.Y, setup.model, setup.dΩ, setup.Ω
    h_cf, f_cf, alpha_cf, g_cf = setup.h_cf, setup.f_cf, setup.alpha_cf, setup.g_cf
    V_free, Q_free = setup.V_free, setup.Q_free

    form, c_1, c_2 = formulation.form, formulation.c_1, formulation.c_2
    solver_picard, solver_newton = iter_solvers.picard, iter_solvers.newton

    phys_cfg = config.physical_properties
    stab_cfg = config.numerical_method.stabilization
    sol_cfg  = config.numerical_method.solver
    diag_cache = isnothing(diagnostics_cache) ? Dict{String, Any}() : diagnostics_cache
    pi_u = nothing
    pi_p = nothing


    local final_x0 = x0
    local success = false
    local eval_time = 0.0
    iter_count_ref = Ref(0)

    check_porous_solver_parameters(stab_cfg, sol_cfg)

    # Primitive unpacking (Critical for Gridap performance)
    method = stab_cfg.method
    max_osgs_iters = stab_cfg.osgs_iterations
    osgs_tol = stab_cfg.osgs_tolerance
    ftol = solver_newton.nls.ftol
    stagnation_tol = solver_newton.nls.stagnation_noise_floor
    tau_reg_lim = phys_cfg.tau_regularization_limit
    freeze_cusp = sol_cfg.freeze_jacobian_cusp

    # ==============================================================================
    # ALGEBRAIC INITIALIZATION BOUNDS
    # Even for OSGS, we universally boot up the global PDE using the zero-projection
    # (ASGS) formulation to securely map the non-linear fields into the exact Newton
    # quadratic basin before fragmenting the state via staggered projections.
    # ==============================================================================
    res_fn_init(x, y) = build_stabilized_weak_form_residual(x, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing)
    jac_picard_init(x, dx, y) = build_picard_jacobian(x, dx, y, setup, formulation, phys_cfg; pi_u=nothing, pi_p=nothing, mult_mom=1.0, mult_mass=1.0)
    jac_newton_init(x, dx, y) = build_stabilized_weak_form_jacobian(x, dx, y, setup, formulation, phys_cfg, freeze_cusp, ExactNewtonMode(); pi_u=nothing, pi_p=nothing)

    op_picard_init = FEOperator(res_fn_init, jac_picard_init, X, Y)
    op_newton_init = FEOperator(res_fn_init, jac_newton_init, X, Y)

    base_nls_global = solver_newton.nls
    # Verbatim re-wrap of the configured Newton solver (no field overrides) —
    # the explicit FESolver(_with_overrides(base_nls_global)) form expresses
    # "use the configured solver verbatim" without depending on `solver_newton`'s
    # FESolver wrapper identity.
    solver_newton_asgs = FESolver(_with_overrides(base_nls_global))

    x0_backup = copy(get_free_dof_values(x0))

    # [P4] Ping-pong scheduling (opt-in; default off ⇒ bit-identical one-way cascade). When enabled, the
    # Stage-I Picard segment uses a gain-then-return target so it stops the moment it re-enters Newton's
    # basin. The gain target is applied here (single point) so build_iter_solvers / call sites need not.
    pingpong_enabled = sol_cfg.pingpong_enabled
    pingpong_max_swaps = sol_cfg.pingpong_max_swaps
    solver_picard_gain = pingpong_enabled ?
        FESolver(_with_overrides(solver_picard.nls; picard_gain_target=sol_cfg.pingpong_picard_gain_orders)) :
        solver_picard

    # Stage I — Algorithm B (RobustNonlinearCascade), ASGS operators, restore=x0_backup.
    eval_time = @elapsed begin
        success = _initialize_asgs_state!(x0, x0_backup, op_newton_init, op_picard_init,
                                          solver_newton_asgs, solver_picard, ftol,
                                          diag_cache, iter_count_ref;
                                          pingpong_enabled=pingpong_enabled,
                                          pingpong_max_swaps=pingpong_max_swaps,
                                          solver_picard_gain=solver_picard_gain)
    end

    # `mms_plateau_success::Union{Bool,Nothing}` is `nothing` when MMS verification is
    # disabled; otherwise it reports whether the MMS plateau was formally established
    # (independent of the inner-solver `success` flag). Splitting these allows callers
    # to detect the "solver converged but MMS budget exhausted" regression case
    # explicitly. See plan Fix 6 / P-007.
    mms_enabled = !isnothing(mms_cfg) && mms_cfg.enabled
    _mms_plateau() = mms_enabled ? get(diag_cache, "mms_plateau_reached", false) : nothing

    if !success
        diag_cache["pi_u"] = pi_u
        diag_cache["pi_p"] = pi_p
        return false, _mms_plateau(), x0, iter_count_ref[], eval_time
    end

    final_x0 = x0

    if method == "ASGS"
        if !isnothing(mms_cfg) && mms_cfg.enabled
            eval_time += _run_asgs_mms_extension!(final_x0, op_newton_init, solver_newton,
                                                  mms_cfg, ftol, diag_cache, iter_count_ref)
            diag_cache["pi_u"] = pi_u
            diag_cache["pi_p"] = pi_p
            return success, _mms_plateau(), final_x0, iter_count_ref[], eval_time
        else
            println("\n      [+] ASGS Formulation exclusively resolved. Exiting solver module.")
            diag_cache["base_convergence_reached"] = true
            diag_cache["mms_stop_reason"] = "base_convergence_only"
            diag_cache["pi_u"] = pi_u
            diag_cache["pi_p"] = pi_p
            return success, _mms_plateau(), final_x0, iter_count_ref[], eval_time
        end
    end

    if method == "OSGS"
        # ==============================================================================
        # ORTHOGONAL SUBGRID SCALE (OSGS)
        # Iteratively tracks and orthogonalizes the sub-grid unresolvable scales against
        # the finite element basis explicitly by isolating $(I - \\Pi_{h})(\\mathcal{R})$.
        # In this formulation, the space for the SGS is taken as $\\mathcal{X}_{h0}^{\\perp}$.
        # Solving the globally-coupled, monolithic nonlinear system with the projection operations
        # is computationally prohibitive. Instead, we solve the OSGS system in a staggered, iterative
        # scheme: the global FE residual projection ($\\boldsymbol{\\pi}_h$) from the previous
        # iteration is held fixed during the underlying nonlinear Newton homotopy to compute
        # the updated primary fields ($U_h$), which are then used to re-project the continuous
        # residual until discrete equilibrium ($x_{diff} < osgs_{tol}$) is achieved.
        # ==============================================================================
        println("\n      [+] Commencing Orthogonal Subgrid Scale (OSGS) fixed-point recursive relaxation loop (allocating for max iterations: $max_osgs_iters)...")

        # Gridap L2 Projections inherit boundary conditions from the FESpace they map onto.
        # Stabilizing the sub-grid field natively without requiring full DG looping. We enforce
        # completely unconstrained spaces explicitly to prevent physical wall velocities from crushing
        # orthogonal projection boundary convergence rates.
        U, P = X
        V, Q = Y

        V_proj = V_free !== nothing ? V_free : V
        Q_proj = Q_free !== nothing ? Q_free : Q
        U_proj = TrialFESpace(V_proj)
        P_proj = TrialFESpace(Q_proj)

        # --- Precompute and Cache L2 Mass Matrices ---
        # The left-hand side mass-matrices for the L2 projections remain functionally invariant
        # across the non-linear OSGS scheme, allowing us to factorize them exactly once.
        println("\n      [+] Initiating pre-assembly & structural factorization for static OSGS L2 Mass Matrices...")
        M_u = assemble_matrix((u,v) -> ∫(v ⋅ u)dΩ, U_proj, V_proj)
        M_p = assemble_matrix((p,q) -> ∫(q * p)dΩ, P_proj, Q_proj)

        # The L² mass matrices are symmetric positive-definite by construction. Cholesky
        # via CHOLMOD is the mathematically honest factorization here: it's faster than
        # the previous `LUSolver()` (no partial pivoting overhead), uses less memory,
        # and preserves the symmetry of operations downstream. The factor is reused
        # across every OSGS outer iteration and every MMS-plateau re-projection.
        ls_u = CholeskySolver()
        ls_p = CholeskySolver()
        num_u_fac = numerical_setup(symbolic_setup(ls_u, M_u), M_u)
        num_p_fac = numerical_setup(symbolic_setup(ls_p, M_p), M_p)

        # OSGS staggered relaxation — Algorithm C (OSGSFractionalRelaxation).
        # Stage I's `success` (always `true` here — we short-circuit returned earlier
        # on Stage I failure) is threaded through so that the original semantics
        # "exhaust OSGS budget without explicit convergence/failure ⇒ keep Stage I success"
        # are preserved (see H7 docstring).
        (success, pi_u, pi_p, osgs_elapsed) = _run_osgs_relaxation!(
            success, final_x0, X, Y, V_free, Q_free, setup, formulation, phys_cfg,
            stab_cfg, sol_cfg, solver_newton, ftol, stagnation_tol,
            freeze_cusp, mms_cfg,
            M_u, M_p, num_u_fac, num_p_fac,
            U_proj, V_proj, P_proj, Q_proj,
            diag_cache, iter_count_ref)

        eval_time += osgs_elapsed

        # H7 already wrote diag_cache["pi_u"] and diag_cache["pi_p"].
        return success, _mms_plateau(), final_x0, iter_count_ref[], eval_time
    end
end
