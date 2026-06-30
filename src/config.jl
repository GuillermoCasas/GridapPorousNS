#=
src/config.jl

Strongly-typed mirror of the JSON config schema (`config/porous_ns.schema.json`) plus the loaders
that parse a JSON file into these structs and validate it. The struct tree here is the in-memory
shape every downstream stage consumes (mesh build, FE spaces, formulation, solver). JSON3 + StructTypes
do the field-by-field deserialization; `validate!` enforces the physical/numerical invariants the
solver relies on.

Policy: NO silent defaults. `load_frozen_config` (the production path) demands every numerical field be
present in the file — a missing field is a configuration error, not something to backfill. The template
merging in `load_config_with_base_template` / `load_config_from_dict` is a test-harness convenience only.
=#

using JSON3
using StructTypes

"""
Physical inputs to the porous Navier-Stokes problem.

- `nu`               kinematic viscosity ν (> 0); sets the viscous/Reynolds scale.
- `f_x`, `f_y`       components of the prescribed body force f.
- `eps_val`          porosity ε of the medium (> 0); appears in the inertial/reaction scaling.
- `reaction_model`   selects the reaction law σ(α,u): "Constant_Sigma" or the Forchheimer/Ergun form.
- `sigma_constant`   scalar resistance σ for the "Constant_Sigma" reaction model (unused by Forchheimer-Ergun).
- `sigma_linear`     Forchheimer-Ergun coefficient of the linear/viscous Darcy term a(α) = sigma_linear·((1-α)/α)².
- `sigma_nonlinear`  Forchheimer-Ergun coefficient of the inertial term b(α) = sigma_nonlinear·(1-α)/α (the |u| coefficient).
- `u_base_floor_ref`, `h_floor_weight`, `epsilon_floor`  parameters of the SmoothVelocityFloor
  regularization: a small lower bound on |u| so |u|-dependent reaction and τ terms stay finite as u→0.
- `velocity_magnitude_derivative_floor`  additive floor ε_d on |u| in the Exact-Newton derivative terms
  (dσ/du, dτ/du), regularizing ∂|u|/∂u = u/|u| at a stagnation point (|u|=0). Jacobian-only — zeroed in
  Picard and absent from the residual — so it cannot move the converged solution. See
  theory/velocity_floor_regularization/.
- `tau_regularization_limit`  cap that keeps the stabilization parameters τ₁/τ₂ from blowing up in
  the vanishing-velocity / vanishing-resistance limit.
"""
Base.@kwdef struct PhysicalProperties
    nu::Float64
    f_x::Float64
    f_y::Float64
    eps_val::Float64            # PHYSICAL compressibility ε_phys: enters BOTH residual and Jacobian (mass LHS
                                # and the manufactured source); may be 0.
    numerical_epsilon::Float64  # NUMERICAL penalty ε_num (Codina iterative penalty). Lagging ε_num·p to the
                                # iterate cancels it in the RESIDUAL and leaves ε_num·dp ONLY in the JACOBIAN's
                                # pressure block — a Newton-step regularization that vanishes at convergence
                                # (no consistency error, no outer loop). Bounded: ε_num ≤ C₂·c₁·inf{α²τ₁/h²}
                                # (article.tex eq:UpperBoundOnEpsilon) or stability is lost. 0 ⇒ off.
    reaction_model::String
    sigma_constant::Float64
    sigma_linear::Float64
    sigma_nonlinear::Float64
    u_base_floor_ref::Float64
    h_floor_weight::Float64
    epsilon_floor::Float64
    velocity_magnitude_derivative_floor::Float64   # ε_d: additive floor on |u| regularizing ∂|u|/∂u = u/|u|
                                                   # in the Exact-Newton dσ/du, dτ/du terms. Jacobian-only
                                                   # (zeroed in Picard, absent from the residual) ⇒ cannot
                                                   # move the converged solution. See
                                                   # theory/velocity_floor_regularization/.
    tau_regularization_limit::Float64
end

"""
Geometry / inhomogeneity description of the porous domain.

- `alpha_0`        baseline value of the inverse-permeability field α (drives the reaction tensor σ(α)).
- `r_1`, `r_2`     radii delimiting the porous-inclusion region where α varies (inner/outer extent).
- `bounding_box`   `[xmin, xmax, ymin, ymax, ...]` extents of the Cartesian domain meshed by Gridap.
"""
Base.@kwdef struct DomainConfig
    alpha_0::Float64
    r_1::Float64
    r_2::Float64
    bounding_box::Vector{Float64}
end

"""
Polynomial orders of the Taylor-Hood-style velocity/pressure FE pair.

- `k_velocity`  Lagrangian order of the velocity space (the `kv` in the solver / τ constants).
- `k_pressure`  Lagrangian order of the pressure space.

Equal order (`k_velocity == k_pressure`) is the VMS-stabilized case the τ constants c₁=4k⁴, c₂=2k²
are tuned for; the stabilization is what makes that inf-sup-unstable pairing well-posed.
"""
Base.@kwdef struct ElementSpacesConfig
    k_velocity::Int
    k_pressure::Int
end

"""
Choice of subgrid-scale stabilization.

`method` is either:
- "ASGS" — Algebraic Subgrid Scale: identity projection, π_h = 0 (paper §4, recover by dropping the
  orthogonal projection).
- "OSGS" — Orthogonal Subgrid Scale: a single coupled solve where π = Π(R(u)) (the L² projection of the
  strong residual) is re-projected at every residual evaluation, with a frozen-π sparse-local Jacobian
  — see src/solvers/osgs_solver.jl. The coupled OSGS solve runs at the shared Newton budget/ftol; it
  has no separate iteration or tolerance knob.
"""
Base.@kwdef struct StabilizationConfig
    method::String
end

"""
Mesh generation and the refinement ladder for convergence studies.

- `partition`               cells per axis for a single run, e.g. `[16, 16]` → 16×16 Cartesian grid.
- `convergence_partitions`  the sequence of per-axis refinements an MMS/convergence sweep marches
  through to measure the O(h^{k+1}) rate.
- `element_type`            "QUAD" (Cartesian quads) or "TRI" (simplexified triangles).
"""
Base.@kwdef struct MeshConfig
    partition::Vector{Int}
    convergence_partitions::Vector{Int}
    element_type::String
end

"""
Linear (inner) solver backend for the monolithic (u, p) Jacobian solve of every Newton/Picard iterate.
The CHOICE of backend does not change the converged solution (both solve `J Δx = −R` to tolerance); it
trades robustness/speed against peak memory:

- `method = "LU"`        — Gridap's sparse direct `LUSolver` (one backsolve per Newton step; exact, the
                           previously-hardcoded default, but its 3D fill-in can exhaust RAM on fine meshes).
- `method = "ILU_GMRES"` — restarted GMRES left-preconditioned by an incomplete-LU factor
                           (`ILUGMRESSolver`); far lower peak memory for large 3D systems at the cost of
                           inner Krylov iterations.

The `ilu_*`/`gmres_*` fields are the ILU drop tolerance and GMRES restart/rel-tol/iteration cap; they are
consumed only when `method == "ILU_GMRES"` but are ALWAYS required (no silent default), per the
no-implicit-defaults rule.

`allow_unpreconditioned_fallback` is the [C.1] honesty policy for the ILU_GMRES path: when the ILU
factorization fails, `false` (the safe default) makes the linear solve fail loudly (so the nonlinear
cascade rolls back / falls to Picard rather than silently running a near-hopeless unpreconditioned solve),
while `true` lets GMRES attempt the solve UNPRECONDITIONED with a loud warning. Either way a GMRES that
does not reach `gmres_rel_tol` within `gmres_maxiter` is reported as a FAILED step, never accepted as
exact — see `src/solvers/linear_solvers.jl` and docs/formulation-audit-2026-06-24.md §C.1.
"""
Base.@kwdef struct LinearSolverConfig
    method::String                 # "LU" (direct, exact) or "ILU_GMRES" (low-memory iterative)
    ilu_drop_tolerance::Float64    # ILU drop tolerance τ (smaller ⇒ denser/stronger preconditioner)
    gmres_restart::Int             # GMRES Krylov subspace size before restart
    gmres_rel_tol::Float64         # relative residual tolerance for GMRES convergence
    gmres_maxiter::Int             # cap on GMRES iterations
    allow_unpreconditioned_fallback::Bool  # [C.1] ILU-failure policy: false ⇒ fail loud; true ⇒ identity Pl + warn
end

"""
All knobs for the safeguarded nonlinear solver in `src/solvers/`.

The solver orchestrates Exact-Newton ↔ Picard fallbacks, an Armijo line search, and homotopy. Fields
below are grouped by role: iteration budgets, convergence tolerances, line-search / robustness guards,
and diagnostics. (The regime-adaptive Re/Da iteration budgets are a HARNESS-frame feature — see
test/extended/harness_dynamic_budget.jl — not part of the production solver.)
"""
Base.@kwdef struct SolverConfig
    # --- Iteration budgets ---
    # [harness-frame] The regime-adaptive Re/Da budget knobs (`dynamic_picard_*` / `dynamic_newton_*` /
    # `dynamic_ftol_*`) were RELOCATED out of this production struct: nothing in `src/` ever read them —
    # they are consumed only by the MMS/Cocquet harnesses, which now own the defaults in
    # test/extended/harness_dynamic_budget.jl. Re/Da are GLOBAL dimensionless numbers (they need
    # characteristic scales U, L that only a benchmark fixes), so they are test-frame quantities; a
    # production adaptive budget would instead key on an element-wise cell-Péclet |u|h_K/ν (already
    # embodied by τ_{1,NS}). See docs/formulation-audit-2026-06-24.md §A.1.
    picard_iterations::Int                         # base Picard sweep count (frozen-coefficient globalizer)
    newton_iterations::Int                         # base Exact-Newton iteration cap
    # --- Convergence tolerances ---
    ftol::Float64                                  # residual-norm tolerance ‖R‖ for declaring convergence
    picard_ftol::Float64                           # absolute ‖R‖ ftol for the Picard solver (looser than Newton's; the ping-pong handoff itself fires via pingpong_picard_gain_orders)
    xtol::Float64                                  # step-size tolerance ‖Δu‖ for stagnation/convergence
    # --- Scale-free convergence gate (convergence_criterion.jl; the authoritative success test) ---
    # The nonlinear solve stops when two DIMENSIONLESS, segment-independent measures fall below these:
    # ε_M = ‖r_M‖/D_M (momentum residual over the force-magnitude envelope) and
    # ε_C = ‖εp+∇·(αu)−g‖/(‖∇(αu)‖+‖g‖) (mass residual over a flux-gradient+source envelope; g = mass
    # source, so it → 0 even for a forced/manufactured problem). Unlike the re-anchored relative ftol, D_M is a PHYSICAL
    # scale read from the current iterate, so the gate is IDENTICAL across ping-pong segments — which is
    # what removes the re-anchoring pathology (a fresh segment no longer demands another ×(1/ftol) drop
    # below wherever the previous one stopped). See docs/solver/nonlinear-convergence-criterion-prompt.md.
    eps_tol_momentum::Float64                      # tol_M: ε_M ≤ tol_M (ε_M ∈ [0,1], → 0 at the discrete solution)
    # tol_C: ε_C ≤ tol_C (ε_C floors at O(h^{kv}); set above that floor, ≤ √d).
    # [known-looseness] The 3D k=2 sweeps run this at 0.8 — deliberately loose, for the mass-residual
    # scale sensitivity documented in the k=2-gate lesson, but it accepts an iterate whose mass residual
    # is up to 80% of the flux-gradient envelope and is the ONLY gate on the continuity balance (tol_M is
    # tight at 1e-9). Flagged for future tightening / a separate pure-divergence (‖∇·(αu)‖/‖∇(αu)‖ ≤ √d)
    # check; not changed here. See docs/formulation-audit-2026-06-24.md §C.3.
    eps_tol_mass::Float64
    # --- Line search & robustness guards ---
    max_increases::Int                             # how many merit-function increases are tolerated before bailing
    freeze_jacobian_cusp::Bool                     # hold the Jacobian fixed across an ill-conditioned cusp step
    armijo_c1::Float64                             # Armijo sufficient-decrease constant c₁ ∈ (0,1)
    divergence_merit_factor::Float64               # merit-ratio (≥1) above which a step is judged divergent
    stagnation_noise_floor::Float64                # ‖R‖ floor below which further "progress" is treated as noise
    condition_noise_floor_baseline::Float64        # baseline for the conditioning-aware noise floor
    condition_noise_floor_absolute_min::Float64    # hard lower bound for that noise floor
    condition_noise_floor_safety_factor::Float64   # multiplier inflating the noise floor for safety margin
    noise_floor_success_max_ftol_multiple::Float64 # honest-exit gate: accept noise-floor success only within this × ftol (large/Inf disables)
    linesearch_alpha_min::Float64                  # smallest step length α the line search will try
    max_linesearch_iterations::Int                 # cap on backtracking steps per line search
    linesearch_contraction_factor::Float64         # α-shrink ratio per backtrack ∈ (0,1)
    # --- Diagnostics / experimental modes ---
    run_diagnostics::Bool                          # emit extra solver diagnostics
    ablation_mode::String                          # selects a term-ablation variant for studies
    experimental_reaction_mode::String             # "standard" enables the constant-σ reaction projection trim (paper §4.4)
    # [iterator-scheduling] These knobs default OFF/inert: with the base_config.json values the cascade
    # runs Newton→Picard→Newton with no extra stall/ping-pong logic; a user opts in by setting them.
    # P1 — no-progress stall guard (engine in nonlinear.jl `_safe_solve_inner!`):
    newton_stall_window::Int                       # window of iterations checked for stall; 0 ⇒ disabled (Newton runs its full budget)
    newton_stall_min_rel_improvement::Float64      # relative ‖R‖∞ drop counting as "progress" within the stall window
    # P4 — adaptive Newton↔Picard ping-pong (orchestrator-level, asgs_solver.jl):
    pingpong_enabled::Bool                         # false ⇒ one-way Newton→Picard→Newton cascade
    pingpong_max_swaps::Int                        # hard cap on Newton↔Picard alternations
    pingpong_picard_gain_orders::Float64           # orders of magnitude ‖R‖∞ must drop in Picard before returning to Newton
    # [residual-divergence guard] Consecutive ‖R‖∞ increases (beyond divergence_merit_factor) that abort a
    # Newton solve → Picard. 0 ⇒ disabled (Newton runs its full budget even while diverging).
    newton_residual_divergence_patience::Int
    # --- OSGS Anderson acceleration (osgs_solver.jl) ---
    # OFF by default. When disabled the OSGS coupled solve is the existing single inexact-Newton path
    # (bit-identical). When enabled, the OSGS stage runs a staggered outer fixed-point (freeze π, solve,
    # re-project) with Anderson mixing of the (u,p) iterate, which the inexact-Newton's dropped ∂π/∂u
    # makes only linearly convergent. See src/solvers/accelerators.jl (NONL-03).
    osgs_anderson_enabled::Bool                    # opt in to the Anderson-accelerated staggered OSGS outer loop
    osgs_anderson_depth::Int                       # m: Anderson history depth (how many past differences are mixed)
    osgs_anderson_relaxation::Float64              # β: relaxation/mixing factor on the fixed-point residual
    osgs_anderson_safety_factor::Float64           # Powell-style restart threshold (too-large extrapolation → reset history)
    osgs_anderson_max_outer::Int                   # cap on the staggered outer iterations
    # --- OSGS JFNK (Jacobian-Free Newton-Krylov) recovery of the dropped ∂π/∂u coupling (osgs_solver.jl) ---
    # OFF by default. When disabled the OSGS coupled solve is the existing single inexact-Newton path
    # (bit-identical); mutually exclusive with osgs_anderson_enabled (validate! rejects both on). When
    # enabled, the coupled Newton step solves the FULL tangent matrix-free: J_full·v ≈ [F(U+εv)−F(U)]/ε
    # (the residual re-projects π, so the FD captures ∂π/∂u), via inner GMRES preconditioned by the
    # already-assembled+factored frozen-π Jacobian. Phase-0 gate (docs/solver/jfnk-phase0-preconditioner-gate.md)
    # measured this preconditioner viable (1–21 inner iters); the frozen-π preconditioner uses the
    # config's numerical_epsilon, which Phase-0 recommends keeping 0 for JFNK (a nonzero ε_num pulls the
    # preconditioner off the residual tangent and degrades the Krylov count).
    osgs_jfnk_enabled::Bool                        # opt in to matrix-free full-tangent (∂π/∂u) inner GMRES
    osgs_jfnk_gmres_rel_tol::Float64               # η: inner-GMRES forcing tolerance (inexact-Newton; ~1e-2)
    osgs_jfnk_gmres_maxiter::Int                   # cap on inner-GMRES iterations per Newton step
    osgs_jfnk_gmres_restart::Int                   # GMRES restart (Krylov subspace size before restart)
    osgs_jfnk_fd_epsilon::Float64                  # Brown–Saad FD base b in ε = b·(1+‖U‖)/‖v‖ (~√eps ≈ 1e-8)
    # --- Codina iterative-penalty method (article.tex §5.2 line ~1383, codina1993iterative) ---
    # OFF by default (bit-identical: the mass residual carries no ε term, only the legacy Jacobian ε_num·dp).
    # When enabled AND numerical_epsilon > 0, the mass-equation residual carries the iterative penalty
    # ε_num·(pⁿ − pⁿ⁻¹) — LHS ε_num·pⁿ plus the lagged RHS ε_num·pⁿ⁻¹ — solved as an OUTER fixed-point loop
    # that HOLDS pⁿ⁻¹ fixed within each pass and updates it between passes. It pins the constant-pressure null
    # mode (well-posedness; the 3D all-Dirichlet case is ill-posed at ε=0, article.tex line 1375) and vanishes
    # at convergence (pⁿ=pⁿ⁻¹), so the converged solution is NOT altered. The matching ε_num·dp is already in
    # the Jacobian, so residual+Jacobian stay consistent. Outer-loop drift tolerance reuses `xtol`.
    iterative_penalty_enabled::Bool
    iterative_penalty_max_iters::Int               # cap on the outer iterative-penalty passes
    # --- OSGS: skip the ASGS Stage-I boot (osgs_solver.jl / solver_core.jl) ---
    # OFF by default (the boot runs, bit-identical legacy). The ASGS boot is a code-side globalization
    # safeguard, NOT part of the paper algorithm (alg:StationarySystem runs the OSGS staggered iteration
    # directly from the initial guess). For a warm/exact guess it converges ASGS to the ASGS root first — a
    # DIFFERENT fixed point — from which the OSGS solve can be badly conditioned (the 3D overshoot). When the
    # eps_pert homotopy provides the cold-start globalization the boot was for, skipping it (OSGS solves
    # straight from the guess) is paper-faithful. No effect on ASGS-method runs.
    osgs_skip_asgs_boot::Bool
    # --- Linear (inner) solver backend ---
    linear_solver::LinearSolverConfig              # LU (direct, exact) vs ILU_GMRES (low-memory iterative)
end

"""
Everything about the discretization and solve, bundled together.

- `element_spaces`        FE polynomial orders (the Taylor-Hood-style pair).
- `stabilization`         ASGS vs OSGS selection.
- `solver`                nonlinear-solver budgets, tolerances, and guards.
- `mesh`                  partition/refinement ladder and element type.
- `viscous_operator_type` viscous term form: "DeviatoricSymmetric" (canonical, ∇·(2μ∇^s u)),
  "SymmetricGradient", or "Laplacian" (the latter two are legacy variants).
"""
Base.@kwdef struct NumericalMethodConfig
    element_spaces::ElementSpacesConfig
    stabilization::StabilizationConfig
    solver::SolverConfig
    mesh::MeshConfig
    viscous_operator_type::String
end

"""
Where results are written.

- `directory`  output folder for VTK/HDF5 artifacts.
- `basename`   filename stem for the exported fields.
"""
Base.@kwdef struct OutputConfig
    directory::String
    basename::String
end

"""
Root config object: the complete, self-contained description of one simulation. This is what the loaders
return and what `src/run_simulation.jl` consumes end to end.
"""
Base.@kwdef struct PorousNSConfig
    physical_properties::PhysicalProperties
    domain::DomainConfig
    numerical_method::NumericalMethodConfig
    output::OutputConfig
end

# Register every config struct with StructTypes so JSON3 can map JSON objects onto them field-by-field
# (field names must match the JSON keys exactly).
# StructTypes definitions
StructTypes.StructType(::Type{PhysicalProperties}) = StructTypes.Struct()
StructTypes.StructType(::Type{DomainConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{ElementSpacesConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{StabilizationConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{MeshConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{LinearSolverConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{SolverConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{NumericalMethodConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{OutputConfig}) = StructTypes.Struct()
StructTypes.StructType(::Type{PorousNSConfig}) = StructTypes.Struct()

"""
    validate!(cfg::PorousNSConfig)

Assert the physical and numerical invariants the solver depends on, then return `cfg` unchanged. Each
`@assert` carries the precise contract it guards (e.g. ν > 0, Armijo c₁ ∈ (0,1), an OSGS/ASGS-only
method string). Several checks encode ordering relationships between knobs — e.g. `picard_ftol`
must be ≥ `ftol` because Picard is a smoother handing off to Newton, not a precise solver. Raises on the
first violated invariant.
"""
function validate!(cfg::PorousNSConfig)
    # Physical
    @assert cfg.physical_properties.nu > 0 "Kinematic viscosity 'nu' must be > 0"
    # eps_val is now the PHYSICAL compressibility ε_phys — it may be 0 (no physical compressibility); the
    # NUMERICAL penalty ε_num provides stability separately. Both must be nonnegative; well-posedness for an
    # all-Dirichlet incompressible problem needs ε_phys + ε_num > 0 (or a pinned pressure), left to the caller.
    @assert cfg.physical_properties.eps_val >= 0 "eps_val (physical ε) must be >= 0"
    @assert cfg.physical_properties.numerical_epsilon >= 0 "numerical_epsilon (penalty ε) must be >= 0"
    # Reaction coefficients must be nonnegative so σ stays symmetric positive-semidefinite (the paper's
    # standing assumption, eq:DBFResistanceTerm): a(α)=σ_linear·((1-α)/α)² ≥ 0 and b(α)=σ_nonlinear·(1-α)/α
    # ≥ 0 for α ∈ (0,1], and the constant law σ ≡ σ_constant. A negative coefficient makes σ an energy
    # source rather than a sink and breaks coercivity.
    @assert cfg.physical_properties.sigma_constant >= 0 "sigma_constant must be >= 0 (σ SPSD)"
    @assert cfg.physical_properties.sigma_linear >= 0 "sigma_linear must be >= 0 (σ SPSD)"
    @assert cfg.physical_properties.sigma_nonlinear >= 0 "sigma_nonlinear must be >= 0 (σ SPSD)"
    # Velocity-magnitude floor (SmoothVelocityFloor): the floored speed sqrt(u·u + u_floor²) is C∞ only
    # when u_floor > 0, which the Newton differentiability of the |u|-dependent reaction/τ terms requires.
    # With h_floor_weight = 0 (the MMS/Cocquet setting) the floor rests entirely on u_base_floor_ref, so
    # demand at least one of them strictly positive; epsilon_floor is the denominator guard and must be > 0.
    @assert cfg.physical_properties.u_base_floor_ref >= 0 "u_base_floor_ref must be >= 0"
    @assert cfg.physical_properties.h_floor_weight >= 0 "h_floor_weight must be >= 0"
    @assert cfg.physical_properties.epsilon_floor > 0 "epsilon_floor must be > 0 (velocity-floor denominator guard)"
    @assert cfg.physical_properties.velocity_magnitude_derivative_floor > 0 "velocity_magnitude_derivative_floor must be > 0 (regularizes u/|u| in the Exact-Newton dσ/du, dτ/du terms)"
    @assert cfg.physical_properties.u_base_floor_ref > 0 || cfg.physical_properties.h_floor_weight > 0 "velocity floor must be strictly positive (u_base_floor_ref > 0 or h_floor_weight > 0) so SmoothVelocityFloor stays C¹"

    # Domain / porosity field (previously unvalidated): α ∈ (0,1] keeps σ finite and the MMS u_ex = α⁻¹·S
    # bounded; the radial transition needs r_1 < r_2; bounding_box is [xmin,xmax,ymin,ymax,(zmin,zmax)].
    dom = cfg.domain
    @assert 0.0 < dom.alpha_0 <= 1.0 "domain.alpha_0 (porosity) must be in (0, 1]"
    @assert dom.r_1 < dom.r_2 "domain.r_1 must be < domain.r_2 (porosity transition annulus)"
    @assert length(dom.bounding_box) >= 4 && iseven(length(dom.bounding_box)) "domain.bounding_box must have an even length >= 4 ([xmin,xmax,ymin,ymax,...])"

    # Solver
    sol = cfg.numerical_method.solver
    @assert sol.ftol > 0 "Solver ftol must be > 0"
    @assert sol.picard_ftol >= sol.ftol "Solver picard_ftol must be >= ftol (Picard is a smoother, not a precise solver)"
    @assert sol.xtol > 0 "Solver xtol must be > 0"
    # Scale-free convergence gate (the AUTHORITATIVE production success test, injected by solve_system):
    # converged ⇔ ε_M ≤ eps_tol_momentum ∧ ε_C ≤ eps_tol_mass. Both must be strictly positive.
    @assert sol.eps_tol_momentum > 0 "Solver eps_tol_momentum (ε_M gate) must be > 0"
    @assert sol.eps_tol_mass > 0 "Solver eps_tol_mass (ε_C gate) must be > 0"
    @assert 0.0 < sol.armijo_c1 < 1.0 "Armijo c1 must be strictly between 0 and 1"
    @assert sol.divergence_merit_factor >= 1.0 "Divergence merit factor must be >= 1.0"
    @assert sol.noise_floor_success_max_ftol_multiple >= 1.0 "noise_floor_success_max_ftol_multiple must be >= 1.0 (use a large value / Inf to disable the honest-exit gate)"
    # [iterator-scheduling] opt-in scheduling knobs; the "0 disables …" wording mirrors their inert default
    @assert sol.newton_stall_window >= 0 "newton_stall_window must be >= 0 (0 disables the no-progress stall guard)"
    @assert 0.0 <= sol.newton_stall_min_rel_improvement < 1.0 "newton_stall_min_rel_improvement must be in [0, 1)"
    @assert sol.newton_residual_divergence_patience >= 0 "newton_residual_divergence_patience must be >= 0 (0 disables the residual-divergence → Picard handoff)"
    @assert sol.pingpong_max_swaps >= 0 "pingpong_max_swaps must be >= 0 (0 disables Newton↔Picard ping-pong)"
    @assert sol.pingpong_picard_gain_orders > 0.0 "pingpong_picard_gain_orders must be > 0"
    @assert sol.newton_iterations >= 1 "Newton iterations must be >= 1"
    @assert sol.max_linesearch_iterations >= 1 "Linesearch iterations must be strictly bounded >= 1"
    @assert 0.0 < sol.linesearch_contraction_factor < 1.0 "Linesearch contraction map alpha must strictly be in (0, 1)"
    @assert sol.osgs_anderson_depth >= 1 "osgs_anderson_depth must be >= 1"
    @assert sol.osgs_anderson_relaxation > 0.0 "osgs_anderson_relaxation must be > 0"
    @assert sol.osgs_anderson_safety_factor > 0.0 "osgs_anderson_safety_factor must be > 0"
    @assert sol.osgs_anderson_max_outer >= 1 "osgs_anderson_max_outer must be >= 1"
    # OSGS JFNK knobs (required even when disabled — no silent default). The two opt-in OSGS paths are
    # mutually exclusive: both on is a configuration error, not a precedence rule to guess.
    @assert !(sol.osgs_jfnk_enabled && sol.osgs_anderson_enabled) "osgs_jfnk_enabled and osgs_anderson_enabled are mutually exclusive OSGS paths; enable at most one"
    @assert 0.0 < sol.osgs_jfnk_gmres_rel_tol < 1.0 "osgs_jfnk_gmres_rel_tol (inner-GMRES forcing η) must be in (0, 1)"
    @assert sol.osgs_jfnk_gmres_maxiter >= 1 "osgs_jfnk_gmres_maxiter must be >= 1"
    @assert sol.osgs_jfnk_gmres_restart >= 1 "osgs_jfnk_gmres_restart must be >= 1"
    @assert sol.osgs_jfnk_fd_epsilon > 0.0 "osgs_jfnk_fd_epsilon (Brown–Saad FD base) must be > 0"
    @assert sol.iterative_penalty_max_iters >= 1 "iterative_penalty_max_iters must be >= 1"

    # Linear (inner) solver backend — the ilu_*/gmres_* knobs are required even for LU (no silent default),
    # but only constrained when ILU_GMRES is actually selected.
    lsc = sol.linear_solver
    @assert lsc.method in ("LU", "ILU_GMRES") "linear_solver.method must be \"LU\" or \"ILU_GMRES\""
    if lsc.method == "ILU_GMRES"
        @assert lsc.ilu_drop_tolerance > 0 "linear_solver.ilu_drop_tolerance must be > 0 when method=ILU_GMRES"
        @assert lsc.gmres_restart >= 1 "linear_solver.gmres_restart must be >= 1 when method=ILU_GMRES"
        @assert lsc.gmres_rel_tol > 0 "linear_solver.gmres_rel_tol must be > 0 when method=ILU_GMRES"
        @assert lsc.gmres_maxiter >= 1 "linear_solver.gmres_maxiter must be >= 1 when method=ILU_GMRES"
    end

    # Stabilization
    stab = cfg.numerical_method.stabilization
    @assert stab.method in ("ASGS", "OSGS") "Stabilization method must be ASGS or OSGS"

    # Formulation Operator validation
    @assert cfg.numerical_method.viscous_operator_type in ("DeviatoricSymmetric", "SymmetricGradient", "Laplacian") "viscous_operator_type strictly expects DeviatoricSymmetric, SymmetricGradient, or Laplacian"
    
    return cfg
end

# Warn about any JSON key in `dict` that does not correspond to a field of struct type `T`. Catches
# typos before they are silently dropped by deserialization. `$schema` at the root is whitelisted.
function _check_unknown_keys(T::Type, dict::AbstractDict, path::String="")
    allowed_keys = string.(fieldnames(T))
    for key in keys(dict)
        if path == "root" && key == "\$schema"
            continue
        end
        if !(key in allowed_keys)
            @warn "Unknown JSON config key '$key' at $path in struct $(T). Please check for typos."
        end
    end
end

# Walk the whole nested config dict, applying the unknown-key check at each level against the matching
# struct type, so a typo anywhere in the tree (not just the root) is reported with its path.
function _check_unknown_keys_hierarchical(dict::AbstractDict)
    _check_unknown_keys(PorousNSConfig, dict, "root")
    
    if haskey(dict, "physical_properties") && dict["physical_properties"] isa AbstractDict
        _check_unknown_keys(PhysicalProperties, dict["physical_properties"], "physical_properties")
    end
    if haskey(dict, "domain") && dict["domain"] isa AbstractDict
        _check_unknown_keys(DomainConfig, dict["domain"], "domain")
    end
    if haskey(dict, "numerical_method") && dict["numerical_method"] isa AbstractDict
        nm = dict["numerical_method"]
        _check_unknown_keys(NumericalMethodConfig, nm, "numerical_method")
        
        if haskey(nm, "element_spaces") && nm["element_spaces"] isa AbstractDict
            _check_unknown_keys(ElementSpacesConfig, nm["element_spaces"], "numerical_method.element_spaces")
        end
        if haskey(nm, "stabilization") && nm["stabilization"] isa AbstractDict
            _check_unknown_keys(StabilizationConfig, nm["stabilization"], "numerical_method.stabilization")
        end
        if haskey(nm, "solver") && nm["solver"] isa AbstractDict
            _check_unknown_keys(SolverConfig, nm["solver"], "numerical_method.solver")
            if haskey(nm["solver"], "linear_solver") && nm["solver"]["linear_solver"] isa AbstractDict
                _check_unknown_keys(LinearSolverConfig, nm["solver"]["linear_solver"], "numerical_method.solver.linear_solver")
            end
        end
        if haskey(nm, "mesh") && nm["mesh"] isa AbstractDict
            _check_unknown_keys(MeshConfig, nm["mesh"], "numerical_method.mesh")
        end
    end
    if haskey(dict, "output") && dict["output"] isa AbstractDict
        _check_unknown_keys(OutputConfig, dict["output"], "output")
    end
end

# Recursively merge `override` into `base` in place: nested dicts are merged key-by-key, any other value
# (including arrays) replaces the base value wholesale. Used by the template-inheriting loaders.
function deep_merge!(base::AbstractDict, override::AbstractDict)
    for (k, v) in override
        if haskey(base, k) && isa(base[k], AbstractDict) && isa(v, AbstractDict)
            deep_merge!(base[k], v)
        else
            base[k] = v
        end
    end
    return base
end

"""
    load_config_with_base_template(override_path::String="")

Test-harness convenience loader: deep-merges the user's JSON file onto `base_config.json` so a partial
config inherits any fields it omits, then checks keys, deserializes, and validates. **Not** the
production entry point — use `load_frozen_config` for self-contained configs where every numerical field
is intentionally supplied, since template inheritance can mask a missing field.
"""
function load_config_with_base_template(override_path::String="")
    base_config_path = joinpath(@__DIR__, "..", "config", "base_config.json")
    base_raw = read(base_config_path, String)
    base_dict = copy(JSON3.read(base_raw, Dict{String, Any}))

    if !isempty(override_path) && isfile(override_path)
        test_raw = read(override_path, String)
        test_dict = JSON3.read(test_raw, Dict{String, Any})
        deep_merge!(base_dict, test_dict)
    end

    _check_unknown_keys_hierarchical(base_dict)
    cfg = JSON3.read(JSON3.write(base_dict), PorousNSConfig)
    return validate!(cfg)
end

"""
    load_frozen_config(exact_path::String)

Strict production loader: parse the JSON at `exact_path` directly into a `PorousNSConfig` with NO template
inheritance. Every field must be present in the file — a missing numerical field surfaces as a
deserialization/validation error rather than being silently backfilled. Checks for unknown keys, then
validates.
"""
function load_frozen_config(exact_path::String)
    if !isfile(exact_path)
        error("Config file not found: $exact_path")
    end
    raw = read(exact_path, String)
    dict = JSON3.read(raw, Dict{String, Any})
    _check_unknown_keys_hierarchical(dict)
    cfg = JSON3.read(raw, PorousNSConfig)
    return validate!(cfg)
end

"""
    load_config_from_dict(override::AbstractDict)

Programmatic counterpart to `load_config_with_base_template`: merge an in-memory `override` dict onto
`base_config.json` (rather than a file), then check keys, deserialize, and validate. Convenient for
sweeps that build configs on the fly.
"""
function load_config_from_dict(override::AbstractDict)
    base_config_path = joinpath(@__DIR__, "..", "config", "base_config.json")
    base_raw = read(base_config_path, String)
    base_dict = copy(JSON3.read(base_raw, Dict{String, Any}))
    
    deep_merge!(base_dict, override)
    _check_unknown_keys_hierarchical(base_dict)
    cfg = JSON3.read(JSON3.write(base_dict), PorousNSConfig)
    return validate!(cfg)
end

"""
    load_config(path::String="")

Production single-file entry point. A thin alias for `load_frozen_config`: strict, with no silent
inheritance from `base_config.json`. Callers that need template inheritance (e.g. the MMS sweep harness)
should call `load_config_with_base_template` or `load_config_from_dict` explicitly.
"""
function load_config(path::String="")
    return load_frozen_config(path)
end
