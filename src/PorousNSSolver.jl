#=
Entry module for the stabilized Darcy-Brinkman-Forchheimer / porous Navier-Stokes
solver. The `include` order below IS the dependency graph: each layer may only use
names defined by layers above it (models → formulations → stabilization → solvers →
metrics/problems → IO → driver). Reordering can break parse-time name resolution.
=#
module PorousNSSolver

using Gridap
using GridapGmsh
using JSON3
using StructTypes
using WriteVTK
using HDF5

# Core: strongly-typed configuration structs + the JSON loader. No silent defaults;
# every numerical control is an explicit field. Everything downstream consumes these.
include("config.jl")

# Geometry: the configurable element characteristic size h(K) that feeds the stabilization τ₁/τ₂.
include("geometry.jl")

# Models: the physical laws of the porous medium.
#   porosity        — the porosity/permeability field α(x) and its derived quantities.
#   regularization  — the SmoothVelocityFloor that keeps |u| away from 0 in σ(α,u).
#   reaction        — the resistance tensor σ(α,u): Forchheimer-Ergun and constant-σ laws.
include("models/porosity.jl")
include("models/regularization.jl")
include("models/reaction.jl")

# Formulations: the continuous VMS weak form, expressed in Gridap terms.
#   viscous_operators  — the viscous term ∇·(2μ∇^s u): DeviatoricSymmetric (canonical),
#                        SymmetricGradient, LaplacianPseudoTraction (legacy variants).
#   projection         — the L² projection π_h of the strong residual onto the FE space
#                        (the orthogonal-subscale machinery; identity for ASGS).
#   continuous_problem — assembles the PaperGeneralFormulation from the above pieces.
include("formulations/viscous_operators.jl")
include("stabilization/projection.jl")
include("formulations/continuous_problem.jl")

# Stabilization parameters τ₁ (eq:Tau1) and τ₂ (eq:Tau2), with constants c₁, c₂ from
# get_c1_c2; these scale the subgrid contribution element-by-element.
include("stabilization/tau.jl")

# Scale-free outer-iteration stopping criterion (eps_M term-envelope + eps_C flux-gradient ratio),
# kept separate from the iteration machinery. Depends only on the weak forms above. solve_system wires
# it in UNCONDITIONALLY as the AUTHORITATIVE success gate (build_convergence_probe → both stage solvers),
# so it is the production convergence test, not merely a diagnostic. See
# docs/solver/nonlinear-convergence-criterion-prompt.md.
include("solvers/convergence_criterion.jl")

# Solvers: the nested nonlinear solve and its orchestration.
#   linear_solvers — the inner linear backends (e.g. ILU-preconditioned GMRES).
#   nonlinear      — SafeNewtonSolver: Armijo line search + divergence/stagnation guards.
#   accelerators   — Anderson acceleration for the slow OSGS coupled fixed-point.
include("solvers/linear_solvers.jl")
include("solvers/nonlinear.jl")
include("solvers/accelerators.jl")
# Shared core + top-level orchestrator. Defines FETopology / VMSFormulation (the data
# bundles the method files name in their signatures), the SolutionVerifier seam,
# CascadePolicy / _pingpong_cascade! (Exact-Newton ↔ Picard ↔ homotopy fallbacks), and
# solve_system. MUST precede the method files: they resolve these names at parse time.
include("solvers/solver_core.jl")
# ASGS branch: Stage-I bootstrap (_initialize_asgs_state!) and STAGE_I_* policies.
# ASGS uses identity projection (π_h = 0). MUST follow solver_core.jl.
include("solvers/asgs_solver.jl")
# OSGS branch: the L² residual projections, solve_osgs_stage! (the coupled solve — π_h is
# re-projected at each Newton evaluation), and OSGS_INNER_POLICY. MUST follow solver_core.jl.
include("solvers/osgs_solver.jl")
# MMS plateau verifier (Algorithm D): checks r_max ≤ τ_error across refinements via the
# SolutionVerifier seam, so it MUST come after solver_core.jl.
include("solvers/mms_verification.jl")
# Error norms and reference comparisons used by the MMS/Cocquet convergence studies.
include("metrics.jl")

# Problems and IO.
#   mms_paper_2d   — the 2D manufactured solution from the paper (exact u, p, forcing).
#   io             — VTK/HDF5 export of the computed fields.
#   run_simulation — the end-to-end pipeline: load config → mesh → spaces → formulation
#                    → solve → export.
include("problems/mms_paper_2d.jl")
include("io.jl")
include("run_simulation.jl")

export run_simulation
export load_config
export PorousNSConfig
export element_size_field, element_size_convention, ELEMENT_SIZE_CONVENTIONS
export compute_reference_errors
export compute_reference_errors_multimask
export compute_trial_projection_errors
export compute_mode_decomposition
export compute_corner_excluded_norm
export ILUGMRESSolver
export ILUFactorizationFailure, GMRESNotConvergedError   # [C.1] typed linear-solve failures (honest non-convergence)
export CholeskySolver
export LinearSolverConfig
export instantiate_linear_solver   # config (LinearSolverConfig) -> concrete LU | ILU_GMRES backend
export AndersonAccelerator   # accelerator for the slow OSGS coupled fixed-point

end # module
