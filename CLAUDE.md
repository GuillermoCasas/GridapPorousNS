# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this codebase is

A `Gridap.jl` Finite Element solver for the stabilized Darcy-Brinkman-Forchheimer / Porous Navier-Stokes equations. The implementation is a literal transcription of continuous **Variational Multiscale (VMS)** mathematics, supporting both **ASGS** (Algebraic Subgrid Scale) and **OSGS** (Orthogonal Subgrid Scale) stabilizations. The driving correctness criterion is convergence under the Method of Manufactured Solutions across `(Re, Da, h)` sweeps.

The code is structured around a paper-faithful core (`src/formulations/`, `src/stabilization/`) plus a safeguarded nonlinear solver (`src/solvers/`) that orchestrates Exact-Newton ↔ Picard fallbacks, line search, and homotopy.

## Commands

### Tests (three tiers, by runtime budget)

```bash
# BLITZ (< 5s per file): residual/Jacobian/tau/projection/MMS exactness unit checks.
julia -O0 -t 1 test/run_blitz_tests.jl

# QUICK (5s - 2m per file): formulation smoke + interpolation/projection + OSGS orthogonality.
julia --project=. test/run_quick_tests.jl

# EXTENDED (> 2m): utilities and long-horizon validation.
julia --project=. test/run_extended_tests.jl

# Full suite (all three tiers):
julia --project=. test/runtests.jl
```

The runners emit a warning when a file's runtime crosses the next tier's bound — that is a signal to move the file, not to ignore.

### Parameter sweeps / benchmarks

```bash
# Method of Manufactured Solutions sweep over (Re, Da, h).
cd test/extended/ManufacturedSolutions
julia --project=../../.. run_test.jl test_config.json
# Run any sub-combination of factors without authoring a new config:
julia --project=../../.. run_test.jl test_config.json --filter Re=1e6,etype=QUAD,kv=1
# Multiple concurrent launches share ONE results DB (content-addressed keys + file lock):
for k in 1 2 3 4; do julia --project=../../.. run_test.jl test_config.json --shard $k/4 & done; wait
python analyze_results.py --h5 results/<db>.h5 --config data/test_config.json   # post-sweep analysis
# (See test/extended/ManufacturedSolutions/README.md for the full harness guide.)

# Cocquet reference experiment (convergence study; compares VMS P1/P1 & P2/P2 against the
# unstabilized Galerkin Taylor-Hood P2/P1 "Cocquet" method via the comparison_runs config).
cd test/extended/CocquetExperiment
julia --project=../../.. run_convergence.jl paper_comparison.json
python plot_convergence.py
```

The sibling directories `test/extended/Cocquet{Alpha1,Deviatoric,LinearReaction,AllDirichlet}` are
single-variable diagnostic flips off this benchmark (see `docs/cocquet/convergence-analysis.md`);
`test/extended/CocquetFormMMS` is the manufactured-solution sibling. The exact Cocquet (unstabilized
Galerkin) formulation is documented in `theory/cocquet/cocquet_formulation.tex`.

### Single simulation from a JSON config

```julia
using PorousNSSolver
run_simulation("config/base_config.json")
```

`config/base_config.json` is the canonical example; `config/porous_ns.schema.json` defines the full schema.

## Architecture

`src/PorousNSSolver.jl` is the entry module; the include order there is the authoritative dependency graph (models → formulations → stabilization → solvers → IO → driver).

### Pipeline (`src/run_simulation.jl`)

1. `load_frozen_config(path)` — parses a self-contained JSON into strongly-typed configs (`PhysicalProperties`, `NumericalMethodConfig`, etc.). **No silent defaults**: missing numerical fields are configuration errors, not opportunities to backfill (see "Hard rules" below). (`load_config_with_base_template` is the test-harness variant that merges partial overrides onto `base_config.json`.)
2. `_build_default_mesh` — `CartesianDiscreteModel` (QUAD or simplexified TRI) with conventional tag IDs: inlet=7, outlet=8, walls=1..6.
3. `build_fe_spaces` — Lagrangian Taylor–Hood–style `(kv, kp)` pair, wrapped in a `MultiFieldFESpace` for the monolithic `(u, p)` system.
4. `build_formulation` — assembles a `PaperGeneralFormulation` from a viscous operator (`DeviatoricSymmetric` | `SymmetricGradient` | `LaplacianPseudoTraction`), a reaction law (`ForchheimerErgunLaw` | `ConstantSigmaLaw`), a projection policy, and a `SmoothVelocityFloor` regularization.
5. `solve_system` — drives the nested solver in `src/solvers/`, then `export_results` writes VTK/HDF5.

### Key invariants (do not break casually)

- **ASGS ≠ OSGS.** ASGS uses identity projection (zero projection state). OSGS requires an iterative `L²` projection of the strong residual and a staggered outer fixed-point on top of the inner Newton. Don't reuse one branch's assumptions in the other.
- **`ExactNewtonMode` vs `PicardMode` are mathematical contracts, not perf toggles.** Exact Newton includes `∂τ/∂u` and `∂Π/∂u` derivative terms; Picard freezes them. Never relabel a Picard simplification as Exact Newton, and never silently drop ExactNewton terms.
- **Adjoint sign in stabilization.** The convective adjoint `L*_conv(v) = -a·∇v` must be paired with `+a·∇v` weighting in the stabilization term. Flipping this produces anti-SUPG / anti-diffusion and explodes coercivity at parameter extrema.
- **Viscous operator.** `DeviatoricSymmetricViscosity` (`∇·(2μ∇^s u)`) is the canonical operator. The Laplacian / pseudo-traction variants are legacy and must stay explicitly labeled; do not silently substitute them.
- **OSGS stopping.** State drift uses a continuous `L²` functional (`∫(e_u·e_u)dΩ`), not raw `ℓ∞` on interleaved DOF arrays — the latter corrupts velocity/pressure scaling. Distinguish *state drift* from *projection drift*; conflating them masks regressions.
- **MMS plateau verification.** Multi-pass `r_max ≤ τ_error` across refinements. Weakening plateau criteria to pass a flaky sweep is forbidden.

See [docs/lessons_learned.md](docs/lessons_learned.md) for the append-only ledger of past regressions and the canonical fixes. Read it before touching `src/formulations/`, `src/stabilization/`, or `src/solvers/`.

Documentation is indexed in two places: [docs/README.md](docs/README.md) (investigation & status docs — MMS and Cocquet clusters) and [theory/README.md](theory/README.md) (LaTeX sources plus the canonical paper↔code references). Each topic has one canonical doc; the rest carry status headers pointing to it.

### Theory anchors

The implementation is a literal transcription of [theory/paper/article.tex](theory/paper/article.tex) — *"A stabilized finite element method for incompressible, inertial flows in inhomogeneous porous media"* (Casas, González-Usúa, Codina, de-Pouplana). When in doubt, the paper is authoritative.

- **Continuous problem** — strong form, momentum + mass: `eq:StrongMomentumEquation`, `eq:StrongMassEquation`. The reaction tensor `σ(α, u)` is symmetric positive semi-definite; the Forchheimer form is `σ = a(α) + b(α)|u|` (`eq:DBFResistanceTerm`).
- **Stabilized OSGS system** — `eq:OSGSProblem` (Eqs. 4.10a–d). ASGS is recovered by setting `π_h = 0`. The staggered linearized iteration is `eq:LinearizedOSGSProblem`; the pseudocode driving `porous_solver.jl` is `alg:StationarySystem`.
- **Stabilization parameters** — `τ₁` (`eq:Tau1`), `τ₂` (`eq:Tau2`), with `τ_{1,NS}` from `eq:TauNavierStokes`. The numerical constants `c₁ = 4k⁴`, `c₂ = 2k²` (paper Remark after `eq:conditions_on_num_param`) are what `get_c1_c2` returns for equal-order interpolation.
- **Reaction projection trim** — Section 4.4 mentions that for constant `σ` the reaction term is omitted from the orthogonal projection (its `L²` projection is exactly zero on the FE space). This is what `ProjectResidualWithoutReactionWhenConstantSigma` implements when `experimental_reaction_mode == "standard"` and `reaction_model == "Constant_Sigma"`.
- **Documented divergences from the paper** — [docs/solver/paper-code-divergences.md](docs/solver/paper-code-divergences.md) catalogues each apparent code/paper mismatch and classifies it. Highlights worth knowing:
  - The `(1/α)∇·(αa)v` term is intentionally absent from `convective_adjoint` (paper Sec. 5, line ~800 — kept out to preserve the `A² − B²` symmetry in the stability estimate).
  - `convective_adjoint` returns `+α a·∇v` (positive sign), because the stabilization bilinear form subtracts the adjoint (`B_S` definition under `eq:OSGSProblem`). Flipping the sign in code reproduces the "Anti-SUPG" failure.
  - OSGS projection is computed on **unconstrained** spaces `V_free/Q_free` (no Dirichlet) — projecting on the Dirichlet-constrained space introduces an `O(1)` boundary residual that breaks `O(h^{k+1})` MMS convergence.

### Solver safeguards (`src/solvers/`)

`SafeNewtonSolver` (in `nonlinear.jl`) implements: Armijo merit-function line search, divergence/stagnation guards keyed on `stagnation_noise_floor` and `divergence_merit_factor`, and bounded `max_increases`. The orchestration in `porous_solver.jl` chains Exact Newton → Picard globalization → homotopy dilution. These safeguards are intentional design, not clutter — do not weaken them in pursuit of speed.

`dynamic_picard_re_threshold` / `dynamic_picard_da_threshold` automatically widen Picard's iteration budget at high Re / Da. `dynamic_ftol_ceiling` couples the residual tolerance to mesh resolution `O(h^{kv+1})` so the linear solve doesn't oversolve relative to discretization error.

### Documentation ontology

The codebase tags design choices explicitly. Preserve the labels when editing:

- `[paper-faithful]` — strict map of continuous analytical math.
- `[code-actual]` — numerical reality (safeguards, floors, noise-floors).
- `[debugging-lore]` — workarounds for Gridap AST / JIT issues.
- `[legacy]` — kept for regression comparison; never present as canonical.
- `[known-fragility]` — sign / threshold invariants that break under algebraic rearrangement.
- `[must-test]` — assertions tied to MMS exactness.

## Hard rules

### No magic numbers, no implicit defaults

This is enforced at the cultural level (`.agents/rules/no-hard-coded-parameters.md`). Every tolerance, threshold, damping factor, floor, iteration cap, line-search constant, and noise floor must:

1. Live in `config/porous_ns.schema.json` and the corresponding `Base.@kwdef struct` in `src/config.jl`.
2. Be threaded explicitly through the call chain (no closure-local fallbacks, no `tol = something(cfg.tol, 1e-8)`).
3. Fail loudly on missing input — `load_frozen_config` must not invent values.

If you need a new numerical control, add it to the schema, the config struct, the JSON, and the consuming function — in that order. Do not introduce inline literals like `1e-8`, `0.5`, `100` in solver/formulation code unless they are mathematically universal (e.g. `2` in `2μ∇^s u`).

### Verification gate

Per `.agents/rules/fast-verification.md`: after editing anything in `src/formulations/`, `src/stabilization/tau.jl`, `src/models/reaction.jl`, or `src/solvers/nonlinear.jl`, run Blitz immediately. For changes to assembly, residual/Jacobian construction, or solver orchestration, run Quick after Blitz. For convergence-study or MMS-touching changes, also run Extended. A change is not complete until the relevant tiers pass with no failures and no tier-warning messages.

### Test file naming

Suffix test files with `_test.jl` (e.g. `tau_blitz_test.jl`), not prefix.

## Repository conventions

- Julia project: `Project.toml` pinned to `Gridap 0.18.6`. `Manifest.toml` is gitignored; resolve locally with `Pkg.instantiate()`.
- Outputs (VTK, HDF5, sweep results) go under `results/` directories that are gitignored.
- `theory/` holds **only the LaTeX sources** now: `theory/paper/` (the SIAM article + its build dependencies, incl. `theory/paper/siam/` for the class/bst), `theory/cocquet/` (the Cocquet-et-al. formulation notes + reference PDF), plus `osgs_algorithm.tex` and `centered_encoding.tex`. All meta-documentation — observations, to-dos, process notes, and the paper↔code references — lives under `docs/` (organized by topic: `docs/solver/`, `docs/cocquet/`, `docs/mms/`, `docs/paper/`). See `docs/README.md` and `theory/README.md` for the indexes. When implementation diverges from the paper, update `docs/solver/paper-code-divergences.md`.
- `.agents/skills/` contains per-domain skill prompts (regression guards, doc architect, config strictness). They formalize the review checks above.
