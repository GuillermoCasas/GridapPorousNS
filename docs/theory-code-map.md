# Theory ↔ Code Map

The stable reference for how this `Gridap.jl` solver realizes the continuous VMS mathematics of
[`theory/paper/article.tex`](../theory/paper/article.tex) — *"A stabilized finite element method for
incompressible, inertial flows in inhomogeneous porous media"* (Casas, González-Usúa, Codina,
de-Pouplana) — plus the algorithm boxes of
[`theory/osgs_algorithm/osgs_algorithm.tex`](../theory/osgs_algorithm/osgs_algorithm.tex).

When implementation appears to diverge from the paper, this doc is the ledger that classifies the
divergence and justifies it. Update it whenever code and paper drift.

This doc consolidates four former solver docs (paper-code divergences ledger, algorithm↔code mapping,
the scale-free convergence-criterion spec, and the normalization/encoding-invariance audit). The
`[paper-faithful]` / `[code-actual]` / `[code-divergent-superior]` / `[deferred]` labels and the
`eq:`/`article.tex` references are preserved throughout. Detailed investigation evidence lives under
[`docs/`](README.md); the LaTeX sources live under [`theory/`](../theory/).

**Contents**

1. [Algorithm-box → code map](#1-algorithm-box--code-map)
2. [Divergence ledger (code vs paper)](#2-divergence-ledger-code-vs-paper)
3. [Scale-free nonlinear convergence criterion (spec)](#3-scale-free-nonlinear-convergence-criterion-spec)
4. [Normalization / encoding-invariance audit](#4-normalization--encoding-invariance-audit)

---

## 1. Algorithm-box → code map

Each named algorithm box in
[`osgs_algorithm.tex`](../theory/osgs_algorithm/osgs_algorithm.tex) maps **1:1** to a single Julia
function. After the shared-core extraction (commonalities vs per-method differences split), the solver
lives in four files:

- [`src/solvers/solver_core.jl`](../src/solvers/solver_core.jl) — the **shared core + orchestrator**:
  the FE containers (`FETopology` / `VMSFormulation` / `StageSolvers`), the `SolutionVerifier` seam, the
  shared cascade machinery (`CascadePolicy` type + `cascade_step_outcome` + `_pingpong_cascade!`), the
  FE-solve plumbing (`safe_fe_solve!` / `_solve_one_step!` / `_record_stage!`), and the orchestrator
  `solve_system`. This is the only solver file that names **both** methods (it dispatches between them).
- [`src/solvers/asgs_solver.jl`](../src/solvers/asgs_solver.jl) — **ASGS only**: the Stage-I boot
  `_initialize_asgs_state!` and the `STAGE_I_POLICY` / `STAGE_I_N2_POLICY` cascade-policy values.
- [`src/solvers/osgs_solver.jl`](../src/solvers/osgs_solver.jl) — **OSGS only**: the L²-projection
  helpers (`discrete_l2_projection`), the coupled OSGS solve (`solve_osgs_stage!`), and the
  `OSGS_INNER_POLICY` value.
- [`src/solvers/mms_verification.jl`](../src/solvers/mms_verification.jl) — the optional Algorithm-D
  MMS plateau verification (`MMSPlateauVerifier`), decoupled behind the `SolutionVerifier` seam.

The shared Newton kernel (Algorithm A) is [`src/solvers/nonlinear.jl`](../src/solvers/nonlinear.jl).
Helpers are file-local (leading-underscore convention) and not exported.

> Reviewers should re-check this table whenever the solver files are touched. (Pre-split, all of these
> lived in a single `src/solvers/porous_solver.jl`.)

| Algorithm / section | Code symbol | Code file |
|---|---|---|
| Algorithm O — `SimulationOrchestration` | `solve_system` | `solver_core.jl` |
| Algorithm A — `ExactNewtonPipeline` | `_safe_solve_inner!` (via `SafeNewtonSolver`) | `nonlinear.jl` |
| Algorithm B — `RobustNonlinearCascade` (Stage I) | `_initialize_asgs_state!` | `asgs_solver.jl` |
| Algorithm C — `CoupledOSGSSolve` (single Newton; per-eval re-projection of `π`; frozen-`π` Jacobian; Picard fallback gated on `pingpong_enabled`; `stall_window=0`) | `solve_osgs_stage!` | `osgs_solver.jl` |
| Algorithm D — `VerifyMMSPlateau` (ASGS branch) | `on_asgs_converged!(::MMSPlateauVerifier, …)` | `mms_verification.jl` |
| Algorithm D — `VerifyMMSPlateau` (OSGS branch) | `on_osgs_converged!(::MMSPlateauVerifier, …)` | `mms_verification.jl` |
| Scale-free convergence criterion — the **authoritative** outer-iteration success gate (`ε_M ≤ tol_M ∧ ε_C ≤ tol_C`, momentum/mass dimensionless residual measures) | `evaluate_convergence` (momentum envelope `momentum_force_envelope`, mass measure `mass_criterion`); `solve_system` injects it via `build_convergence_probe` into each solver's `conv_probe` | `convergence_criterion.jl` (probe in `solver_core.jl`) |

### The verification seam (Algorithm D)

Algorithm D is **not** inlined in the solver. The core (`solve_system` / `solve_osgs_stage!`) is
verification-blind: at each convergence point it invokes a hook on a `SolutionVerifier`
(`on_asgs_converged!` / `on_osgs_converged!`). Production passes `NoVerification` (multiple dispatch
resolves both hooks to no-ops), so it costs nothing and the core never names MMS, reads an oracle, or
writes an `mms_*` key. The MMS harnesses pass an `MMSPlateauVerifier`, which owns the
manufactured-solution oracle and the plateau loop.

The asymmetry in the two hook signatures (`on_asgs_converged!` takes a `step_once!` closure;
`on_osgs_converged!` does not) reflects the **real** asymmetry in the algorithm: the ASGS path plateaus
by extra single-Newton cycles, while the OSGS coupled solve is a single solve evaluated once.

### What is *not* in this mapping

- `safe_fe_solve!` (the try/catch + tuple-unwrap wrapper in `solver_core.jl`) and `_solve_one_step!`
  (the raw single-step variant the ASGS verifier drives) are Julia-side utilities, not paper boxes.
- The `SafeNewtonSolver` constructor / `_with_overrides` (`nonlinear.jl`) is pure plumbing.

### Iterator-scheduling helpers (no paper anchor — efficiency, default-off)

Scheduling/plumbing, not paper algorithm boxes; every behavioural switch defaults OFF so the shipped
config reproduces prior results bit-identically.

- `build_iter_solvers` (`nonlinear.jl`) — single construction point for the `(picard, newton)`
  `FESolver` pair, used by both `run_simulation.jl` (production) and `run_test.jl` (MMS harness).
- `cascade_step_outcome` / `CascadePolicy` (`solver_core.jl`) — the shared Algorithm-B accept/reject
  verdict (type + interpreter). Parameterized by one `CascadePolicy` value per method, each declared in
  its own method file: `STAGE_I_POLICY` / `STAGE_I_N2_POLICY` (`asgs_solver.jl`) and `OSGS_INNER_POLICY`
  (`osgs_solver.jl`), as `(accept_noise_floor, accept_soft_stall, max_iters_caught_is_failure)`. Truth
  tables pinned by `test/blitz/cascade_policy_symmetry_blitz_test.jl`.
- `_pingpong_cascade!` (`solver_core.jl`) — **opt-in** adaptive Newton↔Picard ping-pong (shared by the
  Stage-I boot and the OSGS coupled solve) that replaces the one-way Newton→Picard→Newton cascade when
  `pingpong_enabled`. Runs Newton until it stalls, a Picard segment that stops the moment `‖R‖∞` has
  dropped `pingpong_picard_gain_orders` orders (stop_reason `picard_gain_reached`), then back to Newton,
  bounded by `pingpong_max_swaps`. Each segment is tagged honestly via `_record_stage!`
  (`…:PP[swap]:N` / `:P`).

---

## 2. Divergence ledger (code vs paper)

The canonical map of every apparent mismatch between the literal mathematical theory in
[`theory/paper/article.tex`](../theory/paper/article.tex) and the concrete `Gridap.jl` execution. Any
discrepancy introduced by numerical-stability limits, algebraic bounds, discrete-compilation behaviors,
or Julia/LLVM restrictions MUST be recorded here.

### 2.1 Sub-grid mass-balancing approximation — `[paper-faithful]`

**Location**: `src/formulations/continuous_problem.jl` — `strong_adjoint_momentum`.

**Apparent divergence**: The strict integration-by-parts of the subgrid convective velocities forces the
exact test-side adjoint mapping to include the scalar compressibility term
`(1/α)∇·(α u) v`. In the code, `strong_adjoint_momentum` omits this term explicitly.

**Alignment**: Not a divergence. [`article.tex` line 800](../theory/paper/article.tex#L800) explicitly
justifies removing the `(1/α)∇·(α a) v_h` term across the whole theory:

> *"Note that, strictly speaking, one has the term `(1/α)∇·(α a) v_h` in the expansion of `L* V_h`. The
> inclusion of such term generates a number of crossed terms in (StabilityEstimate) that actually harm
> stability. [...] Here, we have opted for simplifying the formulation by removing the aforementioned
> term from `L* V_h`, leading to a simpler formulation with similar stability properties."*

The code omitting it is a literal transcription of the simplified VMS operator specified in the theory.
(Kept out to preserve the `A² − B²` symmetry in the stability estimate.)

### 2.2 Viscous operator and its formal adjoint — `[paper-faithful]`

**Location**: [`src/formulations/viscous_operators.jl`](../src/formulations/viscous_operators.jl).

**Paper theory**: The strong viscous operator and its formal adjoint `L*V` per
[`article.tex:479`](../theory/paper/article.tex#L479) both involve the divergence of the (deviatoric or
full symmetric) strain tensor:

- deviatoric: `∇·εᵈ(u) = ½Δu + (½ − 1/d)∇(∇·u)`
- symmetric-gradient: `∇·ε(u) = ½Δu + ½∇(∇·u)`

The `∇(∇·u)` contribution is dimension-dependent for the deviatoric variant (coefficient `0` in 2D,
`+1/6` in 3D) and always present for the symmetric-gradient variant.

**Code reality**: Both the strong operator and its formal adjoint use the same dimension-aware
Hessian-evaluation operations:

- [`strong_viscous_operator(::DeviatoricSymmetricViscosity, …)`](../src/formulations/viscous_operators.jl#L145) — `EvalDivDevSymOp(Δ(u), ∇∇(u))`
- [`adjoint_viscous_operator(::DeviatoricSymmetricViscosity, …)`](../src/formulations/viscous_operators.jl#L163) — `EvalDivDevSymOp(Δ(v), ∇∇(v))`
- [`strong_viscous_operator(::SymmetricGradientViscosity, …)`](../src/formulations/viscous_operators.jl#L93) — `EvalStrongViscSymOp(Δ(u), ∇∇(u))`
- [`adjoint_viscous_operator(::SymmetricGradientViscosity, …)`](../src/formulations/viscous_operators.jl#L111) — `EvalStrongViscSymOp(Δ(v), ∇∇(v))`

`EvalDivDevSymOp` and `EvalStrongViscSymOp` are dimension-dispatched callable structs (one method each for
`d=2` and `d=3`) computing the exact divergence-of-strain expansion. Gridap evaluates `∇∇(u)` for trial
and `∇∇(v)` for test fields cleanly on Lagrangian elements; for `k_v = 1` the Hessian is identically zero,
so the `∇(∇·)` contribution vanishes regardless of dimension — exactly as the analytic operator demands.

**Historical note (audit P-004)**: Before the Fix-4a commit, the adjoint dropped the `∇(∇·v)`
contribution and returned only `0.5·Δ(v)`. In 2D deviatoric this was harmless (`½ − 1/d = 0`); in 3D
deviatoric and in any dimension for symmetric-gradient it lost formal symmetry against `L*V`. The fix
re-uses the strong-operator machinery on `v`, so the adjoint is now exact in any dimension. The
orthogonality smoke test (`osgs_orthogonality_quick_test.jl`, `SymmetricGradientViscosity`) saw a small
numerical shift (OSGS L2 `3.39545818e-02 → 3.33665493e-02`); orthogonality `‖Π_h(R_h(u))‖ ≈ 10⁻¹⁴` was
unchanged. The MMS sweep (`DeviatoricSymmetricViscosity` per `base_config.json`) is bit-identical to the
post-`fa8aaec` baseline.

### 2.3 Adjoint streamline mapping positivity — `[paper-faithful]` `[known-fragility]`

**Location**: `src/formulations/continuous_problem.jl` — `strong_adjoint_momentum`.

**Apparent divergence**: Naively one expects `L*_conv = −α a·∇v`; the code evaluates `strong_adjoint_momentum`
with a **positive** sign (`+α a·∇v`). Reversing it triggers catastrophic divergence at high Reynolds
(the "Anti-SUPG" failure).

**Alignment**: Not a divergence, but a mandatory requirement of the stability proof's coercivity.
[`article.tex` Eq. 39 / line 554](../theory/paper/article.tex#L554) constructs the VMS stabilization
bilinear form by **subtracting** the adjoint: `− Σ_K ⟨L* U_h, τ L U_h⟩`. For the velocity test function,
`−L*_conv u_h = +α a·∇u_h = A`. Multiplied by the strong residual (same positive `A`) this forms the
`(A − B)·(A + B) = A² − B²` symmetry ([`article.tex` Eq. 50 / line 797](../theory/paper/article.tex#L797)),
giving the positive-definite bound `+‖τ₁^{1/2} α X(U_h)‖²_h`. So the positive code evaluation is
structurally identical to evaluating `−L*` inside the paper's VMS inner product.

### 2.4 Jacobian scalar-singularity regularizations — `[code-actual]`

**Location**: `src/models/reaction.jl`, `src/solvers/nonlinear.jl`.

**Paper theory**: Jacobian bounds over nonlinear-parameter expansions are treated as continuously
differentiable across local phase transitions.

**Code reality**: A numerical flooring coefficient injects a safe non-zero element bounded by `O(1e-15)`
(via `SmoothVelocityFloor`), governing the continuous Jacobian limit precisely where analytical
derivatives of `|u|` fracture as `|u| → 0`.

### 2.5 Simplified stabilization parameters (`eq:Tau1Final` / `eq:Tau2Final`) — `[paper-faithful]`

**Location**: [`src/stabilization/tau.jl`](../src/stabilization/tau.jl) — `compute_tau_1`, `compute_tau_2`.

**Apparent divergence**: Relative to the full paper definitions, the code drops:

1. The `ε h²` contribution from `τ₂`'s denominator (full: [`eq:Tau2`, article.tex:755](../theory/paper/article.tex#L755); simplified: [`eq:Tau2Final`, L778](../theory/paper/article.tex#L778)).
2. The porosity-gradient `(h/|k₀|)|∇α|` term inside `C_α` for `τ₁` (full: [`eq:Tau1`, article.tex:754](../theory/paper/article.tex#L754); simplified: [`eq:Tau1Final`, L777](../theory/paper/article.tex#L777)).

**Alignment**: Both omissions are explicitly justified in the paper:

- For `τ₂`, [`article.tex` L762](../theory/paper/article.tex#L762): *"The second term in `eq:Tau2` is only strictly necessary for large `ε`."*
- For `τ₁`, [`article.tex` L764–768](../theory/paper/article.tex#L764): *"the second term in the definition of `C_α` is in fact unnecessary if `(h/|k₀|)|∇α| ≲ α`. That is, if the porosity changes are well resolved by the mesh. We will assume this to hold in the following, leaving issues related to steep porosity gradients to future work."*

The simplified §4.2 analysis uses `eq:Tau1Final` / `eq:Tau2Final` directly
([`article.tex` L775](../theory/paper/article.tex#L775)); [`tau.jl:5–16`](../src/stabilization/tau.jl#L5)
implements those final forms verbatim. (For equal-order interpolation the numeric constants are
`c₁ = 4k⁴`, `c₂ = 2k²`, per the Remark after `eq:conditions_on_num_param`, returned by `get_c1_c2`.)

**Empirical verification of the well-resolved-porosity assumption**: every current MMS sweep config uses
`SmoothRadialPorosity` with `r₁=0.2, r₂=0.4` (transition width `Δr = 0.2`). Worst case
`h|∇α|/α ≈ h·(1−α₀)/Δr / α₀`: at the coarsest mesh (`h=0.1`, `α₀=0.5`) `≈ 0.5`; at the finest
(`h=0.003`, `α₀=0.5`) `≈ 0.015`. The Cocquet experiment uses `α ≡ 1` so `|∇α| = 0` trivially. The
assumption holds across the entire sweep.

**Regression anchor**: [`test/blitz/tau_blitz_test.jl`](../test/blitz/tau_blitz_test.jl) `@testset "Tau1/Tau2
simplified paper form is intentional [P-001, P-008]"` locks in both simplifications: (1) `compute_tau_2`
does not take `physical_epsilon` and matches the closed form `h²/(c₁ α τ_NS + reg)`; (2) `compute_tau_1`
is bit-identical when `med.grad_alpha` varies by orders of magnitude while local `α` is fixed. A future
audit re-raising **P-001** ("τ₂ missing ε·h²") or **P-008** ("τ₁ missing C_α") should reach this section
first and then fail the anchor test rather than file a regression.

**Re-evaluation triggers**:

- Switch to `eq:Tau2` (with `ε h²`) if a future config drives `ε` large enough that `ε h²` becomes
  comparable to `c₁ α τ_NS`. `physical_epsilon` already flows through `phys_cfg`; the switch is a local
  change in `compute_tau_2` and its derivative `compute_dtau_2_du`.
- Switch to `eq:Tau1` (with `C_α`) if a future config has `h|∇α|/α ≳ 1` (steep, under-resolved porosity
  gradient). `grad_alpha` is already plumbed through `MediumState`.

> **Related (not a ledger entry): the `4k⁴` = `c₁` constant for 3D structured tets.** The 3D-P2 MMS
> investigation **RESOLVED (2026-07-06)** that `4k⁴` is *under-margined* (not wrong) for high-`C_inv`
> structured tets — the remedy is an element-aware `c₁` per
> [`article.tex` line 910](../theory/paper/article.tex#L910), **not** a code change (paper `c₁` is correct).
> Full detail — the anti-coercive viscous subscale, the `C_inv²` table, the proofs — in
> [`docs/mms/p2-3d.md`](mms/p2-3d.md) §A.

### 2.6 OSGS preconditioning & linearization architecture — `[code-divergent-superior]`

**Location**: [`src/solvers/osgs_solver.jl`](../src/solvers/osgs_solver.jl) — `solve_osgs_stage!` +
`discrete_l2_projection`. The orchestrator (`solve_system` + the shared Newton→Picard→Newton cascade) is
[`src/solvers/solver_core.jl`](../src/solvers/solver_core.jl).

**Paper theory**: §6.2 (Eq. 107a–107c) presents a *staggered* iterative scheme for OSGS: an outer
fixed-point freezing the projection `π_h^{m−1}`, a Picard (Oseen) momentum solve
`B_S(u^{m−1}, U_h^m)`, then a `⟨W_h, π_h^m⟩` update. This implies two architectural boundaries: the
global momentum linearization and the orthogonal tracking subspace.

**Code reality**: The code does **not** implement the staggered outer loop. As of the 2026-06-08 leaning,
OSGS has a **single coupling mode** — the *coupled* solve — the only route to the OSGS fixed point. It
diverges from the staggered presentation in two ways, both reaching the same converged orthogonal
residual `R̃ = R − Π(R)`:

1. **Coupled single Newton solve (no staggering lag)**: Instead of freezing `π_h` across an outer
   relaxation loop, one Newton solve runs whose **residual re-projects `π_h = Π(R(u))` from the current
   iterate at every residual evaluation**. The Jacobian stays the **local frozen-`π` form** (sparse — not
   the prohibitive monolithic `∂π/∂u` tangent), so this is a Picard-type coupling on the projection while
   the rest of the tangent is the exact-Newton (`ExactNewtonMode()`) form. Removing the staggered freeze
   eliminates the lag that made the outer map contract only linearly; the per-evaluation re-projection is
   the cheap Cholesky-cached mass solve (`discrete_l2_projection`). Converges to the same OSGS fixed point,
   targeting roughly ASGS iteration counts. (The per-eval projection cost motivates the JFNK
   linear-convergence fix — see the JFNK canonical doc + `theory/osgs_algorithm/osgs_algorithm.tex`.)
2. **Topologically unconstrained orthogonal projection**: The projection is computed on the
   **unconstrained** spaces `V_free` / `Q_free` (no Dirichlet). Projecting on the Dirichlet-constrained
   space forces boundary nodes to mirror extreme Dirichlet velocities, annihilating the exact `L²`
   projection and introducing an unphysical `O(1)` boundary residual that breaks the `O(h^{k+1})` MMS
   convergence. The free-space projection protects optimal interpolation, mathematically and empirically.

### 2.7 OSGS pressure projection — constant-mode treatment — `[deferred]` (OPEN QUESTION)

**Location**: [`src/solvers/osgs_solver.jl`](../src/solvers/osgs_solver.jl) — the `discrete_l2_projection`
call site for the pressure projection `pi_p_x` inside `solve_osgs_stage!`'s coupled residual
([line 152](../src/solvers/osgs_solver.jl#L152)); also the `Q_proj = Q_free` selection
([line 126](../src/solvers/osgs_solver.jl#L126)). `Q_free` is the unconstrained `TestFESpace` built in
[`src/run_simulation.jl`](../src/run_simulation.jl#L399).

**Status**: Explicitly considered, intentionally **not** applied as of the Phase-5 batch. The companion
plan item **Phase 5 §3.6** (pressure-mean-removal) is excluded but kept on the long-term ledger.

**Operator the code uses today**: At every residual evaluation, the pressure-side projection `π_h^p` is an
unweighted `L²` projection of the strong mass residual `R_p` onto the FE space `Q_h`, built from `Q_free`
— a `TestFESpace` with **no Dirichlet and no zero-mean constraint**. So `π_h^p` in general carries a
non-zero constant mode equal to the `L²`-mean of `R_p`.

**What the paper says about the pressure gauge** ([`article.tex` L407](../theory/paper/article.tex#L407)):

> The pressure belongs to `Q₀ := L²(Ω)` in general, and to `Q₀ := {q ∈ L²(Ω) | ∫_Ω q dΩ = 0}` when the
> boundary conditions are all-Dirichlet (constraining to this subspace fixes the free constant when
> `ε = 0`; for `ε > 0` this is met automatically).

So in the **all-Dirichlet velocity** regime — every current MMS config (`small_test_config.json`,
`test_config.json`, all `probe_k*.json`) — the continuous pressure space is `L²₀(Ω)` (mean-zero), gauge-
fixed by the `ε > 0` penalty in the perturbed continuity equation, which "imposes that the average
pressure is zero" ([`article.tex` L1333](../theory/paper/article.tex#L1333)). The OSGS projection `π_h`
is defined in `eq:NonlinearResidualProjection` as the `L²` projection onto `X_{h0}` with **no** explicit
mean-removal clause.

**The candidate change (§3.6 of the plan)** — the cheap post-hoc form:

```julia
pi_p_next = discrete_l2_projection(R_p, ...)
pi_p_next_mean = sum(∫(pi_p_next)dΩ) / sum(∫(1.0)dΩ)
pi_p_next = pi_p_next - pi_p_next_mean
```

i.e. subtract the `L²` mean, restricting `π_h^p` to `Q_h ⊖ {1}`.

**Why plausibly beneficial**: The constant mode in `π_h^p` is inert for the velocity-side stabilization
gradient (the gradient annihilates constants), but it is carried along by the per-evaluation re-projection
`π_h^p = Π(R_p(u))` — every residual evaluation re-derives it from the current iterate's mass residual.
Mean removal would strip that constant-mode noise and is expected to slightly improve pressure-rate
measurements in MMS sweeps at small `α₀`, where the constant-mode pollution scales unfavourably.

**Why §3.6 is not in Phase 5** — three independent concerns:

1. **Regime dependence.** All-Dirichlet velocity: `Q₀ = L²₀`, so mean removal is *more* paper-faithful
   than the current unconstrained projection. Mixed BCs (open outlet, e.g. Cocquet): `Q₀ = L²`, no
   constraint — mean removal would *remove a physically meaningful constant mode*, making the code *less*
   faithful. A correct §3.6 must be conditional on the BC regime (from the configured Dirichlet tags at
   config-load); the plan's recipe is unconditional.
2. **Two distinct implementations.** *Option A* (cheap, plan's recipe): project on full `Q_free`, then
   subtract the `L²` mean — one extra inner product per OSGS iteration, a post-hoc orthogonalization.
   *Option B* (cleaner, more invasive): build the projection space as `Q_free ⊖ {1}` explicitly (Lagrange
   multiplier on the Gram matrix, or a constrained `TestFESpace`), making the operator "projection onto a
   closed subspace" — unambiguously paper-compatible. Option B is rigorous; Option A is defensible only if
   accompanied by **this** divergence entry documenting the post-hoc mean removal as engineering hygiene,
   not a redefinition of `π_h`.
3. **The continuous gauge is already fixed by the iterative penalty `ε > 0`.** The mass equation carries an
   `ε·p` term whose lagged previous-iterate form (the Codina iterative penalty) drives `p` toward
   zero-mean. §3.6 addresses **intermediate-iteration pollution** of the convergence diagnostic, not a
   steady-state correctness issue; the expected MMS-rate gain is in the measurement noise floor, not the
   converged solution.

   > **Clarification (2026-06-24 — `eps_phys` vs `eps_num`).** The gauge-fixing `ε` is the **numerical
   > iterative penalty** (`numerical_epsilon` / `eps_num`), implemented **Jacobian-only** (lagging
   > `ε p^{n−1}` to the RHS so it cancels at convergence) — *not* a physical compressibility. The physical
   > `ε_phys` (`physical_epsilon`) is **0** for the incompressible problem
   > ([`eq:StrongMassEquation`, article.tex L239](../theory/paper/article.tex#L239), "mainly for numerical
   > reasons"; the 2D examples take `ε = 0`, L1098). A 2026-06-24 working-tree bug that set a non-zero
   > `eps_phys` default in the 3D MMS harness (silently solving a *compressible* problem, with `ε p_ex`
   > injected into the oracle source) was reverted — see `lessons_learned.md`. The canonical spec for the
   > gate's incompressible ε handling is §3 below (§4.1 of the criterion spec).

**Re-evaluation triggers** — reconsider §3.6 when any one becomes true:

1. A **non-all-Dirichlet test config** is added (open-outlet MMS variant, traction BCs) — the regime
   branch is unavoidable and is the right moment to land Option B with a `bc_regime`-aware switch.
2. A **post-Phase-5 MMS sweep shows pressure-rate sub-optimality traceable to constant-mode noise** in
   `π_h^p`. Symptom: coupled-solve pressure residual stagnates with a `π_h^p` whose `L²` decomposition is
   dominated by the constant mode. Diagnostic: dump `sum(∫(pi_p_x)dΩ)` per residual evaluation vs the full
   `‖π_h^p‖_{L²}` on a coarse mesh.
3. **Phase 6 §2.1 continuation runs at `α₀ = 0.05`** (narrow channel) show corner-cell pressure-rate
   degradation (constant-mode pollution scales unfavourably at small `α₀`).
4. The ledger gains **another mass-side entry** interacting with the pressure-space definition — then the
   pressure-space treatment becomes part of a larger formal review and §3.6 should be settled definitively.

**What NOT to do** (captured for future sessions):

- **Do not** apply mean removal to the pressure variable `p` itself (mutating `final_x0`). The `ε > 0`
  gauge fixing is a soft feedback; slamming `p` to zero-mean every iteration fights it and destabilizes
  Newton. §3.6 modifies only `π_h^p` (the projection used inside the stabilization term), not `p`.
- **Do not** land Option A without also landing the BC-regime conditional. An unconditional mean removal at
  a future open-outlet test would degrade rates silently.
- **Do not** treat §3.6 as a prerequisite for Phase 6. Phase 6 §2.1 can proceed without it.

**In scope for the current Phase 5 batch (without §3.6)**: §3.1, §3.5, §5.1, §5.2 only. Expected effect of
omitting §3.6: pressure rates may not improve as much as theoretically possible; they are not expected to
*regress*. The post-Phase-5 baseline is the reference against which a future §3.6 commit is judged.

### 2.8 Two-way asymmetry in the legacy `max_iters_caught` exception path — `[code-actual]`

**Location**: [`src/solvers/asgs_solver.jl`](../src/solvers/asgs_solver.jl) — `_initialize_asgs_state!`
(Stage I). The MMS-extension site is the decoupled verifier hook `on_asgs_converged!` in
[`src/solvers/mms_verification.jl`](../src/solvers/mms_verification.jl). The shared cascade and
`safe_fe_solve!` they both call live in [`src/solvers/solver_core.jl`](../src/solvers/solver_core.jl).

**Paper theory**: Algorithm B (`osgs_algorithm.tex` §"The shared cascade") describes a
Newton→Picard→Newton cascade reused at these call sites, mode-agnostic about *how* a non-converged Newton
exit is detected — only the success/failure binary matters.

**Apparent divergence**: The legacy Gridap exception path (`"Reached maximum iterations"`, caught as
`:max_iters_caught` by `safe_fe_solve!`) is treated *differently* at the two surviving sites:

- Stage I: structural failure → Picard fallback fires (matches the "Stage I quadratic-basin guarantee").
- ASGS-MMS extension: single-iteration partial success, `iter_count` increments, no fallback, **no** log
  line.

**Alignment**: `[code-actual]` — documented additions to Algorithm B, not divergences. The modern
`SafeNewtonSolver` exits cleanly with `stop_reason = "max_iters_stagnation"` on the `:ok` path; the
exception path is purely defensive against legacy / non-`SafeNewtonSolver` instances and is normally never
taken. The paper enumerates all three policies in `osgs_algorithm.tex` §"The shared cascade" (paragraph
"Legacy `max_iters_caught` exception path").

**Re-evaluation trigger**: If a custom non-`SafeNewtonSolver` is plugged in, or Gridap's exception contract
changes, re-audit which policy each site should apply.

### 2.9 `eval_time` reporting convention — `[code-actual]`

**Location**: [`src/solvers/solver_core.jl`](../src/solvers/solver_core.jl), the `eval_time`
return-tuple slot of `solve_system` (accumulates `@elapsed` from each stage; the OSGS portion is
`osgs_elapsed` from `solve_osgs_stage!`).

**Paper theory**: Algorithm O describes the orchestration without committing to a wall-clock reporting
boundary.

**Implementation reality**: `eval_time` = cumulative wall time of three regions, accumulated in
`solve_system`:

1. Stage-I cascade (`@elapsed` around `_initialize_asgs_state!`, `asgs_solver.jl`).
2. ASGS-MMS extension (`eval_time += @elapsed on_asgs_converged!(...)`, `mms_verification.jl`).
3. OSGS coupled solve (`eval_time += osgs_elapsed`, `@elapsed` *inside* `solve_osgs_stage!` around the
   single coupled `begin … end` block only).

`eval_time` **excludes**: OSGS mass-matrix assembly + Cholesky factorisation (run-once setup, inside
`solve_osgs_stage!` but outside its inner `@elapsed`); the ASGS-MMS extension setup (oracle call,
local-solver construction); the post-coupled-block `pi_u`/`pi_p` diagnostic writes; all `diag_cache`
writes after the timed regions.

**Alignment**: `[code-actual]` — `eval_time` is the *iterative* wall time, not the *total* per-call wall
time. Wrap the whole `solve_system` call in `@elapsed` for total cost; it will exceed `eval_time` by the
OSGS setup cost (non-trivial on large meshes).

**Re-evaluation trigger**: If a `setup_eval_time` field is added (or a diagnostics-cache refactor lands),
update this entry.

### 2.10 OSGS MMS stop reason reports solver success, not verification success — `[paper-faithful]` (post P-007 / Fix-6)

**Location**: [`src/solvers/mms_verification.jl`](../src/solvers/mms_verification.jl),
`on_osgs_converged!` (~lines 163–174) — the decoupled verifier hook the OSGS branch of `solve_system`
calls after the coupled solve converges.

**Paper theory**: `osgs_algorithm.tex` Algorithm C (OSGS branch with MMS hook) + Algorithm D (plateau
verifier). The paragraph "Budget exhaustion is not a verification failure of the solver" in §"How
Algorithm D hooks into the core" states the contract explicitly.

**Implementation reality**: The coupled OSGS solve is a single Newton solve, not a staggered budget that
can "exhaust." When MMS verification is active and the coupled solve converges, `on_osgs_converged!` sets
`diag["mms_plateau_reached"] = true` with `diag["mms_stop_reason"] = "coupled_single_solve"`; if the
converged `‖e_u‖_{L²}` exceeds `rate_check_factor · h^{kv+1}` (the pre-asymptotic high-Da coercivity gap),
the reason is instead `"coupled_at_suboptimal_rate"` and the solver still returns success. Either way the
*solver* succeeded (the OSGS fixed point was reached); the second flag carries whether the converged error
met the optimal-rate budget. The legacy `"mms_budget_exhausted"` reason now belongs solely to the ASGS-MMS
extension hook (`on_asgs_converged!`).

**Alignment**: `[paper-faithful]` — the two-flag split (`solver_success`, `mms_plateau_success`) is the
resolution of audit finding **P-007 / Fix-6** ("solver success conflated with verification success").
Callers must read both flags.

**Re-evaluation trigger**: If the Fix-6 caller migration changes the return-tuple shape or adds an
`overall_verification_success` convenience flag, update this entry.

### 2.11 Scale-free nonlinear convergence criterion — `[code-actual]`

**Location**: [`src/solvers/convergence_criterion.jl`](../src/solvers/convergence_criterion.jl) (the pure
criterion). Injected by `solve_system` via `build_convergence_probe`
([`solver_core.jl:462`](../src/solvers/solver_core.jl#L462)) and consumed as `conv_probe` inside
`_safe_solve_inner!` ([`nonlinear.jl`](../src/solvers/nonlinear.jl), the `scale_free = solver.conv_probe
!== nothing` gate ~L742 and the authoritative-success break ~L849). Config: `eps_tol_momentum` /
`eps_tol_mass` ([`config/base_config.json`](../config/base_config.json),
[`src/config.jl:137`](../src/config.jl#L137)).

**Paper theory**: The paper prescribes no discrete *stopping criterion* for the nonlinear iteration of
`alg:StationarySystem` — the algorithm boxes describe the fixed-point map and assume an abstract "until
converged" test. The continuous convergence statement is in the energy/stability norms of §4
(`eq:StabilityEstimate`), not a per-iterate algebraic residual threshold.

**Implementation reality**: The authoritative success gate is **scale-free**: converged iff `ε_M ≤ tol_M`
**and** `ε_C ≤ tol_C`, where

- `ε_M = ‖r_M‖ / D_M` — the assembled stabilized momentum residual (velocity block) over a *dynamic*
  force-magnitude envelope `D_M = ‖α u·∇u‖ + ‖2∇·(α ν Πˢ∇u)‖ + ‖α ∇p‖ + ‖σ(α,u) u‖ + ‖f‖` (Philosophy A:
  every term assembled through the same weak form, same Euclidean norm as the numerator);
- `ε_C = ‖r_C‖ / D_C` — the Route-B **"Philosophy-A" algebraic** mass gate: the assembled stabilized mass
  residual over a term-magnitude envelope, **symmetric with `ε_M`**, gated at the same `~1e-9` level. The
  earlier strong-form / flux-gradient measure `‖ε p + ∇·(α u) − g‖ / (‖∇(α u)‖_F + ‖g‖)` (with the
  pure-divergence self-check `‖∇·(α u)‖/‖∇(α u)‖ ≤ √d`) is now the **diagnostic `eps_C_strong`**, no longer
  the gate. A **`residual_floor_reached` scale-free accept** accepts machine-floor-converged cells whose
  `ε_C` cannot reach `tol` because `D_C` collapses for near-divergence-free flow.

Because `D_M` (and the mass envelope) are *measured from the current iterate and known material data*, the
gate carries **no a-priori scale** (no `U`, `L`, `P`, `Re`, `Da`) and is the SAME threshold across
ping-pong cascade segments. This deliberately supersedes the OLD per-field **re-anchored** relative-ftol
gate (`effective_ftol = relative_ftol·‖R₀‖`, `f_norm = ‖R‖/effective_ftol`), whose re-anchoring forced each
segment to demand another `×(1/ftol)` residual drop. **That relative gate was then REMOVED (commit
`d1fac8e`): `relative_ftol_per_field` is gone, so the `conv_probe===nothing` fallback now uses the uniform
scalar `ftol` per field.**

The old re-anchored gate survives ONLY (a) as the `conv_probe === nothing` fallback — the Cocquet
unstabilized-Galerkin runs and the kernel unit tests, which use the scalar `ftol` — and (b) as the `f_norm`
trace diagnostic (merit normalization + per-iteration history), which no longer decides success. The
`degenerate` flag (a denominator at the machine-eps underflow floor) stands in for the spec's `k ≥ 1` rule,
rejecting the trivial all-zero entry while accepting an already-converged developed iterate.

**Alignment**: `[code-actual]` — an *addition* the paper leaves unspecified, not a contradiction. The
criterion is decoupled by design (`convergence_criterion.jl` decides *whether* an iterate is converged;
`nonlinear.jl` / `solver_core.jl` decide *how* to step) and is a pure read-only observer of a
self-consistent `(iterate, residual)` pair — a probe failure degrades to `NaN` and can never perturb the
solve. Full rationale (Philosophy A vs B, the `−g` subtraction, the `√d` self-check, why the §6
pressure-normalized fallback is intentionally NOT implemented) is in §3 below.

**Re-evaluation trigger**: If a future config drives the stabilization excess so `ε_M` persistently sits
`≫ 1` on a *known-converged* case, that is NOT a loose tolerance — it means `r_M` and the `D_M` term
decomposition have drifted apart (different quadrature/space/sign); chase the consistency bug, not the
criterion. Also revisit if the `conv_probe === nothing` fallback set is migrated onto the scale-free gate,
or if a pressure-scale (§6) fallback is ever genuinely needed for an open-outlet regime where `‖∇(α u)‖`
is not robustly bounded.

---

## 3. Scale-free nonlinear convergence criterion (summary + pointer)

The **permanent derivation** of the scale-free stopping criterion for the outer nonlinear (Picard/Newton)
iteration now lives in LaTeX at
[`theory/scale_free_gate_note/scale_free_gate_note.tex`](../theory/scale_free_gate_note/scale_free_gate_note.tex)
(moved out of `docs/` 2026-07-11 — permanent theory belongs in `theory/`). Implemented in
[`src/solvers/convergence_criterion.jl`](../src/solvers/convergence_criterion.jl); the operational spec and the
Route-B status live in [`docs/mms/convergence-2d.md`](mms/convergence-2d.md) §1.

**In one line.** Stop when `max(ε_M, ε_C) ≤ tol`, both dimensionless and computable from the current iterate +
material/mesh data alone (**no a-priori `U`/`L`/`P`/`Re`/`Da`** — those are bespoke to a manufactured-solution
test; a production criterion measures the needed scale from the iterate instead):

- **Momentum** `ε_M = ‖r_M‖ / D_M`, `D_M = ‖α u·∇u‖ + ‖2∇·(ανΠ∇u)‖ + ‖α∇p‖ + ‖σ(u)u‖ + ‖f‖` — the assembled
  stabilized residual over a *dynamic term-magnitude envelope* (Philosophy A: measure what the solver actually
  drives to zero). Regime-robust because the momentum terms *balance* at the solution, so whichever mechanism
  dominates (viscous as `Re,Da→0`, convection as `Re→∞`, resistance as `Da→∞`) sets the scale automatically.
- **Mass (production, Route-B "Philosophy A"):** the algebraic `ε_C = ‖r_C‖ / D_C` — the assembled mass residual
  over its own term envelope, symmetric with `ε_M`, gated at the same `~1e-9`; plus a `residual_floor_reached`
  accept for near-divergence-free flow where `D_C → 0` (the mass envelope collapses for essentially
  incompressible flow, flooring `ε_C` a decade above `tol` even when converged).
- **Mass (diagnostic `eps_C_strong`):** the flux-gradient ratio `ε_C = ‖∇·(α u)‖ / ‖∇(α u)‖ ≤ √d`. The `√d`
  bound (`∇·q = tr(∇q)`, `(tr A)² ≤ d(A:A)` ⇒ `‖∇·q‖² ≤ d‖∇q‖_F²`) is a cheap correctness self-check — a value
  above `√d` signals a quadrature/projection/assembly bug. This flux-gradient measure is the derivation's
  scale-free mass gate; it was **demoted from the gate to a diagnostic when Route-B landed**.

**Hard constraint (do not violate):** the criterion must never introduce, request, or hard-code a characteristic
scale. Element-level `Re_h`, `Da_h` built from `h_K` and the iterate are allowed (they are measured, not a-priori)
and appear only in the documented — but **not implemented** — pressure-normalized fallback. Full rationale, the
edge cases (trivial guess; uniform porosity → classical NS; all-Dirichlet pressure indeterminacy; the ε>0
iterative penalty; Newton-vs-Picard must use the genuine nonlinear residual; tolerance-above-floor), and that
fallback: the `theory/` note above.
---

## 4. Normalization / encoding-invariance audit (historical — one durable invariant)

*(P5, 2026-06-04.)* This audit classified every nonlinear-solver gate by whether both sides of its inequality
carry the same units. It is now **historical**: the inner per-field ftol / noise-floor / honest-exit gates
(#1–#3) were **superseded** by the scale-free `ε_M`/`ε_C` criterion (§3, §2.11) and survive only on the
`conv_probe === nothing` fallback (Cocquet unstabilized-Galerkin + kernel tests); the two OSGS outer drift
gates (#5/#6) were **removed** with the staggered loop in the 2026-06-07 coupled-only leaning. All surviving
gates were dimensionally sound; the only P5 code change was removing the dead `_resolve_solution_scale_per_field`
helper.

**The one durable invariant — divergence safeguard, do not regress the Picard branch.**
`eval_safeguard_termination_bounds!` uses a merit *ratio* in `:newton` mode (`Φ_new/Φ_old > divergence_merit_factor`)
but a residual-inf-norm *ratio* in `:picard` mode — both dimensionless. The mode split is load-bearing: a Φ-based
test in Picard mode caused spurious `merit_divergence_escaped` exits
(`test/extended/ManufacturedSolutions/diagnostics/probe_stiff_findings.md`). Keep the `:picard` branch
residual-based.
---

## See also

- [`docs/formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) — the deep
  theory↔code + results audit that corroborates the ledger entries above.
- [`docs/mms/p2-3d.md`](mms/p2-3d.md)
  — the for-scrutiny record behind the `4k⁴`/`c₁` note in §2.5.
- [`docs/cocquet/investigation-synthesis.md`](cocquet/investigation-synthesis.md) — cross-topic
  investigation synthesis.
- [`theory/paper/article.tex`](../theory/paper/article.tex) — the SIAM article (authoritative).
- [`theory/osgs_algorithm/osgs_algorithm.tex`](../theory/osgs_algorithm/osgs_algorithm.tex) — the
  algorithm boxes mapped in §1.
