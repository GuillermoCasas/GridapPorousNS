# JFNK Phase-0 gate — is the frozen-π Jacobian a viable inner-GMRES preconditioner?

**Status: GATE PASSED — GO → Phase-1 IMPLEMENTED (2026-06-26).** This is the mandatory Phase-0 measurement
that authorized the JFNK implementation for the OSGS coupled solve (`solve_osgs_stage!`,
[osgs_solver.jl](../../src/solvers/osgs_solver.jl)); it passed, and the implementation has since landed
behind the opt-in flag `osgs_jfnk_enabled` (see "Phase-1: implemented" at the end). The theory frontier this
validates is `\ref{sec:jfnk}` of
[theory/osgs_algorithm/osgs_algorithm.tex](../../theory/osgs_algorithm/osgs_algorithm.tex); the cost-model
motivation is the project memory note `jfnk-osgs-cost-model-and-preconditioner-question`.

## The question

The OSGS coupled solve is an inexact Newton on `F(U)=0` with `F` embedding `π(U)=Π(R(U))` (re-projected at
every residual evaluation). The true tangent is

```
dF/dU = J_frozen − C,   C = ∫ L*(V)·τ·Π(dR·dU)   (the dense ∂π/∂u coupling)
```

The solver keeps the **exact frozen-π tangent** `J_frozen` (commit 27393f6, finding A.3) and deliberately
drops `C` (a dense global `M⁻¹` coupling). JFNK recovers `C` matrix-free: `J_full·v ≈ [F(U+εv)−F(U)]/ε`
(the residual already re-projects π, so the FD captures `C` for free), preconditioned by the
already-assembled, already-factored `J_frozen`.

**Decisive open question:** is `J_frozen` (the *free* preconditioner) good enough to precondition the inner
GMRES for `J_full`? A throwaway measurement gave `ρ_prec = max|eig(J_frozen⁻¹·C)| ≈ 1e5–1e9 ≫ 1`, which
would make `J_frozen` useless — but that run used `eps_val=0` and no pressure-level constraint, leaving
`J_frozen` **singular on the constant-pressure mode**. This re-measures cleanly.

## Harness

Extends the committed [test/quick/osgs_frozen_pi_jacobian_quick_test.jl](../../test/quick/osgs_frozen_pi_jacobian_quick_test.jl)
machinery (`_a3_problem`: spaces, formulation, `live_pi`, `res_vec`, `jac_mat`), threading a configurable
`eps_val` and velocity amplitude. Equal-order P1/P1 QUAD (where OSGS pressure stabilization matters),
`SymmetricGradient + ConstantSigma`, the same setup as the original cost-model measurement. Scripts (throwaway):
`scratchpad/phase0_jfnk_precond.jl`, `phase0_anderson_baseline.jl`, `phase0_coldstart_robustness.jl`.

### The two epsilons (kept rigorously distinct — they behave oppositely here)

| | symbol | where it lives | role |
|---|---|---|---|
| physical | `eps_val` (ε_phys) | **both** residual & Jacobian mass-LHS (`eps_val·p`) | residual-consistent; pins the constant-pressure mode |
| numerical | `numerical_epsilon` (ε_num) | **only** the Jacobian (2,2) Galerkin block (`(eps_val+ε_num)·dp`); cancels in the residual | Codina iterative penalty; root-preserving Newton-step regularizer |

Consequences exploited in the measurement:
- The residual `F` (hence the matrix-free `J_full` action, which FD's the residual) depends **only on ε_phys**.
- `J_full − J_frozen(ε_num=0)` is therefore the **pure** `∂π/∂u` (the genuine JFNK dropped term); a nonzero
  ε_num would pollute that difference with `−ε_num·M_p`, so the canonical dropped term is measured at ε_num=0.
- `J_frozen(ε_num) = J_frozen(0) + ε_num·M_p_block`. Because ε_num is root-preserving, it can be added to the
  GMRES **preconditioner** for the same (ε_num-independent) full-J system without biasing the JFNK root — a
  free conditioning knob. We swept it to test whether it helps. **It does not** (see below).

## Results

### Headline: N_c vs N_j (cold start → 1e-9, every cell)

`N_c` = Newton steps with the frozen-π tangent (= the current solver). `N_j` = Newton steps with the full
FD Jacobian (= ideal JFNK).

| cell | N_c (frozen-π = current) | N_j (full-J = JFNK) |
|---|---|---|
| n=6 ν=1 σ=1 | **60, not converged** (1.5e-8) | **2**, converged (4e-11) |
| n=6 ν=0.01 σ=100 | **60, DIVERGED** (3e8) | **2**, converged (2e-10) |
| n=6 ν=0.01 σ=100 amp=1 | **60, DIVERGED** (4e38) | **3**, converged (4e-16) |
| n=10 ν=1 σ=1 | **60, not converged** (5e-9) | **2**, converged (2e-11) |
| n=10 ν=0.01 σ=100 amp=1 | **60, DIVERGED** (2e1) | **3**, converged (6e-17) |

The dropped `∂π/∂u` is **large**, not small: `‖ΔJ‖_F/‖J_full‖_F` = 0.15 (mild) up to **0.90–0.97** (stiff).
It is what caps the frozen-π rate, and on the stiff/convective cells the current path doesn't merely slow —
it **diverges**.

### Decisive metric: J_frozen-preconditioned GMRES iterations (to 1e-2 rel., the inexact-Newton inner tol)

With the **free** preconditioner — the pure A.3 frozen-π tangent, **ε_num = 0**, at production ε_phys:

| cell | GMRES iters → 1e-2 | ρ_prec (cross-check) | matrix-free FD action |
|---|---|---|---|
| n=6 ν=1 σ=1 (ε_phys=1e-8) | **1** | 0.88 | 1 |
| n=6 ν=1 σ=1 (ε_phys=1e-6) | **3** | 0.88 | 3 |
| n=6 ν=0.01 σ=100 (ε_phys=1e-8) | **4** | 67.5 | 4 |
| n=6 ν=0.01 σ=100 amp=1 (ε_phys=1e-6) | **16** | 48.2 | 16 |
| n=10 ν=1 σ=1 (ε_phys=1e-8) | **1** | 0.87 | 1 |
| n=10 ν=0.01 σ=100 amp=1 (ε_phys=1e-6) | **16** | 29.4 | 16 |

The honest matrix-free directional-FD action reproduces the dense iteration counts exactly.

**Cold/mid-convergence robustness** (GMRES iters to 1e-2 at Newton steps 0–3, ε_num=0 preconditioner):

| cell | step 0 (cold) | step 1 | step 2 | step 3 |
|---|---|---|---|---|
| n=6 ν=1 σ=1 | 4 | 4 | 1 | 1 |
| n=6 ν=0.01 σ=100 amp=1 | 13 | 19 | 16 | 1 |
| n=10 ν=0.01 σ=100 amp=1 | 12 | 21 | 16 | 1 |

GMRES stays bounded along the whole trajectory: ≤4 for mild cells; a **~19–21 peak** mid-convergence for the
hardest convective cells, dropping to 1 near the root.

### The ρ_prec "1e5–1e9" was pure contamination

Reproducing the throwaway setup (ε_phys=0, no pin): `ρ_prec = 7.6e4`, `cond(J_frozen)=1.6e18`, and the
**worst eigenvector's overlap with the constant-pressure direction = 1.0000**. Deflating that one mode gives
`ρ_defl = 0.74 < 1`. So the bad number was 100% the constant-pressure null mode. Even there, GMRES took **1
iteration** to 1e-2 — because GMRES dispatches a single outlier eigenvalue in O(1) steps. **ρ_prec is a red
herring for the GO decision; the GMRES iteration count is the honest metric.**

This also refines the theory's prediction (`\ref{sec:jfnk-change}`): the sufficient condition
`‖J_frozen⁻¹·C‖ < 1` holds for the mild cells (ρ_prec≈0.88) but is **violated** for stiff/convective cells
(ρ_prec≈29–67). GMRES converges fast anyway because the super-unit eigenvalues are few and clustered — a
polynomial Krylov method beats the worst-case spectral-norm bound.

### ε_num as a preconditioner regularizer: not beneficial (answers the two-ε question)

Sweeping ε_num in the preconditioner (root-preserving) consistently **hurt or was neutral**:

| cell | ε_num=0 | 1e-6 | 1e-4 | 1e-2 |
|---|---|---|---|---|
| n=6 ν=1 σ=1 (ε_phys=1e-6) | **3** | 3 | 11 | 6 |
| n=6 ν=0.01 σ=100 (ε_phys=1e-8) | **4** | 28 | 41 | 17 |
| n=10 ν=1 σ=1 (ε_phys=1e-8) | **1** | 8 | 18 | 6 |

ε_num deliberately pulls the Jacobian **off** the residual's true tangent, so adding it to the preconditioner
just introduces preconditioner↔operator mismatch. The residual-consistent physical `ε_phys` already makes
`J_frozen` an excellent preconditioner (the pressure null mode is a single GMRES-trivial outlier). **JFNK
should use the ε_num=0 frozen-π tangent as the preconditioner.** (ε_num remains useful for its actual purpose
— Jacobian-only pressure stabilization where the discretization is unstable, e.g. P2/3D — but not as a JFNK
preconditioner regularizer.)

### Baseline to beat: the opt-in Anderson accelerator

Factorizations to drive the true residual `‖F‖∞ < 1e-9` (the cost unit that dominates in 3D):

| cell | current (inexact-N) | bare staggered | Anderson staggered | **JFNK (ideal)** |
|---|---|---|---|---|
| n=6 ν=1 σ=1 | 60 (stuck 1e-8) | 76 (stuck 2e-8) | **32** ✓ | **2** ✓ |
| n=6 ν=0.01 σ=100 | 60 (diverged 3e8) | 362 (stuck 3e1) | 418 (stuck 2e-1) | **2** ✓ |
| n=6 ν=0.01 σ=100 amp=1 | 60 (diverged 4e38) | 372 (stuck 3e1) | 411 (stuck 3e-1) | **3** ✓ |
| n=10 ν=0.01 σ=100 amp=1 | 60 (diverged 2e1) | 360 (stuck 8e0) | 355 (stuck 4e-2) | **3** ✓ |

On the stiff/convective cells — the regime that motivates this work — **both** the current path **and** Anderson
**fail to converge**. JFNK is the only method that converges, and it does so in 2–3 Newton steps.

## Cost arithmetic & decision

Per Newton step JFNK costs **one** `J_frozen` factorization (the preconditioner — same cost as one current
Newton step) **+** `G` GMRES iters (each = one matrix-free residual eval + one cached back-substitution; **no
new factorization**). Total JFNK cost ≈ `N_j` factorizations + `N_j·G` back-substitutions.

- **Mild:** `N_j=2`, `G≈1–4` → ~2 factorizations + ~6 substitutions. vs Anderson 32, current 60. **~16–30× fewer factorizations.**
- **Stiff/convective:** `N_j=3`, `G≈16` (peak ~21) → ~3 factorizations + ~50 substitutions. vs Anderson 355–418 (non-converging), current 60 (diverging). **~100× fewer factorizations, and the only convergent method.**

In 3D, where a factorization dominates a back-substitution by orders of magnitude, this is decisive.

### Verdict: **GO**

- `J_frozen` (ε_num=0, at the physical ε_phys) is a **viable, free preconditioner**.
- Decisive metric: GMRES converges in **1–4 iters (mild)** / **≤16–21 iters (stiff/convective)** per Newton step.
- `N_j` (2–3) `≪` `N_c` (60, non-converging/diverging).
- The hardest convective cells sit at the GO/CONDITIONAL boundary on **raw** Krylov count (~16–21 peak), but
  pass the CONDITIONAL cost-arithmetic test overwhelmingly (≈3 factorizations vs 60–418), and JFNK is the only
  method that converges on them at all.
- No saddle-point / multigrid preconditioner is needed for these 2D equal-order cells.

## Carry-overs into Phase 1

- **Preconditioner = `J_frozen` with ε_num=0** (the assembled A.3 ExactNewton tangent, reusing
  `jac_fn_coupled`). Do **not** add ε_num to the JFNK preconditioner.
- **3D watch item.** These measurements are 2D equal-order. The theory's `‖J_frozen⁻¹·C‖<1` condition is
  already violated by stiff 2D cells (GMRES survives via clustering); 3D may push `G` higher. The built-in
  safety net is the **C.1 honesty machinery** (commit 985ebff, `GMRESNotConvergedError`,
  [linear_solvers.jl](../../src/solvers/linear_solvers.jl)): a struggling inner GMRES is surfaced, not
  swallowed, and the Picard / frozen-π fallback catches it. If 3D `G` is routinely large, *that* is the
  empirical trigger to add a real saddle-point preconditioner (block/Schur — PCD/LSC/SIMPLE — or Vanka/MG;
  τ₁ already seeds a discrete pressure-Laplacian in the (2,2) block; τ~h ⇒ rediscretize per MG level).
- **Inexact line search.** The merit slope identity `D = −2Φ` is exact only for the exact Newton step; the
  Krylov (forcing-term η) step satisfies it only up to the inner residual. Relax the Armijo slope test to the
  inexact-Newton form `D ≤ −2Φ(1−η)` (osgs_algorithm.tex `\ref{sec:jfnk-change}`).
- **Behavior preservation.** With `osgs_jfnk_enabled=false`, the existing path must stay byte-identical
  (verify, as Anderson did). Knobs (inner rel-tol/η, max iters, restart, FD-ε policy) go in schema +
  `SolverConfig` + base_config with fail-loud `validate!` (repo hard rule: no magic numbers).

## Phase-1: implemented (2026-06-26)

The GO was acted on. JFNK is wired in opt-in behind `osgs_jfnk_enabled` (default false), mutually exclusive
with `osgs_anderson_enabled` (`validate!` rejects both on). Design — the theory's "change exactly one thing:
the inner linear solve":

- **`JFNKLinearSolver`** ([linear_solvers.jl](../../src/solvers/linear_solvers.jl)) — a drop-in matrix-free
  `LinearSolver`. Its `solve!(dx, ns, b)` runs left-preconditioned GMRES on `J_full·dx = b` where the
  mat-vec is the directional FD of the coupled residual (`JFNKMatVec`, Brown–Saad ε) and the preconditioner
  is the factored frozen-π Jacobian `A` Gridap already assembled (`JFNKPrecond`). On non-convergence it
  raises `GMRESNotConvergedError` ([C.1]).
- **`_osgs_jfnk_solve!`** ([osgs_solver.jl](../../src/solvers/osgs_solver.jl)) — plugs that solver into the
  existing `SafeNewtonSolver` (via a new `ls=` override on `_with_overrides`), so the outer Newton, Armijo
  line search, divergence/stall guards, per-field gate, and C.1 honesty are inherited unchanged. The FD base
  point x_k is threaded in via a Ref written by a thin wrapper around `jac_fn_coupled` (Gridap evaluates the
  jacobian at the iterate right before the inner solve). On structural failure it falls back to the standard
  frozen-π coupled solve from the best iterate — never worse than the current solver.
- **Two-ε:** the preconditioner is `jac_fn_coupled`, which carries the config's `numerical_epsilon`; Phase-0
  recommends keeping `ε_num=0` for JFNK (a nonzero ε_num degrades the Krylov count). `base_config` has
  `numerical_epsilon=0`.
- **Line search:** reuses the exact-slope `D=−2Φ` test verbatim; at η≈1e-2 the inexact `(1−η)` relaxation is
  a 1% conservative effect (safe).

**A/B integration result** (full `solve_system` path, the orthogonality MMS cell, P2/P1): frozen-π = **17
Newton iters / 8.96 s**, stopping at ‖F‖∞≈3e-5 (soft-stall accepted); JFNK = **6 iters / 6.73 s**, reaching
‖F‖∞≈4e-8 — *faster and more fully converged*; the converged L2 velocity errors agree to 5 significant
figures (the 4e-6 residual reflects frozen-π's under-convergence, exactly what JFNK fixes).

**Verification:** Blitz 240/240, Quick 76/76 (incl. `test/quick/osgs_jfnk_quick_test.jl`: matrix-free action
== true full tangent; production `JFNKLinearSolver.solve!` → same root as a full-Jacobian Newton to 1e-6;
starved-GMRES → `GMRESNotConvergedError`), and `test/extended/jfnk_equivalence_extended_test.jl` (8/8,
4m14s: ASGS byte-identical, OSGS same MMS root, JFNK iters ≤ frozen-π).
