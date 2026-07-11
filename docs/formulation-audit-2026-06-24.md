# Formulation & Solver Audit — 2026-06-24

**Scope.** A deep, independent re-audit of the whole `src/` tree against the theory
(`theory/paper/article.tex` and the `theory/*` notes), plus a forensic re-examination of the
convergence *results* and the test harness/reporting that produces them. Three questions:

1. Is the theory faithfully transcribed into the code? Where not, what are the remaining inconsistencies and the recommended fixes?
2. Where does the formulation/solver make the algorithm **fragile or inaccurate**, and what is a viable alternative?
3. Where can it be **simplified / reorganised** for clarity, efficiency, or elegance?

**Method.** (a) A line-by-line independent read of every core file in `src/`. (b) A 9-domain
multi-agent audit of `src/` against the paper, with every finding adversarially re-verified by an
independent code-grounding skeptic *and* a theory-grounding skeptic (53 raw findings → 34 upheld, 19
refuted/trivial). (c) An independent recomputation of the 2D and 3D convergence rates straight from the
raw result files (HDF5 + JSON), *not* trusting the summary docs. (d) A fresh control experiment
(`/tmp/audit_3d_structured.jl`, log under `results/debug_results/`) run on a structured Kuhn tet mesh of
uniform, refinement-invariant quality, to separate "mesh quality" from "method/formulation" as the cause
of the 3D rate anomalies.

> **Provenance of this document.** Every code finding below was confirmed by reading the cited
> `file:line`. Every results claim was recomputed from the raw error arrays. The control-experiment
> numbers come from the run logged at
> `test/extended/ManufacturedSolutions3D/results/debug_results/audit_3d_structured.log`.

---

## 0. Status & what remains

> **Trimmed working copy** (dated-provenance: original audit + full 34-finding write-ups and the complete
> resolved detail are preserved verbatim at commit `a31f191` and its follow-up commits). Resolved findings
> live in the commits cited; this document now holds the still-standing findings and their arguments.

**Headline that still stands.** The continuous VMS formulation is faithfully transcribed — the strong
residual, both Jacobians, the adjoint sign conventions, the deviatoric/symmetric viscous expansions
(2D & 3D), τ₁/τ₂, the OSGS projection policies, σ(α,u), the porosity field + all four derivatives, and
both MMS oracles each match the paper. **No correctness bug was found in the weak-form assembly.** This
headline was reached by the adversarial two-lens verification described under Method (53 raw findings → 34
upheld, each re-checked by an independent code-grounding skeptic and a theory-grounding skeptic). The
remaining work is in (i) a couple of doc↔code contracts, (ii) the convergence *results* and their
harness/reporting, and (iii) hygiene/fragility gaps.

**3D-P2 c₁ + OSGS-P2 coupling: RESOLVED** — see findings.md section 3 and open-questions.md section 4
(this audit's earlier c1-discrepancy / saddle-point-required framing is SUPERSEDED).

---

## Part A — Theory ↔ code consistency

### A.0 What is faithful (re-verified, not assumed)

- **Strong momentum/mass residual** (`continuous_problem.jl` `eval_strong_residual_u/p`) matches
  `eq:StrongMomentumEquation`/`eq:StrongMassEquation` term-by-term, including the IBP pressure form
  `−p(α∇·v + ∇α·v)` (= `−∫ p ∇·(αv)`) and `∇·(αu) = α∇·u + u·∇α`.
- **Adjoint sign discipline.** `strong_adjoint_momentum` returns `+α(∇v)'·u` and `B_S` subtracts the
  adjoint; the `σv` term is subtracted. Matches Eq.39/Eq.50 and the documented A²−B² symmetry. The
  `(1/α)∇·(αa)v` omission is the paper's own simplification (article.tex ~L800).
- **Viscous operators** (`viscous_operators.jl`). `∇·ε^d(u)=½Δu+(½−1/d)∇(∇·u)` with coefficient `0` in
  2D and `+1/6` in 3D; `∇·ε(u)=½Δu+½∇(∇·u)`. The 2αν factor (μ=αν), the weak Jacobian (linear ⇒ `du`
  for `u`), and the self-adjoint reuse on `v` are all correct. The 3D MMS oracle (`src/problems/mms_paper.jl`, the dimension-generic oracle) uses the
  matching `∇·(2αν∇ᵈu)=2ν(∇ˢu·∇α−⅓(∇·u)∇α)+ανΔu+(αν/3)∇(∇·u)`.
- **τ₁/τ₂** (`tau.jl`) implement `eq:Tau1Final`/`eq:Tau2Final` exactly: `τ₁=1/(α·τ_NS⁻¹+σ)`,
  `τ₂=h²/(c₁ α τ_NS)`, `τ_NS=(c₁ν/h²+c₂|w|/h)⁻¹`, `c₁=4k⁴, c₂=2k²`. The dropped `εh²` and `C_α` terms
  are the paper's own §4.2 simplifications. The dτ/du derivatives are the correct chain rule.
- **OSGS projection policies** (`projection.jl`) implement the §4.4 `(1−Π)` trim correctly; the
  unconstrained `V_free/Q_free` projection space matches the divergences-ledger requirement.
- **Reaction** (`reaction.jl`): `σ=a(α)+b(α)|u|`, `a=σ_lin((1−α)/α)²`, `b=σ_nonlin(1−α)/α`; `dσ/du`
  is the exact `b·(u·du)/|u|`.
- **Porosity** (`porosity.jl`): `α`, `∇α`, the scalar Laplacian, and the full Hessian were each derived
  by hand and match; the Hessian trace equals the Laplacian (mutually consistent), and the logistic is
  genuinely C∞ across the annulus boundaries. The MMS oracles are exact for the same operators the
  solver assembles.
- **ε_num** is Jacobian-pressure-block only in *both* Jacobian builders and absent from the residual and
  the VMS subscale — cancels in the residual, vanishes at convergence, as documented.

### A.4 [LOW] Config-strictness gaps relative to the repo's own hard rule

- The JSON schema declares `required` for exactly one object (`linear_solver`); presence is enforced in
  practice by the no-default `@kwdef` structs, but the schema does not encode it
  ([porous_ns.schema.json](config/porous_ns.schema.json)). Add `required` arrays mirroring the structs.
- `config/base_config.json` omits the now-required `eps_val` and fails `load_frozen_config`
  (documented; **DRIV-01**). Either add `eps_val` to the canonical example or document that it is
  intentionally incomplete.
- Doc bug: the `PhysicalProperties` struct docstring ([config.jl:23](src/config.jl#L23)) calls `eps_val`
  *"porosity ε of the medium (>0)"* — it is the compressibility/pressure-penalty ε (α is porosity), and
  it may be 0. The inline field comment is correct; the struct docstring is not.

---

## Part B — Convergence results vs theoretical expectations

> This is the part where the user's intuition ("the clues are in the results") pays off. **Do not trust
> the summary docs** — every number below was recomputed from the raw error arrays, and the headline
> conclusions of the existing 3D investigation are re-tested against a clean control mesh.

### B.0 2D (k=1 QUAD) is honest and optimal — recomputed

Recomputing finest-pair rates straight from
`previous_results/validated_k1_quad_N640/phase1_quad_k1.h5` (48 configs): **every** cell has L²u ≥ 1.93
(median 2.03), H¹u ≥ 1.00 (median 1.01), L²p median 2.01. Zero sub-optimal at the finest pair. The
HDF5-stored `rate_u_l2` attribute is a *global* least-squares fit (≈1.7–1.96, dragged down by coarse
meshes) and is systematically lower than the finest-pair rate — so a report's headline depends on which
estimator it cites — but the method is genuinely optimal. **The 2D story is sound; the problem is
3D-specific.** (Minor reporting note: state which rate estimator a table uses.)

### B.1 [HIGH] 3D P1-ASGS is sub-optimal even on a perfect uniform mesh — method-intrinsic, not mesh quality

**The control experiment.** The structured Kuhn simplex mesh has uniform, refinement-invariant element
quality (constant inverse-estimate constant `C_inv` across levels), so it removes "mesh quality" as a
variable. Driving the *same* paper §5.2 case (Deviatoric, Constant-σ, Re=Da=1, α₀=0.5, paper c₁) through
the *same* `solve_one` the committed sweeps use:

| method | h-pair | L²u rate | H¹u rate | L²p rate |
|---|---|---|---|---|
| **P1 ASGS** | 0.149→0.099 | 1.24 | 0.92 | 0.84 |
| | 0.099→0.075 | 1.06 | 0.67 | 0.83 |
| | 0.075→0.060 | 1.27 | 0.88 | 0.95 |
| **P1 OSGS** | 0.149→0.099 | **2.21** | **0.99** | 1.66 |
| | 0.099→0.075 | **1.77** | 0.85 | 1.93 |

(Optimal: L²u = 2, H¹u = 1, L²p = 1.) On the **identical mesh and c₁**, ASGS sits at L²u ≈ 1.2 across
all three pairs while OSGS is at 2.21. ASGS's H¹u (≈ 0.9) is near-optimal — so the deficiency is
specifically the **L²-velocity extra (Aubin–Nitsche) order**, which ASGS fails to achieve in 3D and OSGS
recovers. This reproduces the committed `frontal_c1x1` numbers (recomputed: OSGS L²u ≈ 2.0 optimal,
ASGS L²u ≈ 1.3 sub-optimal) — but the structured-Kuhn control proves the gap **cannot be a mesh-quality
artifact**, because a same-mesh ASGS↔OSGS difference is by construction method-dependent.

**Interpretation.** This is consistent with standard VMS theory: OSGS stabilises only the part of the
strong residual *orthogonal* to the FE space, removing the FE-projectable "consistency error" that ASGS's
full-residual stabilisation retains. That retained error is adjoint-inconsistent and caps the L²
superconvergence at the H¹ order. In 2D the codebase shows ASGS *is* optimal (B.0, median L²u 2.00); the
3D loss appears tied to the genuinely-3D tetrahedral discretisation of the (z-extruded) solution and the
deviatoric trace coupling — but the precise mechanism is secondary to the reproducible empirical fact.

**Why this matters for the existing narrative.** `docs/mms/p2-3d.md` and the
project memory frame 3D sub-optimality as *mesh quality / `C_inv` vs the c₁=4k⁴ budget*, "fixed by the
Frontal mesh." That framing is correct **for OSGS** (a better mesh lifts OSGS to optimal) but **wrong for
ASGS**: the ASGS L²-order loss is method-intrinsic and survives a perfect uniform mesh. Reports that
present "3D P1 optimal" should qualify it as *OSGS-optimal, ASGS-suboptimal*.

**Re-confirmation (2026-07-04, ratio-to-interpolant — the cleaner discriminator).** The ASGS-vs-OSGS
comparison above is strengthened by measuring each against the *nodal interpolant* (the best the FE space
allows) on the structured Kuhn ladder:

| method | ratio→interp: (8,8,2)→(16,16,4)→(24,24,6) | L²u rate vs interp rate |
|---|---|---|
| **P1 ASGS** | 1.47 → 2.40 → **3.03** (grows) | 1.2 vs interp **1.85** |
| **P1 OSGS** | 0.93 → 0.84 (pinned ~1) | tracks interp |

This is decisive on two fronts. (i) The *interpolant itself* only reaches rate ≈1.85 (not 2) at these
coarse meshes (the α-annulus preasymptotics — same effect as P2), so the raw ASGS <2 rate is **not all
defect**; but (ii) ASGS lags *even the preasymptotic interpolant* and the gap **widens** (ratio 1.47→3.03),
while OSGS *pins* at the interpolant (~0.9) on the **same mesh**. So the space demonstrably admits the
optimal order (OSGS gets it) and ASGS specifically loses it — a genuine L²-order (Aubin–Nitsche) defect,
not a mesh or preasymptotic artifact. (The earlier ASGS-vs-OSGS framing understated the preasymptotic
share; ratio-to-interpolant separates the two cleanly.)

**Recommended action.** (1) Document the ASGS 3D L²-order deficiency honestly as a method property, not a
mesh defect. (2) Treat OSGS as the load-bearing 3D method (as the code already leans). (3) If ASGS 3D
optimality is wanted, that is a genuine formulation research question (the orthogonal-projection
consistency term), worth a paper footnote — not a c₁ tuning.

### B.2 [HIGH] The committed "3D-P2 divergence" is dominated by `success=False` solves plotted as valid data

*(Resolved — the plotter/rate-computation now gate on `success`; the finding and its argument are kept below.)*

Recomputing the committed P2 rates straight from
`previous_results/convergence3d/convergence3d_results_frontal_c1x1_20260623.json` (Frontal mesh, paper
c₁):

| level | h | success | L²u | H¹u | L²p |
|---|---|---|---|---|---|
| 0 | 0.123 | **True** | 0.0272 | 0.905 | 0.268 |
| 1 | 0.094 | **True** | 0.0123 | 0.501 | 0.126 |
| 2 | 0.078 | **False** | 0.0620 | 2.863 | 0.656 |

The level 0→1 rates (both `success=True`) are **L²u = 2.92, H¹u = 2.17, L²p = 2.77 — fully optimal for
P2** (opt 3/2/2). The "divergence" is entirely the level-2 step, which **did not converge**
(`success=False`) and whose error values (0.062, 2.863) are whatever the failed solve happened to stop
at. The `pre_frontal` (Delaunay) file is the same story inverted: its level-1 solve *fails* and spikes,
while the *finer* level-2 solve (`success=True`) recovers to L²u 0.0062.

**The reporting does not gate on `success`.** [plot_convergence3d.py](test/extended/ManufacturedSolutions3D/plot_convergence3d.py)
`_plot_by_kv` reads the top-level `hs`/`l2u`/`h1u`/`l2p` arrays and `_seg_slope` computes a slope across
*every* consecutive pair, with no `success` filter; the top-level error arrays in the JSON drop the
per-level `success` flag (kept only in `levels[]`). So a failed solve is plotted as a data point and its
wild slope is annotated as a convergence rate. The "P2 diverges under refinement" headline is largely an
artifact of mixing non-converged solves into the rate table.

This does **not** fully exonerate the formulation. The committed files conflate two effects: (i) the
failed-solve contamination above, and (ii) a **genuine** under-stabilization defect. The "optimal"
frontal pair (level 0→1, L²u ≈ 2.9) is at *coarse* h (0.12→0.094); the clean-mesh control (B.5) goes
finer and shows the P2 error **turning around and growing** on converged fine pairs. So the right
reading is: **the reporting overstates the divergence (failed solves), but a real P2-3D defect exists
underneath it** — see findings.md section 3 for the resolved root cause.

**Lesson.** (1) The plotter and any rate computation **exclude or visibly flag** `success=False` levels
(the data is in `levels[]`). (2) Re-derive the "errors grow under refinement" claim using only converged
levels on a fixed-quality mesh before attributing it to a mechanism. (3) **Never compute a convergence
rate across a non-converged solve.**

### B.3 [Med] ASGS and OSGS report byte-identical errors at every shared-failed level

*(Resolved — the harness now marks an OSGS solve that degenerated to the ASGS boot state distinctly; the finding and its argument are kept below.)*

Across both committed P2 files, the *only* levels where ASGS and OSGS agree to all 16 digits are exactly
the levels where **both** report `success=False`:

```
frontal   lvl2: ASGS(False, l2u=0.06198525425452791) == OSGS(False, l2u=0.06198525425452791)
pre_frontal lvl1: ASGS(False, l2u=0.06196690717763982) == OSGS(False, l2u=0.06196690717763982)
```

At every mixed- or both-success level the two methods differ (as independent solves must). Two
independent nonlinear solves with different stabilisation cannot produce bit-identical errors unless they
returned the *same field* — i.e. a shared failed/fallback state (most likely a degenerate independent
remesh at that level on which both solves fail identically, before the method-specific terms can
diverge), or a data-handling artifact. Either way the failed-level numbers are doubly untrustworthy.

**Lesson.** Flag identical ASGS/OSGS error tuples as a red flag; surface the per-level mesh quality
(min dihedral/radius-ratio) so a bad remesh is visible; and (with B.2) exclude these levels from the rate
table.

### B.4 [Med] The most recent committed 3D results are not reproducible from the current harness

The committed `convergence3d_results_frontal_c1x1_20260623.json` records
`mesh_algorithm = "gmsh_Frontal_alg4_independent_remesh"` and a 6-level P1 ladder, but **no function now
in `smoke3d.jl` produces that string or that ladder**: the sweep paths use `build_nested_family` and
write `mesh="nested_red"`; `build_sequence("frontal")` ([mesh3d.jl](test/extended/ManufacturedSolutions3D/mesh3d.jl))
defines only three lcs `[0.137,0.098,0.070]`. All of `mesh3d.jl`, `smoke3d.jl`, `plot_convergence3d.py`
are uncommitted-modified in the working tree. So the exact driver that produced the most recent committed
numbers is not recoverable from `HEAD`+worktree — a violation of the project's own
parameters→results-traceability rule.

A related minor provenance smell: `smoke3d.jl`'s `build_config` sets `physical_properties.eps_val = 1e-8`,
but `solve_one` builds the formulation with `eps_phys = 0.0` (default), so the formulation's ε is 0 and
the config's `1e-8` is dead for the solve — the stored config value is not the one used.

**Recommended action.** Restore/commit the exact driver (the Frontal independent-remesh sweep function)
that produced the committed 3D files, or re-run them with a committed, named function and archive the
config snapshot alongside, per `reproducible-results.md`. Make `build_config`'s `eps_val` and the
formulation's `eps_phys` agree (or document why they differ).

### B.5 P2-3D under-stabilization — RESOLVED

The structured-Kuhn control (uniform, refinement-invariant quality) ran P2 ASGS+OSGS at paper c₁ and
ASGS at c₁×4; the c₁×1-fails/grows vs c₁×4-collapses-L²u-~40× numbers are real Gridap data. The c₁ root
cause and the OSGS-P2-3D coupling are **RESOLVED** — see findings.md section 3 and open-questions.md
section 4. (This audit's earlier c1-discrepancy / element-family-coercivity-deficit framing is
SUPERSEDED.)

### B.6 [Med] A failed OSGS solve silently reports the ASGS Stage-I state under the OSGS label

*(Resolved — the OSGS-degenerated-to-ASGS case is now marked distinctly; the finding and its mechanism are kept below.)*

The control reproduced the byte-identical anomaly (B.3) **and revealed its mechanism.** Every P2-OSGS
solve at paper c₁ failed (`success=False`) and reported an error **bit-identical to the standalone
P2-ASGS run at the same level** — including at (20,20,5) where ASGS *succeeded* (1.3101e-2) and OSGS
*failed* but still reported 1.3101e-2. Because the OSGS path runs the **ASGS Stage-I boot first**
([solver_core.jl:511](src/solvers/solver_core.jl#L511)) and the coupled solve, on failure, leaves the
iterate at that ASGS state, **a non-converging OSGS solve degenerates to ASGS and reports ASGS's error
under the OSGS label.** [osgs_solver.jl:163-168](src/solvers/osgs_solver.jl#L163) already flags this exact
failure mode as a `[known-fragility]` for the stall sensor; the control shows it also happens via
`linesearch_failed`. When OSGS *does* converge (all P1 cases), it produces genuinely different, more
accurate results — so the degeneration is silent and only bites on the hard (P2) cases that most need
OSGS.

**Lesson.** When the OSGS coupled solve does not reach its own convergence verdict, mark the result
distinctly (e.g. `method="OSGS(degenerated→ASGS)"` or a separate flag) so a sweep cannot record an ASGS
error in an OSGS column. Combined with B.2/B.3, this stops the failed-solve numbers from masquerading as
method comparisons.

---

## Appendix 1 — Control experiment: full 3D structured-Kuhn data

Paper §5.2 case (Deviatoric, Constant-σ, Re=Da=1, α₀=0.5), structured Kuhn simplex mesh, domain
(0,1)×(0,1)×(0,0.3). Per-segment rates between consecutive levels (optimal: L²u=k+1, H¹u=k, L²p=k).
`†` = a segment that includes a `success=False` solve (rate is meaningless).

| case | h-pairs | L²u rates | H¹u rates | L²p rates | verdict |
|---|---|---|---|---|---|
| P1 ASGS c₁×1 | 0.149/0.099/0.075/0.060 | 1.24, 1.06, 1.27 | 0.92, 0.67, 0.88 | 0.84, 0.83, 0.95 | **sub-optimal** (L²u≈1.2) |
| P1 OSGS c₁×1 | same | 2.20, 1.78, 2.08 | 0.99, 0.86, 0.98 | 1.67, 1.93, 1.60 | **optimal** |
| P2 ASGS c₁×1 | same | 1.64†, 5.24, **−0.82** | 0.63†, 4.08, **−1.96** | 0.45†, 5.14, **−1.14** | **defect** (errors grow on converged fine pair; coarse solves fail) |
| P2 OSGS c₁×1 | same | all †† | all †† | all †† | all solves failed → reports ASGS state (B.6) |
| P2 ASGS c₁×4 | same | 1.67, 2.67, 2.30 | 0.83, 1.76, 1.26 | 0.94, 1.73, 1.48 | **converges, climbing to optimal**; ≈40× smaller errors |

Headline reads: P1 ASGS sub-optimal vs P1 OSGS optimal on the *same* mesh (B.1); P2 at paper c₁
under-stabilizes on a *uniform* mesh (B.5 — RESOLVED, see findings.md section 3); a failed OSGS solve
reports the ASGS state (B.6).

## Appendix 3 — Method & reproducibility

- Multi-agent audit run record (53 findings, per-finding two-lens verdicts): workflow `wf_9dc5a56d-800`.
- 2D rate recomputation: `previous_results/validated_k1_quad_N640/phase1_quad_k1.h5` (48 configs).
- 3D committed-data reanalysis: `previous_results/convergence3d/convergence3d_results_{frontal_c1x1,pre_frontal}_*.json`.
- 3D control experiment: driver `/tmp/audit_3d_structured.jl`, log
  `test/extended/ManufacturedSolutions3D/results/debug_results/audit_3d_structured.log`.
- Every code `file:line` in this document was opened and read; every results claim was recomputed from
  raw error arrays, per the "do not trust the reports" directive.
