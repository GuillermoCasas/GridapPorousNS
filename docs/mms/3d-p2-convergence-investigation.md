# 3D ASGS sub-optimality (P₁ & P₂): a c₁/coercivity fixed point, not mesh quality

> **⚠️ SUPERSEDED (2026-06-30).** This doc's headline conclusion — *"paper c₁=4k⁴ under-budgets coercivity
> for 3D tetrahedra; c₁ is the lever"* — was **WRONG**. It was masking a missing term: the **Codina
> iterative penalty** (`ε_num·(pⁿ−pⁿ⁻¹)` in the mass residual, article.tex §5.2 line 1383) was never
> implemented in the residual (only the Jacobian), so the all-Dirichlet 3D problem was ill-posed at ε=0.
> Adding it makes **ASGS-3D converge at PAPER c₁ with optimal rate** (c₁×4 only shrank the error constant,
> which masked this). Kratos solves this 3D case at paper c₁ — there is no dimensional c₁. The new canonical
> doc is **[3d-iterative-penalty-fix-and-osgs-coupling.md](3d-iterative-penalty-fix-and-osgs-coupling.md)**;
> read it first. The §4 "iteration/Jacobian exonerated" content here remains valid; the c₁ verdict does not.
>
> **Status (historical):** was canonical for 3D tetrahedral MMS (paper §5.2) — ASGS vs OSGS,
> the c₁ lever, ε, and mesh quality. Companion to the 2D [convergence-status.md](convergence-status.md).
> Last updated 2026-06-24. Harness: `test/extended/ManufacturedSolutions3D/` (`smoke3d.jl`, `mms3d.jl`,
> `mesh3d.jl`). Convergence results: `results/k<kv>/TET/<mesh_sequence>/`; mesh sequences +
> mosaics: `results/meshes/<seq>/`.
>
> **2026-06-24 revision.** The earlier hypothesis of this doc — *"gmsh-Delaunay bad element quality
> exhausts the C_inv margin; regularize the mesh and convergence returns at the uniform paper c₁"* — was
> **tested and falsified**. A **structured Kuhn** tet family (uniform, refinement-invariant quality,
> qmin≈0.618) still gives **sub-optimal ASGS** at paper c₁ (P₁) and **erratic/diverging P₂**, while
> **OSGS is optimal on the same meshes**. So the deficit is a **coercivity-limited discrete fixed point**
> that the paper's `c₁ = 4k⁴` under-budgets for 3D tetrahedra *even at good quality*; mesh quality
> *modulates* it but is not the root. The nonlinear iteration, the analytic Jacobian, and the solver
> orchestration are all **rigorously exonerated** (§4).

## TL;DR (verdict)

For the §5.2 manufactured solution on tetrahedra:

- **P₁ ASGS is mildly SUB-OPTIMAL** at paper c₁ — L²u ≈ 1.2–1.4 (optimal 2), H¹u ≈ 0.85 (optimal 1) — even
  on a **perfect, constant-quality structured mesh**, while **P₁ OSGS is OPTIMAL** there (L²u ≈ 1.9–2.0,
  L²p super-convergent). The ASGS↔OSGS gap is the *only structural difference* — the OSGS orthogonal
  projection of the residual — so it is **not** a mesh-quality effect.
- **P₂ DIVERGES / is erratic** at paper c₁ (errors grow under refinement) — on both gmsh-unstructured *and*
  perfect structured meshes — confirming the same c₁/coercivity deficit, sharper at P₂.
- **c₁ is the single lever.** On a *frozen* mesh, scaling c₁ up reduces the converged error and lifts the
  rate (mesh-frozen A/B: ASGS L²u 1.21→1.70, L²p 0.58→1.31; the (16,16,4) point L²u 0.0113→0.0062 at
  c₁×4). c₁ does **not** move H¹u (≈0.83→0.86) — H¹u is the genuinely quality-bound quantity on unstructured
  meshes. This is exactly the paper's coercivity condition `c₁ > 2ξ C_inv²`.
- **The nonlinear iteration is NOT the cause.** Verified directly (§4): the analytic ExactNewton Jacobian
  is the exact ∂R (Taylor test, O(ε²) to round-off); the logged residual is the true assembled residual; a
  bare Newton with zero orchestration lands on production's solution. The sub-optimal field is a genuine
  **converged discrete fixed point**, and c₁ changes that fixed point.
- **ε is the numerical iterative penalty only, and `eps_phys` must be 0** (incompressible). A transient
  working-tree bug had set a non-zero `eps_phys` default (solving a *compressible* problem); reverted. ε is
  rate-irrelevant for P₁ (§5).

**Why OSGS optimal but ASGS sub-optimal here, when the high-Da story is the opposite?** Same mechanism,
opposite sign by regime — see the reconciliation in §6 and `docs/solver/osgs-reaction-dominated-rate.md`.

---

## 1. The observation

Paper §5.2: z-extruded manufactured field on a `(0,1)×(0,1)×(0,0.3)` slab, `(α₀, Re, Da) = (0.5, 1, 1)`,
ConstantSigma reaction, Deviatoric viscous operator, tetrahedra, ASGS and OSGS. Optimal rates: L²u
`O(h^{k+1})`, H¹u/L²p `O(h^k)`.

- **P₁:** ASGS L²u ≈ 1.2–1.4 (sub-optimal), H¹u ≈ 0.85, L²p ≈ 0.9 — converges but short of optimal. OSGS
  L²u ≈ 1.9–2.0, H¹u ≈ 0.95, L²p super-convergent — **optimal**.
- **P₂:** errors **grow under refinement** (e.g. H¹u 0.45→1.02 as h halves on early gmsh ladders;
  non-monotone L²u 0.022→0.049→0.056→0.011 on a structured ladder, one mesh `ok=false`). "Diverges."

The original gmsh-unstructured P₂ ladder (`exp12`, ASGS, Deviatoric, ε off), against the achieved tet
size h = (6√2·V)^{1/3}:

| level | h | cells | L²u | L²p | H¹u | Newton |
|------:|------:|------:|------:|------:|------:|:------|
| 0 | 0.206 | 259  | 0.02288 | 0.1284 | 0.4537 | Exact-Newton, ‖R‖≈2.2e-16, success |
| 1 | 0.103 | 2072 | 0.02303 | 0.2568 | 1.015  | Exact-Newton, ‖R‖≈7.5e-15, success |

L²u flat, **L²p and H¹u nearly double as h halves** — the discrete solution gets *worse* under refinement,
while the solve is healthy (`success=True`, deep ‖R‖). **The solve is fine; the discretization is not.**

---

## 2. The decisive control: a perfect structured mesh (falsifies "mesh quality")

The earlier hypothesis predicted: *regularize the mesh → optimal convergence returns at paper c₁, isolating
mesh quality as the cause.* We ran exactly that control — a **structured Kuhn** tet family
(`simplexify(CartesianDiscreteModel)`, `structured_kuhn_model`), which has **uniform, refinement-invariant**
element quality (qmin ≈ 0.618), eliminating mesh quality as a variable. Results (paper c₁, `eps_phys=0`,
`results/k1/TET/structured/`):

| method | L²u (opt 2) | H¹u (opt 1) | L²p (opt 1) |
|---|---|---|---|
| **OSGS** | 2.03 → 1.92 → 1.91 | 0.94 → 0.92 → 0.96 | 1.79 → 1.69 → 1.74 |
| **ASGS** | 1.16 → 1.27 → **1.40** | 0.82 → 0.84 → 0.92 | 0.84 → 0.97 → 1.09 |

> **Provenance.** The per-segment table above is the **constant Nx:Nz-aspect** ladder
> ((8,8,2)→(16,16,4)→(24,24,6)→(32,32,8)), which holds Kuhn quality exactly constant (qmin 0.618) — the
> cleanest illustration. The **committed** `results/k<kv>/TET/structured/` is the *shared* LU-feasible
> ladder ((10,10,3)→(16,16,4), so P₁ and P₂ sit on identical grids for comparison); it carries the same
> headline — global L²u: **ASGS 1.30, OSGS 2.26** — but with mild rate noise because those partitions do
> *not* hold a constant aspect (qmin wobbles 0.657/0.618/0.650/0.618). Both ladders were generated by a
> one-off structured-Kuhn sweep over a partition list (driver not retained — recreatable via `smoke3d.jl`'s
> `solve_one` on `structured_kuhn_model` partitions).

**The prediction failed.** On a flawless, constant-quality mesh:

- **OSGS is fully optimal** (L²u ≈ 1.9, H¹u ≈ 0.95, L²p super-convergent).
- **ASGS L²u is still ~1.4** — sub-optimal — and **P₂ is still erratic/divergent** at paper c₁
  (`results/k2/TET/structured/`: ASGS L²u 0.0216→0.0493→0.0561→0.0109, non-monotone, H¹u up to 2.6, one
  `ok=false`; **OSGS-P₂ fails the convergence gate on all four meshes** — `ok=false`, non-monotone, global
  rates *negative* (L²u −0.02, H¹u −1.07). Both methods are unstable for P₂-3D at paper c₁).

Therefore the ASGS↔OSGS L²u gap, and the P₂ divergence, are **not** mesh quality — they persist when mesh
quality is perfect and constant. The *only* structural difference between the two curves is the OSGS
orthogonal projection of the strong residual. The deficit is a property of the **ASGS discrete problem at
paper c₁**, modulated (not caused) by mesh quality.

> **Boundary conditions were also cleared.** The harness imposes the full exact-velocity Dirichlet on all
> six faces (incl. the top/bottom z-faces), i.e. the fully-consistent MMS setup — not a natural/slip
> condition — so BCs are not a 3D-only source of error.

---

## 3. What changes it: the stabilization constant c₁ (mesh frozen)

**Mesh-frozen A/B (the clean isolation).** On the *identical* nested pair (L0,L1), changing only c₁:

| quantity | c₁×1 rate | c₁×4 rate | optimal | L1 error: ×1 → ×4 |
|---|---|---|---|---|
| L²u | 1.21 | **1.70** | 2 | 0.01129 → 0.00724 (−36%) |
| H¹u | 0.83 | **0.86** | 1 | 0.2039 → 0.1910 (−6%) |
| L²p | 0.58 | **1.31** | 1 | 0.1763 → 0.0627 (−64%) |

With the mesh held byte-for-byte fixed, c₁ is unambiguously the lever for **L²u and L²p** (errors fall,
rates lift); **H¹u barely moves** (quality-bound on unstructured tets). The single-mesh verification (§4)
shows the same: structured (16,16,4) ASGS L²u **0.0113 (c₁×1) → 0.0062 (c₁×4)**.

**Original gmsh P₂ A/B (`exp13`, ASGS, ε off, levels 0–1):**

| c₁ | rates L²u / H¹u / L²p |
|---:|---|
| **×1** (paper 4k⁴=64) | **−0.01 / −1.16 / −1.00** (diverging) |
| **×4** (256) | **+2.15 / +1.10 / +1.69** (converging) |

At paper c₁ every P₂ rate is negative (errors grow); at c₁×4 every rate flips positive (and errors 3–13×
smaller). Among ε / tolerances / mesh, **only c₁** flips P₂-3D from diverging to converging.

---

## 4. The nonlinear iteration is NOT the cause (rigorously verified)

Earlier this doc *asserted* the nonlinear iteration was healthy (deep ‖R‖). 2026-06-24 we **proved** it, on
the structured (16,16,4) P₁ ASGS cell at c₁×1 and c₁×4 (a one-off diagnostic, not retained; method per
`docs/solver/coupled-only-leaning-and-jfnk-plan.md` §4). Three checks, both c₁:

1. **[A] Residual-identity.** The number logged as `f(x) inf-norm` is the *raw assembled* `‖R(U)‖∞` (no
   gate/scaling) — confirmed by independent re-assembly: production converges to 2.14e-7, our assembly gives
   2.14e-7; the interpolant gives 2.7e-3. (Production *decides* convergence via the dimensionless scale-free
   gate `ε_M,ε_C`, but the residual it reports is the true one.) **Caveat / interesting detail:** the
   residual *floors* at 2.14e-7, not machine-zero — this is the constant-pressure null mode / global-mass
   quadrature mismatch with `eps_phys=0` (exactly the well-posedness issue the paper's ε>0 addresses on
   unstructured meshes). It is **~5 orders below** the 1e-2 discretization error, so it does not affect the
   rates.
2. **[B] Jacobian Taylor test.** `‖R(U+εδU) − R(U) − ε·J(U)δU‖` drops **~100×/decade** (c₁×1:
   8.7e-12 → 8.7e-14 → 1e-15 floor; c₁×4: 5.3e-12 → 5.3e-14 → 1e-15) — textbook O(ε²). The analytic
   ExactNewton Jacobian **is** the exact derivative of R. No Jacobian bug.
3. **[C] Bare Newton from the interpolant** (analytic J, plain LU, no Picard/homotopy/verifier/scale-free
   gate) lands on production's solution: **velocity DOFs agree to ~3e-6**, MMS errors identical
   (L²u 0.011311 = 0.011311 at ×1; 0.0061694 = 0.0061694 at ×4). The 43.9 total DOF gap is purely the
   unpinned pressure constant (`eps_phys=0`), removed by the mean-subtraction in the error norm.

**Verdict:** iteration, Jacobian and orchestration are exonerated. The sub-optimal field is a genuine
converged discrete fixed point — consistent with the structural fact that, for a deeply-converged Newton,
the converged solution is a property of **R** (the discrete weak form), not of **J**. **c₁ changes that
fixed point** (0.0113→0.0062 on a fixed mesh), so the lever is the discrete problem's coercivity, not the
solver.

---

## 5. The ε's: `eps_phys=0` (incompressible), `eps_num` numerical-only

The equation is **incompressible**. The only ε is the **numerical iterative penalty** `numerical_epsilon`
(`eps_num`), implemented **Jacobian-only**: lagging `ε_num·p` to the iterate leaves `ε_num·dp` in the
Jacobian pressure block, which **vanishes at convergence** — article.tex `eq:StrongMassEquation` (line 239),
the "mainly numerical reasons / iterative penalty" remark (line 241), and the 3D-MMS remark (lines
1375–1383). The 2D examples use ε=0 outright (line 1098).

- **The physical compressibility `eps_phys` (formulation `eps_val`) must be 0.** A transient *working-tree*
  bug (2026-06-24) had set `solve_one`'s `eps_phys` default to a non-zero `1e-4·α₀/(ν(1+Re+Da)) ≈ 1.67e-5`,
  putting `ε·p` in the residual **and** `ε·p_ex` in the oracle `g` — i.e. solving a genuinely *compressible*
  problem. Reverted to `0.0` (committed-correct). See the `lessons_learned` entry.
- **ε is rate-irrelevant for P₁.** With `eps_phys=0` the P₁ structured rates are identical to 4 digits vs
  the buggy run. And the historical A/B (`exp12b`, `eps_mult ∈ {0,1,20}` scaling `eps_num`) gave identical
  converged errors (20× slightly *worse*, consistent with `ε ≤ C₂·c₁·inf{α²τ₁/h²}`, `eq:UpperBoundOnEpsilon`).
  ε buys **pressure well-posedness**, not coercivity or rate.

The canonical spec for the ε distinction and the scale-free convergence gate is
`docs/solver/nonlinear-convergence-criterion-prompt.md` (its §4.1 already states the incompressible,
ε≈1e-4-iterative-penalty handling).

---

## 6. Mesh-quality mechanics (modulates, not root) + the OSGS reconciliation

Mesh quality is real but enters as the constant `C_inv`, not as the cause of the ASGS↔OSGS gap:

- **Independent gmsh remeshes SCATTER.** Each level is a fresh, differently-flawed realization whose error
  level is set by its own quality, so per-segment "rates" are realization noise, not h-convergence — we
  measured *impossible* P₂ per-segment rates (5.81, 8.49) and a 2× error spread at fixed h. **Do not fit a
  rate across independent unstructured remeshes.**
- **Nested red refinement DEGRADES min-quality** (qmin 0.466→0.247→0.151 on a Frontal+optimize base) — it
  is *not* quality-preserving (the `mesh3d.jl` comment claiming otherwise was corrected). So neither gmsh
  strategy holds quality fixed; only the structured Kuhn family does (§2).
- **gmsh Frontal-Delaunay (alg 4) + Netgen/Relocate3D optimize** lifts the quality tail vs plain Delaunay
  (alg 1) and improves H¹u somewhat — but does **not** close the ASGS L²u gap (that needs c₁).

**OSGS-optimal-here vs OSGS-worse-at-high-Da (reconciliation).** This doc finds OSGS *better* than ASGS
(low Da=1, 3D). `docs/solver/osgs-reaction-dominated-rate.md` + `theory/osgs_reaction_note/` find OSGS
*worse* at high Da (reaction-dominated). These are **opposite orderings in different regimes, same
mechanism**: OSGS projects the FE-representable part `Π_h R` out of the subscale. At **high Da** the
reaction term `σu` lives largely in the FE space, so OSGS discards reactive stabilization → weaker velocity
control (OSGS worse). At **low Da** the convective/pressure residual dominates, and removing its
FE-representable part is exactly what cleans up the velocity-L² (OSGS better; ASGS retains that part and
pays for it). So the two findings are consistent, not contradictory — the OSGS↔ASGS ordering is
regime-dependent.

---

## 7. Hypothesis (refined): paper c₁ under-budgets coercivity for 3D tets

The paper's stability estimate (`lemma:Stability`, line ~916) needs the **coercivity condition**
(`eq:conditions_on_num_param`, line ~905): `c₁ > 2ξ C_inv²`, where `C_inv` (inverse-estimate constant,
h-independent) depends on element **order** (∼k², so `C_inv²∼k⁴`) **and shape**. The Remark (line ~910)
chooses `c₁ = 4k⁴, c₂ = 2k²`, the **4** absorbing `2ξ` and the geometry part.

**Refined claim (post structured-control):** the paper's uniform `c₁ = 4k⁴` is **insufficient for 3D
tetrahedra even at good (structured) element quality** — the geometry/dimension margin in the constant **4**
is too small for 3D tets, so `c₁ = 4·2⁴ = 64` falls below `2ξ C_inv²` at P₂ (divergence) and is marginal at
P₁ (L²u ≈ 1.4). Poor unstructured quality (large geometric `C_inv`) makes it *worse* and *noisier*, but the
deficit is already present on a flawless mesh. Raising c₁ restores `c₁ > 2ξ C_inv²` (§3). This is consistent
with the paper's own statement that the effective c₁ is element-type-dependent — but the new fact is that
the dependence bites in 3D **independent of mesh quality**, which the earlier "it's just the gmsh mesh
generator" reconciliation did not capture.

**Open:** the paper reports optimal P₂-3D slopes at the same uniform c₁. If even a perfect structured mesh
needs c₁ > paper-c₁, the remaining explanations are (a) the paper's effective c₁/τ scaling differed, (b) a
subtle viscous/τ 3D term, or (c) the paper's mesh-diameter `h` convention vs our `(6√2V)^{1/3}` shifts τ.
H¹u being quality-bound (c₁-immovable) also suggests a separate, mesh-shape-limited component on unstructured
grids. Not yet resolved.

---

## 8. P₂ specifics: instability + the memory wall

- **P₂ is unstable at paper c₁ even on structured meshes** (§2) — non-monotone errors, occasional
  `ok=false` (the solver reaches the scale-free gate but on a refinement-worsening branch). c₁×4 tames the
  catastrophic divergence (`exp13`) but does not reach optimal on accessible meshes.
- **P₂-3D is memory-capped.** Direct LU OOMs on this 32 GB machine past ~30–40k DOF; `(24,24,6)` ≈ 104k DOF
  auto-switches to ILU-GMRES, which stalls on the indefinite (u,p) saddle point (>3 h, no convergence — the
  ILU is a weak preconditioner there). So P₂ structured is confined to a narrow LU-feasible h-window
  ((10,10,3)→(16,16,4)), and the OSGS finest meshes are unreachable. This is a hardware/linear-solver limit,
  not a method limit — and is the motivation for the deferred **JFNK** plan
  (`coupled-only-leaning-and-jfnk-plan.md` §4). Recorded in `docs/known_issues.md`.

---

## 9. Next steps (status)

1. ~~Regularize the mesh (structured tet) and rerun P₂ at uniform c₁ — predict convergence returns,
   isolating mesh quality~~ — **DONE and FALSIFIED (§2):** structured Kuhn still sub-optimal/erratic at paper
   c₁; the cause is c₁/coercivity, not quality.
2. **Bound the c₁ requirement.** Sweep c₁×{1,2,4,8} on the structured family (P₁ and the LU-feasible P₂
   window) and find the smallest c₁ that makes ASGS L²u→2 and P₂ monotone. If a modest constant suffices,
   it argues for an element-type-aware c₁ (a labelled `[code-actual]` 3D-tet deviation), not a per-mesh hack.
3. **Characterize the retained-`Π_h R` mechanism** directly: measure `‖Π_h R‖` in the converged ASGS field,
   and test an "ASGS-minus-projectable-part" variant — does it recover OSGS-like L²u? This explains *why*
   the ASGS fixed point is less accurate (mechanism, not just lever).
4. **H¹u quality ceiling:** on unstructured grids H¹u is c₁-immovable. Either accept it as the genuine
   shape-quality limit, or test a quality-bounded mesher (TetGen `-q`) that gmsh cannot match (gmsh caps
   ~0.4; red refinement degrades).
5. **JFNK for P₂** to clear the memory wall (§8) and reach the finer P₂ meshes.

---

## Appendix A: the numerical-ε implementation

The Codina iterative penalty (`numerical_epsilon`, ε_num) is **Jacobian-only**, not an outer loop. Lagging
`ε_num·p` to the iterate gives `ε_num·pᵏ⁺¹ − ε_num·pᵏ = ε_num·dp`, so the **residual** uses `eps_val`
(=`eps_phys`, which **must be 0** for the incompressible MMS) and the **Jacobian** mass term uses
`eps_phys + ε_num` (`continuous_problem.jl`, both Jacobian builders). It is a pressure-block Newton-step
regularization that **vanishes at convergence** — no consistency error, no manufactured-source change,
bit-identical at `ε_num = 0`. Bounded by `ε_num ≤ C₂·c₁·inf{α²τ₁/h²}` (`eq:UpperBoundOnEpsilon`).

## Appendix B: experiment ledger

| id | what | result |
|---|---|---|
| `exp12`  | P₂ ASGS gmsh ε-off ladder | errors grow under refinement; Newton converges every level |
| `exp12b` | P₂ ASGS A/B `eps_mult ∈ {0,1,20}` (scales ε_num) | identical errors; 20× slightly worse ⇒ ε_num does not fix it |
| `exp13`  | P₂ ASGS A/B `c1_mult ∈ {1,4}`, ε off | rates −0.01/−1.16/−1.00 (×1) → +2.15/+1.10/+1.69 (×4): c₁ is the lever |
| structured P₁ (2026-06-24) | ASGS+OSGS, paper c₁, `eps_phys=0`, Kuhn qmin≈0.618 | OSGS optimal (L²u≈1.9); ASGS sub-optimal (L²u≈1.4) — **falsifies mesh-quality** |
| structured P₁ c₁ A/B | ASGS c₁×1 vs c₁×4, frozen mesh | L²u 1.21→1.70, L²p 0.58→1.31, H¹u 0.83→0.86 |
| mesh-frozen verification | (16,16,4) P₁ ASGS, c₁×1/×4 | [A] raw‖R‖=logged (floors 2.14e-7); [B] Jacobian Taylor O(ε²); [C] bare-Newton=production; L²u 0.0113→0.0062 |
| structured P₂ (2026-06-24) | ASGS+OSGS, paper c₁, `eps_phys=0`, shared LU-feasible ladder (10,10,3)→(16,16,4) | ASGS erratic/non-monotone (L²u 0.022→0.049→0.056→0.011, one `ok=false`); OSGS **all four `ok=false`**, global rates negative (−0.02 / −1.07 / −0.66) — **P₂ unstable even on a perfect mesh, both methods** |
