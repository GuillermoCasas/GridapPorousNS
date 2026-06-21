# 3D P₂ convergence failure — investigation, traces, and hypothesis

> **Status:** Canonical for the 3D-P₂ (tetrahedral, paper §5.2) convergence failure.
> Companion to the 2D [convergence-status.md](convergence-status.md). Last updated 2026-06-21.
> Harness: `test/extended/ManufacturedSolutions3D/` (`smoke3d.jl`, `mms3d.jl`, `mesh3d.jl`).
> Raw experiment scripts/logs: `test/extended/ManufacturedSolutions3D/results/debug_results/exp12*`, `exp13*`.

## TL;DR (verdict)

For the §5.2 manufactured solution on **quadratic (P₂) tetrahedra**, the discrete errors **grow under
mesh refinement** (e.g. H¹-velocity 0.45 → 1.02 as h halves), while **every other configuration converges
optimally** (all 2D cases; 3D P₁). The failure is **NOT**:

- nonlinear convergence — Newton reaches ‖R‖ ≈ 1e-11…1e-15 (far below the gate), `success=True`;
- the tolerances/gates — eps_M, ftol, eps_C are all satisfied with margin;
- the formulation equations — the analytic residual/Jacobian forms match (P₂ Hessian exact, 3D
  deviatoric grad-div verified);
- the mesh red-refinement — independent remeshes diverge the same way;
- the numerical compressibility ε — it is Jacobian-only and vanishes at convergence; an A/B across
  `eps_mult ∈ {0, 1, 20}` gives **identical** converged errors (ε is for pressure well-posedness, exactly
  as article.tex §5.2 states, not for coercivity/rate).

The single lever that **does** change it is the stabilization constant **c₁**: scaling c₁ up restores
convergence. **Most-likely hypothesis:** our gmsh-Delaunay P₂ tetrahedra have a larger inverse-estimate
constant `C_inv` (poorer worst-case shape quality) than the uniform `c₁ = 4k⁴` accommodates, so the
coercivity condition `c₁ > 2ξ C_inv²` (article.tex `eq:conditions_on_num_param`) is **violated** at P₂ in
3D and the method loses coercivity. The paper obtained optimal P₂-3D slopes on its own (commercial-app)
unstructured mesh with the *same* c₁; the difference is the **mesh generator / element quality**, i.e. a
**numerical-parameter (c₁) calibration tied to mesh quality** — not a formulation-equation bug.

---

## 1. The observation

Paper §5.2: z-extruded manufactured field on a `(0,1)×(0,1)×(0,0.3)` slab, `(α₀, Re, Da) = (0.5, 1, 1)`,
unstructured tetrahedra, P₁ (4 meshes) and P₂ (3 meshes), ASGS and OSGS. Optimal rates are L²u `O(h^{k+1})`,
H¹u/L²p `O(h^k)`. P₁ matches; **P₂ diverges**.

`exp12` (ASGS, Deviatoric, ε off), measured against the achieved tet size h = (6√2·V)^{1/3}:

| level | h | cells | L²u | L²p | H¹u | Newton |
|------:|------:|------:|------:|------:|------:|:------|
| 0 | 0.206 | 259  | 0.02288 | 0.1284 | 0.4537 | Exact-Newton, ‖R‖≈2.2e-16, success |
| 1 | 0.103 | 2072 | 0.02303 | 0.2568 | 1.015  | Exact-Newton, ‖R‖≈7.5e-15, success |

L²u is flat, **L²p and H¹u nearly double as h halves** — the discrete solution gets *worse* under
refinement. The finest paper mesh (`exp12` level-2 / trace `traj_…kv2…ASGS_N23.json`, 27968 P₂-tets,
h=0.0434) likewise solves cleanly — initial ‖R‖=1.07e-3 → final ‖R‖=2.78e-11, `success=True`, 2
iterations — yet carries the same growing error. **The solve is healthy; the discrete solution is not.**

---

## 2. What was ruled out (and how)

### 2a. Nonlinear convergence — NO
Every P₂ solve reaches the Exact-Newton quadratic basin and stops on `ftol_reached` with ‖R‖ between
2e-16 and 3e-11. The pathological "huge step" seen earlier (step ≈15) is the near-singular *pressure*
block from the §5.2 indeterminacy, which the solver absorbs (and which ε regularizes — see 2e); it does
not prevent convergence. A converged residual of 1e-11 with a growing discretization error is the
signature of a **stable solver on an unstable discretization**, not the reverse.

### 2b. Tolerances and gates — NO (second look)
| control | value (k=2) | achieved | limiting? |
|---|---|---|---|
| `eps_tol_momentum` (eps_M gate) | 1e-9 | ‖R‖/D_M ⇒ eps_M ~1e-11 or smaller | no — satisfied with ≥2 orders margin |
| `ftol` (→ `dynamic_ftol`) | 1e-12 | ‖R‖ reaches 2e-16…3e-11 | no — solves to/below ftol |
| `eps_tol_mass` (eps_C gate / tol_C) | 0.8 | eps_C ≈ 0.27 | no — passes; looseness intentional (mass-residual scale sensitivity, see the k=2-gate lesson) |
| `numerical_epsilon` | 1e-4·ε_ref | — | irrelevant to errors (2e) |

The k=2 gate (`eps_tol_momentum=1e-9`, `ftol=1e-12`, `eps_tol_mass=0.8`) is exactly the tightened gate the
2D work established; it is satisfied with margin here. **Tolerances are not stopping the solve early and
are not the cause.**

### 2c. Formulation equations — NO
The earlier diagnosis confirmed the analytic momentum/mass residual and Jacobian forms match between the
oracle and the assembled operator, the discrete P₂ Hessian is exact to 1e-13 on base+refined meshes, and
the 3D deviatoric identity `∇·(2αν∇ᵈu) = 2ν(∇ˢu·∇α − ⅓(∇·u)∇α) + ανΔu + (αν/3)∇(∇·u)` is implemented and
oracle-matched (`mms3d.jl`). A wrong term would not be cured by a pure *magnitude* scaling of c₁ (2d).

### 2d. Mesh red-refinement artifact — NO
Independent `build_box_tet_model` remeshes (not nested red-refinement) diverge identically, so the
behavior is intrinsic to the P₂-on-tet discretization, not a refinement-bookkeeping artifact.

### 2e. The numerical compressibility ε — NO (A/B, `exp12b`)
After the redesign, ε is a **Jacobian-only** pressure-block regularization: lagging `ε·p` to the iterate
cancels it in the residual, leaving `ε·dp` only in the Jacobian, which **vanishes at convergence** (see
[the ε note](#appendix-a-the-numerical-ε-redesign)). So it cannot move a converged solution. Confirmed:

| eps_mult | level 0 (h=0.206) L²u/L²p/H¹u | level 1 (h=0.103) L²u/L²p/H¹u |
|---:|---|---|
| 0  | 0.02288 / 0.1284 / 0.4537 | 0.02303 / 0.2568 / 1.015 |
| 1  | 0.02288 / 0.1284 / 0.4537 | 0.02312 / 0.2579 / 1.019 |
| 20 | 0.02289 / 0.1284 / 0.4539 | 0.02422 / 0.2712 / 1.068 |

Errors are identical (level 0 byte-identical; level 1 within ~0.4% — the residual `ε·dp` at the
ftol-limited iterate) and **20× ε is slightly *worse***, consistent with the stability bound
`ε ≤ C₂·c₁·inf{α²τ₁/h²}` (`eq:UpperBoundOnEpsilon`). This is exactly what article.tex §5.2 says ε is for:
*"the unstructured nature of the mesh makes it difficult to ensure that the boundary conditions … are
compatible with mass conservation … leading to an ill-posed problem. Thus … we have resorted to the
compressibility, taking ε > 0, which makes the problem well-posed [and] eliminates the indeterminacy in
the pressure."* ε buys **well-posedness / pressure-indeterminacy removal**, not coercivity or rate.

---

## 3. What changes it: the stabilization constant c₁

`exp13` (ASGS, Deviatoric, **ε off in both arms**, `c1_mult ∈ {1, 4}`, levels 0–1):

| c₁ | level 0 (h=0.206) L²u/L²p/H¹u/H¹p | level 1 (h=0.103) L²u/L²p/H¹u/H¹p | rates L²u / H¹u / L²p |
|---:|---|---|---|
| **×1** (paper 4k⁴=64) | 0.02288 / 0.1284 / 0.4537 / 2.317 | 0.02303 / 0.2568 / 1.015 / 11.0 | **−0.01 / −1.16 / −1.00** (diverging) |
| **×4** (256) | 0.008046 / 0.1087 / 0.1815 / 1.756 | 0.001807 / 0.03359 / 0.0848 / 1.247 | **+2.15 / +1.10 / +1.69** (converging) |

The result is unambiguous: at the paper's uniform c₁ every rate is **negative** (errors grow, H¹p blows up
2.3 → 11); at c₁×4 every rate flips **positive** and the absolute errors are 3–13× smaller. (The c₁×4 rates
are not yet the optimal P₂ values — 3 / 2 / 2 — because this is a deliberately coarse pre-asymptotic
2-mesh ladder; the point is the sign flip, not the asymptote.) The c₁ knob — and, among ε / tolerances /
mesh, **only** the c₁ knob — flips P₂-3D from diverging to converging, matching the prior `exp6` finding.

---

## 4. Hypothesis: loss of coercivity via an under-budgeted `C_inv`

The paper's stability estimate (`lemma:Stability`, article.tex line ~916) holds under the **coercivity
condition** (`eq:conditions_on_num_param`, line 905):

```
c₁ > 2 ξ C_inv²
```

where `C_inv` is the constant in the inverse estimate `‖∇^m ψ_h‖ ≤ C_inv h^{-(l-m)} ‖∇^l ψ_h‖` (line 857),
**independent of h** but **dependent on element order and shape**. The Remark (line 910) is explicit:

> *"the optimal value of c₁ depends on the element types involved through the inverse estimate constant.
> In particular, for elements of the same polynomial order k …, taking c₁ = 4k⁴, c₂ = 2k² turns out to be
> effective … and was the choice made in all the numerical experiments … consistent with the quadratic
> dependence of `C_inv` on the polynomial order, which is known to grow as k²."*

So `4k⁴` budgets for `C_inv ∼ k²` (the **order** dependence: `C_inv² ∼ k⁴`), with the constant **4**
absorbing `2ξ` and the **element-geometry / shape-quality** part of `C_inv`. The geometry part is *not*
universal — it blows up for distorted (sliver / large-dihedral) tetrahedra.

**Claim:** our gmsh-Delaunay P₂ tetrahedra (`mesh3d.jl`, `algorithm=1`) have a worse worst-case shape
quality — hence a larger geometric `C_inv` — than the paper's commercial-meshing-app tetrahedra. At P₂ in
3D the geometry margin in the constant **4** is exhausted, `c₁ = 4·2⁴ = 64` drops below `2ξ C_inv²`, the
discrete bilinear form loses coercivity, and the error grows under refinement. Scaling c₁ up restores
`c₁ > 2ξ C_inv²` and convergence — which is precisely what §3 shows.

This single hypothesis explains the whole pattern:

- **P₁ converges, P₂ fails:** `C_inv` grows with order; `4k⁴` covers the *order* growth but the residual
  *geometry* margin is tighter at P₂, where the absolute `C_inv` is larger.
- **2D converges, 3D-P₂ fails:** structured 2D triangles/quads have well-controlled `C_inv`; unstructured
  3D tets have a far worse shape-quality tail (this codebase has already seen mesh topology control
  convergence — see the Cocquet structured-vs-Delaunay finding).
- **Newton is healthy:** coercivity is a property of the *discrete operator*, independent of the nonlinear
  iteration — so ‖R‖→1e-11 while the solution is still wrong.
- **ε does nothing:** ε targets pressure indeterminacy (a *kernel*/well-posedness issue), not the viscous
  coercivity defect; and it vanishes at convergence.
- **c₁ is the lever:** it is exactly the knob in the coercivity condition.

### Reconciliation with article.tex
The paper reports **optimal** P₂-3D slopes (tab:3DL2 ≈ 3.18) on **unstructured** tets with the **same**
uniform `c₁ = 4k⁴` and **ε > 0** for well-posedness. We reproduce that recipe and diverge. With the
equations, tolerances, nonlinear solver and ε all eliminated, the remaining degree of freedom is the
**mesh** (generator and element-quality distribution) feeding `C_inv` in `c₁ > 2ξ C_inv²`. The failure is
therefore a **numerical-parameter calibration** (c₁ vs the mesh's `C_inv`), *consistent with* the paper's
own statement that the effective c₁ is element-type-dependent — not a defect in the transcribed
formulation.

---

## 5. Testable predictions / next steps (not yet done)

1. **Measure the gmsh mesh quality** (min dihedral angle, radius-ratio, the `(6√2V)^{1/3}` vs true-diameter
   spread) across the P₂ ladder. Prediction: a heavy bad-quality tail that worsens at finer levels.
2. **Regularize the mesh** (gmsh `Optimize`/`OptimizeNetgen`, or a structured tet pattern) and rerun P₂ at
   the *uniform* `c₁ = 4k⁴`. Prediction: convergence returns with no c₁ change — isolating mesh quality as
   the cause.
3. **Quality- / element-aware c₁** as the pragmatic fix: raise c₁ for P₂-tet only, behind a config knob, as
   an explicitly-labeled `[code-actual]` deviation from the paper's uniform c₁ (2D stays paper-faithful).
   This is a band-aid that treats the symptom; (1)+(2) should be tried first.
4. **h convention:** τ uses h = `(6√2·V)^{1/3}` (regular-tet edge from volume). It scales τ magnitude but
   not the coercivity *threshold* (`C_inv` is h-independent), so it is secondary; still worth checking
   against the paper's "mesh of diameter h" convention.

---

## Appendix A: the numerical-ε redesign

The Codina iterative penalty (`physical_properties.numerical_epsilon`, ε_num) is implemented as a
**Jacobian-only** term, not an outer loop. Lagging `ε_num·p` to the current nonlinear iterate makes
`ε_num·pᵏ⁺¹ − ε_num·pᵏ = ε_num·dp`, so:

- the **residual** uses `ε_phys` only (the physical compressibility `eps_val`) — unchanged;
- the **Jacobian** mass term uses `ε_phys + ε_num` (`continuous_problem.jl` `mass_term_jac`, both Jacobian
  builders).

It is a pressure-block Newton-step regularization (resolves the §5.2 indeterminacy / near-singular
pressure block) that **vanishes at convergence** — no consistency error, no manufactured-source change,
bit-identical at `ε_num = 0` (blitz 183/183, quick 53/53). Bounded by `ε_num ≤ C₂·c₁·inf{α²τ₁/h²}`
(`eq:UpperBoundOnEpsilon`). The prior outer-loop implementation was over-engineered and was reverted.

## Appendix B: experiment ledger

| id | what | result |
|---|---|---|
| `exp12`  | P₂ ASGS ε-off ladder (levels 0–2) | errors grow under refinement; Newton converges every level |
| `exp12b` | P₂ ASGS A/B `eps_mult ∈ {0,1,20}`, levels 0–1 | identical errors; 20× slightly worse ⇒ ε does not fix it |
| `exp13`  | P₂ ASGS A/B `c1_mult ∈ {1,4}`, ε off, levels 0–1 | rates −0.01/−1.16/−1.00 (c₁×1) → +2.15/+1.10/+1.69 (c₁×4): c₁ is the lever |
| `exp6` (prior) | c₁×4 on the failing P₂ cell | restored convergence (corroborates `exp13`) |
