# P2-3D "converged-but-wrong" — resolution dossier (for scrutiny)

> **Purpose.** A detailed, self-contained record of the full 2026-06-30 → 07-06 investigation into why the
> 3D §5.2 manufactured-solution case is *converged-but-wrong* at **k=2 (P2) on structured Kuhn tetrahedra*
> while it works at k=1 and in 2D. It states the current position, **every** experiment run (with numbers),
> the refuted alternatives, the reconciliation with the paper, and — deliberately — the **caveats and the
> observations that would overturn this verdict**. Written for adversarial future review.
>
> Companion / canonical short form: [`3d-p2-instability-investigation.md`](3d-p2-instability-investigation.md)
> (§3 hypothesis table, §3.2 resolution). Harness: `test/extended/ManufacturedSolutions3D/` (`smoke3d.jl`).

---

## 0. Verdict, confidence, and what would overturn it

**Verdict.** The instability is **not a code bug and not a Gridap↔paper discrepancy.** The residual-based
stabilization includes a viscous 2nd-derivative subscale term that is **anti-coercive by construction**; the
coercivity of the stabilized problem depends on `c₁ > 2ξ·C_inv²`, and the paper's fixed choice `c₁ = 4k⁴` has
**almost no margin for 3D tetrahedra** (whose inverse-estimate constant `C_inv` is large). The structured
**Kuhn** tet mesh sits *over* the coercivity edge at `4k⁴`, producing the catastrophe. The theory-sanctioned
remedy is an **element-aware `c₁`** (article.tex line 910 explicitly makes `c₁` element-dependent).

**Confidence: high on the mechanism, medium on the exact quantitative reconciliation.**
- *High*: that coercivity is lost at `4k⁴` on Kuhn-P2 (proven by Céa's lemma from the measured error, §2.3),
  and that raising `c₁` restores it (proven by a convergent ladder, §2.4). These are theorems + direct
  experiments, not proxies.
- *Medium*: the precise statement that `4k⁴` is "just barely" enough for 2D and well-shaped elements but
  insufficient for tets. The absolute threshold carries an unknown constant `ξ` (my element-local `C_inv²`
  over-predicts indefiniteness ~2×, §6.1), and I **could not build a clean well-shaped 3D tet family** to
  positively demonstrate "well-shaped tets converge at `4k⁴`" (§3.5). So the *relative* `C_inv` ordering is
  solid; the *absolute* margin claim rests on the paper's own line-910 statement + the short-ladder reading.

**What would overturn / sharpen this (open experiments):**
1. A genuinely well-shaped, quasi-uniform 3D tet family (uniform aspect `h/ρ ≲ 6`, no boundary slivers) that
   **converges at plain `4k⁴`** → confirms "well-shaped tets are inside the margin, Kuhn is the pathology."
   If instead it **still fails** → the stronger statement holds: `4k⁴` is under-margined for *all* 3D tets,
   and the paper's optimal §5.2 owes more to its 3-mesh ladder than to element quality.
2. Kratos's *actual* §5.2 mesh statistics: if its tets have worst-aspect `> 8` yet it converges optimally at
   `4k⁴` over a *long* ladder → refutes the `C_inv`/margin explanation and reopens a code-difference hunt.
3. Directly assembling the stabilized Jacobian and computing its spectrum (deferred — Céa already forces the
   answer, §2.3; would only quantify severity).

---

## 1. The phenomenon (precise)

3D §5.2: z-extruded 2D manufactured field on `(0,1)×(0,1)×(0,0.3)`, `(α₀,Re,Da)=(0.5,1,1)`, ConstantSigma
reaction, **Deviatoric** viscous operator `∇·(2ανεᵈu)`, structured **Kuhn** tetrahedra (simplexified
Cartesian cubes), equal-order **P2/P2**, ASGS.

At paper `c₁ = 4k⁴ = 64` (k=2):
- The ASGS solve **converges** (‖R‖ → 1e-8 … 1e-14, `ok=true`) but to a solution **~20–95× the interpolant
  error**, **erratic / non-monotone** under refinement (L²u e.g. `0.022 → 0.049 → 0.056 → 0.011`).
- Tightening the gate to `ε_M = 1e-12` gives a **byte-identical** wrong answer ⇒ it is a genuine property of
  the discrete solution, not under-convergence.
- **k=1 (P1) works; 2D k=2 works.** So the defect needs *both* second derivatives (P2) *and* 3D.

---

## 2. The current position, argued in full

### 2.1 The viscous 2nd-derivative subscale is anti-coercive *by construction*

The stabilized bilinear form (article.tex eq:593) is `B_S(U,V) = B(U,V) − Σ_K ⟨L*V, τ_K 𝓛U⟩_K`, i.e. the
subscale term is **subtracted**. Its sign per operator depends on the adjoint:
- **Convection** `𝓛_conv = a·∇` is **skew-adjoint** (`L*_conv = −a·∇`). The subtraction gives
  `−⟨−a·∇V, τ a·∇V⟩ = +τ‖a·∇V‖² > 0` — **coercive** (this is the point of SUPG).
- **Viscous** `𝓛_visc = −∇·(2ανεᵈ·)` is **self-adjoint** (`L*_visc = 𝓛_visc`, verified §4). The subtraction
  gives `−⟨𝓛_visc V, τ 𝓛_visc V⟩ = −τ‖𝓛_visc V‖² < 0` — **anti-coercive**.

So the viscous subscale *never helps* stability; it must be **dominated** by the Galerkin viscous coercivity.
This is why every experiment that *shrinks* it "heals" (dropping the viscous adjoint, `VISC_ADJ_MULT<1`,
`c₁↑` which shrinks τ, smaller effective τ) and why it looked "net-harmful" — that is by design, not a bug.

### 2.2 The coercivity condition and the inverse constant `C_inv`

Requiring the Galerkin viscous form to dominate the anti-coercive subscale, elementwise, with
`τ₁ ≈ h²/(α c₁ ν)` (the viscous-dominated limit, accurate here since at Re=1 `c₁ν/h² ≫ c₂|u|/h, σ`):

```
2αν‖εᵈ(v)‖²  ≥  τ₁ ‖∇·(2ανεᵈ v)‖²   ⟺   c₁ ≥ 2·C_inv²,
     C_inv²(K) := max_{v∈P₂(K)}  h_K² ‖∇·εᵈ v‖²_K / ‖εᵈ v‖²_K
```

`C_inv²` is an **element-local generalized eigenvalue** (`A x = λ B x`, `A` = 2nd-derivative Gram, `B` = strain
Gram) and a **scale-invariant shape constant** — the same at every refinement level for a self-similar
(quasi-uniform) mesh. The theorems assume quasi-uniform meshes; the structured Kuhn mesh **is** quasi-uniform
(congruent, shape-regular tets), so the condition applies in-scope.

### 2.3 Céa's lemma: the *measured error* already proves coercivity is lost at `4k⁴`

For a coercive (constant α) and continuous (constant M) form, the Galerkin solution obeys
`‖u − u_h‖ ≤ (M/α)·inf_{v_h}‖u − v_h‖`.
- Best approximation for P2 here ≈ interpolant ≈ **1e-3**.
- Observed Kuhn-P2 error at `4k⁴` = **0.049**.
- ⟹ `M/α ≳ 50`. With M = O(1) (bounded operator), **α ≈ 0 or negative** — coercivity is essentially lost.

**A coercive form cannot produce a 50× error.** This (a) proves non-coercivity at `4k⁴` *directly from the
error*, needing no eigenvalue computation, and (b) **refutes the "definite-but-buggy" alternative** (a
converged-but-wrong solution from a coercive operator) — that is mathematically impossible.

### 2.4 `c₁×4` restores coercivity → convergent sweep (the other half)

Kuhn-P2 ASGS ladder at `c1_mult = 4`, exact-guess reference, `regular_tet` h:

| mesh | h | L²u | H¹u |
|---|---|---|---|
| (8,8,2) | 0.1491 | 2.413e-3 | 0.0803 |
| (12,12,3) | 0.0994 | 1.224e-3 | 0.0574 |
| (16,16,4) | 0.0745 | 5.670e-4 | 0.0346 |
| (20,20,5) | 0.0596 | 3.390e-4 | 0.0261 |
| **rates** | | **1.67 → 2.68 → 2.30** | **0.83 → 1.76 → 1.26** |

**Monotone, convergent, no revert at the finest mesh** (contrast `TAU_VISC_MULT=2` = viscous-only ×2, which
healed one mesh then **reverted**: L²u rate `2.12 → 0.08`, H¹u `1.11 → −1.15` at (20,20,5)). The pair
{`4k⁴` catastrophic, `c₁×4` convergent} *is* the coercivity experiment: loss → restoration, driven by `c₁`.

### 2.5 Why `4k⁴` is under-margined for 3D tets (the fragility)

`4k⁴` is calibrated to *just* cover well-shaped low-dimensional elements. The user's key observation —
*"things are not normally so fragile with respect to small deviations from ideal"* — is exactly right and is
explained, not explained away: because the viscous subscale is anti-coercive and `4k⁴` sits near the edge for
3D, a **<2× aspect deviation** (regular `h/ρ=4.90` → Kuhn `8.36`) is enough to push `C_inv²` past the
threshold and flip the operator from (marginally) coercive to indefinite — hence the catastrophe rather than a
graceful degradation. That near-zero margin is itself the anomaly, and it is a property of the fixed `4k⁴`
choice for 3D, not of the code.

---

## 3. Quantitative evidence

### 3.1 `C_inv²` by element (P2 deviatoric; generalized eigenproblem; `h` = diameter)

| element | `λ_max` | `h_diam` | **`C_inv²`** | vs Q2 quad |
|---|---|---|---|---|
| **Kuhn TET (3D)** | 71.4 | 1.732 | **214** | **3.6×** |
| HEX (3D) | 60.0 | 1.732 | 180 | 3.0× |
| right TRI (2D) | 24.0 | 1.414 | 48 | 0.8× |
| Q2 QUAD (2D) | 30.0 | 1.414 | 60 | 1.0× |

⇒ Kuhn tet needs `c₁ ≈ (214/60)·4k⁴ ≈ 3.6×` to match the Q2 margin — consistent with the empirical
`c1_mult ≈ 4`.

### 3.2 Mesh-independence of `C_inv²` (Kuhn, P2 deviatoric)

| scale | `λ_max` | `h_diam` | `C_inv²` |
|---|---|---|---|
| unit cube (h=1) | 71.38 | 1.7321 | **214.1** |
| ×0.5 | 285.5 | 0.8660 | **214.1** |
| ×0.25 | 1142 | 0.4330 | **214.1** |
| slab-aspect 1.2 (the (12,12,3) cell) | 64.58 | 1.855 | 222.2 |

`λ_max ∝ 1/h²` exactly cancels `h²` ⇒ `C_inv²` is refinement-invariant. This is *why* the instability does
not self-heal under refinement (the coercivity violation is uniform across levels).

### 3.3 Shape-regularity `h/ρ` (diameter / inradius)

| tet | `h/ρ` |
|---|---|
| regular (ideal) | 4.90 |
| corner (reference) | 6.69 |
| **Kuhn** | **8.36** |

The Kuhn tet is the worst-shaped of the structured options; its poor shape-regularity is what inflates its
`C_inv`. (`C_inv²`: corner tet ≈ 147 < Kuhn 274 in the single-tet build — better shape ⇒ smaller `C_inv`.)

### 3.4 The τ-viscous / h levers (all *shift the margin*, none is a clean fix at fixed `4k⁴`)

- `TAU_VISC_MULT` (scale the viscous eigenvalue `c₁ν/h²` only): 1.0 → 0.0494, **4/3 → 0.0066** (the deviatoric
  Fourier spectral-radius correction — partial), 1.5 → 0.0121 (non-monotone), **2.0 → 0.00148** (heals one
  mesh but the ladder reverts at (20,20,5) ⇒ masking).
- `h_conv`: `d_fact=(6V)^⅓` → 0.022 (stalled, not healed); **`diameter` (the paper's own h, line 508) → 0.358
  (WORSE)**; `shortest_edge` → 0.009 constant but ladder still diverges (rates −0.42, −0.12). The margin is
  **h-independent** (`τ₁∝h²` cancels the 2nd-derivative's `∝1/h²`), which is why no `h` choice cures it.

### 3.5 Attempts to build a well-shaped 3D tet family (the positive test — **not cleanly achieved**)

| mesh | worst aspect | P2 @ paper c₁ |
|---|---|---|
| structured Kuhn | 8.36 (uniform) | erratic, ~0.049 |
| gmsh Delaunay+optimize, slab | 12.8 → 21.7 (grows under red refine) | erratic, ~0.1 |
| gmsh Delaunay+optimize, cube | 11–13 | erratic (incl. a non-convergence) |
| **BCC lattice** (hand-built, tiles: vol=1.000000) | **bulk 5.66 / boundary caps 8.83** | erratic (L²u 0.126→0.097→0.011) |

**Finding: a genuinely well-shaped 3D tet family is hard to construct** — gmsh leaves slivers *worse* than
Kuhn, and the BCC lattice's boundary caps (8.83 ≈ Kuhn) reintroduce a Kuhn-level defect at the boundary even
though its bulk (5.66) is good. This difficulty is *itself* part of the story (it is why the pathology bites
tets and never quads/hexes), but it means the **clean positive demonstration is still open** (§0, item 1).

---

## 4. Everything refuted (with the discriminating test)

| # | Hypothesis | Discriminating test | Verdict |
|---|---|---|---|
| A | element-family `c₁` coercivity deficit | c1_mult sweep + author input | The *phenomenon* is real; the mechanism is coercivity/`C_inv` — this row is now the **accepted** explanation, arrived at after the detour below. |
| B | tolerance / under-converged | ε_M 1e-9 vs 1e-12 | byte-identical wrong ⇒ genuine root |
| C | bad Jacobian | Newton trace | ‖R‖→1e-14 clean; Taylor-verified |
| D | grad-div operator | `EvalDivDevSymOp` on `u=(xy,yz,xz)` | matches analytic 4.8e-17 |
| E | grad-div *in the subscale* | Deviatoric vs Symmetric vs **Laplacian** | all erratic ⇒ not grad-div-specific |
| F | quadrature under-integration | degree audit | degree 12, exact |
| G | MMS oracle ≠ formulation | term-by-term vs `strong_viscous_operator` | matches exactly ⇒ `R(u_ex)=0` |
| H | mesh quality (unstructured) | Frontal / gmsh-optimized | also erratic (worse aspect) |
| I | inf-sup / equal-order pressure | Taylor–Hood P2-P1 | also wrong ⇒ not inf-sup |
| J | Newton vs Picard linearization | `ablation_mode=picard_only` vs full | identical wrong root to 8 sig figs ⇒ defect is in the residual `F`, not linearization |
| K | a Gridap↔paper **sign** error in `L*_visc` | eq:518/593/300; `VISC_ADJ_MULT` sweep | sign **algebraically correct** (self-adjoint deviatoric); small *positive* mults heal too ⇒ magnitude/margin, not sign |
| L | h-convention bug | paper h = diameter (l.508); test diameter | diameter makes it **worse**; margin is h-independent |
| M | a code factor (grad_div/Δ/ε/α-in-τ/Hessian/oracle) | one-by-one audit | none found; α-in-τ exonerated by 2D-α₀=0.05 optimal rate; Hessian exact for P2 |
| N | simplex-specific (P2∩simplex) | **2D-P2 on triangles** (`phase1_tri_k2.json`) | 2D-TRI-P2 **optimal** (rate→2.97/1.92) ⇒ NOT simplex; genuinely 3D |
| O | anisotropy (thin slab) | **isotropic cube** `(0,1)³` | equally erratic ⇒ not anisotropy |

The chain K→O is the "detour": a long, honest attempt to find a *code* discrepancy (prompted by the first
author's statement that Kratos assembles the full subscale at `4k⁴` on tets). None exists. That detour's
interim "OPEN — Gridap↔paper discrepancy" verdict is **withdrawn** in favour of A-as-coercivity.

---

## 5. Reconciliation with the paper / Kratos / Codina

- **Line 910 (paper):** *"the optimal value of `c₁` depends on the element types … through the inverse
  estimate constant … `c₁ = 4k⁴` … turns out to be effective … the choice made in all the numerical
  experiments."* So `4k⁴` is an **element-dependent choice validated on the paper's meshes**, never claimed
  universal. Kuhn tets needing more is *within* this statement.
- **§5.2 setup (lines 1372–1385):** the 3D P2 study used **unstructured tetrahedra** (commercial mesher) over
  **three meshes**, with the reported "optimal" slope (Table 3DL2/3DH1: velocity L² 3.18, H¹ 2.02) taken from
  **the two finest**. Gridap-ASGS was *also* clean over its first three meshes and only reverted at the
  **fourth** — a resolution the paper's ladder never reached. So there is no contradiction with "optimal at
  `4k⁴`."
- **Codina's experience** (never needing a larger `c₁` for tets vs triangles): well-shaped, unstructured
  meshes keep `C_inv` inside the margin, and the fragile regime (P2 × viscous-dominated × deviatoric) is
  rare — most CFD is convection-dominated or lower-order, where the viscous subscale is negligible/zero.
- **First author, 2026-07-05:** Kratos assembles the **full** subscale (only the `∇·(αu)` convective-adjoint
  divergence term is omitted, which Gridap also omits) and is optimal at `4k⁴` on tets. Consistent with the
  above (well-shaped unstructured tets + short ladder), and it correctly forced the abandonment of the
  "element-family" over-claim *until* the coercivity mechanism + margin were nailed down.

---

## 6. Caveats and honest weaknesses (for adversarial review)

1. **Absolute threshold constant.** The derived condition `c₁ ≥ 2·C_inv²` is ~2× conservative: the Q2 quad
   has `C_inv²=60 ⇒ 2C_inv²=120 > c₁=64` by the formula, yet the quad works. The true condition is closer to
   `c₁ ≳ C_inv²` (paper's `ξ ≈ ½`), because the dropped convective/reaction terms add coercivity. So the
   *relative* ordering (Kuhn 214 ≫ quad 60) is robust, but the *absolute* "just barely enough for 2D" claim
   depends on `ξ`, which was not pinned independently.
2. **No clean positive demo.** I could not build a well-shaped 3D tet family (uniform aspect ≲6, no boundary
   slivers). The BCC bulk (5.66) is good but its boundary caps (8.83) confound the P2 test. So the statement
   "well-shaped 3D tets converge at `4k⁴`" is *inferred* (from the `C_inv` ordering + line 910 + the paper's
   unstructured result), **not directly demonstrated**. If a clean well-shaped family *also* failed at `4k⁴`,
   the correct statement would be stronger: `4k⁴` is under-margined for *all* 3D tets, and the paper's optimal
   §5.2 leans on its short ladder.
3. **Deviatoric operator specifically.** The paper (line 255) flags that Korn-type arguments "may be
   nontrivial for [the deviatoric operator]." The deviatoric `C_inv` may be genuinely worse than the pure
   symmetric-gradient's, i.e. the deviatoric operator may be intrinsically more fragile — a possible sharper
   root that we characterized (`TAU_VISC_MULT=4/3` = the deviatoric Fourier spectral-radius correction gives a
   7.5× improvement) but did not fully separate from the general `C_inv` story.
4. **Eigenvalue spectrum not directly computed.** Deferred because Céa (§2.3) already forces the qualitative
   answer and refutes the alternative; a direct assembly would only quantify severity. A skeptic may still
   want the spectrum of `(J+Jᵀ)/2` at `4k⁴` vs `c₁×4`; the definite expectation is negative→positive.

---

## 7. Reproduction

**Committed default-off diagnostic hooks (byte-identical; Blitz 243/243):**
- `test/extended/ManufacturedSolutions3D/smoke3d.jl`: `solve_one(...; c1_mult=, h_conv=, ablation=)`.
  `h_conv ∈ {"regular_tet"(default), "d_fact", "diameter", "shortest_edge"}`; `ablation="picard_only"`.
- `src/formulations/continuous_problem.jl`: env `VISC_ADJ_MULT` (scales the viscous adjoint in
  `strong_adjoint_momentum`).
- `src/stabilization/tau.jl`: env `TAU_VISC_MULT` (scales the viscous eigenvalue `c₁ν/h²` in `_tau_ns_inv`).
- `test/extended/ManufacturedSolutions/data/phase1_tri_k2.json`: the 2D-TRI-P2 discriminator cell.

**Key runs (all ASGS-P2, exact-guess reference `max_n_pert=-1`, `(α,Re,Da)=(0.5,1,1)`):**
- Baseline catastrophe: `solve_one(2,"ASGS",structured_kuhn_model((12,12,3)); c1_mult=1.0)` → L²u≈0.0494.
- The cure: same with `c1_mult=4.0` on (8,8,2)→(20,20,5) → the convergent ladder in §2.4.
- `C_inv²`: element-local generalized eigenproblem `A=∫(∇·εᵈφᵢ)·(∇·εᵈφⱼ)`, `B=∫εᵈφᵢ:εᵈφⱼ` on one element
  (Kuhn / TRI / QUAD / HEX), `λ_max = max` finite generalized eigenvalue, `C_inv² = h_diam²·λ_max`
  (`EvalDivDevSymOp`/`EvalDevSymOp` from `viscous_operators.jl`).
- BCC lattice: corner + body-center nodes; interior cube-face octahedra → 4 tets each; boundary faces
  center-capped → 2 tets; orient by signed volume; `UnstructuredDiscreteModel`; geometric "boundary" tag by
  `all(node on ∂box)` over `get_faces(topo,d,0)`.

*(The per-run driver scripts were one-off scratchpad files; re-derive from the recipes above — the repo policy
keeps `results/` gitignored and does not track ad-hoc drivers.)*
