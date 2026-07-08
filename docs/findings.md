# Findings — settled results & debugging conclusions

Living index of the **settled state** of every MMS / Cocquet / solver investigation in this codebase: what is RESOLVED, the canonical numbers, and the verdict. One section per area. For full experiment-by-experiment detail see the preserved evidence dossiers under [`docs/`](README.md) (the 3D-P2 coercivity resolution dossier, the OSGS-reaction investigation synthesis, and the 2026-06-24 formulation audit) and the LaTeX sources under [`theory/`](../theory/). Cross-references to code use `file:line`.

Status tags: **RESOLVED** / **OPEN** / **REFUTED** carry the same meaning as in the source docs.

---

## 1. 2D MMS convergence — RESOLVED (optimal across the whole grid; Route-B algebraic mass gate)

**Verdict: the stabilized equal-order method converges at full optimal order across the entire `(Re, Da, α₀, h)` grid at k=1 and k=2, on both QUAD and TRI.** The primary correctness criterion is met.

### The manufactured problem (the clean case)

`test/extended/ManufacturedSolutions/run_test.jl` builds, per `(Re, Da, α₀, k, element)`: **Dirichlet on all 8 tags** (`run_test.jl:381,391`) ⇒ Aubin–Nitsche duality holds ⇒ the velocity-`L²` extra order `k+1` is expected; **constant reaction** `σ_c = Da·α∞·ν/L²` (`run_test.jl:76,400`, always `ConstantSigmaLaw`) ⇒ `Da` is a benign linear knob on a *coercive* term, not a difficulty axis; **`SmoothRadialPorosity`** logistic `α₀→1` across `r∈[0.2,0.4]` — infinitely smooth but *steep* at small `α₀`, the only spatial-gradient stressor. Optimal targets: velocity `L² = O(h^{k+1})`, velocity `H¹ = O(h^k)`; pressure (equal-order) `L² = O(h^{kp}) = O(h)`.

Grid: `Re, Da ∈ {1e-6, 1, 1e6}`, `α₀ ∈ {1.0, 0.5, 0.05}` (27 physics cells) × {ASGS,OSGS} × `k∈{1,2}` × {QUAD,TRI}; meshes `N ∈ {10,20,40,80,160,320(,640)}`.

The **trouble axis is `Re × α₀`, not `Re × Da`.** All difficulty concentrates in one extreme corner: high `Re` **and** low porosity `α₀` — see §2 (the fold).

### Completed sweeps (Route-B algebraic mass gate)

The mass convergence gate is now the **Philosophy-A algebraic** `ε_C = ‖r_C‖/D_C` (pressure block of the assembled residual, symmetric with the momentum gate `ε_M = ‖r_M‖/D_M`, gated ~1e-9). The old loose `eps_tol_mass = 0.8` gate is demoted to the diagnostic `eps_C_strong`. Core commit `e455f36`; essential companion fix `3b76864` (below). Gate spec: [`solver/nonlinear-convergence-criterion-prompt.md`](solver/nonlinear-convergence-criterion-prompt.md); config knobs `eps_tol_momentum` / `eps_tol_mass` in `SolverConfig` (`src/config.jl`), consumed by `convergence_criterion.jl` and the `"scale_free"` path in `src/solvers/nonlinear.jl` (`_safe_solve_inner!`).

**Final verdict — all three families (canonical `results/k<kv>/<etype>/results.h5`, completed 2026-07-03):**

| family | cells | NaN | median L²u rate | behavior-preservation |
|---|---|---|---|---|
| k1 QUAD | 48 | 0 | 2.00 (optimal) | median rel Δe_u **6.5e-7** vs gold `validated_k1_quad_N640` @N=320; 40/48 within 1% |
| k1 TRI | 48 | **2** (pre-existing) | 2.00 (optimal) | the 2 NaN (Re=1,Da=1,α=0.05) were **also NaN in the baseline**; Route B **fixed 2** other α=0.05 NaN cells (4 → 2) |
| k2 QUAD | 48 | 0 | 3.00 (optimal), min 2.79 | vs pre-Route-B baseline @N=160: median rel Δe_u **1.97e-11** (byte-identical), 48/48 within 1% |

The remaining α=0.05 NaN cells are the known curved-interface-on-structured-mesh difficulty (the fold, §2), not a Route-B effect.

**The `3b76864` (`residual_floor_reached`) fix — a genuine Route-B gate limitation and its resolution.** Route B as first committed (`e455f36`) floored at k2 fine meshes: for a near-divergence-free converged flow the mass envelope `D_C = ‖∫q∇·(αu)‖ + ‖∫q εp‖ + ‖∫q g‖` collapses (dominant `∇·(αu)`→0), so `ε_C = ‖r_C‖/D_C` **floors ~1e-8** (a decade above `tol_C = 1e-9`) *even though the solution is fully converged* (`ε_M ~ 1e-14`, residual at the ~3e-12 machine floor). The pure gate then rejects a converged solution and burns the entire homotopy-perturbation fallback (`eps_pert 1.0→0.1→0.0`) on the largest cells. Fix `3b76864` honors the honest-exit valve (`noise_floor_success_max_ftol_multiple`) under the scale-free branch: accept iff (1) not degenerate, (2) `ε_M ≤ tol_M` (rejects high-Re fold stalls), and (3) the per-field residual is at the machine/noise floor. No new magic number, no gate loosening. In the clean k2 sweep this valve **fired 263×** (the most common stop reason; every high-Re k2 cell ends on it) — it is what makes the symmetric tight `eps_tol_mass = 1e-9` viable at k2.

### The k=1 QUAD success (canonical baseline, 2026-06-10, N=10→640)

- **Velocity optimal everywhere.** L² rate (finest pair N=320→640) ≥ **1.93** on all 48 cells (median 2.00 ASGS / 2.07 OSGS); H¹ rate ≥ **1.00** everywhere.
- **Pressure optimal everywhere.** Every cell's finest-segment L² rate is **1.5–2.4** — at or above the nominal O(h) for all 48 cells, consistently super-optimal. Zero sub-optimal cells. (Pitfall: pressure's optimal order for equal-order P1/P1 is O(h), **not** the velocity O(h²).)
- **OSGS ≈ 2× more accurate than ASGS** at the same rate (finest-mesh-error ratio 0.50 velocity, 0.41 pressure) — the orthogonal projection buys accuracy at higher iteration cost.
- **Behavior-preserving:** errors byte-identical to the pre-scale-free-gate archive on every overlapping mesh — the ε_M/ε_C gate changed *when* the solver stops, not *where*.
- **Expected caveats (not defects):** Re=1e6 @ N=10 is `NaN` (boundary layers ~Re^{-1/2}=1e-3 ≪ h=0.1, hopeless on 10×10); the three Re=1e6/α₀=0.05 cells are the coarse-mesh fold (§2). `analyze_results.py`'s per-pair detector still flags ≈29/48 as fold/no-root — a conservative artifact (pre-asymptotic coarse meshes, the N=10 NaN pulling global fits, super-convergent tails); its own rate-check reports **0 sub-optimal and 0 super-convergent**. Read the per-pair ratios, not the one-word verdict.

### The high-order (k≥2) gate lesson — RESOLVED

**k≥2 MMS needs a tighter momentum gate than k=1.** At the k=1 default `eps_tol_momentum = 1e-6`, high-order / high-Re cells stop early at a 5–10× worse solution, collapsing the last-segment rate (k=2 L²u 160→320 rate ≈3.3 → ≈1.7, with ~⅕ the iterations, ≈47 vs ≈309). Signature: *few iterations + fine-mesh error several× above optimal + last-interval rate below O(h^{k+1}), while coarse meshes look fine.* Setting **`eps_tol_momentum = 1e-9`** (sweep-wide for k=2, in `data/phase1_quad_k2.json`) recovers optimal O(h³) exactly (N=80: 1.23e-6 vs old 1.232e-6 ✓; N=160: 1.38e-7 vs 1.377e-7 ✓). Do **not** over-tighten (1e-12 → NaN).

Root cause was the **gate, not JFNK, not the inner-GMRES tolerance** — a clean isolation on the worst cell (Re=1e6, Da=1, α=1.0, k2 OSGS) gave byte-identical errors for OLD frozen-π, NEW JFNK (η=1e-2), NEW frozen-π, and NEW JFNK with tight inner tol (η=1e-6): all 1.46e-6 @N=80 / 2.34e-7 @N=160. The "Fix A" that made the scale-free probe *authoritative* (was diagnostic-only) exposed that the 1e-6 gate is too loose for k=2 high-Re. **For 3D: expect to set `eps_tol_momentum = 1e-9` (or tighter) — check the gate FIRST before blaming stabilization/mesh/solver.**

### Provenance / historical

- **k≤320 baseline table** (H¹ rates + inner-iteration costs) is the frozen N≤320 pre-asymptotic reference. ASGS is optimal (H¹≈1.0) and cheap (2–6 inner Newton steps, up to ~24 at Re=1e6); OSGS matches ASGS at Da≤1 (43–104 inner steps — the linear frozen-π rate, now cut by JFNK, §5).
- **Analysis workflow:** single tool `analyze_results.py` (true-root detection → `flagged_cells.json`; merged paper table; per-config plots; honest true-root `Converged` column). Report marks: verified / `*` recovered-at-fine / `‡` genuine sub-optimal / `ˢ` super-convergent / `**` fold-best-root / `N/A`. One-sided slope acceptance (`slope ≥ target − tol`) keeps super-convergence from being mislabelled sub-optimal.
- **k2 P2 guardrails on 32GB:** use ≤2 shards (P2 N=320 LU is multi-GB; 4+ OOM); `pgrep -f run_test.jl` and kill stragglers before launching (pause/resume can orphan shards, ~13GB zombies caused a false "tight-gate regression" alarm — reproduce single-process before alarming). k1 (P1) tolerates 6 shards.

Source docs consolidated here: `mms/convergence-status.md`, `mms/convergence-baseline.md`, `mms/route-b-2d-sweep-status.md`, `mms/high-order-convergence-gate-and-jfnk.md`.

---

## 2. The high-Re / low-α MMS fold — RESOLVED (coarse-mesh turning point; FE-optimal above it)

**Verdict: the failure at the extreme corner is a genuine discrete solution-branch fold (turning point), NOT a solver or Jacobian bug. It recedes with mesh refinement; once a root exists (~N=512 for k=1/TRI), a direct exact-guess Newton solve reaches it, and the corner converges optimally.**

Cells: `Re=1e6, α₀=0.05, Da∈{1e-6,1,1e6}`, ASGS and OSGS, each element family — the only MMS cells that fail from the exact-solution guess at coarse N.

### Decisive diagnosis (the fold is real)

- **A1 — Jacobian is exact.** Assembled Exact-Newton `J·v` vs centered FD of the residual at `u_ex`: best relative error **4.8e-12**, clean ε-convergence in both blocks ⇒ the "inconsistent τ-derivative" hypothesis is dead.
- **A2 — no root exists.** Heavy Newton *and* Picard from `u_ex` (budget 500, `noise_floor=1e-12`) both **stall at ‖R‖≈5e-2**, L²u ≈ 0.12–0.17.
- **Continuation folds in every parameter direction** (warm-started): Da folds at ≈5e5 (target 1), Re at ≈5.6 (target 1e6), α at ≈0.16 (target 0.05, N=40). Adaptive step-halving confirms a TRUE fold, not a step-size artifact.
- **The fold recedes with mesh:** α-fold ≈ 0.24 (N=10) → 0.16 (N=40) → 0.106 (N=80) → …; solutions above the fold are clean (machine-zero residual).
- **A3 — landscape.** cell-Péclet ≈ 4.7e4 in the α=0.05 core at N=10 (convective `c₂|u|/h` dominates viscous `c₁ν/h²` by ~10⁴–10⁵). The velocity floor never activates ⇒ coarse-mesh coercivity limit, not a regularization artifact.
- **Why α is the only viable continuation axis:** Da/Re continuation hold α=0.05 fixed (stiff layer present the whole way) and fold immediately; α-continuation starts at α=1 (easy) and *relieves* the layer.

### Recovery — the corner converges optimally / super-convergently

C24 (`Re=1e6, Da=1, α₀=0.05`) continued to the target α=0.05, at machine-zero residuals:

| N | h | L²u | H¹u | ‖R‖∞ |
|---|---|---|---|---|
| 512 | 1.953e-3 | 2.648e-4 | 8.558e-2 | 6.3e-10 |
| 768 | 1.302e-3 | 7.746e-5 | 5.623e-2 | 3.6e-9 |
| 1024 | 9.766e-4 | 3.273e-5 | 4.206e-2 | 4.1e-12 |

| pair | rate L²u | rate H¹u |
|---|---|---|
| 512→768 | **3.03** | **1.04** |
| 768→1024 | **2.99** | **1.01** |

**H¹u ≈ 1.0** (textbook-optimal for k=1); **L²u ≈ 3.0** (super-convergent, above the nominal 2), holding steady — the opposite of a sub-optimal plateau. This is the pre-asymptotic-erosion hypothesis, demonstrated: coarse-mesh slopes are depressed because the steep α-layer is under-resolved (at N≤40 the steep core is resolved by only 1–4 cells; by N=512–1024 by ~100–200), then climb to optimal once the layer is resolved. C21 (`Da=1e-6`, same corner) is **bit-identical** to C24 (`‖R‖=6.26e-10, L²u=2.648e-4, H¹u=8.558e-2`) — the fold is driven by the α-layer + high Re, **not Da** (σ ∝ Da is negligible against the convective scale ≈19).

**Direct confirmation for the standard (non-folding) α₀=0.05 cells:** low/unit-Re α₀=0.05 ASGS cells read ≈1.88–1.95 at the sweep's N=160→320 finest pair (the values in the article's P1 table) — pre-asymptotic, not a rate loss: the worst cell (`Re=1, Da=1e-6, α₀=0.05`, ASGS, TRI k=1) extended one mesh climbs **1.888 (160→320) → 1.960 (320→512)**, matching the QUAD k=1 sweep where every α₀=0.05 cell recovers to ≥1.93 by N=640.

### The production path (2026-06-17): direct exact-guess solve supersedes continuation

Continuation is only needed to *reach* a root when none exists at coarse N. Once the fold clears (~N=512 for k=1/TRI), the interpolated exact solution is an excellent guess for the existing root, so plain Newton converges directly (~3 iters). Drivers: `run_corner_article.jl` (ASGS, direct Newton N=512 → mesh-step N=768, ~6 LU total vs ~70 for an α-ramp — ~25 min/cell vs ~4 h/cell); `run_corner_osgs.jl` + `osgs_corner_lib.jl` (OSGS, warm-started from the ASGS root, which is O(h²) away).

| Corner family | Cells | Status | Path |
|---|---|---|---|
| P1 / TRI k=1 | Re=1e6, α₀=0.05, Da∈{1e-6,1,1e6}, ASGS+OSGS | ✅ **DONE** | direct solve, base N=512 + step N=768 (folds for N≤320) |
| Q2 / QUAD k=2 | same 6 cells | ✅ **DONE** | direct solve N=160→320 — **does NOT fold** (k=2 resolves the α-layer ~2× better, fold clears at ~half the N) |

Both corners fill `results/paper_tables.tex` (zero `n.c.`). k=2 needs no extrapolation (converges at N=320 directly); the blanket `skip_cells [1e6, *, 0.05]` was over-conservative for k=2.

**Honest-exit gate (fixed in core, `5ae4c25`).** The production solver previously reported past-the-fold corners as "converged" at a noise-floor stall (‖R‖≈1e-5, ~25× wrong). The `noise_floor_success_max_ftol_multiple` gate (`k_nf`; default `Inf`=off, sweep uses `10`) makes the sweep reject fold stalls; its dimensionless analogue is the scale-free `residual_floor_reached` accept (`3b76864`, §1).

### Caveats / open

- **OSGS slope inflation caveat:** the OSGS corner coupled solve converges slowly-linearly (frozen-π) and is stopped at the production residual, not a tight true root; the FME is reliable (warm-from-ASGS, OSGS≈ASGS) but the OSGS *slope* can be mildly inflated when the coarse (N=512) point is not fully settled (e.g. Da=1e6 reads 2.63 vs ASGS 2.11).
- **OPEN — Gridap-vs-Kratos magnitude offset:** Gridap corner FME are **~3–12× larger** than the article's (Kratos) values, norm-dependent (vel L²: Gridap 7.9e-5 @N=768 vs paper 1.1e-5 @N=640; pressure L² ~5×; H¹ ~2.4×). **The rates agree** (≈2–3) and Gridap TRI matches Gridap QUAD continuation to ~2%, so the discretization is internally consistent — the offset is a code-vs-code calibration question (candidates: `U_c`/`P_c` normalization, porosity-field definition, MMS amplitude). Worth reconciling before the table is taken as a literal paper reproduction.

Source doc: `mms/fold-recovery.md`.

---

## 3. 3D-P2 MMS "converged-but-wrong" — RESOLVED (`4k⁴` under-margined for high-`C_inv` structured tets)

**Verdict (2026-07-06): NOT a bug and NOT a Gridap↔paper discrepancy.** The viscous 2nd-derivative subscale is **anti-coercive by construction** (the viscous operator is self-adjoint, so `B_S = B − Σ⟨L*V, τ𝓛U⟩` carries `−τ‖𝓛_visc V‖²`, unlike the *coercive* `+τ‖a·∇V‖²` from skew-adjoint convection — which is why removing/shrinking it "heals"). It must be dominated by the coercivity condition `c₁ > 2ξ·C_inv²`. **The paper's fixed `c₁ = 4k⁴` has almost no margin for 3D tets** (large `C_inv`), so a slightly poorly-shaped structured mesh sits over the coercivity edge. The remedy is an **element-aware `c₁`**, which article.tex **line 910** explicitly prescribes ("the optimal value of `c₁` depends on the element types … through the inverse estimate constant"). This is fully within the quasi-uniform theory the theorems assume (the structured Kuhn mesh is quasi-uniform — congruent, shape-regular tets), and consistent with the paper's optimal §5.2 result (which used *unstructured* tets over *only 3 meshes*).

> **The earlier "Gridap↔paper discrepancy / c₁ masks a bug" framing (2026-07-05) is WITHDRAWN.** c₁ is not masking a bug — it is the theorem's own coercivity knob, and `4k⁴` is simply an element-dependent choice that is under-margined here.

**Full for-scrutiny record** (every experiment + numbers, the C_inv table, mesh-independence, shape-regularity, the c₁×4 ladder, τ/h levers, the BCC attempts, the refuted-alternatives table A–O, the paper reconciliation, and the honest caveats / what would overturn the verdict): [`docs/mms/3d-p2-coercivity-resolution-dossier.md`](mms/3d-p2-coercivity-resolution-dossier.md).

### The symptom

3D §5.2 MMS: z-extruded field on (0,1)×(0,1)×(0,0.3), `(α₀,Re,Da)=(0.5,1,1)`, ConstantSigma, structured Kuhn tets. **P1 works** (ASGS/OSGS optimal-to-sub-optimal, §7). **P2 is converged-but-wrong:** ASGS-P2 converges (residual → 1e-8…1e-14) but to a solution **20–95× the interpolant error**, erratic / non-monotone under refinement (SymmetricGradient even *diverges*). The P2 interpolant of `u_ex` is optimal (L²u ≈ 0.0011 @ h=0.099, converges monotonically); every stabilized solve lands 12–44× worse:

| (12,12,3) | L²u | ×interp |
|---|---|---|
| interpolant | 0.00112 | 1× |
| ASGS Deviatoric | 0.0494 | 44× |
| ASGS SymmetricGradient | 0.0298 | 27× |
| ASGS Laplacian | 0.0130 | 12× |
| OSGS (frozen-π one-shot) | 0.0157 | 14× |

### The proof (two ways)

- **Céa's lemma settles coercivity from the error itself** (no eigenvalue run). Best approx (P2 interpolant) ≈ 1e-3; observed Kuhn-P2 error at `4k⁴` = **0.049** ⟹ `M/α ≳ 50` ⟹ coercivity `α ≈ 0`. A coercive form *cannot* produce a 50× error, so the catastrophe **is** loss of coercivity — refuting any "definite-yet-buggy" reading.
- **`c₁×4` restores a monotone convergent sweep** (the other half). Kuhn-P2 ASGS ladder (8,8,2)/(12,12,3)/(16,16,4)/(20,20,5) at `c1_mult=4`: L²u 2.41e-3 → 1.22e-3 → 5.67e-4 → 3.39e-4, rates 1.67 → 2.68 → **2.30** (H¹u 0.83 → 1.76 → 1.26) — **monotone, no revert at the finest mesh** (contrast τ×2, which reverted).
- **Element-local inverse constant `C_inv²`** (generalized eigenproblem, P2 deviatoric), **mesh-independent** (identical at h=1, ½, ¼): Kuhn TET **214** vs Q2 quad **60**, right-TRI 48, hex 180 — Kuhn tet needs `c₁ ≈ (214/60)·4k⁴ ≈ 3.6×`, matching the empirical `c1_mult=4`. Shape-regularity `h/ρ`: regular 4.90, corner 6.69, **Kuhn 8.36** (worst).
- **Reconciliation:** the paper's optimal §5.2 P2-3D (slope 3.18/2.02) used **unstructured tets** over **only 3 meshes** (rate from the two finest). Gridap-ASGS was *also* clean over its first 3 meshes and only reverted at the 4th — a resolution the paper's ladder never reached. No contradiction. A genuinely well-shaped 3D tet family is hard to build (gmsh worst-aspect 11–13; a hand-built BCC lattice gives bulk aspect 5.66 but boundary caps 8.83 ≈ Kuhn, still erratic at paper c₁).

### Refuted alternatives (do not re-chase)

| Hypothesis | Verdict |
|---|---|
| **c₁ under-budgets coercivity for the element family** (2026-07-03 reading) | ❌ REFUTED as *interpretation* (2026-07-05): the first author confirms Kratos runs the full subscale at paper c₁ on tets optimally, so `4k⁴` is not the *cause*; but the *data* (c₁ controls it, ×4 pins) is real and is exactly the coercivity-margin experiment — see the RESOLVED verdict, which reframes it as element-aware margin, not a bug |
| Tolerance / under-converged | ❌ byte-identical wrong answer at ε_M 1e-9 vs 1e-12 (L²u=0.049370 both) |
| Bad Jacobian / NR floor | ❌ residual descends cleanly to 1e-8…1e-14; P2-P1 hits 1.5e-14 |
| grad-div term in the viscous op | ❌ `EvalDivDevSymOp` matches analytic to **4.8e-17** |
| grad-div in the subscale | ❌ Deviatoric / Symmetric / **Laplacian (no grad-div)** all erratic |
| Quadrature under-integration | ❌ degree 12 for P2 (exact to P11), sufficient |
| MMS oracle ≠ formulation | ❌ deviatoric forcing matches `strong_viscous_operator` exactly |
| Mesh quality (Kuhn low-quality) | ❌ Frontal+optimized meshes also erratic (9.5–25× interp; only a ~2–4× constant helped) — mesh-independent |
| inf-sup / pressure instability | ❌ Taylor-Hood P2-P1 also wrong (L²u 57–95×, absolute L²p ~0.4 same/worse), converges to 1.5e-14 |
| Newton vs Picard | ❌ **identical wrong root to 8 sig figs** (0.049370051 vs 0.049370052) via different paths ⇒ defect is in the assembled **residual `F`**, not the linearization/solver; also closes the spurious-adjacent-root loophole |
| Simplex-specific (P2∩simplex) | ❌ **2D-P2 on TRIANGLES is OPTIMAL** (`data/phase1_tri_k2.json`, same cell: L²u rate 2.39→2.90→**2.97**, H¹u→**1.92**, tracking QUAD to ~1.3×) ⇒ the bug is genuinely **3D-only** |
| Anisotropy (thin slab) | ❌ isotropic cube (0,1)³ equally erratic (L²u 0.0229→0.0351→0.0235) |
| h-convention (`smoke3d.jl:222` extra √2) | ❌ real 2D/3D inconsistency + a strong lever (√2 drop cut L²u 0.0494→0.0219, 2.25×) but **NOT the fix** (still ~10× interp, stalled); the paper's h *is* the diameter (line 508), which makes it *worse* (0.358); margin is h-independent (`τ∝h²` cancels the 2nd-derivative's `∝1/h²`) |
| Viscous-adjoint sign flip | ❌ algebraically correct (eq:518/593/300); small **positive** `VISC_ADJ_MULT` heals too ⇒ magnitude/margin, not sign — do NOT flip |
| Any spurious code factor | ❌ exhaustive hunt negative (grad-div, Laplacian coeff, deviatoric-½-split, ε, strong≡adjoint byte-identical, α-in-τ exonerated by 2D-α₀=0.05 optimal, Hessian exact, oracle matches) |

The destabilizer is localized to the **viscous adjoint `L*_visc(v)=∇·(2ανεᵈv)`** in `⟨L*_visc(v), τ₁R_u⟩`: an env-gated `VISC_ADJ_MULT` scan showed only full-strength +1.0 is catastrophic (L²u 0.0494) while anything ≤0.5 heals to interpolant level (0.00136–0.00180), with the minimum at mult ≤ 0 — an *uncontrolled anti-coercive term*, ~2× over `c₁ > 2ξC_inv²`.

### The iterative-penalty well-posedness fix (separate from the c₁ resolution)

3D all-Dirichlet with ε=0 is **ill-posed** (constant-pressure null mode). The **Codina iterative penalty** adds `ε_num·(pⁿ − pⁿ⁻¹)` to the **mass residual** (the code previously had `ε_num` only in the Jacobian, where it cancels in the residual and vanishes at convergence). This fix — with the gated `p_prev` handling — gives well-posedness (Blitz 240/240). **Its "P2 root cause = penalty" claim is WITHDRAWN:** the penalty fixes *well-posedness*, not the P2 *accuracy* defect. Its *original* "NOT c₁" instinct is vindicated (paper c₁ is correct; the P2 defect is the element-margin issue above, neither penalty nor c₁). Also fixed in the same work: spurious-root homotopy acceptance (reference-root match, `5ecc0ca`) and an OSGS→ASGS `initial_ftol` recording leak (`osgs_advanced_off_entry` guard). Canonical: [`mms/3d-iterative-penalty-fix-and-osgs-coupling.md`](mms/3d-iterative-penalty-fix-and-osgs-coupling.md).

### 3D official sweep results (2026-06-30, paper c₁, regular Kuhn mesh, self-describing `results/k*/TET/structured/`)

- **OSGS P1 SOLVED** — robust (`eps_used=1`) **and** fully optimal (recipe: iterative-penalty + boot-skip + JFNK + reference-root homotopy).
- **ASGS P1 robust** (L²u structured-limited — the genuine 3D P1-ASGS L²-order sub-optimality on structured tets; **OSGS recovers it**).
- **ASGS P2 non-monotone** at paper c₁ (the under-margin above; orthogonal to the penalty fix).
- **OSGS P2 still ∂π/∂u-open** (`ok=false` but accurate) — see below.

### OSGS-P2 is additionally unsolvable — OPEN (separate from the discretization issue)

Even granting the discretization, OSGS-P2 cannot be *solved*: the damped staggered π-iteration is **violently non-contractive** (drift ratio ρ ≈ 8–65 for ω=1.0 down to 0.1, every ω diverges — mechanism: π = Π(R(u)) with R∋∇²u amplifies high-frequency content ~1/h²; Anderson can't rescue it). **JFNK is not budget-fixable** (inner GMRES stalls at rel-res 0.01–0.16, non-monotone across maxiter 30/100/300 ⇒ the matrix-free `Jᵥ` is noisy because R re-projects π inside every FD probe). A *single* frozen-π solve from the interpolant gives L²u=0.0157 — the fixed point exists and is reasonable; only the iteration to it is unstable. A real saddle-point/MG preconditioner would be needed, but is only worthwhile *after* the discretization is fixed.

Source docs: `mms/3d-p2-instability-investigation.md` (verdict), `mms/3d-iterative-penalty-fix-and-osgs-coupling.md`. Diagnostic hooks (default-off): `smoke3d.jl` `ablation`/`h_conv`; `continuous_problem.jl` `VISC_ADJ_MULT`; `tau.jl` `TAU_VISC_MULT`; `data/phase1_tri_k2.json`.

---

## 4. OSGS reaction-dominated rate (high Da) — RESOLVED (pre-asymptotic; recovers by N=640)

**Verdict (2026-06-10): the high-Da OSGS velocity-rate loss is a genuine, pre-asymptotic OSGS coercivity gap that recovers to the optimal rate — NOT a bug, and NOT an asymptotic order ceiling.**

The reaction-dominated OSGS velocity H¹ rate climbs `0.57 → 0.54 → 0.58 → 0.73 → 1.11 → 1.85` (α₀=1, Da=1e6, N=10→640): flat ≈0.7 through N≤320 (the value earlier reported as the "defect"), then **recovers to ≥1.0** once the N=320→640 pair is in (α₀=0.5 reaches 1.60). It is **independent of porosity** α₀ (as severe at α₀=1, no porosity layer) ⇒ pure reaction-dominance, a different axis from the high-Re/low-α fold (§2).

### The A/B evidence (staggered ≡ coupled; the suboptimality lives in the fixed point)

Cell `Re=1, Da=1e6, α₀=1, k=1, QUAD` (chosen: no porosity layer, no fold — only the large constant reaction). ASGS is textbook-optimal end to end (H¹→1.0, L²→2.0). OSGS-staggered ≡ OSGS-coupled to ~4 sig figs at every mesh:

| pair | ASGS L² / H¹ | OSGS L² / H¹ (staggered = coupled) |
|---|---|---|
| 10→20 | 1.72 / 1.10 | 1.58 / 0.57 |
| 20→40 | 1.87 / 1.10 | 1.56 / 0.54 |
| 40→80 | 1.94 / 1.06 | 1.57 / 0.57 |
| 80→160 | 1.99 / 1.03 | 1.66 / 0.71 |
| 160→320 | 2.03 / 1.02 | 1.55 / 0.74 |

⇒ the suboptimality is a property of the **OSGS discrete fixed point**, not the route that reaches it.

### Mechanism (theory corrected 2026-06-09)

Because `σu_h ∈ V_h`, the orthogonal projection **annihilates the reactive residual** `(I−Π)(σu_h)=0` (annihilation probe: `‖(I−Π)(σu_h)‖/‖σu_h‖ = 3e-16` at quadrature degree 2/4/8). So OSGS loses the reactive stabilization square `−‖τ₁^{1/2}σu‖²` that gives ASGS its H¹-strength `σ̃_α ~ α(1+Re_h)ν/h²` velocity control; OSGS controls velocity only at strength σ, weaker by the **mesh Damköhler** `Da_h = σh²/(c₁ν) ∝ 1/N²`. The gap therefore **closes as h→0** ⇒ pre-asymptotic. Signature confirmed on both axes: degrades only at Da=1e6, healthy again at Re=1e6.

> **Theory-sign correction (2026-06-09):** the constant-σ annihilation means OSGS *retains* the full Galerkin σ on ‖u‖, whereas ASGS *drains* to `σ_a = σ − τ₁σ² < σ`. So OSGS's coercivity constant is the **larger** of the two — the dip is NOT a coercivity loss but a pre-asymptotic consistency/approximation transient. (The historical TL;DR "OSGS loses coercivity" reads the gap backwards; the empirical recovery stands regardless.) Companion note `theory/osgs_reaction_note/osgs_reaction_note.tex`.

**Da-sweep confirmation** (Re=1, α=1, OSGS): optimal wherever `Da_h ≪ 1` (Da=1: H¹ 1.00; Da=10²: H¹ 1.00; Da=10⁴: super-convergent recovery H¹ 1.63 as `Da_h` crosses 1; Da=10⁶: degraded but climbing). A bug would hit every Da.

### The trim is innocent; full-residual OSGS is *unstable*

The reaction-projection trim (drop `σu` from the projection for constant σ, article.tex §580, `ProjectResidualWithoutReactionWhenConstantSigma`) is correct and **coefficient-robust at the fixed point** (it removes a quantity annihilated anyway; `continuous_problem.jl:203`). Switching to `ProjectFullResidual` at Da=10⁶ **stalls/diverges at every mesh** (→NaN at N=80) and reports the ASGS Stage-I fallback (H¹ bit-identical to ASGS) — because although the *residual* equals the trim's, the *Jacobians differ*: the trim's `−σ·du` term is load-bearing for solver stability. **This is why the paper trims; the trim is not removable.**

> **Methodological caution (recorded deliberately):** a 22-agent code/theory audit's *synthesis* reached the WRONG verdict ("code bug; full-residual recovers; trim is the cause", conf 88) by reading the ASGS-fallback number of an unstable full-OSGS solve as "full-OSGS optimal." Its own component agents disagreed (trim rated 8/100). Caught by two direct probes (annihilation = 3e-16; full-OSGS stall). A confident multi-agent synthesis is not a substitute for a direct numerical probe of the load-bearing assumption.

### The optional formulation lever (not needed; the rate recovers)

The rate cure — if a term-by-term rate at all h were wanted — is a **split / term-by-term OSGS**: keep the reactive `σu` in the stabilization with ASGS (identity-projection) treatment while convective/pressure terms keep the orthogonal projection, restoring `−‖τ₁^{1/2}σu‖²` at all h. This is a formulation change (new projection policy `ProjectResidualSplitReaction` + matching Jacobian + its own MMS verification + a paper-divergence entry), **not yet attempted** and not required (the rate recovers by N=640). Rejected alternatives: full-residual OSGS (solver-unstable) and constrained-space projection (breaks the O(h^{k+1}) boundary property). OSGS iteration cost was high (30–104 inner steps, the dropped ∂π/∂u linear rate) — now cut by JFNK (§5).

Full investigation synthesis: [`docs/cocquet/investigation-synthesis.md`](cocquet/investigation-synthesis.md). Source doc: `solver/osgs-reaction-dominated-rate.md`.

---

## 5. JFNK + Anderson for OSGS — both LANDED

### JFNK (matrix-free full-tangent OSGS solve) — LANDED and verified

**Verdict: JFNK recovers the dropped dense `∂π/∂u` coupling, cuts the OSGS Newton count by ~16–100×, and reproduces the frozen-π root exactly.** Opt-in behind `osgs_jfnk_enabled` (default off, mutually exclusive with `osgs_anderson_enabled`).

The OSGS coupled solve is an inexact Newton on `F(U)=0` with `π(U)=Π(R(U))` re-projected every eval; the true tangent is `dF/dU = J_frozen − C`, `C = ∫ L*(V)·τ·Π(dR·dU)` (the dense `∂π/∂u`). The solver keeps the exact frozen-π tangent (`27393f6`) and drops `C`, making it only **linearly** convergent. JFNK recovers `C` matrix-free (`J_full·v ≈ [F(U+εv)−F(U)]/ε`, the residual already re-projects π), preconditioned by the already-factored `J_frozen`.

**Phase-0 gate PASSED (2026-06-26).** The dropped `∂π/∂u` is **large**, not small (`‖ΔJ‖_F/‖J_full‖_F` = 0.15 mild → **0.90–0.97** stiff). `N_c` (frozen-π = current solver) is **60, not converged/diverged** on every measured cell; `N_j` (full-J = JFNK) is **2–3, converged**. The free preconditioner (`J_frozen`, ε_num=0, at production ε_phys) needs **1–4 GMRES iters (mild) / ≤16–21 (stiff/convective)** to the 1e-2 inner tol. The earlier "ρ_prec ≈ 1e5–1e9" was **100% constant-pressure null-mode contamination** (`ε_phys=0`, no pin; deflating that one mode gives ρ_defl=0.74; GMRES dispatches the single outlier in O(1) steps anyway) — ρ_prec is a red herring; the GMRES count is the honest metric. **ε_num as a preconditioner regularizer: not beneficial** (root-preserving but introduces preconditioner↔operator mismatch; use ε_num=0 for the JFNK preconditioner). Cost: ~2 factorizations (mild) / ~3 (stiff) vs Anderson 32–418 (often non-converging) and current 60 (diverging) — decisive in 3D. On stiff/convective cells, **JFNK is the only method that converges**.

**Phase-1 LANDED (2026-06-26).** `JFNKLinearSolver` (`src/solvers/linear_solvers.jl`) — drop-in matrix-free `LinearSolver`, left-preconditioned GMRES on `J_full·dx = b` (mat-vec = directional FD of the coupled residual via `JFNKMatVec`, Brown–Saad ε; preconditioner = factored frozen-π `A` via `JFNKPrecond`; raises `GMRESNotConvergedError` on non-convergence [C.1]). `_osgs_jfnk_solve!` (`src/solvers/osgs_solver.jl`) plugs it into the existing `SafeNewtonSolver` — "change exactly one thing: the inner linear solve" — so outer Newton, Armijo line search, divergence/stall guards, per-field gate, and C.1 honesty are inherited unchanged; falls back to the frozen-π coupled solve on structural failure (never worse). **A/B integration** (orthogonality MMS cell, P2/P1): frozen-π = 17 iters / 8.96 s stopping at ‖F‖∞≈3e-5; **JFNK = 6 iters / 6.73 s reaching ‖F‖∞≈4e-8** — faster and more fully converged, L2 velocity errors agree to 5 sig figs. **Verified:** Blitz 240/240, Quick 76/76, `test/extended/jfnk_equivalence_extended_test.jl` 8/8 (ASGS byte-identical, OSGS same MMS root, JFNK iters ≤ frozen-π).

**Benign JFNK fallback (~5% of 2D sweep cells, 14/288):** outer-Newton safeguards firing (line-search depletion / divergence guard), NOT inner-GMRES/preconditioner failures. Two regimes: the convective corner (Re=1e6 × coarse h — the full-tangent step is too aggressive for the re-projecting merit, frozen-π's gentler steps win — the "Newton-vs-Picard is a wash" lesson at the extreme) and boot-already-converged low-Re cells (matrix-free FD of a noise-floor residual, divergence guard correctly bails). Every fallback cell still gives the validated optimal error. **For 3D: expect the same on the convective corner; do not chase 100% JFNK coverage.**

**3D watch item:** if 3D inner-GMRES `G` is routinely large, that is the trigger to add a real saddle-point/MG preconditioner (block/Schur — PCD/LSC/SIMPLE — or Vanka/MG; τ₁ already seeds a discrete pressure-Laplacian in the (2,2) block). Config: `osgs_jfnk_enabled`, `osgs_jfnk_gmres_*`, `osgs_jfnk_fd_epsilon`.

### Anderson-accelerated staggered OSGS — LANDED

**Verdict: Anderson cuts the staggered outer-iteration count by ≈1.4–2.2×.** Opt-in behind `osgs_anderson_enabled` (default off, OFF is bit-identical). When on, the OSGS stage runs the staggered fixed point (`_osgs_anderson_outer!`): freeze `π_k = Π(R(x_k))`, solve the frozen-π nonlinear system (exact-tangent Newton, 1–2 iters), extrapolate `x_{k+1}` via `AndersonAccelerator.update!` (L²-weighted by the block (u,p) mass matrix), stop on relative state drift < xtol. Both schemes reach the same discrete OSGS fixed point `R − Π(R) = 0`.

| case | accelerator | outer iters | inner Newton | L2u error |
|---|---|---|---|---|
| **MILD** (8×8, σ=1) | Picard m=0 | 33 | 40 | 2.46593e-2 |
| | Anderson m=5 | **18** | 23 | 2.46593e-2 |
| | Anderson m=10 | **15** | 20 | 2.46593e-2 |
| **STIFF** (16×16, σ=1e3) | Picard m=0 | 19 | 21 | 9.69220e-4 |
| | Anderson m=5 | **14** | 16 | 9.69216e-4 |
| | Anderson m=10 | **12** | 14 | 9.69215e-4 |

Deeper history helps more (m=10 beats m=5 both cases); same fixed point (L2u matches to ~6 sig figs). Config (`numerical_method.solver`): `osgs_anderson_enabled` (false), `osgs_anderson_depth` (5), `osgs_anderson_relaxation` (1.0), `osgs_anderson_safety_factor` (10.0, Powell restart), `osgs_anderson_max_outer` (50). **Note:** on the stiff/convective reaction cells (the JFNK Phase-0 baseline), **both** the current path **and** Anderson *fail to converge* — Anderson is infrastructure for the linear-rate bottleneck, not a substitute for JFNK there.

Source docs: `solver/jfnk-phase0-preconditioner-gate.md`, `solver/osgs-anderson-acceleration.md`.

---

## 6. CocquetFormMMS — moderate-α clean; the low-α×high-Re corner is a coarse-mesh fold (FE-optimal above it)

`test/extended/CocquetFormMMS`: equal-order stabilized VMS (P1/P1 & P2/P2, ASGS+OSGS) vs unstabilized Galerkin **Taylor–Hood P2/P1** under one MMS sweep, with the **full Forchheimer–Ergun reaction** `σ = a(α)+b(α)|u|` (`σ_lin=0.3, σ_nl=1.75`, the Cocquet coefficients). Grid: `α₀ ∈ {0.5, 0.1}`, `Re ∈ {1, 1e5}`, Da=1 (a no-op under Forchheimer — `α₀` is the effective-Da knob, `Da_eff ≈ 40` at α=0.1, ≈2 at α=0.5), TRI N=10…160, gate `eps_tol_momentum = 1e-9`.

| Cell | k=1 (ASGS/OSGS) | k=2 (ASGS/OSGS) | Taylor–Hood |
|---|---|---|---|
| **α=0.5, Re=1** | ✅ 5/5 | ✅ 5/5 | ✅ 5/5 |
| **α=0.5, Re=1e5** | ✅ 5/5 | ✅ 5/5 | ✅ 5/5 |
| **α=0.1, Re=1** | ✅ 5/5 | ASGS 4/5, OSGS 3/4 | ✅ 5/5 |
| **α=0.1, Re=1e5** | fold N≤80, ✅ **optimal at N≥160** | ASGS root N=40,80; OSGS N=80 | fails at corner (§4.3) |

### §2 — Moderate porosity (α=0.5): the definitive clean result

**Complete, clean convergence for *every* method** (k=1, k=2, ASGS, OSGS, Taylor–Hood) at both Re. Velocity optimal (L² rate ≈2.2 for k=1), pressure converges. ASGS and OSGS are genuinely distinct (α=0.5/Re=1e5/N=80: ASGS L²u≈3.2e-4 vs OSGS≈4.2e-4). (The ASGS≠OSGS fix: the per-cell config previously hard-coded `method="ASGS"`, so OSGS silently ran ASGS byte-identical; each method now gets its own config.)

### §4.1 — α=0.1×Re=1e5 is a coarse-mesh solution-branch fold — RESOLVED for k=1 (2026-07-07)

The α=0.1×Re=1e5 "failure" is a **genuine coarse-mesh turning-point fold** — no root with ‖R‖≤tol exists for N≤80 — that **recedes with mesh refinement**; NOT a solver bug or stabilization defect. Evidence: the fold recedes as N↑ and deepens as α↓ (α=0.9/0.5 converge every mesh; **α=0.2 folds at N≤20, converges at N=40; α=0.1 folds at N≤80, has a TRUE root at N=160** — ASGS ‖R‖=1.3e-7 L²u=1.84e-3 in 7 it; OSGS ‖R‖=1.0e-6 L²u=2.09e-3 in 48 it). It is not a basin/initial-guess problem (the harness already inits from the exact interpolant + perturbation-homotopy and still folds at N≤80) — mirroring the sister harness's A1/A2 tests (exact Jacobian 4.8e-12; heavy Newton and Picard from `u_ex` both stall at ‖R‖≈5e-2, §2 / [`mms/fold-recovery.md`](mms/fold-recovery.md)).

**Recovery (extend the ladder above the fold, N=[160,320], cold exact-guess — no mesh-continuation, no `src/` change):**

| method | N=160 (‖R‖) | N=320 (‖R‖) | rate L²u | rate H¹u | rate L²p |
|---|---|---|---|---|---|
| ASGS | L²u=1.84e-3 (1.3e-7) | L²u=2.28e-4 (1.7e-7) | **3.01** | **1.07** | 3.03 |
| OSGS | L²u=2.09e-3 (1.5e-6) | L²u=2.44e-4 (5.6e-7) | **3.10** | **1.10** | 3.12 |

H¹u ≈ 1.07/1.10 is textbook-optimal O(h) for k=1; L²u ≈ 3.0 super-optimal — **identical to the sister harness's α=0.05 corner** (H¹≈1.0, L²≈3.0), cross-validating the 2-point slope. So the equal-order stabilized method converges optimally at the corner once past the fold; the fold is the paper's intrinsic 1/α₀ degradation pushing the *coarse-mesh nonlinear discrete map* past a turning point, not a loss of FE order. The k=2 corner already has clean roots at N=40 & N=80 (clears the fold ~2× earlier); extending to N=160 to firm its rate is a cheap remaining follow-up.

### §4.3 — mechanism: Newton is exact; TH is convectively unstable at the corner; σ̃_α unconfirmed (2026-07-08)

- **Newton is EXACT for the Cocquet formulation — the fold is NOT a linearization bug.** The permanent test `test/extended/cocquet_jacobian_consistency_extended_test.jl` assembles the Exact-Newton Jacobian for **SymmetricGradient + Forchheimer** vs a centered FD of the residual: **‖J−J_fd‖/‖J_fd‖ ≈ 1e-8…1e-11** for k=1 *and* k=2 at corner-like params, and Newton converges **quadratically** (‖R‖: 2.9e-4→1.3e-5→2.7e-8→2.3e-13). This closed a real coverage gap (the velocity-dependent `∂σ/∂u` Exact-Newton tangent for the Cocquet combo was previously unguarded — `picard_jacobian_equivalence` checks only Picard mode, `osgs_frozen_pi_jacobian` uses ConstantSigma). ⇒ the fold is a genuine property of the correctly-linearized nonlinear problem.
- **The "Taylor–Hood converges everywhere" claim is FALSE at the corner (correction).** TH (Galerkin P2/P1) at α=0.1×Re=1e5 does **not** reach a root: its residual **stalls at O(1)** (‖R‖ = 650→190→50→13→**3.1** at N=10→160, vs ~1e-9 when it works), velocity error **flat at 0.40** (rate 0). TH is *convectively unstable* here (inf-sup for pressure but no SUPG for convection) — **pressure** converges optimally (L²p rate 2.0, via LBB) while **velocity** is garbage. So the real contrast is **VMS folds hard at coarse mesh (NaN) but converges to an accurate root at N≥160, where TH still cannot** (res 3.1, L²u 0.40). **VMS is the *better* method at the corner.** (At Re=1, TH converges cleanly, rate 2.94 — confirming the corner failure is the high-Re convective instability.)
- **Reaction-out-of-stabilization A/B — INCONCLUSIVE (σ̃_α not confirmed).** Via a temporary gated `STRIP_REACTION_FROM_STAB` diagnostic (since **REVERTED** — not paper-faithful; default-off was byte-identical, Blitz 243/243; the stripped formulation was self-consistent, J-vs-FD ~1e-11, quadratic): removing σ from the stabilization (τ₁, 𝓛U, 𝓛*V, and derivatives) while keeping the coercive Galerkin `(v,σu)` cleared **only 1 of 3** folding meshes (N=40, to an inaccurate root L²u=0.092, ~3× too large); N=20 and N=80 still fail from the exact guess. **Confound:** the strip also enlarges τ₁ (removes σ from its denominator), and σ is genuinely entangled in the stabilization scale (σ̃_α itself contains τ₁) — so it neither confirms nor cleanly refutes σ̃_α. It does show the fold is **not reducible to a single removable term**; reaction-in-stab is at most a partial contributor.
- **c₁ — not the lever.** At high Re, τ_NS⁻¹ is convection-dominated (`c₂|u|/h ≫ c₁ν/h²`), so raising c₁ barely moves the stabilization. A c₁×4 probe (via `C1_MULT` env hook, default 1.0 byte-identical): α=0.1 folds at all N at paper c₁, and c₁×4 converges only N=10 (L²u≈0.428, large) and still folds at N=20/40 — *not* a convergent sweep; α=0.2 keeps the same pattern but N=40 error is ~5× smaller (0.0736→0.0151). Marginal help, no rescue — UNLIKE 3D-P2, and expected: this is **k=1**, where the viscous 2nd-derivative subscale (the term c₁ fixed for P2 tets, §3) is *identically zero*, so c₁ acts only through τ_NS. (A c₁×64 confirmation was killed under CPU contention.) The mild help is weak evidence of a coercivity/stabilization component (consistent with the σ̃_α Layer-2 hypothesis).

**Bottom line: the exact fold mechanism is OPEN.** Ruled out: a Jacobian/linearization bug (Newton exact) and c₁. The leading hypothesis is the paper's **`σ̃_α` coercivity weakening** (ASGS estimate, `article.tex` §`sec:StabilityASGS`, `eq:SigmaAlpha`: `σ̃_α = τ_{1,NS}⁻¹σ/(τ_{1,NS}⁻¹+σ/α_K)` collapses as α→0; TH keeps full σ) — paper-grounded and consistent with every observation (reaction-magnitude driven, low-α specific, fold-recedes-with-mesh) but **not confirmed** (the paper deems σ̃_α *benign for the rate* — the "weaker coercivity ⇒ nonlinear-solver fold" link is an unproven extension, and the strip test was τ₁-confounded). A cleaner isolation (strip σ from 𝓛U/𝓛*V only, holding τ₁ physical) is deferred — the practical deliverable (§4.1) does not depend on it. **The fold is reaction-magnitude driven, not nonlinearity-driven:** a linear-reaction control at α=0.1 with matched `Da_eff` (`σ_nl=0`) folds identically ⇒ the `b(α)|u|` Forchheimer `∂σ/∂u` basin-shrink is not the cause. (Layer 1 — high-Re τ₁-saturation, `theory/tau_saturation_note` — is dormant at Re=1e5, a Re≳1e6 effect, so it does not explain this fold.)

### ASGS vs OSGS reaction handling — do not conflate

σ̃_α is the **ASGS** estimate. **OSGS** removes the reaction from the orthogonal projection (for constant σ its orthogonal subscale is exactly zero, `article.tex` ~line 619), which the paper says *"facilitates the convergence of the nonlinear iterations"* — a documented reaction/convergence concern ASGS does not phrase the same way. The trim `ProjectResidualWithoutReactionWhenConstantSigma` is **OSGS-only** (`src/stabilization/projection.jl:76-80`) and fires only for `Constant_Sigma`; under **Forchheimer** (what the sweep uses) the reaction stays in the stabilization for **both** methods — consistent with both folding at low α.

Full detail: [`docs/cocquet/investigation-synthesis.md`](cocquet/investigation-synthesis.md) (Cocquet synthesis) and [`docs/mms/3d-p2-coercivity-resolution-dossier.md`](mms/3d-p2-coercivity-resolution-dossier.md) (the c₁ mechanism). The τ-saturation note is `theory/tau_saturation_note/tau_saturation_note.tex`. Source doc: `cocquet/cocquet-form-mms-status.md`.

---

## Cross-references

- **Full evidence dossiers:** [`docs/mms/3d-p2-coercivity-resolution-dossier.md`](mms/3d-p2-coercivity-resolution-dossier.md), [`docs/cocquet/investigation-synthesis.md`](cocquet/investigation-synthesis.md), [`docs/formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md).
- **Theory (LaTeX):** [`theory/paper/article.tex`](../theory/paper/article.tex) (the authoritative formulation), `theory/osgs_reaction_note/osgs_reaction_note.tex`, `theory/tau_saturation_note/tau_saturation_note.tex`, `theory/osgs_algorithm/osgs_algorithm.tex`.
- **Living companions:** [`lessons_learned.md`](lessons_learned.md) (regression ledger), [`known_issues.md`](known_issues.md) (open code-correctness items), [`solver/paper-code-divergences.md`](solver/paper-code-divergences.md), [`solver/algorithm-code-mapping.md`](solver/algorithm-code-mapping.md), [`solver/nonlinear-convergence-criterion-prompt.md`](solver/nonlinear-convergence-criterion-prompt.md).
