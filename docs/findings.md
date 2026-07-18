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

The mass convergence gate is now the **Philosophy-A algebraic** `ε_C = ‖r_C‖/D_C` (pressure block of the assembled residual, symmetric with the momentum gate `ε_M = ‖r_M‖/D_M`, gated ~1e-9). The old loose `eps_tol_mass = 0.8` gate is demoted to the diagnostic `eps_C_strong`. Core commit `e455f36`; essential companion fix `3b76864` (below). Gate spec: [`theory-code-map.md`](theory-code-map.md) §3; config knobs `eps_tol_momentum` / `eps_tol_mass` in `SolverConfig` (`src/config.jl`), consumed by `convergence_criterion.jl` and the `"scale_free"` path in `src/solvers/nonlinear.jl` (`_safe_solve_inner!`).

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

Source docs consolidated here: `mms/convergence-2d.md`, `mms/convergence-baseline.md`.

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

Source doc: `mms/convergence-2d.md`.

---

## 3. 3D-P2 MMS "converged-but-wrong" — RESOLVED (`4k⁴` under-margined for high-`C_inv` structured tets)

**Verdict (2026-07-06): NOT a bug and NOT a Gridap↔paper discrepancy.** The viscous 2nd-derivative subscale is **anti-coercive by construction** (the viscous operator is self-adjoint, so `B_S = B − Σ⟨L*V, τ𝓛U⟩` carries `−τ‖𝓛_visc V‖²`, unlike the *coercive* `+τ‖a·∇V‖²` from skew-adjoint convection — which is why removing/shrinking it "heals"). It must be dominated by the coercivity condition `c₁ > 2ξ·C_inv²`. **The paper's fixed `c₁ = 4k⁴` has almost no margin for 3D tets** (large `C_inv`), so a slightly poorly-shaped structured mesh sits over the coercivity edge. The remedy is an **element-aware `c₁`**, which article.tex **line 910** explicitly prescribes ("the optimal value of `c₁` depends on the element types … through the inverse estimate constant"). This is fully within the quasi-uniform theory the theorems assume (the structured Kuhn mesh is quasi-uniform — congruent, shape-regular tets), and consistent with the paper's optimal §5.2 result (which used *unstructured* tets over *only 3 meshes*).

> **The earlier "Gridap↔paper discrepancy / c₁ masks a bug" framing (2026-07-05) is WITHDRAWN.** c₁ is not masking a bug — it is the theorem's own coercivity knob, and `4k⁴` is simply an element-dependent choice that is under-margined here.

**Full for-scrutiny record** (every experiment + numbers, the C_inv table, mesh-independence, shape-regularity, the c₁×4 ladder, τ/h levers, the BCC attempts, the refuted-alternatives table A–O, the paper reconciliation, and the honest caveats / what would overturn the verdict): [`docs/mms/p2-3d.md`](mms/p2-3d.md).

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

### Independent clean-room confirmation of the coercivity margin (2026-07-09)

An external from-scratch recomputation (monomial bases, exact integration, nothing shared with the code) **reproduced the element inverse constants** `C_inv²` = 214 (Kuhn TET) / 60 (Q2 quad) / 48 (P2 right-TRI) / 180 (Q2 hex) exactly, and added two numbers the repo lacked:

- **Regular TET (best-possible tet) = 66.67** — even an *ideal* tet exceeds the quad's 60 and the `c₁=4k⁴=64` budget (in diameter units), so *no* tetrahedron sits comfortably inside the 2D margin. The deviatoric symmetric-gradient operator on the Kuhn tet is worse still: **282.4**.
- A **single element-independent threshold `ξ* ∈ (1.42, 2.13]`** fits every 2D and 3D data point once `c₁` and `C_inv²` are expressed in the **same** `h`-convention. The apparent "inconsistency" was an apples-to-oranges units error (comparing `c₁=64` in harness-`h` against `C_inv²` in diameter). Converting to consistent diameter units gives ratio ≈2.13 (quads, works) vs ≈0.71 (Kuhn tets, fails) — one threshold, all families.
- The dimension-dependent **Fourier** correction (the deviatoric viscous symbol's longitudinal eigenvalue `(2−2/d)·αν|k₀|²` = 4/3 in 3D vs 1 in 2D) supplies only ~4/3 of the needed increase; the **dominant** driver is geometric — the Kuhn 1:√2:√3 edge anisotropy carries ~3.3× more high-frequency content than the quad → total ≈3.6×, matching the empirical `c1_mult=4`.

Because this is a pure element-geometry eigenvalue argument (independent of any formulation transcription), it corroborates the coercivity-margin verdict on its own. The same external audit also raised **code observations** (joint c₁/c₂ scaling in the multiplier hooks; a possible `tau_reg_lim` unit inconsistency; the 3D harness's `h`-convention bypass) — captured as **verify-first** leads in [`pending-tasks.md`](pending-tasks.md) §2c, not enshrined here until checked against the code.

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

3D all-Dirichlet with ε=0 is **ill-posed** (constant-pressure null mode). The **Codina iterative penalty** adds `ε_num·(pⁿ − pⁿ⁻¹)` to the **mass residual** (the code previously had `ε_num` only in the Jacobian, where it cancels in the residual and vanishes at convergence). This fix — with the gated `p_prev` handling — gives well-posedness (Blitz 240/240). **Its "P2 root cause = penalty" claim is WITHDRAWN:** the penalty fixes *well-posedness*, not the P2 *accuracy* defect. Its *original* "NOT c₁" instinct is vindicated (paper c₁ is correct; the P2 defect is the element-margin issue above, neither penalty nor c₁). Also fixed in the same work: spurious-root homotopy acceptance (reference-root match, `5ecc0ca`) and an OSGS→ASGS `initial_ftol` recording leak (`osgs_advanced_off_entry` guard). Canonical: [`mms/p2-3d.md`](mms/p2-3d.md).

### 3D official sweep results (2026-06-30, paper c₁, regular Kuhn mesh, self-describing `results/k*/TET/structured/`)

- **OSGS P1 SOLVED** — robust (`eps_used=1`) **and** fully optimal (recipe: iterative-penalty + boot-skip + JFNK + reference-root homotopy).
- **ASGS P1 robust** (L²u structured-limited — the genuine 3D P1-ASGS L²-order sub-optimality on structured tets; **OSGS recovers it**).
- **ASGS P2 non-monotone** at paper c₁ (the under-margin above; orthogonal to the penalty fix).
- **OSGS P2 SOLVED** (2026-07-09) — a preconditioner-only c₁×4 inflation reaches the paper-c₁ root; see below.

### OSGS-P2-3D ∂π/∂u coupling — RESOLVED (2026-07-09, preconditioner-only c₁×4)

The blocker was **`ρ_prec = ρ(J_frozen⁻¹·∂π/∂u) ≈ 1249`** at paper c₁ (2D ref ≈ 0.88): the frozen-π tangent is a hopeless preconditioner for the coupled `∂π/∂u` system, so JFNK-GMRES stalled and the solver **sat at the exact-guess interpolant** — `success=false` was *correct*. A **preconditioner-only c₁×4 inflation** (`osgs_jfnk_precond_c1_mult`, default-off) drops `ρ_prec` to **≈3.8** (a U-shaped optimum) while the residual `F` stays at paper c₁, so the converged root is unchanged (`‖F‖→1.4e-12`). On (12,12,3): **`success=true`, `eps_used=1`, quadratic Newton (5 iters)**. Landed default-off (Blitz 272/272, Quick 85/85); the OSGS-P2 structured recipe uses `mult=4` + `jfnk_maxiter=80`.

**Corrects the earlier "accurate but `ok=false`" reading:** that was an artifact of starting from the exact-solution **interpolant** — the solver never descended off it, so the reported L²u=0.0012 was the *interpolation* error, not a reached root. Reaching the paper-c₁ root for the first time makes its true error visible: velocity accurate (L²u=0.00123 ≈ interpolant) but **pressure ~15× larger (L²p=0.045)** — the paper-c₁ P2-3D pressure is genuinely under-stabilized (c₁×4 *in the residual* gives L²p=0.0029). That pressure accuracy question is §3 (the coercivity-margin story), now directly measurable. Refuted en route: stronger-`ε_num` (made GMRES worse), constant-pressure gauge deflation (ρ_prec unchanged — unlike 2D), damped staggering (its Picard contraction rate *is* ρ_prec=1249), and a classic Schur / approximate-`J_frozen` preconditioner (already exact `J_frozen⁻¹`; the deficit is the *dropped* `∂π/∂u`, which the c₁-inflated preconditioner approximates).

Source doc: `mms/p2-3d.md`. Diagnostic hooks (default-off): `smoke3d.jl` `ablation`/`h_conv`; `continuous_problem.jl` `VISC_ADJ_MULT`; `tau.jl` `TAU_VISC_MULT`; `data/phase1_tri_k2.json`.

### Element-aware c₁ made exact — irregular-mesh sub-optimality is NOT coercivity (2026-07-12)

`theory/numerical_constants/c1_dimension_note.tex` gives the exact elementwise coercivity floor `c₁*(K) = 2ĉ²(K)` (pure shape constant); `test/extended/ManufacturedSolutions3D/element_c1.jl` transcribes it and **reproduces the note's Table to ~1e-14** (`ĉ²(Kuhn)=214` **is** the `C_inv²≈214` above; P1 ⇒ `c₁*≡0`). Measured on nested_red (`c1_distribution_probe.jl`), the red-refined **quality tail grows** so `c₁*/64` reaches p99 7.6 / max 14.9 at L2 (Kuhn is flat ≈2.87) ⇒ a fraction is elementwise sub-coercive at `×4`. **But the same-mesh study `smoke3d.jl c1study_nested_red` REFUTES that as the lever:** `×4`→`×14.9` moves the k=2 finest `L2u` rate by **Δ≈±0.05** (2.63–2.68; L2 byte-identical = the ILU-GMRES-uncertified interpolant). So the nested_red k=2 sub-optimality is **mesh-quality + hardware, not tail-coercivity** — `c₁` is the lever only for *uniform* sub-coercivity (Kuhn @ paper c₁). k=1 control: element-aware→`mult=1` is *worse* accuracy than `×4` ⇒ element-aware c₁ is the coercivity **floor**, not the accuracy optimum. Artifacts: `nested_red_<strategy>/` result leaves, `compare_c1study.py`. Source doc: `mms/p2-3d.md` §A.

### Interpolation reference DECOMPOSES the 3D sub-optimality — direct confirmation (2026-07-17)

The prior verdict ("irregular sub-optimality is mesh-quality, not coercivity") was reached indirectly (a
`c₁` A/B). `test/extended/ManufacturedSolutions3D/run_interpolation_reference3d.jl` now measures it
**directly**: the nodal-interpolation error of the exact fields on the *same* Kuhn and (committed-base)
nested-red sequences, normalized identically to the solver rows (`calc_errors3d`), at the same quadrature
(`4k+4`). Inserted as reference rows in `tab:3DL2`/`tab:3DH1`. At the tables' **adopted `c₁=16k⁴`**,
`α₀=0.5`:

- **Velocity is at the interpolation floor on both families, in value AND slope.** Efficiencies (method
  FME / interpolant FME): ℙ₁ u H¹ 1.02/1.02 (reg), 0.91/0.93 (irr); ℙ₂ u L² 1.04/1.00 (reg), **1.03/1.00
  (irr)**; ℙ₂ u H¹ 1.02/1.07 (reg), 0.93/0.91 (irr); ℙ₁ u L² 1.5/0.8 (reg) — the one case slightly above,
  still same order. Where the method slope is sub-optimal the **interpolant slope is too**: irregular ℙ₂
  u L² method 2.55/2.56 vs interpolant **2.67**; irregular ℙ₁ u H¹ method 0.83/0.85 vs interpolant
  **0.71** (regular interpolant is 0.94 — the red-refined quality tail depresses the *best-approximation*
  rate). ⇒ the 3D velocity sub-optimality is the mesh's approximation capacity, **not the formulation**.
  This is the direct evidence the review's C6 (`docs/review_numerics_vs_theory.md`) said the paper's
  "element-quality tail" attribution lacked; it can now be stated as a decomposition, not an assertion.
- **Pressure is NOT at the floor** (13–770× above; OSGS H¹ reaches 1.29), but this is **not** a 3D or
  element-quality defect: it is the expected viscous-regime one-order pressure loss, and the 2D viscous
  baseline is *further* from its floor (ℚ₂ p L² 4964×, ℙ₁ p H¹ normalized 2.61>1). It appears equally on
  regular and irregular families. So the pressure "sub-optimality" is theory-consistent (the working-norm
  order loss), a different mechanism from the velocity.

**Scope / still open.** This is at `c₁=16k⁴` (the tables' constant) — it does **not** address the
`open-questions.md §3` caveats about whether plain `4k⁴` is fragile for 3D tets in general (Reading A vs
B; the un-built well-shaped-tet positive test), which are a separate `c₁`-margin question. And it does
not explain **C7**: the OSGS pressure H¹ = **1.29** printed *identically* on three different rows (regular
ℙ₁, irregular ℙ₁, irregular ℙ₂) — different meshes, orders, slopes. The magnitude is unremarkable given
the viscous baseline, but the exact triple is a coincidence to check against the raw data (transcription
slip vs a genuine mesh-independent saturation); the in-progress 3D re-run (`smoke3d.jl sweep_structured`
at `c₁=16k⁴`, 0 `success=false`, with per-mesh `success` flags now recorded) will settle it. Source doc:
`docs/paper-revision-plan.md` §8.

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

The reaction-projection trim (drop `σu` from the projection for constant σ, article.tex §580, `ProjectResidualWithoutReactionWhenConstantSigma`) is correct and **coefficient-robust at the fixed point** (it removes a quantity annihilated anyway; `src/stabilization/projection.jl:63-76`). Switching to `ProjectFullResidual` at Da=10⁶ **stalls/diverges at every mesh** (→NaN at N=80) and reports the ASGS Stage-I fallback (H¹ bit-identical to ASGS) — because although the *residual* equals the trim's, the *Jacobians differ*: the trim's `−σ·du` term is load-bearing for solver stability. **This is why the paper trims; the trim is not removable.**

> **Methodological caution (recorded deliberately):** a 22-agent code/theory audit's *synthesis* reached the WRONG verdict ("code bug; full-residual recovers; trim is the cause", conf 88) by reading the ASGS-fallback number of an unstable full-OSGS solve as "full-OSGS optimal." Its own component agents disagreed (trim rated 8/100). Caught by two direct probes (annihilation = 3e-16; full-OSGS stall). A confident multi-agent synthesis is not a substitute for a direct numerical probe of the load-bearing assumption.

### The optional formulation lever (not needed; the rate recovers)

The rate cure — if a term-by-term rate at all h were wanted — is a **split / term-by-term OSGS**: keep the reactive `σu` in the stabilization with ASGS (identity-projection) treatment while convective/pressure terms keep the orthogonal projection, restoring `−‖τ₁^{1/2}σu‖²` at all h. This is a formulation change (new projection policy `ProjectResidualSplitReaction` + matching Jacobian + its own MMS verification + a paper-divergence entry), **not yet attempted** and not required (the rate recovers by N=640). Rejected alternatives: full-residual OSGS (solver-unstable) and constrained-space projection (breaks the O(h^{k+1}) boundary property). OSGS iteration cost was high (30–104 inner steps, the dropped ∂π/∂u linear rate) — now cut by JFNK (§5).

Full investigation synthesis: [`docs/cocquet/investigation-synthesis.md`](cocquet/investigation-synthesis.md). Source dossier (archived): [`archive/osgs-reaction-dominated-rate.md`](archive/osgs-reaction-dominated-rate.md).

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

Source doc: [`solver/jfnk-phase0-preconditioner-gate.md`](solver/jfnk-phase0-preconditioner-gate.md) (the standalone Anderson dossier was merged into this §5, 2026-07-11).

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

The α=0.1×Re=1e5 "failure" is a **genuine coarse-mesh turning-point fold** — no root with ‖R‖≤tol exists for N≤80 — that **recedes with mesh refinement**; NOT a solver bug or stabilization defect. Evidence: the fold recedes as N↑ and deepens as α↓ (α=0.9/0.5 converge every mesh; **α=0.2 folds at N≤20, converges at N=40; α=0.1 folds at N≤80, has a TRUE root at N=160** — ASGS ‖R‖=1.3e-7 L²u=1.84e-3 in 7 it; OSGS ‖R‖=1.0e-6 L²u=2.09e-3 in 48 it). It is not a basin/initial-guess problem (the harness already inits from the exact interpolant + perturbation-homotopy and still folds at N≤80) — mirroring the sister harness's A1/A2 tests (exact Jacobian 4.8e-12; heavy Newton and Picard from `u_ex` both stall at ‖R‖≈5e-2, §2 / [`mms/convergence-2d.md`](mms/convergence-2d.md)).

**Recovery (extend the ladder above the fold, N=[160,320], cold exact-guess — no mesh-continuation, no `src/` change):**

| method | N=160 (‖R‖) | N=320 (‖R‖) | rate L²u | rate H¹u | rate L²p |
|---|---|---|---|---|---|
| ASGS | L²u=1.84e-3 (1.3e-7) | L²u=2.28e-4 (1.7e-7) | **3.01** | **1.07** | 3.03 |
| OSGS | L²u=2.09e-3 (1.5e-6) | L²u=2.44e-4 (5.6e-7) | **3.10** | **1.10** | 3.12 |

H¹u ≈ 1.07/1.10 is textbook-optimal O(h) for k=1; L²u ≈ 3.0 super-optimal — **identical to the sister harness's α=0.05 corner** (H¹≈1.0, L²≈3.0), cross-validating the 2-point slope. So the equal-order stabilized method converges optimally at the corner once past the fold; the fold is the paper's intrinsic 1/α₀ degradation pushing the *coarse-mesh nonlinear discrete map* past a turning point, not a loss of FE order. The k=2 corner already has clean roots at N=40 & N=80 (clears the fold ~2× earlier); extending to N=160 to firm its rate is a cheap remaining follow-up.

### §4.3 — mechanism: Newton is exact; TH is the *worse* method at the corner; σ̃_α unconfirmed

- **Newton is EXACT for the Cocquet formulation — the fold is NOT a linearization bug.** Exact-Newton J vs centered FD for **SymmetricGradient + Forchheimer**: **‖J−J_fd‖/‖J_fd‖ ≈ 1e-8…1e-11** (k=1 *and* k=2), quadratic ‖R‖ descent (2.9e-4→1.3e-5→2.7e-8→2.3e-13). Now guarded by `test/extended/cocquet_jacobian_consistency_extended_test.jl` (closed a real gap — the velocity-dependent `∂σ/∂u` tangent for this combo was previously unguarded). ⇒ the fold is a genuine property of the correctly-linearized problem.
- **"Taylor–Hood converges everywhere" is FALSE at the corner.** TH (Galerkin P2/P1) at α=0.1×Re=1e5 does **not** reach a root: ‖R‖ stalls at O(1) (650→190→50→13→**3.1** at N=10→160), velocity error **flat at 0.40** (rate 0) — convectively unstable (LBB pressure but no SUPG), so **pressure** converges (L²p rate 2.0) while **velocity** is garbage. So **VMS is the *better* method at the corner** (accurate root at N≥160 where TH cannot). (At Re=1, TH converges cleanly, rate 2.94.)
- **The fold is reaction-magnitude driven, not nonlinearity-driven:** a linear-reaction control (`σ_nl=0`, matched `Da_eff`) folds identically ⇒ the Forchheimer `∂σ/∂u` basin-shrink is not the cause.

**The exact fold mechanism is OPEN** — leading hypothesis is the paper's `σ̃_α` reaction-in-stabilization coercivity weakening (ASGS estimate `eq:SigmaAlpha`, collapses as α→0). The ruled-out list (Jacobian bug; c₁-as-lever), the τ₁-confounded `STRIP_REACTION` strip test, and the ASGS-vs-OSGS reaction-handling distinction (the OSGS-only trim `ProjectResidualWithoutReactionWhenConstantSigma`, `src/stabilization/projection.jl:66-80`, fires only for `Constant_Sigma`; under Forchheimer the reaction stays in the stabilization for both methods) are detailed once in [`open-questions.md`](open-questions.md) §1, its canonical home.

Full detail: [`docs/cocquet/investigation-synthesis.md`](cocquet/investigation-synthesis.md) (Cocquet synthesis) and [`docs/mms/p2-3d.md`](mms/p2-3d.md) (the c₁ mechanism). The τ-saturation note is `theory/tau_saturation_note/tau_saturation_note.tex`. Source doc: `cocquet/cocquet-form-mms-status.md`.

---

## 7. Resolved code-correctness issues (folded from the retired `known_issues.md`, 2026-07-10)

Tracked as open code bugs, now fixed — kept here so retiring `known_issues.md` loses nothing. **Open**
code-correctness items now live in [`pending-tasks.md`](pending-tasks.md) (§2 code–theory consistency, §6 input/output).

- **`cfg.phys.f_x`/`cfg.phys.f_y` would throw** — `run_simulation.jl` read `cfg.phys.f_x`, but the field is
  `physical_properties`. **FIXED 2026-06-04 (P6):** corrected to `cfg.physical_properties.f_x`/`.f_y`; the
  production path is now exercised by `test/quick/production_schedule_smoke_quick_test.jl`.
- **`config/base_config.json` was missing the required `eps_val`** — **RESOLVED 2026-07-04.** The field was
  renamed to **`physical_epsilon`** and `base_config.json` now carries `"physical_epsilon": 0.0`, so
  `load_frozen_config("config/base_config.json")` (and the `run_simulation` example) loads cleanly.
  `validate!` asserts `physical_epsilon >= 0` (it may be 0 — it is ε_phys, not the porosity). See
  [`formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md) §A.4.
- **Dead helper after the covariance fix** — `_resolve_solution_scale_per_field` + the `x_per_field_raw`
  buffer were no longer called (the gate uses the frozen initial-residual scale `‖R₀‖` directly).
  **RESOLVED 2026-06-04 (P5):** both removed.
- **High-Da OSGS rate "defect"** — **RESOLVED 2026-06-10: pre-asymptotic, recovers to ≥1.0 at N=640** (not an
  order ceiling). Full account in §4 above; permanent mechanism in
  [`../theory/osgs_reaction_note/osgs_reaction_note.tex`](../theory/osgs_reaction_note/osgs_reaction_note.tex).

---

## 8. Machine-checked a priori theory (Coq) — RESOLVED (the whole chain; trusted base 50 rows, residue 2 items)

**Verdict (2026-07-13): the paper's entire a priori chain is machine-checked.** `lemma:Stability`, `lemma:Continuity`, `lem:continterp` and `thm:convergence` are complete Coq theorems (`proof_verification/coq-formal/`, 17 files — the 15-file a priori chain plus the two non-vacuity witnesses added 2026-07-17 — ~789 theorems). All compile and are re-verified by `coqchk` (Coq 8.18.0); `Print Assumptions` on each returns exactly three standard-library axioms and **no user axiom**: two from the classical-reals construction (`ClassicalDedekindReals.sig_not_dec`, `sig_forall_dec`) plus functional extensionality (`FunctionalExtensionality.functional_extensionality_dep`). Full map, and the complete 50-row hypothesis inventory: [`proof_verification/coq_coverage.tex`](../proof_verification/coq_coverage.tex).

### The `c₁` margin, now quantified — this is what §3 was hitting empirically

§3 concluded that `c₁ = 4k⁴` is *under-margined* for high-`C_inv` 3D tets, against the coercivity condition `c₁ > 2ξ·C̄_inv²`. The formalisation makes the two halves of that statement precise, and both are now theorems (`StabilityAlgebra.v`):

- **The sharp threshold is a factor of two lower than the paper states.** Positivity of the two stability coefficients needs only `c₁ > ξ·C̄_inv²` (with `ξ > 2`), not `c₁ > 2ξ·C̄_inv²` — `stability_constants_positive_sharp`. So the paper's condition is *sufficient, not necessary*.
- **The extra factor of two buys a quantitative floor, not mere positivity** — `C_stab_margin`. Under `c₁ > 2ξ·C̄_inv²` one gets `C_u > 1/2` and `2 − 4C̄_inv²/c₁ > 1`, hence a coercivity constant that is **free of `C̄_inv` altogether**:
  ```
  C_stab  ≥  min{ 2(1 − 2/ξ),  1/2,  1 − C₂ }.
  ```
  Under the sharp condition alone the constants stay positive but `C_u → 0` as `c₁ ↓ ξ·C̄_inv²`, and the floor is lost.

**Why this matters for §3.** It explains the empirical fold exactly: sitting just above the *positivity* threshold is not enough — coercivity degrades continuously as `c₁` approaches it, so a mesh with a large `C̄_inv` (structured Kuhn tets) lands in the regime where `C_stab` is positive but tiny, which is indistinguishable in practice from loss of coercivity. The element-aware `c₁` remedy is what restores the margin, and `C_stab_margin` says how much margin a given `c₁` actually buys. Note also that the relevant constant is `C̄_inv` (the *weighted* inverse constant of `lem:winv`, `= √(d·δ_α)·C_inv + C_α`), strictly larger than the bare `C_inv` — `rem:winvconst`.

### Other settled results from the formalisation

- **σ = 0 is admissible.** The development assumes exactly `H:data` (`σ ≥ 0`, `ε ≥ 0`, `0 ≤ C₂ < 1`); the reaction-free (pure Navier–Stokes) limit is a genuine instance of the stability and convergence theorems.
- **The implemented τ₂ is covered.** `abstract_convergence_implemented`: `thm:convergence` holds for the τ₂ the solver actually forms (`eq:Tau2`, with its `ε h²` term) with the constant inflated by at most `√(1+C₂) < √2`.
- **Non-vacuity is machine-checked for three of the four abstract theorems**, each with a *non-degenerate* conclusion (strictly positive on both sides, correct inequality direction), each headline `Print Assumptions`-clean (the same three stdlib axioms, no new user axiom). Load-bearingness is established *inside the witness files* by kernel-checked adversarial lemmas — the `w_*_forced` / `w_*_sharp` / `w_*_live` / `w_atoms_nonzero` / `w_*_pos` families — each establishing that the witness datum it targets is *pinned or saturated* by the hypothesis (not free), so corrupting that datum breaks the lemma in the kernel; these are compiled Coq lemmas, not a separate runnable "mutation-test" harness (there is none). The concrete per-hypothesis instances are the bullets below (e.g. `du(A):=1/4` breaks `Horth`; `C_I:=0` breaks the witness).
  - **`abstract_stability`** — two witnesses (`NonVacuity.v`). Witness 1 is the Darcy corner `|a|_∞ = 0` (`7/8 ≥ 7/16`); since the refactor its `B_S = 7/8` is **computed** from the eighteen-term form, not posited. Witness 2 adds a **strictly positive advection field** (`am = 1`, two-element mesh, `5069/567 ≥ 167/42`), **closing G1**: with the witness data the `|a|`-carrying hypotheses `H_face_c`/`HI_cxu` are *false* at `am = 0` (they read `|2| ≤ ½·0`), so `am > 0` is necessary, not cosmetic. Witness 2 also makes `c₂` live (its value is numerically dead at `am = 0`) and turns a saturated internal velocity estimate strict.
  - **`abstract_continterp`** (`NonVacuityInterp.v`, 39 hypotheses) and **`abstract_convergence`** (`NonVacuityConv.v`, 44) — the **first witnesses ever** for these (**closing G2**), both with strictly positive advection (`am = (1,2)`, `am = 2`), a genuine two-element interior-face mesh (not `Empty_set`, not `Fl := []`), every vector-atom family nonzero on both elements, and a **concrete rational lower bound** on the right-hand side (`|B_S| = 72181/8400 ≤ 131 ≤ RHS` for continterp; `NErr ≤ C_conv·Ψ` with `Ψ, NErr > 0` for convergence) — not merely `RHS > 0`, which a reader could satisfy with `10⁻³⁰`.
  - **`CI_pos : 0 < C_I` is now witnessed** — discharged *strictly and jointly* with the other 38 / 43 hypotheses (`C_I = 1` in continterp, `C_I = 4` in convergence). This is the single hypothesis the 2026-07-17 refactor made strictly stronger (from `CI_nonneg : 0 ≤ C_I`), and it lived only in these two theorems, so until this run it had **no witness at all**; its joint satisfiability with the full bundle is now compiled, not merely argued. Mutating `C_I := 0` — which still satisfies the *old* `CI_nonneg` — breaks the witness, so the mutation targets the strengthening precisely.
  - **`abstract_continuity` still has no witness** — the one remaining non-vacuity gap.
  - **Still exercised only trivially (disclosed, and this is a real limit).** The skew-form diagonal identities — `H_skew_diag` (stability, convergence), `H_skew` / `H_ibp_vp` (continterp) — read arithmetically as `0 = ±0`. This is *forced*, not idle: on the diagonal an antisymmetric form must vanish, so `0 = ±0` is the *best attainable* standard, not a shortcut. The witnesses meet them with **nonzero cancelling summands** (e.g. `1·2 + 1·(−2) = 0`), and the identity is **load-bearing** — it *pins* the assembled face data (`FBp = FBc = 2` in continterp) that would otherwise escape all control and let the whole face bundle go vacuous. Stability witness 1 additionally satisfies three of its four data-hypotheses at the `0 = 0` / `0 ≤ 0` boundary (the deliberate Darcy corner); witness 2 carries the non-trivial certification for those. Minor caveats, all flagged in the file banners: stability witness 2's mesh is *uniform* (the atoms vary across elements, the mesh data does not), `Heps` holds with *equality* (the extremal admissible `ε`, the hardest choice for an upper bound, not a degenerate one), and the carrier is 1-D `R`.
- **The convergence witness's Galerkin orthogonality is a genuine cancellation, not the `E := −W` trap.** `Horth` — the crux of `thm:convergence` — equates two eighteen-term sums, `⟨L*W, L W⟩` and `⟨L*E, L W⟩`; the witness solves one atom (`du(A) = 3/16`) to force `Σ BS_WW = 3931/32 = −Σ BS_EW`. Such an equality can be faked by taking the error `E := −W`, which collapses the *bounded* quantity to zero and makes the whole convergence bound vacuously `0 ≤ 0`. That escape is provably excluded here: (i) `NErr = 91 > 0`, so the error is not the negated solution; (ii) `du` does not occur in the error norm `perErr` (which pairs `gu+gv, uu+vv, pp+qq, xe+xw, divu+divv` — never `du`), so the solved knob **cannot** distort the bounded quantity; (iii) no *pairwise* `BS_WW[i] = −BS_EW[i]` holds — the cancellation is global across all eighteen terms, and mutating `du(A)` to `1/4` breaks `Horth`. So the pivotal hypothesis of the a priori chain is discharged on nonzero, non-self-cancelling data.
- **Residual trust is two items.** Of the 50 hypotheses, 24 are data/mesh conditions (nothing to prove). Of the 26 analytic ones, 16 are instances of two textbook facts (inverse + Bramble–Hilbert estimates) and 7 are the divergence theorem plus face bookkeeping. The genuine residue is `Horth` and the face-estimate bundle — the Lean 4 targets ([`LEAN_ROADMAP.md`](../proof_verification/LEAN_ROADMAP.md)).

### The 2026-07-17 revision: two rows removed, and why the count is the weakest part of it

- **The tested identity is now proved, not assumed.** `AbstractStability.v` defines `B_S` as the *same* eighteen-term expression the other three files use, and **proves** the tested identity from two elementary diagonal Green identities (`H_skew_diag`, `H_ibp_diag`, items 28–29) — mirroring what `AbstractConvergence.v` already did for `HBS_W`. **But the row it removes was a phantom.** The old `Variable BS : R` + `Hypothesis HBS : BS = t` closes to `∀b, b = t ⇒ P(b)`, which is logically just `P(t)`: `HBS` was eliminable all along, so `53 → 50` counts one row that carried no logical content. What the change actually buys is the *reconciliation* — the number `AbstractStability.v` bounds from below is now the same closed expression the other three bound from above, checked by the kernel instead of by a reader comparing two displays — plus the difference-of-squares expansion `HBS` used to smuggle in, now machine-checked. The price is real and disclosed: **the stability lemma's own base goes up, 16 → 17**, and three of `HBS`'s four bundled pieces (including idempotence of `𝒫`) *moved* into the (I4) reading obligation rather than being discharged. The global count still falls only because items 28–29 were already assumed for the convergence theorem.
- **`IU_nonneg`/`IP_nonneg` are derivable.** `0 ≤ ‖uu k‖ ≤ C_I · IU k` transfers the sign as soon as `C_I > 0`, so both rows go — at the price of strengthening `CI_nonneg : 0 ≤ C_I` to `CI_pos : 0 < C_I`. A **trade, not a free lunch**: `{CI_pos}` is strictly stronger than `{CI_nonneg, IU_nonneg, IP_nonneg}` (compiled countermodel). It is credibility-neutral — `IU = h_K^{k+1}|u|_{H^{k+1}(K)} ≥ 0` holds by definition and `C_I > 0` for any real interpolation estimate — and nearly free, since `C_I` occurs only monotonically (in upper bounds and the output constant), so any old model survives at `C_I + 1` with a weaker constant. Since this run the strict `CI_pos` is no longer *only* argued: the `continterp`/`convergence` non-vacuity witnesses discharge it jointly with the full bundle at `C_I = 1` and `C_I = 4` (§ non-vacuity above).
- **Why the same trick fails for `am_nonneg`, `hK_pos`, `aK_pos`.** `Hw_cxu`'s right-hand side carries a factor `‖uu k‖` that vanishes whenever `u_h ≡ 0` on an element, collapsing the bound to `0 ≤ 0` and telling you nothing about the sign. `HI_uu`'s does not — `C_I · IU k` has no vanishing factor. `hK`/`aK` occur only on the right of upper bounds with nothing bounding them below. All three are genuinely primitive (compiled countermodels). Norm axioms cannot reach them either: `am`, `hK`, `aK` are abstract `K -> R` fields, not norms of anything — the development deliberately has no geometry.

### A defect it found in the manuscript (amendment F8)

In `lem:winv` the label `eq:winv-conv` sat on the **last** line of the display (the pressure-gradient estimate), while the **convective** line above it had none — so two of the four references to `eq:winv-conv` pointed at the *wrong estimate* (Step 5's "first contribution" and Step 9's "velocity part" are both convective). Fixed: the convective line now carries `eq:winv-conv`, the pressure-gradient line the new `eq:winv-gradp`, and the call sites are re-pointed. Surfaced only because the Coq audit had to cite the two estimates separately (`Hw_cxu`/`Hw_cxv` vs `Hw_gpu`). Recorded in [`proof_verification/AUDIT.md`](../proof_verification/AUDIT.md) F8.

---

## Cross-references

- **Full evidence dossiers:** [`docs/mms/p2-3d.md`](mms/p2-3d.md), [`docs/cocquet/investigation-synthesis.md`](cocquet/investigation-synthesis.md), [`docs/formulation-audit-2026-06-24.md`](formulation-audit-2026-06-24.md).
- **Theory (LaTeX):** [`theory/paper/article.tex`](../theory/paper/article.tex) (the authoritative formulation), `theory/osgs_reaction_note/osgs_reaction_note.tex`, `theory/tau_saturation_note/tau_saturation_note.tex`, `theory/osgs_algorithm/osgs_algorithm.tex`.
- **Machine-checked theory:** [`proof_verification/coq_coverage.tex`](../proof_verification/coq_coverage.tex) (theory→Coq map + the 50-row hypothesis inventory), [`proof_verification/AUDIT.md`](../proof_verification/AUDIT.md) (hand audit + amendments F1–F8), [`proof_verification/LEAN_ROADMAP.md`](../proof_verification/LEAN_ROADMAP.md).
- **Living companions:** [`pending-tasks.md`](pending-tasks.md) (backlog + open code-correctness items), [`open-questions.md`](open-questions.md) (open questions), [`lessons_learned.md`](lessons_learned.md) (regression ledger), [`theory-code-map.md`](theory-code-map.md) (paper↔code map + divergence ledger + convergence-gate spec).
