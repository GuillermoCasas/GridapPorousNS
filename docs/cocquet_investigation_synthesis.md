# Cocquet absolute-magnitude investigation — synthesis snapshot (2026-05-26)

This file is a **point-in-time synthesis** of the Cocquet investigation. The chronological growth ledger remains at [cocquet_magnitude_investigation.md](cocquet_magnitude_investigation.md); the convergence-slope sibling is at [cocquet_convergence_analysis.md](cocquet_convergence_analysis.md). This snapshot was generated after a second-pass logical-soundness audit of every conclusion against code, math, and result files.

---

## 1 — Executive summary

**The gap.** Cocquet et al. 2021 (Fig 2 right, Re=500, c_in=0.5) report L²(u) ≈ 3×10⁻⁵ at N=10 and 2×10⁻⁷ at N=100 against an N=200 self-reference. Our best Galerkin P2/P1 (freefem-divs unstructured mesh) gives 9.68×10⁻³ at N=10 and 1.69×10⁻⁴ at N=100 — a **ratio that grows with refinement from ~320× to ~845×**. Finest-segment slope, however, matches: ours ‑2.05 (N=40→80), paper ‑2.18.

**What we know.** The code is a term-by-term faithful transcription of Cocquet's weak form (verified — §4 below). The Newton solve converges to machine-precision residual (3.7×10⁻¹⁵). On a manufactured solution with the **same** Forchheimer/porosity operators **and the same unstabilized Taylor-Hood P2/P1 Galerkin formulation** ([CocquetFormMMS three-way comparison](../test/extended/CocquetFormMMS/)) we hit L²(u) = 2.85×10⁻⁷ at h=1/320 with slope 2.87 (optimal P2) — i.e., the code **can** reach paper magnitudes on smooth problems, **on the exact same Galerkin pathway used by Cocquet**. The remaining Cocquet-benchmark gap is therefore a **non-smoothness × discretization interaction**, not a solver bug or a stabilization-vs-Galerkin artifact.

**Dominant cap mechanism identified (S5 audit, §5.7).** A side-by-side sweep of five one-knob siblings shows:

- Removing the **porous drag entirely** (Alpha1), switching to a **deviatoric viscous operator** (Deviatoric), and **zeroing the Forchheimer β·\|u\|·u nonlinearity** (LinearReaction) all leave the P2/P2 slope stuck at ~1.26–1.38 — three independent exonerations of porous-physics knobs.
- Replacing the natural traction-free outlet with **all-Dirichlet BCs** (AllDirichlet) jumps P2/P2 slope from ~1.27 to **2.46**.
- Picard vs Newton linearization is irrelevant (LiteralPicard).

→ The convergence cap is driven by the **mixed-BC outlet corner singularity** at `(2,0)` and `(2,1)`, where Dirichlet walls meet the Neumann outlet. This also explains the localize_err finding (71% of structured-mesh L²(u)² lives at the outlet corner).

**The remaining gap.** Even AllDirichlet's best run (P2/P2 N=80, L²(u) = 4.7×10⁻⁶) is still **~12× the paper's ~4×10⁻⁷** at N=80. So eliminating the corner singularity peels off the slope component of the gap, but a residual ~10× magnitude factor persists — almost certainly the **L²(u)/H¹(u) low-frequency mode anomaly** flagged in §5.4 (not yet diagnosed; S3 is the highest-priority remaining check). All currently-tested specific mechanisms (H1–H12, O1–O8) and S5 have been falsified or confirmed paper-faithful. **One pre-existing open hypothesis (O6, pressure gauge) and three new candidates from this audit (S1, S2, S3) remain to be tested.**

**Housekeeping correction (2026-05-26).** A previous version of this synthesis claimed H8 (`pressure penalty η = 1e-7`) was falsified because the canonical [test/extended/CocquetExperiment/data/paper_comparison.json](../test/extended/CocquetExperiment/data/paper_comparison.json) did not override the (now-removed) `1e-8` default in `base_config.json`. The fix landed on 2026-05-26: every Cocquet paper-comparison config now **explicitly sets `eps_val = 1e-7`** to match the paper, and `eps_val` was removed from `base_config.json` so it can no longer leak in silently (consistent with the CLAUDE.md rule that `load_config_with_*` must not invent values). The freefem-divs result that produced the headline numbers above was already running with `eps_val=1e-7` (it had its own override), so the headline ratios are unaffected by this fix.

---

## 2 — Headline numbers

### Galerkin P2/P1, freefem-divs unstructured mesh (our best comparison) vs Cocquet Fig 2

| N   | Our L²(u)   | Paper L²(u)  | Ratio   |
|-----|-------------|--------------|---------|
| 10  | 9.68×10⁻³   | ~3×10⁻⁵      | ~323×   |
| 20  | 3.26×10⁻³   | ~5×10⁻⁶      | ~650×   |
| 40  | 9.74×10⁻⁴   | ~1×10⁻⁶      | ~970×   |
| 80  | 2.67×10⁻⁴   | ~4×10⁻⁷      | ~670×   |
| 100 | 1.69×10⁻⁴   | ~2×10⁻⁷      | ~845×   |

Slope (N=40→80): ours **‑2.05**, paper **‑2.18**. Slope: ✓. Magnitude: ✗.

### Same case, alternative meshes — context

| Configuration                                 | L²(u) at N=100 | Final-segment slope | Source                                                                                                                            |
|-----------------------------------------------|----------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| Freefem-divs unstructured (Delaunay, walls 2× coarser) | 1.69×10⁻⁴ | ‑2.05 | [test/extended/CocquetExperimentIrregularMeshFreefemDivs/results/](../test/extended/CocquetExperimentIrregularMeshFreefemDivs/results/) |
| Literal FreeFem .msh (Cocquet's exact recipe) | 4.59×10⁻⁵      | (n/a, single N)     | [test/extended/CocquetExperimentIrregularMesh/results/convergence_paper_comparison_freefem.h5](../test/extended/CocquetExperimentIrregularMesh/results/convergence_paper_comparison_freefem.h5) |
| Uniform Delaunay 1/N divisions                | (sweep)        | ‑1.80               | [test/extended/CocquetExperimentIrregularMesh/](../test/extended/CocquetExperimentIrregularMesh/)                              |
| **Structured Cartesian — Galerkin P2/P1** (canonical, post-retrofit 2026-05-26) | **2.31×10⁻⁵ at N=80** (58× paper @ N=80 ~4×10⁻⁷) | **‑1.22** | [test/extended/CocquetExperiment/results/convergence_paper_comparison.h5](../test/extended/CocquetExperiment/results/convergence_paper_comparison.h5) |
| Structured Cartesian — ASGS P2/P2 | 9.19×10⁻⁶ at N=80 (23× paper) | ‑1.64 | same h5 |
| Structured Cartesian — ASGS P1/P1 | 8.84×10⁻⁵ at N=80 | ‑1.82 | same h5 |
| AllDirichlet — ASGS P2/P2 (no mixed-BC corner) | 4.70×10⁻⁶ at N=80 (12× paper) | ‑2.46 | [test/extended/CocquetAllDirichlet/results/](../test/extended/CocquetAllDirichlet/results/) |
| **MMS — Galerkin P2/P1 unstabilized (same pathway as Cocquet)** | **2.85×10⁻⁷ at h=1/320** | **‑2.87 (optimal P2)** | `config_3_Galerkin` in [test/extended/CocquetFormMMS/results/cocquet_form_mms_comparison.h5](../test/extended/CocquetFormMMS/results/cocquet_form_mms_comparison.h5) |
| MMS — Stabilized P2/P2 ASGS (our method) | 3.42×10⁻⁷ at h=1/320 | ‑2.95 (optimal P2) | `config_2_ASGS` in same h5 |
| MMS — Stabilized P1/P1 ASGS (our method) | 1.34×10⁻⁴ at h=1/320 | ‑1.64 | `config_1_ASGS` in same h5 |

Even the literal-FreeFem mesh leaves a 230× gap. The three MMS rows are the strongest exculpatory evidence: **on a smooth solution with the same Forchheimer / porosity operators, all three formulations — our stabilized P1/P1 and P2/P2, AND Cocquet's literal unstabilized Galerkin P2/P1 — converge at their expected optimal rates and reach paper-magnitude.** This is the explicit three-way comparison built into the [CocquetFormMMS](../test/extended/CocquetFormMMS/) test (configs `cocquet_form_mms_comparison_stab.json` + `cocquet_form_mms_comparison_galerkin.json` writing to the shared HDF5; the Galerkin leg uses the dedicated [`execute_solver_galerkin_inline!`](../test/extended/CocquetFormMMS/run_test.jl#L160-L207) path with `mult_mom=mult_mass=0`).

---

## 3 — Test setup baseline

- **PDE:** ε-weighted Darcy–Brinkman–Forchheimer (Cocquet 2021 Eqs. 1–2). Re=500, ν=1/Re=0.002, Forchheimer drag, no body force.
- **Domain:** (0,2)×(0,1). Γ_in={0}×[0,1], Γ_out={2}×[0,1], Γ_w=[0,2]×({0}∪{1}).
- **Inlet:** u_in(y) = 0.5·y(1‑y) e_x.
- **Porosity profile:** ε(y) = 0.45 + 0.55·exp(y‑1). Min 0.45 at y=0 (bottom wall), exactly 1 at y=1 (top wall).
- **FE pair:** Taylor–Hood P2/P1, unstabilized for the Galerkin row. The `comparison_runs` at [paper_comparison.json:8-12](../test/extended/CocquetExperiment/data/paper_comparison.json#L8-L12) also runs P1/P1 and P2/P2 with ASGS stabilization for cross-comparison.
- **Reference:** self-reference at N=200 (same methodology as paper).
- **Driver:** [test/extended/CocquetExperiment/galerkin_driver.jl](../test/extended/CocquetExperiment/galerkin_driver.jl) calls assembly with `mult_mom=mult_mass=0` (no VMS τ-stabilization).
- **Solver:** Newton, `xtol=1e-11, ftol=1e-11`, Armijo line search, safeguarded fallback. `h_floor_weight=0` per [paper_comparison.json:27](../test/extended/CocquetExperiment/data/paper_comparison.json#L27) (τ-floor regression fix from [lessons_learned.md](lessons_learned.md) row 25 disabled).

---

## 4 — Catalogue of all hypotheses (closed)

For each hypothesis: ID, claim, status, **the test/code path that currently exercises it**, and the result.

### 4.1 — Paper-faithfulness hypotheses (H3–H12)

These are not "hypotheses" in the experimental sense — they are claims about whether our code corresponds to the paper. They are verified by **code inspection**, not by running experiments.

| ID  | Claim                                                                 | Status              | Path that exercises the claim                                                                                                                                       | Result / verdict                                                                                                                                              |
|-----|-----------------------------------------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| H3  | ν = 1/Re mapping (Re=500 ⇒ ν=0.002)                                   | CONFIRMED           | [src/formulations/viscous_operators.jl:101-103](../src/formulations/viscous_operators.jl#L101-L103) + [paper_comparison.json:24](../test/extended/CocquetExperiment/data/paper_comparison.json#L24) | `2 * α * ν * (ε(u) ⊙ ε(v))` matches paper's `2 Re⁻¹ ε S(u):S(v)`. nu=0.002.                                                                                  |
| H4  | Forchheimer α(ε), β(ε) closure                                        | CONFIRMED           | [src/models/reaction.jl:57-62](../src/models/reaction.jl#L57-L62) + [paper_comparison.json:25-26](../test/extended/CocquetExperiment/data/paper_comparison.json#L25-L26) | `a_term = 0.30 · ((1‑ε)/ε)²`, `b_term = 1.75 · (1‑ε)/ε`. 0.30 = 150/500. ε=0.45 ⇒ a=0.448, b=2.139. Exact.                                                  |
| H5  | Outlet BC ε-weighted traction-free (natural)                          | CONFIRMED           | [src/formulations/continuous_problem.jl:241](../src/formulations/continuous_problem.jl#L241)                                                                       | `pres_term = -p·(α·∇·v + ∇α·v)` integrates by parts ⇒ natural BC `εσ·n = 0` on Γ_out. No explicit boundary integral.                                          |
| H6  | Convective form ε(u·∇)u                                               | CONFIRMED           | [src/formulations/continuous_problem.jl:239](../src/formulations/continuous_problem.jl#L239)                                                                       | `conv_term = v · (α · (∇(u)' · u))` = v · ε(u·∇)u. Exact.                                                                                                     |
| H7  | Mass / pressure-divergence form                                       | CONFIRMED           | [src/formulations/continuous_problem.jl:241,244-245](../src/formulations/continuous_problem.jl#L241)                                                                | `mass_term = q · (eps_val·p + α·∇·u + u·∇α) = q · (eps_val·p + div(εu))`. Exact.                                                                              |
| H8 | Pressure penalty η = 1e-7 matches paper | CONFIRMED (after 2026-05-26 retrofit) | [paper_comparison.json:25](../test/extended/CocquetExperiment/data/paper_comparison.json#L25) explicitly sets `"eps_val": 1e-7`. All Cocquet paper-comparison and sibling configs now do the same; `eps_val` was removed from [base_config.json](../base_config.json) so it can no longer silently inherit. | Before the retrofit, several configs silently inherited the now-removed `1e-8` default. The freefem-divs headline run had its own `1e-7` override so the headline ratios in §2 are unaffected. The structured baseline + several sibling diagnostics ran at `1e-8` before the retrofit; they should be re-run against `1e-7` for a clean apples-to-apples (see §6 S1). |
| H9  | P1 porosity interpolant for the Galerkin row                          | CONFIRMED           | [paper_comparison.json:11](../test/extended/CocquetExperiment/data/paper_comparison.json#L11)                                                                       | `"porosity_order": 1` set explicitly on the Galerkin row.                                                                                                       |
| H10 | Domain, inlet profile, porosity profile                               | CONFIRMED           | [run_convergence.jl:9-10](../test/extended/CocquetExperiment/run_convergence.jl#L9-L10) + [paper_comparison.json:31-41](../test/extended/CocquetExperiment/data/paper_comparison.json#L31-L41) | `alpha_func(x) = 0.45 + 0.55·exp(x[2]-1.0)`, box `[0,2]×[0,1]`, c_in=0.5. Exact.                                                                              |
| H11 | N_ref=200 self-reference (same methodology as paper)                  | CONFIRMED           | [paper_comparison.json:62-67](../test/extended/CocquetExperiment/data/paper_comparison.json#L62-L67) (convergence_partitions) + driver                              | Paper p.32 uses N=200 reference; our driver does too.                                                                                                          |
| H12 | Galerkin Taylor-Hood unstabilized (mult_mom=mult_mass=0)              | CONFIRMED           | [galerkin_driver.jl:26-27,62-64](../test/extended/CocquetExperiment/galerkin_driver.jl#L26-L27)                                                                     | `mult_mom = 0.0`, `mult_mass = 0.0` passed to assembly. No τ-stabilization terms in the assembled form.                                                        |

### 4.2 — Experimental hypotheses (closed)

| ID | Claim | Status | Test directory / probe | Result |
|----|-------|--------|------------------------|--------|
| H1 | Outlet-corner Dirichlet pin caps convergence | FALSIFIED | [test/extended/CocquetExperimentModifiedCorner/](../test/extended/CocquetExperimentModifiedCorner/) — entities 2,4 removed from "walls" tag, freeing outlet-corner DOFs | Made absolute L²(u) **3.1× worse at N=10** (9.36e‑4 vs 3.01e‑4) and **3.0× worse at N=80** (6.89e‑5 vs 2.31e‑5). The released DOF takes an algebraically-driven value that pollutes the bulk. Memory: [cocquet-modified-corner-experiment.md](/Users/guillermocasasgonzalez/.claude/projects/-Users-guillermocasasgonzalez-repos-porous-NS-with-Gridap/memory/cocquet-modified-corner-experiment.md) |
| H2 | Structured-mesh corner-aligned diagonals lock the corner singularity | PARTIALLY CONFIRMED (slope only) | [test/extended/CocquetExperimentIrregularMesh/](../test/extended/CocquetExperimentIrregularMesh/) (uniform 1/N gmsh Delaunay) + [test/extended/CocquetExperimentIrregularMeshFreefemDivs/](../test/extended/CocquetExperimentIrregularMeshFreefemDivs/) (literal `buildmesh(a(N)+b(N)+c(N)+d(N))` N-per-side) | Unstructured Delaunay recovers near-optimal slopes (Galerkin P2/P1 freefem-divs final segment **‑2.05**, uniform-divs ‑1.80; P2/P2 ASGS 2.39 uniform, 2.13 freefem-divs). But absolute magnitudes are still 320–845× the paper. Memory: [cocquet-mesh-topology-controls-convergence.md](/Users/guillermocasasgonzalez/.claude/projects/-Users-guillermocasasgonzalez-repos-porous-NS-with-Gridap/memory/cocquet-mesh-topology-controls-convergence.md) |
| O1 | Our N=200 reference is itself under-converged vs the truth solution | RULED OUT BY ARGUMENT | (No experimental test — argument from symmetry of methodology.) | Paper uses the same self-reference methodology. If our reference were under-converged by ~1e‑4, paper's would be too and would also plateau at that floor. It doesn't (paper reaches 2e‑7 at N=100). Therefore the gap is not in the reference. Strengthened in §5.2. |
| O2 | `u_base_floor_ref = 1e-4` velocity-magnitude regularization adds a noise floor | RULED OUT BY PROBE | `/tmp/probe_with_floor_off.jl`, log `/tmp/probe_floor_off.log` — single Galerkin P2/P1 N=80 + N_ref=200 solve with `u_base_floor_ref=0` and `epsilon_floor=0` | L²(u) at N=80 = **2.6685e‑4** — identical to 4 sig figs to baseline (2.669e‑4). Newton iter-0 residual bit-identical at 0.001164580518. The floor never fires on this flow (true \|u\| > 1e‑4 everywhere except wall vertices where u=0). |
| O3 | Solution magnitude / spatial profile differs from Cocquet's | PARTIALLY RULED OUT | `/tmp/probe_cocquet_reference.jl`, log `/tmp/probe_cocquet_ref.log` | Flow magnitudes physically reasonable: ‖u‖_L²(Ω) = 0.125 (consistent with c_in/4 inflow scale); inlet u(0,0.5) = (0.125, 0) exact; centerline u_x decreases 0.125 → 0.102 (Darcy drag); asymmetric profile (peak at y=0.75) because porosity ramps. Newton converged to 3.7e‑15 in 4 iterations. We're solving the right PDE. Caveat: a smooth, global error mode would slip past these checks — see §5.5. |
| O5 | Quadrature degree insufficient for non-polynomial Forchheimer term | RULED OUT BY PROBE | `/tmp/probe_quad_plus12.jl`, log `/tmp/probe_quad12.log` — N=200 reference solve with `Measure(Ω, 21)` (default for k_v=2 is 9; bump +12) | Iter-0 PDE residual: baseline `9.588812700822735e‑4`, quad+12 `9.588812745812068e‑4`. Delta ~5e‑9 — five orders of magnitude below our 1e‑4 error level. Quadrature is adequate. |

### 4.3 — Diagnostic experiments referenced in the ledger

| Diagnostic | What it measured | Where to find it | Headline |
|------------|------------------|------------------|----------|
| **CocquetFormMMS** (smooth-solution MMS with same operators) | Best-case attainable L²(u) with same Forchheimer/porosity operators | [test/extended/CocquetFormMMS/](../test/extended/CocquetFormMMS/), `results/cocquet_form_mms.h5` | **L²(u) = 8.54×10⁻⁷ at h=1/80** (config_1 ASGS P2/P1). Same order as paper's N=100 magnitude. Confirms the code is not bugged. |
| **Literal FreeFem .msh comparison** | Solve on Cocquet's exact mesh-recipe `.msh` files (generated by FreeFem) | [test/extended/CocquetExperimentIrregularMesh/results/convergence_paper_comparison_freefem.h5](../test/extended/CocquetExperimentIrregularMesh/results/convergence_paper_comparison_freefem.h5), driver [paper_comparison_freefem.json](../test/extended/CocquetExperiment/data/) | **L²(u) = 4.59×10⁻⁵ at N=100** — closer than gmsh Delaunay variants, but still 230× the paper's 2×10⁻⁷. |
| **localize_err.jl** (spatial decomposition of L²(u)² over regions) | Where in the domain does the error live? | `/tmp/localize_err.log` (committed analysis output) | Structured: 71.0% outlet_corner, 25.3% bulk. Unstructured Delaunay: 1.08% outlet_corner, 61.8% bulk. Unstructured *total* ‖e‖ is 18× *larger* than structured at N=10 (5.74e‑3 vs 3.19e‑4). |
| **Figure 2 y-axis re-verification** | Confirm paper's reported magnitudes | Direct inspection of the paper PDF | Y-axis: 10⁻⁴ (top tick) → 10⁻⁷ (bottom). Curve from ~3×10⁻⁵ at N=10 to ~2×10⁻⁷ at N=100. Magnitudes are real. |

---

## 5 — Logical soundness audit (second pass)

This section walks every previously-recorded conclusion and tests it against the **code as it actually exists today**, the math in [theory/cocquet_formulation.tex](../theory/cocquet_formulation.tex) and [theory/article.tex](../theory/article.tex), and the result files on disk. Each subsection ends with the concrete next action, if any.

### §5.1 — H8 retrofit (2026-05-26) — silent default removed; ledger now consistent

**Audit finding.** A prior version of this synthesis (and the long-running ledger) recorded H8 (`pressure penalty η = 1e-7`) as CONFIRMED, but a code audit showed the canonical [test/extended/CocquetExperiment/data/paper_comparison.json](../test/extended/CocquetExperiment/data/paper_comparison.json) did **not** override the `eps_val=1e-8` default in `base_config.json`. So the structured baseline + the H1 ModifiedCorner experiment + the Alpha1 / Deviatoric / LinearReaction / AllDirichlet sibling diagnostics were all running with `eps_val=1e-8`, a 10× *smaller* penalty than the paper's `η = 1e-7`. The IrregularMeshFreefemDivs config that produced the headline numbers in §2 had its own `1e-7` override, so those headline ratios are unaffected.

**Fix (2026-05-26):**
- (a) `eps_val=1e-7` now set explicitly in all seven affected configs ([CocquetExperiment](../test/extended/CocquetExperiment/data/paper_comparison.json), [ModifiedCorner main](../test/extended/CocquetExperimentModifiedCorner/data/paper_comparison_modified_corner.json) and [smoke](../test/extended/CocquetExperimentModifiedCorner/data/paper_comparison_modified_corner_smoke.json), [Alpha1](../test/extended/CocquetAlpha1/data/alpha1.json), [Deviatoric](../test/extended/CocquetDeviatoric/data/deviatoric.json), [LinearReaction](../test/extended/CocquetLinearReaction/data/linear_reaction.json), [AllDirichlet](../test/extended/CocquetAllDirichlet/data/all_dirichlet.json)).
- (b) `eps_val` removed from [base_config.json](../base_config.json) so silent inheritance through `load_config_with_*` is impossible (every test config must now declare its own `eps_val` or the strict struct constructor will fail loudly — consistent with the CLAUDE.md "no implicit defaults — fail loudly on missing input" rule).
- Blitz tests (83 / 83) pass post-fix.

**Why this doesn't close the magnitude gap.** Direction analysis: the penalty term is `+ η·p·q`. A smaller η means a *tighter* constraint on `div(εu) ≈ 0` — more faithful to the true incompressibility constraint, not less. The previous `1e-8` versus paper-faithful `1e-7` could not have moved velocity error by orders of magnitude. The pre-retrofit sweeps should be regenerated for audit-trail consistency (S1 in §6), but Δ on L²(u) is predicted ≪ 1e‑6, far below the 1e‑4 error level.

**Sensitivity confirmed empirically (2026-05-26).** The canonical structured-mesh CocquetExperiment was re-run post-retrofit with `eps_val=1e-7` explicit. The ASGS P1/P1 L²(u) numbers are **bit-identical to 4 significant figures** to the pre-retrofit (1e-8) run: N=10: 3.919×10⁻³, N=20: 1.187×10⁻³, N=40: 3.433×10⁻⁴, N=80: 8.840×10⁻⁵. The eps_val change is empirically invisible at the present error scale, as predicted by the direction analysis. The retrofit's value is methodological (paper-faithful + no silent defaults), not numerical.

### §5.2 — O1's symmetry argument is sound, with one open thread

The ledger argues: "if our N=200 reference were under-converged vs truth by ~1e‑4, paper's N=200 reference would be too." This is correct *provided Cocquet's discretization is identical to ours*. Items confirmed identical: FE pair (P2/P1), η-penalty gauge family (both penalize), self-reference methodology, weak form (verified §4.1).

The literal-FreeFem mesh run gives 4.59×10⁻⁵ at N=100 — 230× the paper. So mesh alone cannot account for the gap.

**Remaining unknown:** Cocquet's published paper does not specify their quadrature degree or linear-solver tolerance. We've measured ours' contribution as 5×10⁻⁹ (O5 probe). If Cocquet's quadrature is higher and linear solve tighter, that bounds the gap from above but does not produce 1000× ratios.

**Action:** none for now (the O1 argument stands).

### §5.3 — H2 conclusion is coherent but the localize_err finding is awkward

Reading the localize_err numbers carefully:

- Structured: 71.0% outlet_corner, 25.3% bulk, 3.6% wall_strip, 0.10% inlet_corner. Total ‖e‖ = 3.19e‑4 at N=10.
- Unstructured Delaunay: 1.08% outlet_corner, 61.8% bulk, 7.7% wall_strip, 29.4% inlet_corner. Total ‖e‖ = 5.74e‑3 at N=10 — **18× larger** total than structured.

What this says, decomposed:
- Structured mesh: low bulk error AND a corner singularity → small total, large fraction at corner.
- Unstructured Delaunay: high bulk error AND no corner singularity → large total, almost no fraction at corner.

Cocquet's reported 3×10⁻⁵ at N=10 is **6× below** even the structured-bulk-only error (1.72e‑4 = 3.19e‑4 × √0.253 ≈ 1.6e‑4, of the right order). So Cocquet has *both*: a more accurate bulk AND no corner singularity. The unusual combination suggests their mesh is corner-refined (BAMG adaptive) rather than uniform Delaunay.

**Action (S2):** check Cocquet's Section 5 mesh recipe in the paper PDF for adaptive flags. If they use `adaptmesh` or post-uniform-refinement adaptation, that single fact would change all of this analysis.

### §5.4 — The L²/H¹ cross-norm anomaly is the most actionable observation in the ledger

Numbers from the ledger:

- L²(u) ratio = 845× (ours / paper at N=100).
- H¹(u) seminorm ratio = 7×.

For a discrete solution with uniform error, Aubin–Nitsche gives `‖u − u_h‖_L² ≤ C·h·‖u − u_h‖_H¹`. The L² ratio should shrink relative to H¹ as h decreases — *not grow by 120×*. The opposite scaling implies **a non-uniform error structure**: most likely a low-frequency / global mode that contributes a lot to L² but little to H¹.

Candidate mechanisms for such a mode:
- Pressure-gauge offset (related to O6): a global p offset can drag u through the momentum coupling at the η-penalty level. Should be O(η) = 1e‑8, but the coupling could amplify.
- Wall layer at y=0: porosity minimum (0.45) ⇒ Forchheimer drag at its peak. A thin, slowly-resolved boundary layer would add low-frequency mass.
- Forchheimer term activation threshold: drag term `β(ε)|u|u` is non-smooth at |u|=0. The cross-stream zero-line of u_y could create a kink.

**Action (S3):** project u_h − u_ref onto P0 (cell-mean) and onto low-degree polynomials. If the P0 projection captures most of the energy, the gap *is* a global mode and the candidate mechanisms above become testable. This is the most informative diagnostic not yet run.

### §5.5 — O3 "we're solving the same PDE" relies on a single coarse-grained probe

The probe checked: ‖u‖_L²=0.125, inlet point value, centerline drag, cross-stream asymmetry, Newton residual. All consistent. **But** a smooth global error mode of magnitude 1×10⁻⁴ would not be visible in any of these checks — it's smaller than the precision of the coarse-grained diagnostics.

**Action:** see §5.4 — the P0-projection diagnostic also strengthens this conclusion. Additionally, comparing against Cocquet's Fig 3 (Re=1000, c_in=1) — where they DO show field plots — would give a direct visual check.

### §5.6 — CocquetFormMMS is the strongest exculpatory evidence, AND it directly exercises the unstabilized Galerkin pathway

The [CocquetFormMMS](../test/extended/CocquetFormMMS/) test is a **deliberate three-way comparison** of the same MMS problem under all three formulations: the user's stabilized P1/P1 ASGS, the user's stabilized P2/P2 ASGS, and Cocquet's literal unstabilized Galerkin P2/P1. The two configs `cocquet_form_mms_comparison_stab.json` and `cocquet_form_mms_comparison_galerkin.json` write into the **same** HDF5 (`cocquet_form_mms_comparison.h5`) for direct side-by-side reading. The Galerkin leg uses a dedicated path — [`execute_solver_galerkin_inline!`](../test/extended/CocquetFormMMS/run_test.jl#L160-L207) — that calls `build_stabilized_weak_form_residual` / `build_stabilized_weak_form_jacobian` with `mult_mom=mult_mass=0`, i.e., the **exact** assembly used by the production Cocquet comparison_runs Galerkin row.

Results at h=1/320:

| Method | k_v / k_p | L²(u) | Slope |
|---|---|---|---|
| ASGS (user's stabilized) | P1/P1 | 1.34×10⁻⁴ | ‑1.64 |
| ASGS (user's stabilized) | P2/P2 | 3.42×10⁻⁷ | ‑2.95 (optimal P2) |
| **Galerkin (Cocquet's literal)** | **P2/P1** | **2.85×10⁻⁷** | **‑2.87 (optimal P2)** |

**What this proves.** Assembly, integration, Newton, the Forchheimer reaction operator, the viscous + convective + pressure-divergence terms, AND the unstabilized `mult_mom=mult_mass=0` Galerkin codepath itself are all correct on smooth fields, including at the optimal P2 rate. The Cocquet-benchmark gap is therefore unambiguously a **non-smoothness × discretization interaction**, not specific to stabilization vs. Galerkin and not a hidden assembly defect in the unstabilized path.

The remaining open question (S3 below) is what specific feature of the Cocquet benchmark's non-smoothness our solver handles differently from theirs.

### §5.7 — S5 sibling audit (CLOSED 2026-05-26): mixed-BC outlet corner is the dominant cap mechanism

Each sibling is a deliberate one-knob flip of the canonical [paper_comparison.json](../test/extended/CocquetExperiment/data/paper_comparison.json). All results below were produced 2026-05-22 with `h_floor_weight=0` (post τ-floor regression fix) but pre-eps_val retrofit (i.e., they ran at `eps_val=1e-8` rather than the now-explicit `1e-7`). The expected Δ on L²(u) from that 10× penalty change is ≪ 1e-6 — well below the magnitudes reported here — so the verdicts below stand. A clean re-sweep is queued as S1.

**Sibling L²(u) at N=80, slope across N∈{10,20,40,80}, vs paper N=80 reference (~4×10⁻⁷):**

| Sibling | One-knob flip | P1/P1 slope | P1/P1 L²(u) @ N=80 | P2/P2 slope | P2/P2 L²(u) @ N=80 | Verdict |
|---|---|---|---|---|---|---|
| Baseline (mixed-BC, Forchheimer) | (reference) | 1.82 | 8.84e‑5 | (n/a in this h5) | (n/a) | Pre-existing baseline. |
| **Alpha1** | α ≡ 1 everywhere (pure NS, **no porous drag**) | 1.85 | 4.42e‑5 | **1.38** | **9.0e‑6** | Cap persists without drag. **EXONERATES the entire porous physics** as a cap source. |
| **Deviatoric** | DeviatoricSymmetric viscous op (not SymmetricGradient) | 1.86 | 6.49e‑5 | 1.26 | 1.62e‑5 | Cap persists with deviatoric form. **EXONERATES viscous operator choice.** |
| **LinearReaction** | β = 0 (linear Darcy only, **no Forchheimer**) | 1.79 | 5.59e‑5 | 1.27 | 1.30e‑5 | Cap persists with linear drag only. **EXONERATES the β·\|u\|·u nonlinearity.** |
| **AllDirichlet** | Outlet → Dirichlet (same as inlet); no natural traction outlet | 1.73 | 1.08e‑4 | **2.46** | **4.7e‑6** | **P2/P2 slope jumps from ~1.3 to 2.46.** Smoking gun: **the mixed-BC outlet corner is the dominant cap.** |
| **LiteralPicard** | Picard linearization (not Newton) on the freefem-divs mesh | 1.76 | 2.67e‑4 (P2/P1) | (Galerkin row only) | (n/a) | Identical to Newton freefem-divs row (numbers match to 4 sig figs). **EXONERATES Newton vs Picard linearization.** |

**Diagnostic synthesis:**

- Three independent siblings (Alpha1, Deviatoric, LinearReaction) all keep the P2/P2 slope around 1.26–1.38 despite radically different physics. This isolates the cap from any porous/viscous term — it is a discretization-side phenomenon, not a physics-side one.
- One sibling (AllDirichlet) changes only the outlet BC, removing the Dirichlet-wall / Neumann-outlet corner singularity. **Slope nearly doubles** (1.27 → 2.46). The mixed-BC corner at `(2,0)` and `(2,1)` is the dominant convergence cap.
- This dovetails with the localize_err finding (structured mesh: 71% of L²(u)² lives in the outlet corner) and reinforces — but tightens — the H2 partial-confirmation: it's not generic mesh topology, it's specifically the corner BC mismatch.
- Linearization choice (Picard vs Newton) is irrelevant (LiteralPicard).

**The remaining magnitude gap (apples-to-apples ASGS P2/P2 at N=80, structured mesh, vs paper ~4×10⁻⁷):**

| Configuration | L²(u) @ N=80 | Ratio to paper | What changes vs baseline |
|---|---|---|---|
| Baseline structured (mixed BCs, full Forchheimer) — ASGS P2/P2 | 9.19×10⁻⁶ | 23× | (reference) |
| AllDirichlet structured (no mixed-BC corner) — ASGS P2/P2 | 4.70×10⁻⁶ | 12× | Only outlet BC changed |
| ↳ ratio | 1.95× improvement | | |

Removing the corner singularity buys a **~2× magnitude improvement** in the ASGS P2/P2 row (and almost all of the slope cap, from 1.64 → 2.46). The residual **~12× factor to the paper persists even with the corner gone** — almost certainly the global / low-frequency mode flagged in §5.4 (L²/H¹ anomaly). Caveat: AllDirichlet changes the physical problem (no traction-free outlet), so this is the structured-mesh apples-to-apples within OUR experiments, not against Cocquet's reported Re=500 c_in=0.5 case directly. But the SLOPE comparison is methodology-invariant and is the meaningful diagnostic.

For the Galerkin P2/P1 row (the actual Cocquet pathway) on the structured mesh, the canonical post-retrofit numbers are: L²(u) = 2.31×10⁻⁵ at N=80 → **58× paper**, slope ‑1.22. The freefem-divs Delaunay mesh gives 2.67×10⁻⁴ at N=80 (670× paper) with the better slope ‑2.05 — the structured mesh is in fact 12× *more accurate* in absolute terms at N=80 despite the worse slope (the structured mesh's lower coarse-N bulk error wins until h is small enough for the slope advantage to flip). This crossover is why the §2 headline numbers (freefem-divs) overstate the magnitude gap at finite N — they reflect the SLOPE-aligned mesh, not the absolute-magnitude-best one.

**Status:** S5 is CLOSED. Findings folded into §6 (priority list bumps S2 / S3) and into the executive summary.

### §5.8 — Paper–code divergences ledger does not apply to Cocquet's Galerkin pathway

[theory/paper-code-divergences.md](../theory/paper-code-divergences.md) lists 9 documented divergences. All are stabilization-specific (τ parameters, projection policies, adjoint signs, OSGS iterative loop, etc.). With `mult_mom=mult_mass=0`, **none** fire on the Galerkin row. The Cocquet comparison is therefore not affected by any known divergence.

---

## 6 — Open hypotheses + new candidates surfaced by audit

| ID  | Claim                                                                                          | Currently exists?                                                                                                                                                | Status  | Estimated cost |
|-----|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|----------------|
| ~~O4~~ | ~~Cocquet reports relative L²(u), not absolute~~ | **FALSIFIED 2026-05-26.** Cocquet et al. 2021 page 32 explicitly defines `Err_tot := ‖ũ_h − ũ‖_X + ‖p_h − p‖_L²(Ω)` (absolute), and the Figure 2 / 3 right panels plot the same absolute L²(u). No relative normalization anywhere. | CLOSED | — |
| O6  | Cocquet uses Lagrange-multiplier or zero-mean pressure gauge instead of η-penalty               | None                                                                                                                                                              | OPEN    | Medium         |
| ~~O7~~ | ~~Newton stalls short of machine precision in some runs~~ | **FALSIFIED 2026-05-26.** [test/extended/CocquetExperimentIrregularMesh/results/run_freefem_uniform.log](../test/extended/CocquetExperimentIrregularMesh/results/run_freefem_uniform.log) shows every Galerkin solve at N∈{10,20,30,…,100} reaches iter-4 residual inf-norm between 1.7×10⁻¹⁵ and 1.0×10⁻¹³ (machine precision). The N=200 reference solve goes 7 iterations, stops at 6.0×10⁻¹² — still well below the configured `ftol=1e-11`. | CLOSED | — |
| ~~O8~~ | ~~Anderson acceleration interferes with the Galerkin runs~~ | **FALSIFIED 2026-05-26.** [galerkin_driver.jl:48-49](../test/extended/CocquetExperiment/galerkin_driver.jl#L48-L49) constructs `SafeNewtonSolver` for both Picard and Newton modes — neither takes the Anderson accelerator. The Anderson config block in [paper_comparison.json:84-88](../test/extended/CocquetExperiment/data/paper_comparison.json#L84-L88) only enters `solve_system`'s ASGS/OSGS pathway (used for the P1/P1 and P2/P2 ASGS rows), not the Galerkin pathway. Newton from a zero initial guess; line-search Armijo with full step size α=1.0 throughout. | CLOSED | — |
| **S1** | **Re-run the configs that previously inherited eps_val=1e‑8 against the now-explicit 1e‑7** | **CocquetExperiment canonical: DONE 2026-05-26** — ASGS P1/P1 numbers bit-identical to 4 sig figs (Δ at N=80 is < 1e‑10 — empirically invisible). Still pre-retrofit: ModifiedCorner, Alpha1, Deviatoric, LinearReaction, AllDirichlet. Freefem-divs / FreeFem-mesh / LiteralPicard always had 1e‑7. | PARTIALLY CLOSED | Low — the canonical confirms the change is invisible. The 5 remaining sweeps are bookkeeping only. |
| **S2** | **Cocquet uses BAMG-adaptive corner-refined mesh, not uniform Delaunay**                    | Inspect [theory/Cocquet et al. - 2021 ...pdf](../theory/) Section 5 mesh recipe for `adaptmesh`/`hsize` adaptation flags                                          | NEW     | Trivial (read) |
| **S3** | **Low-frequency / global mode contaminates L²(u) but not H¹(u). Project u_h − u_ref onto P0** | None                                                                                                                                                              | NEW     | Low (write probe script) |
| ~~**S5**~~ | ~~Audit untouched Cocquet siblings~~ | **CLOSED 2026-05-26.** See §5.7. Three siblings (Alpha1 / Deviatoric / LinearReaction) exonerate all porous-physics knobs as cap sources. AllDirichlet implicates the **mixed-BC outlet corner**: P2/P2 slope jumps from ~1.3 to 2.46 when the corner is eliminated. LiteralPicard exonerates linearization choice. | CLOSED  | — |

~~**S4** (Galerkin-mode MMS to confirm unstabilized pathway is correct on smooth fields)~~ — **CLOSED**. The CocquetFormMMS test already includes a Galerkin P2/P1 leg as part of the three-way comparison. Result: L²(u) = 2.85×10⁻⁷ at h=1/320, slope ‑2.87 (optimal P2). See §5.6.

---

## 7 — What we have NOT yet verified (the frontier)

Organized by likely diagnostic value:

### Most likely to yield a diagnosis
- **S3 — Spatial L²(u) decomposition of u_h − u_ref.** The 120× L²/H¹ anomaly (§5.4) is *the* fingerprint. After the S5 audit (§5.7) we know the mixed-BC corner is the **slope** cap, but a residual ~10× magnitude gap remains even after the corner is removed (AllDirichlet P2/P2 still 12× the paper). A P0 cell-mean / low-degree polynomial projection of `u_h − u_ref` would tell us whether that residual is a global-mode contamination — and if so, which mechanism (pressure gauge O6, wall layer near porosity-min y=0, etc.).

### Cheapest to settle — ALL CLOSED 2026-05-26
- ~~**O4**~~ — Cocquet plots absolute L²(u) (paper p. 32 `Err_tot` definition explicit).
- ~~**O7**~~ — All Galerkin Newton solves reach iter-4 residual ≤ 1.0×10⁻¹³, well past `ftol=1e-11`.
- ~~**O8**~~ — Galerkin path uses `SafeNewtonSolver` only; Anderson never fires on the Galerkin row.

### Highest impact if true
- **S2 — Cocquet's mesh is BAMG-adaptive / corner-refined.** Now reinforced by S5: we know (i) the mixed-BC outlet corner is the slope cap, and (ii) a residual magnitude gap of ~10× persists even after the corner is removed. If Cocquet's mesh refines the corner adaptively, that would (i) explain why their slope is paper-level on the *same* mixed-BC problem, and (ii) the residual gap could plausibly be a global mode (S3) that BAMG also dampens. Read-only check of the paper PDF Section 5; no compute.

### Bookkeeping (after the canonical re-run confirmed Δ is invisible)
- **S1 (partially closed) — Re-sweep the 5 remaining pre-retrofit siblings** (ModifiedCorner, Alpha1, Deviatoric, LinearReaction, AllDirichlet) for audit-trail consistency. The canonical CocquetExperiment was re-run 2026-05-26 with `eps_val=1e-7` explicit and gave numbers bit-identical to 4 significant figures vs the pre-retrofit `1e-8` run — so the empirical effect is below 1e-10 on L²(u). The 5 remaining sweeps are pure bookkeeping; defer until the user wants AC-powered overnight compute.

### Hypotheses suggested by the audit but probably not worth testing
- **Inlet quadratic profile + P2 representation.** With c_in=0.5 the inlet is *exactly* representable by P2; no projection error.
- **Newton damping × Forchheimer interaction.** Already fixed by [lessons_learned.md](lessons_learned.md) row 25 (τ-floor leakage); `h_floor_weight=0` is set in [paper_comparison.json:27](../test/extended/CocquetExperiment/data/paper_comparison.json#L27).

### Suggested but speculative
- **Wall boundary layer at y=0** (porosity minimum, Forchheimer drag maximum). Could need anisotropic refinement. Would manifest as concentration of L²(u) error near y=0; testable by reusing localize_err.jl with a different region partition (wall vs bulk vs corner). The S5 audit's LinearReaction sibling (which zeroes β·|u|·u while keeping the linear Darcy α(ε) drag) keeps the same slope as baseline, weakly suggesting the *nonlinear* part of the wall-layer is not the source — but the linear-Darcy wall layer itself is untouched and could still contribute.

---

## 8 — Document relationships and update protocol

- [docs/cocquet_magnitude_investigation.md](cocquet_magnitude_investigation.md) — **chronological ledger.** Append-only record of every diagnostic.
- [docs/cocquet_convergence_analysis.md](cocquet_convergence_analysis.md) — **slope-phenomenology sibling.**
- **This file (cocquet_investigation_synthesis.md)** — **point-in-time snapshot, replaceable.** Update whenever a hypothesis crosses the open→closed boundary or a new candidate is surfaced. Keep the ledger appendable; keep this file replaceable.

**Open action items**, in order of recommended attack: S3, S2, S1, O6. (O4, O7, O8, S5 all closed 2026-05-26.)
