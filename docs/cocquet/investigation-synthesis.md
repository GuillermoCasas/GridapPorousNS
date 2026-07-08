# Cocquet absolute-magnitude investigation — synthesis snapshot (2026-05-26)

> **CANONICAL Cocquet reference — start here.** This is the single canonical doc for the Cocquet benchmark. The chronological hypothesis ledger is folded in as the **Appendix** of this file. Companions in this folder: the phased slope investigation [convergence-analysis.md](convergence-analysis.md), the (unsent) [email-questions.md](email-questions.md), and the archived raw transcripts [replicating-cocquet-transcript.md](archive/replicating-cocquet-transcript.md) / [corner-singularity-transcript.md](archive/corner-singularity-transcript.md).

## 1 — Executive summary — SETTLED conclusion

**The convergence cap is MESH-TOPOLOGY.** The sub-optimal Cocquet convergence rate on our runs is a **structured-mesh artifact**: a structured Cartesian-simplexified mesh with edges aligned to the 90° no-slip/traction outlet corner **locks in the corner singularity**, capping the velocity-L² slope. An **unstructured Delaunay mesh (FreeFem `buildmesh` paradigm) recovers $O(h^2)$**, matching Cocquet's Fig. 2. This settles the *slope* question and **withdraws** the earlier Phase 9 "proven mesh- and method-independent corner singularity" and Phase 10 "the paper's $H^3{\times}H^2$ claim is incorrect" verdicts — both rested on the structured mesh (see [convergence-analysis.md](convergence-analysis.md) Phase 11).

**The residual is a MAGNITUDE gap of ~30× (coarse) to ~300× (fine).** After Phase 11–12's literal-FreeFem-mesh reproduction and the corrected figure reading (the [replicating-cocquet](archive/replicating-cocquet-transcript.md) session), the honest gap on the Cocquet Galerkin P2/P1 row is **~30× at N=10 and ~300× at N=100** — i.e. our slope is right but the curve sits on a different baseline. This **corrects two stale readings**: the "~2×" that an earlier Phase 11–12 header claimed (an order-of-magnitude misread of Fig. 2's N=10 point — it is ~3×10⁻⁵, not ~3×10⁻⁴), and the older "320×–845×" `freefem-divs` figures in the body below (a slope-aligned but not magnitude-best mesh; kept for provenance, but read them through the ~30–300× correction). The magnitude gap is most plausibly a *measurement* difference in what Cocquet plots (see the lateral-hypothesis subsection below), not a bug in our code.

**Cocquet's reported numbers** (Fig 2 right, Re=500, c_in=0.5, N=200 self-reference): L²(u) ≈ 3×10⁻⁵ at N=10, ≈ 2×10⁻⁷ at N=100. Our best Galerkin P2/P1 (freefem-divs unstructured mesh) gives 9.68×10⁻³ at N=10 and 1.69×10⁻⁴ at N=100. Finest-segment slope matches: ours ‑2.05 (N=40→80), paper ‑2.18.

---

## 1a — Lateral-hypothesis verdicts (merged from the 2026-05-24 replicating-cocquet session)

Five lateral hypotheses were raised to explain the ~30–300× magnitude gap on the freefem-divs mesh. Four were tested and discarded *from inside the repo*; one is parked pending external code. See the archived [replicating-cocquet-transcript.md](archive/replicating-cocquet-transcript.md) for the full working.

- **H-A (paper plots SQUARED L² norms).** DISCARDED. Plotting `‖e_u‖²` crosses the paper curve between N=20 and N=40 (above at coarse N, below at fine N) and gives slope −3.52 vs the paper's −2.30 — neither the ratios nor the slope match. Squaring is not the explanation.
- **H-B (Cocquet's Picard stopped short of the discrete fixed point at Re=500, c_in=0.5).** DISCARDED. A literal pure-Picard sibling (`CocquetTubeTest/data/literal_picard`, itmax=10, tol=1e-10, no Newton) lands on the true discrete fixed point in 7–10 iterations; its L²(u) is bit-identical (≤0.018%) to the converged-Galerkin solution at every N. Cocquet's stopping rule applied literally closes none of the gap. (Also corrects the earlier "squared-norm" reading of the paper's `‖·‖_{L²(Ω)²}` stopping-rule notation: it is the vector-valued L², not a squared norm.)
- **H-C (they measure `‖I_h u_ref − u_h‖`, a trial-space best-approximation projection, not `‖u_ref − u_h‖`).** DISCARDED. A trial-space-projection metric agrees with the free-DOF metric to ≤1.4% on the load-bearing Galerkin P2/P1 case (the P2-exact inlet + zero walls bound the difference to cross-mesh boundary-node noise). It cannot close even 1% of the gap.
- **H-D (their FreeFem solver iterates on the lifting `w = u − V` and reports the error in `w`, not `u`).** PARKED — unfalsifiable from inside the repo (only the mesh-generation `.edp` is available, not Cocquet's solver `.edp`). Requires external action (corresponding-author email / supplementary-code hunt). Note `V` is P2-exact for the quadratic inlet, so a *divergence-free* lifting would leave the error unchanged; only an h-dependent approximate lifting could matter.
- **H-E (a factor-of-2 in `S(u)` ⇒ effective ν doubled ⇒ smoother solution).** DISCARDED by code reading. `viscous_operators.jl` uses `2·α·ν·(ε(u)⊙ε(v))` with `ε(u)=½(∇u+∇uᵀ)`, matching the paper's `2Re⁻¹ S(u)` with `S(u)=½(∇u+∇uᵀ)`. Effective ν is genuinely `1/Re`, not `2/Re`. No mismatch.

**Net:** nothing we can change in our code, solver, metric, or viscosity convention closes any meaningful part of the ~30–300× magnitude gap. The remaining viable explanation is a *measurement* difference in what Cocquet's (unavailable) solver plots (H-D or something beyond the original five), not a defect in our discretization.

---

## 1b — Supporting findings (pre-settlement synthesis)

> These paragraphs pre-date the mesh-topology settlement (§1) and the ~30–300× correction (§1a). They remain accurate as **sub-findings** — the S5 sibling audit and the CocquetFormMMS exculpation are real — but note that "Dominant cap mechanism" below is now understood as *the structured-mesh manifestation* of the corner singularity: on an unstructured mesh the slope cap disappears (see §1). The paragraphs are kept for provenance and as the supporting evidence for §1.

**What we know.** The code is a term-by-term faithful transcription of Cocquet's weak form (verified — §4 below). The Newton solve converges to machine-precision residual (3.7×10⁻¹⁵). On a manufactured solution with the **same** Forchheimer/porosity operators **and the same unstabilized Taylor-Hood P2/P1 Galerkin formulation** ([CocquetFormMMS three-way comparison](../../test/extended/CocquetFormMMS/)) we hit L²(u) = 2.85×10⁻⁷ at h=1/320 with slope 2.87 (optimal P2) — i.e., the code **can** reach paper magnitudes on smooth problems, **on the exact same Galerkin pathway used by Cocquet**. The remaining Cocquet-benchmark gap is therefore a **non-smoothness × discretization interaction**, not a solver bug or a stabilization-vs-Galerkin artifact.

**Dominant cap mechanism identified (S5 audit, §5.7).** A side-by-side sweep of five one-knob siblings shows:

- Removing the **porous drag entirely** (Alpha1), switching to a **deviatoric viscous operator** (Deviatoric), and **zeroing the Forchheimer β·\|u\|·u nonlinearity** (LinearReaction) all leave the P2/P2 slope stuck at ~1.26–1.38 — three independent exonerations of porous-physics knobs.
- Replacing the natural traction-free outlet with **all-Dirichlet BCs** (AllDirichlet) jumps P2/P2 slope from ~1.27 to **2.46**.
- Picard vs Newton linearization is irrelevant (LiteralPicard).

→ The convergence cap is driven by the **mixed-BC outlet corner singularity** at `(2,0)` and `(2,1)`, where Dirichlet walls meet the Neumann outlet. This also explains the localize_err finding (71% of structured-mesh L²(u)² lives at the outlet corner).

**The remaining gap.** Even AllDirichlet's best run (P2/P2 N=80, L²(u) = 4.7×10⁻⁶) is still **~12× the paper's ~4×10⁻⁷** at N=80. So eliminating the corner singularity peels off the slope component of the gap, but a residual magnitude factor persists. All currently-tested specific mechanisms (H1–H12, O1–O8, S2–S5) and the five lateral hypotheses (H-A…H-E, §1a) have been falsified, confirmed paper-faithful, or closed by the mesh-topology settlement. **The only genuinely-open thread is external:** the ~30–300× magnitude gap is now attributed to a *measurement* difference in Cocquet's unavailable solver (H-D / beyond-the-five, §1a), not to any knob in our code. The earlier "L²(u)/H¹(u) low-frequency global mode" candidate (§5.4) was ruled out by S3a (χ_Ω ≲5%).

**Housekeeping correction (2026-05-26).** A previous version of this synthesis claimed H8 (`pressure penalty η = 1e-7`) was falsified because the canonical [test/extended/CocquetTubeTest/data/structured/paper_comparison.json](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json) did not override the (now-removed) `1e-8` default in `base_config.json`. The fix landed on 2026-05-26: every Cocquet paper-comparison config now **explicitly sets `physical_epsilon = 1e-7`** to match the paper, and `physical_epsilon` was removed from `base_config.json` so it can no longer leak in silently (consistent with the CLAUDE.md rule that `load_config_with_*` must not invent values). The freefem-divs result that produced the headline numbers above was already running with `physical_epsilon=1e-7` (it had its own override), so the headline ratios are unaffected by this fix.

---

## 2 — Headline numbers

> **Read through the §1 correction.** The "320×–845×" ratios in this section are from the `freefem-divs` mesh — slope-aligned but not magnitude-best. The **settled** magnitude gap is **~30× (coarse) → ~300× (fine)** (§1). These rows are kept for provenance.

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
| Freefem-divs unstructured (Delaunay, walls 2× coarser) | 1.69×10⁻⁴ | ‑2.05 | [test/extended/CocquetTubeTest/results/freefem_divisions/](../../test/extended/CocquetTubeTest/results/freefem_divisions/) |
| Literal FreeFem .msh (Cocquet's exact recipe) | 4.59×10⁻⁵      | (n/a, single N)     | [test/extended/CocquetTubeTest/results/freefem_meshes/convergence.h5](../../test/extended/CocquetTubeTest/results/freefem_meshes/convergence.h5) |
| Uniform Delaunay 1/N divisions                | (sweep)        | ‑1.80               | [test/extended/CocquetTubeTest/data/unstructured_gmsh/](../../test/extended/CocquetTubeTest/data/unstructured_gmsh/)                              |
| **Structured Cartesian — Galerkin P2/P1** (canonical, post-retrofit 2026-05-26) | **2.31×10⁻⁵ at N=80** (58× paper @ N=80 ~4×10⁻⁷) | **‑1.22** | [test/extended/CocquetTubeTest/results/structured/convergence.h5](../../test/extended/CocquetTubeTest/results/structured/convergence.h5) |
| Structured Cartesian — ASGS P2/P2 | 9.19×10⁻⁶ at N=80 (23× paper) | ‑1.64 | same h5 |
| Structured Cartesian — ASGS P1/P1 | 8.84×10⁻⁵ at N=80 | ‑1.82 | same h5 |
| AllDirichlet — ASGS P2/P2 (no mixed-BC corner) | 4.70×10⁻⁶ at N=80 (12× paper) | ‑2.46 | [test/extended/CocquetTubeTest/results/all_dirichlet/](../../test/extended/CocquetTubeTest/results/all_dirichlet/) |
| **MMS — Galerkin P2/P1 unstabilized (same pathway as Cocquet)** | **2.85×10⁻⁷ at h=1/320** | **‑2.87 (optimal P2)** | `config_3_Galerkin` in [test/extended/CocquetFormMMS/results/cocquet_form_mms_comparison.h5](../../test/extended/CocquetFormMMS/results/cocquet_form_mms_comparison.h5) |
| MMS — Stabilized P2/P2 ASGS (our method) | 3.42×10⁻⁷ at h=1/320 | ‑2.95 (optimal P2) | `config_2_ASGS` in same h5 |
| MMS — Stabilized P1/P1 ASGS (our method) | 1.34×10⁻⁴ at h=1/320 | ‑1.64 | `config_1_ASGS` in same h5 |

Even the literal-FreeFem mesh leaves a 230× gap. The three MMS rows are the strongest exculpatory evidence: **on a smooth solution with the same Forchheimer / porosity operators, all three formulations — our stabilized P1/P1 and P2/P2, AND Cocquet's literal unstabilized Galerkin P2/P1 — converge at their expected optimal rates and reach paper-magnitude.** This is the explicit three-way comparison built into the [CocquetFormMMS](../../test/extended/CocquetFormMMS/) test (its designed modes `cocquet_form_mms_{vms,vms_k2,taylorhood}.json`; the Galerkin leg uses the dedicated [`execute_solver_galerkin_inline!`](../../test/extended/CocquetFormMMS/run_test.jl#L160-L207) path with `mult_mom=mult_mass=0`).

---

## 3 — Test setup baseline

- **PDE:** ε-weighted Darcy–Brinkman–Forchheimer (Cocquet 2021 Eqs. 1–2). Re=500, ν=1/Re=0.002, Forchheimer drag, no body force.
- **Domain:** (0,2)×(0,1). Γ_in={0}×[0,1], Γ_out={2}×[0,1], Γ_w=[0,2]×({0}∪{1}).
- **Inlet:** u_in(y) = 0.5·y(1‑y) e_x.
- **Porosity profile:** ε(y) = 0.45 + 0.55·exp(y‑1). Min 0.45 at y=0 (bottom wall), exactly 1 at y=1 (top wall).
- **FE pair:** Taylor–Hood P2/P1, unstabilized for the Galerkin row. The `comparison_runs` at [paper_comparison.json:8-12](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L8-L12) also runs P1/P1 and P2/P2 with ASGS stabilization for cross-comparison.
- **Reference:** self-reference at N=200 (same methodology as paper).
- **Driver:** [test/extended/CocquetTubeTest/galerkin_driver.jl](../../test/extended/CocquetTubeTest/galerkin_driver.jl) calls assembly with `mult_mom=mult_mass=0` (no VMS τ-stabilization).
- **Solver:** Newton, `xtol=1e-11, ftol=1e-11`, Armijo line search, safeguarded fallback. `h_floor_weight=0` per [paper_comparison.json:27](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L27) (τ-floor regression fix from [lessons_learned.md](../lessons_learned.md) row 25 disabled).

---

## 4 — Catalogue of all hypotheses (closed)

For each hypothesis: ID, claim, status, **the test/code path that currently exercises it**, and the result.

### 4.1 — Paper-faithfulness hypotheses (H3–H12)

These are not "hypotheses" in the experimental sense — they are claims about whether our code corresponds to the paper. They are verified by **code inspection**, not by running experiments.

| ID  | Claim                                                                 | Status              | Path that exercises the claim                                                                                                                                       | Result / verdict                                                                                                                                              |
|-----|-----------------------------------------------------------------------|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| H3  | ν = 1/Re mapping (Re=500 ⇒ ν=0.002)                                   | CONFIRMED           | [src/formulations/viscous_operators.jl:101-103](../../src/formulations/viscous_operators.jl#L101-L103) + [paper_comparison.json:24](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L24) | `2 * α * ν * (ε(u) ⊙ ε(v))` matches paper's `2 Re⁻¹ ε S(u):S(v)`. nu=0.002.                                                                                  |
| H4  | Forchheimer α(ε), β(ε) closure                                        | CONFIRMED           | [src/models/reaction.jl:57-62](../../src/models/reaction.jl#L57-L62) + [paper_comparison.json:25-26](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L25-L26) | `a_term = 0.30 · ((1‑ε)/ε)²`, `b_term = 1.75 · (1‑ε)/ε`. 0.30 = 150/500. ε=0.45 ⇒ a=0.448, b=2.139. Exact.                                                  |
| H5  | Outlet BC ε-weighted traction-free (natural)                          | CONFIRMED           | [src/formulations/continuous_problem.jl:241](../../src/formulations/continuous_problem.jl#L241)                                                                       | `pres_term = -p·(α·∇·v + ∇α·v)` integrates by parts ⇒ natural BC `εσ·n = 0` on Γ_out. No explicit boundary integral.                                          |
| H6  | Convective form ε(u·∇)u                                               | CONFIRMED           | [src/formulations/continuous_problem.jl:239](../../src/formulations/continuous_problem.jl#L239)                                                                       | `conv_term = v · (α · (∇(u)' · u))` = v · ε(u·∇)u. Exact.                                                                                                     |
| H7  | Mass / pressure-divergence form                                       | CONFIRMED           | [src/formulations/continuous_problem.jl:241,244-245](../../src/formulations/continuous_problem.jl#L241)                                                                | `mass_term = q · (physical_epsilon·p + α·∇·u + u·∇α) = q · (physical_epsilon·p + div(εu))`. Exact.                                                                              |
| H8 | Pressure penalty η = 1e-7 matches paper | CONFIRMED (after 2026-05-26 retrofit) | [paper_comparison.json:25](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L25) explicitly sets `"physical_epsilon": 1e-7`. All Cocquet paper-comparison and sibling configs now do the same; `physical_epsilon` was removed from [base_config.json](../../config/base_config.json) so it can no longer silently inherit. | Before the retrofit, several configs silently inherited the now-removed `1e-8` default. The freefem-divs headline run had its own `1e-7` override so the headline ratios in §2 are unaffected. The structured baseline + several sibling diagnostics ran at `1e-8` before the retrofit; they should be re-run against `1e-7` for a clean apples-to-apples (see §6 S1). |
| H9  | P1 porosity interpolant for the Galerkin row                          | CONFIRMED           | [paper_comparison.json:11](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L11)                                                                       | `"porosity_order": 1` set explicitly on the Galerkin row.                                                                                                       |
| H10 | Domain, inlet profile, porosity profile                               | CONFIRMED           | [run_convergence.jl:9-10](../../test/extended/CocquetTubeTest/run_convergence.jl#L9-L10) + [paper_comparison.json:31-41](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L31-L41) | `alpha_func(x) = 0.45 + 0.55·exp(x[2]-1.0)`, box `[0,2]×[0,1]`, c_in=0.5. Exact.                                                                              |
| H11 | N_ref=200 self-reference (same methodology as paper)                  | CONFIRMED           | [paper_comparison.json:62-67](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L62-L67) (convergence_partitions) + driver                              | Paper p.32 uses N=200 reference; our driver does too.                                                                                                          |
| H12 | Galerkin Taylor-Hood unstabilized (mult_mom=mult_mass=0)              | CONFIRMED           | [galerkin_driver.jl:26-27,62-64](../../test/extended/CocquetTubeTest/galerkin_driver.jl#L26-L27)                                                                     | `mult_mom = 0.0`, `mult_mass = 0.0` passed to assembly. No τ-stabilization terms in the assembled form.                                                        |

### 4.2 — Experimental hypotheses (closed)

| ID | Claim | Status | Test directory / probe | Result |
|----|-------|--------|------------------------|--------|
| H1 | Outlet-corner Dirichlet pin caps convergence | FALSIFIED | [test/extended/CocquetTubeTest/data/modified_corner/](../../test/extended/CocquetTubeTest/data/modified_corner/) — entities 2,4 removed from "walls" tag, freeing outlet-corner DOFs | Made absolute L²(u) **3.1× worse at N=10** (9.36e‑4 vs 3.01e‑4) and **3.0× worse at N=80** (6.89e‑5 vs 2.31e‑5). The released DOF takes an algebraically-driven value that pollutes the bulk. Memory: [cocquet-modified-corner-experiment.md](/Users/guillermocasasgonzalez/.claude/projects/-Users-guillermocasasgonzalez-repos-porous-NS-with-Gridap/memory/cocquet-modified-corner-experiment.md) |
| H2 | Structured-mesh corner-aligned diagonals lock the corner singularity | PARTIALLY CONFIRMED (slope only) | [test/extended/CocquetTubeTest/data/unstructured_gmsh/](../../test/extended/CocquetTubeTest/data/unstructured_gmsh/) (uniform 1/N gmsh Delaunay) + [test/extended/CocquetTubeTest/data/freefem_divisions/](../../test/extended/CocquetTubeTest/data/freefem_divisions/) (literal `buildmesh(a(N)+b(N)+c(N)+d(N))` N-per-side) | Unstructured Delaunay recovers near-optimal slopes (Galerkin P2/P1 freefem-divs final segment **‑2.05**, uniform-divs ‑1.80; P2/P2 ASGS 2.39 uniform, 2.13 freefem-divs). But absolute magnitudes are still 320–845× the paper. Memory: [cocquet-mesh-topology-controls-convergence.md](/Users/guillermocasasgonzalez/.claude/projects/-Users-guillermocasasgonzalez-repos-porous-NS-with-Gridap/memory/cocquet-mesh-topology-controls-convergence.md) |
| O1 | Our N=200 reference is itself under-converged vs the truth solution | RULED OUT BY ARGUMENT | (No experimental test — argument from symmetry of methodology.) | Paper uses the same self-reference methodology. If our reference were under-converged by ~1e‑4, paper's would be too and would also plateau at that floor. It doesn't (paper reaches 2e‑7 at N=100). Therefore the gap is not in the reference. Strengthened in §5.2. |
| O2 | `u_base_floor_ref = 1e-4` velocity-magnitude regularization adds a noise floor | RULED OUT BY PROBE | `/tmp/probe_with_floor_off.jl`, log `/tmp/probe_floor_off.log` — single Galerkin P2/P1 N=80 + N_ref=200 solve with `u_base_floor_ref=0` and `epsilon_floor=0` | L²(u) at N=80 = **2.6685e‑4** — identical to 4 sig figs to baseline (2.669e‑4). Newton iter-0 residual bit-identical at 0.001164580518. The floor never fires on this flow (true \|u\| > 1e‑4 everywhere except wall vertices where u=0). |
| O3 | Solution magnitude / spatial profile differs from Cocquet's | PARTIALLY RULED OUT | `/tmp/probe_cocquet_reference.jl`, log `/tmp/probe_cocquet_ref.log` | Flow magnitudes physically reasonable: ‖u‖_L²(Ω) = 0.125 (consistent with c_in/4 inflow scale); inlet u(0,0.5) = (0.125, 0) exact; centerline u_x decreases 0.125 → 0.102 (Darcy drag); asymmetric profile (peak at y=0.75) because porosity ramps. Newton converged to 3.7e‑15 in 4 iterations. We're solving the right PDE. Caveat: a smooth, global error mode would slip past these checks — see §5.5. |
| O5 | Quadrature degree insufficient for non-polynomial Forchheimer term | RULED OUT BY PROBE | `/tmp/probe_quad_plus12.jl`, log `/tmp/probe_quad12.log` — N=200 reference solve with `Measure(Ω, 21)` (default for k_v=2 is 9; bump +12) | Iter-0 PDE residual: baseline `9.588812700822735e‑4`, quad+12 `9.588812745812068e‑4`. Delta ~5e‑9 — five orders of magnitude below our 1e‑4 error level. Quadrature is adequate. |

### 4.3 — Diagnostic experiments referenced in the ledger

| Diagnostic | What it measured | Where to find it | Headline |
|------------|------------------|------------------|----------|
| **CocquetFormMMS** (smooth-solution MMS with same operators) | Best-case attainable L²(u) with same Forchheimer/porosity operators | [test/extended/CocquetFormMMS/](../../test/extended/CocquetFormMMS/), `results/cocquet_form_mms.h5` | **L²(u) = 8.54×10⁻⁷ at h=1/80** (config_1 ASGS P2/P1). Same order as paper's N=100 magnitude. Confirms the code is not bugged. |
| **Literal FreeFem .msh comparison** | Solve on Cocquet's exact mesh-recipe `.msh` files (generated by FreeFem) | [test/extended/CocquetTubeTest/results/freefem_meshes/convergence.h5](../../test/extended/CocquetTubeTest/results/freefem_meshes/convergence.h5), driver [paper_comparison_freefem.json](../../test/extended/CocquetTubeTest/data/freefem_meshes/) | **L²(u) = 4.59×10⁻⁵ at N=100** — closer than gmsh Delaunay variants, but still 230× the paper's 2×10⁻⁷. |
| **localize_err.jl** (spatial decomposition of L²(u)² over regions) — *script removed in the CocquetTubeTest unification; recoverable from git history* | Where in the domain does the error live? | `/tmp/localize_err.log` (committed analysis output) | Structured: 71.0% outlet_corner, 25.3% bulk. Unstructured Delaunay: 1.08% outlet_corner, 61.8% bulk. Unstructured *total* ‖e‖ is 18× *larger* than structured at N=10 (5.74e‑3 vs 3.19e‑4). |
| **Figure 2 y-axis re-verification** | Confirm paper's reported magnitudes | Direct inspection of the paper PDF | Y-axis: 10⁻⁴ (top tick) → 10⁻⁷ (bottom). Curve from ~3×10⁻⁵ at N=10 to ~2×10⁻⁷ at N=100. Magnitudes are real. |

---

## 5 — Logical soundness audit (second pass)

This section walks every previously-recorded conclusion and tests it against the **code as it actually exists today**, the math in [theory/cocquet/cocquet_formulation.tex](../../theory/cocquet/cocquet_formulation.tex) and [theory/paper/article.tex](../../theory/paper/article.tex), and the result files on disk. Each subsection ends with the concrete next action, if any.

### §5.1 — H8 retrofit (2026-05-26) — silent default removed; ledger now consistent

**Audit finding.** A prior version of this synthesis (and the long-running ledger) recorded H8 (`pressure penalty η = 1e-7`) as CONFIRMED, but a code audit showed the canonical [test/extended/CocquetTubeTest/data/structured/paper_comparison.json](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json) did **not** override the `physical_epsilon=1e-8` default in `base_config.json`. So the structured baseline + the H1 ModifiedCorner experiment + the Alpha1 / Deviatoric / LinearReaction / AllDirichlet sibling diagnostics were all running with `physical_epsilon=1e-8`, a 10× *smaller* penalty than the paper's `η = 1e-7`. The IrregularMeshFreefemDivs config that produced the headline numbers in §2 had its own `1e-7` override, so those headline ratios are unaffected.

**Fix (2026-05-26):**
- (a) `physical_epsilon=1e-7` now set explicitly in all seven affected configs ([structured](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json), [ModifiedCorner main](../../test/extended/CocquetTubeTest/data/modified_corner/paper_comparison_modified_corner.json) and [smoke](../../test/extended/CocquetTubeTest/data/modified_corner/paper_comparison_modified_corner_smoke.json), [Alpha1](../../test/extended/CocquetTubeTest/data/alpha_one/alpha1.json), [Deviatoric](../../test/extended/CocquetTubeTest/data/deviatoric/deviatoric.json), [LinearReaction](../../test/extended/CocquetTubeTest/data/linear_reaction/linear_reaction.json), [AllDirichlet](../../test/extended/CocquetTubeTest/data/all_dirichlet/all_dirichlet.json)).
- (b) `physical_epsilon` removed from [base_config.json](../../config/base_config.json) so silent inheritance through `load_config_with_*` is impossible (every test config must now declare its own `physical_epsilon` or the strict struct constructor will fail loudly — consistent with the CLAUDE.md "no implicit defaults — fail loudly on missing input" rule).
- Blitz tests (83 / 83) pass post-fix.

**Why this doesn't close the magnitude gap.** Direction analysis: the penalty term is `+ η·p·q`. A smaller η means a *tighter* constraint on `div(εu) ≈ 0` — more faithful to the true incompressibility constraint, not less. The previous `1e-8` versus paper-faithful `1e-7` could not have moved velocity error by orders of magnitude. The pre-retrofit sweeps should be regenerated for audit-trail consistency (S1 in §6), but Δ on L²(u) is predicted ≪ 1e‑6, far below the 1e‑4 error level.

**Sensitivity confirmed empirically (2026-05-26).** The canonical structured-mesh variant (`CocquetTubeTest/data/structured`) was re-run post-retrofit with `physical_epsilon=1e-7` explicit. The ASGS P1/P1 L²(u) numbers are **bit-identical to 4 significant figures** to the pre-retrofit (1e-8) run: N=10: 3.919×10⁻³, N=20: 1.187×10⁻³, N=40: 3.433×10⁻⁴, N=80: 8.840×10⁻⁵. The physical_epsilon change is empirically invisible at the present error scale, as predicted by the direction analysis. The retrofit's value is methodological (paper-faithful + no silent defaults), not numerical.

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

**Action (S2): CLOSED.** Superseded by the mesh-topology settlement (§1): a *plain* unstructured Delaunay mesh (no adaptivity) already recovers the paper's $O(h^2)$ slope (Phase 11), so whether Cocquet additionally used BAMG adaptmesh is no longer load-bearing for the settled slope conclusion.

### §5.4 — The L²/H¹ cross-norm anomaly (global-mode candidate — CLOSED by S3a)

> **CLOSED 2026-07-04.** The global-mode reading below was **ruled out**: S3a's domain-mean share χ_Ω is ≲5% for the load-bearing Galerkin P2/P1 and ASGS P2/P2 rows (see the S3a detail in the Appendix). The magnitude gap is *not* a rigid/global mode surviving self-reference. Kept for provenance and because the cross-norm numbers themselves are still valid observations.

Numbers from the ledger:

- L²(u) ratio = 845× (ours / paper at N=100).
- H¹(u) seminorm ratio = 7×.

For a discrete solution with uniform error, Aubin–Nitsche gives `‖u − u_h‖_L² ≤ C·h·‖u − u_h‖_H¹`. The L² ratio should shrink relative to H¹ as h decreases — *not grow by 120×*. The opposite scaling implies **a non-uniform error structure**: most likely a low-frequency / global mode that contributes a lot to L² but little to H¹.

Candidate mechanisms for such a mode:
- Pressure-gauge offset (related to O6): a global p offset can drag u through the momentum coupling at the η-penalty level. Should be O(η) = 1e‑8, but the coupling could amplify.
- Wall layer at y=0: porosity minimum (0.45) ⇒ Forchheimer drag at its peak. A thin, slowly-resolved boundary layer would add low-frequency mass.
- Forchheimer term activation threshold: drag term `β(ε)|u|u` is non-smooth at |u|=0. The cross-stream zero-line of u_y could create a kink.

**Action (S3): DONE — global mode ruled out.** The P0 / domain-mean projection was run (S3a): χ_Ω ≲5% on the load-bearing cases, so the gap is *not* a global mode. No candidate mechanism above survives; the residual magnitude gap is a measurement question (§1a).

### §5.5 — O3 "we're solving the same PDE" relies on a single coarse-grained probe

The probe checked: ‖u‖_L²=0.125, inlet point value, centerline drag, cross-stream asymmetry, Newton residual. All consistent. **But** a smooth global error mode of magnitude 1×10⁻⁴ would not be visible in any of these checks — it's smaller than the precision of the coarse-grained diagnostics.

**Action:** see §5.4 — the P0-projection diagnostic also strengthens this conclusion. Additionally, comparing against Cocquet's Fig 3 (Re=1000, c_in=1) — where they DO show field plots — would give a direct visual check.

### §5.6 — CocquetFormMMS is the strongest exculpatory evidence, AND it directly exercises the unstabilized Galerkin pathway

The [CocquetFormMMS](../../test/extended/CocquetFormMMS/) test is a **deliberate three-way comparison** of the same MMS problem under all three formulations: the user's stabilized P1/P1 ASGS, the user's stabilized P2/P2 ASGS, and Cocquet's literal unstabilized Galerkin P2/P1 — its designed modes `cocquet_form_mms_{vms,vms_k2,taylorhood}.json` (an earlier `comparison_{stab,galerkin}` variant that wrote into a shared HDF5 was superseded by these and removed 2026-07-08). The Galerkin (Taylor–Hood) leg uses a dedicated path — [`execute_solver_galerkin_inline!`](../../test/extended/CocquetFormMMS/run_test.jl#L160-L207) — that calls `build_stabilized_weak_form_residual` / `build_stabilized_weak_form_jacobian` with `mult_mom=mult_mass=0`, i.e., the **exact** assembly used by the production Cocquet comparison_runs Galerkin row.

Results at h=1/320:

| Method | k_v / k_p | L²(u) | Slope |
|---|---|---|---|
| ASGS (user's stabilized) | P1/P1 | 1.34×10⁻⁴ | ‑1.64 |
| ASGS (user's stabilized) | P2/P2 | 3.42×10⁻⁷ | ‑2.95 (optimal P2) |
| **Galerkin (Cocquet's literal)** | **P2/P1** | **2.85×10⁻⁷** | **‑2.87 (optimal P2)** |

**What this proves.** Assembly, integration, Newton, the Forchheimer reaction operator, the viscous + convective + pressure-divergence terms, AND the unstabilized `mult_mom=mult_mass=0` Galerkin codepath itself are all correct on smooth fields, including at the optimal P2 rate. The Cocquet-benchmark gap is therefore unambiguously a **non-smoothness × discretization interaction**, not specific to stabilization vs. Galerkin and not a hidden assembly defect in the unstabilized path.

The remaining open question (S3 below) is what specific feature of the Cocquet benchmark's non-smoothness our solver handles differently from theirs.

### §5.7 — S5 sibling audit (CLOSED 2026-05-26): mixed-BC outlet corner is the dominant cap mechanism

Each sibling is a deliberate one-knob flip of the canonical [paper_comparison.json](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json). All results below were produced 2026-05-22 with `h_floor_weight=0` (post τ-floor regression fix) but pre-physical_epsilon retrofit (i.e., they ran at `physical_epsilon=1e-8` rather than the now-explicit `1e-7`). The expected Δ on L²(u) from that 10× penalty change is ≪ 1e-6 — well below the magnitudes reported here — so the verdicts below stand. A clean re-sweep is queued as S1.

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

Removing the corner singularity buys a **~2× magnitude improvement** in the ASGS P2/P2 row (and almost all of the slope cap, from 1.64 → 2.46). The residual **~12× factor to the paper persists even with the corner gone** — originally attributed to the global / low-frequency mode flagged in §5.4, but that global-mode reading was later ruled out (S3a, χ_Ω ≲5%); the residual magnitude gap is now understood as a measurement difference in Cocquet's unavailable solver (§1a). Caveat: AllDirichlet changes the physical problem (no traction-free outlet), so this is the structured-mesh apples-to-apples within OUR experiments, not against Cocquet's reported Re=500 c_in=0.5 case directly. But the SLOPE comparison is methodology-invariant and is the meaningful diagnostic.

For the Galerkin P2/P1 row (the actual Cocquet pathway) on the structured mesh, the canonical post-retrofit numbers are: L²(u) = 2.31×10⁻⁵ at N=80 → **58× paper**, slope ‑1.22. The freefem-divs Delaunay mesh gives 2.67×10⁻⁴ at N=80 (670× paper) with the better slope ‑2.05 — the structured mesh is in fact 12× *more accurate* in absolute terms at N=80 despite the worse slope (the structured mesh's lower coarse-N bulk error wins until h is small enough for the slope advantage to flip). This crossover is why the §2 headline numbers (freefem-divs) overstate the magnitude gap at finite N — they reflect the SLOPE-aligned mesh, not the absolute-magnitude-best one.

**Status:** S5 is CLOSED. Findings folded into §6 (priority list bumps S2 / S3) and into the executive summary.

### §5.8 — Paper–code divergences ledger does not apply to Cocquet's Galerkin pathway

[docs/solver/paper-code-divergences.md](../solver/paper-code-divergences.md) lists 9 documented divergences. All are stabilization-specific (τ parameters, projection policies, adjoint signs, OSGS iterative loop, etc.). With `mult_mom=mult_mass=0`, **none** fire on the Galerkin row. The Cocquet comparison is therefore not affected by any known divergence.

---

## 6 — Open hypotheses + new candidates surfaced by audit

| ID  | Claim                                                                                          | Currently exists?                                                                                                                                                | Status  | Estimated cost |
|-----|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|----------------|
| ~~O4~~ | ~~Cocquet reports relative L²(u), not absolute~~ | **FALSIFIED 2026-05-26.** Cocquet et al. 2021 page 32 explicitly defines `Err_tot := ‖ũ_h − ũ‖_X + ‖p_h − p‖_L²(Ω)` (absolute), and the Figure 2 / 3 right panels plot the same absolute L²(u). No relative normalization anywhere. | CLOSED | — |
| ~~O6~~  | ~~Cocquet uses Lagrange-multiplier or zero-mean pressure gauge instead of η-penalty~~               | **PARKED 2026-07-04.** Subsumed by the mesh-topology settlement (§1) + the ~30–300× magnitude reading (§1a): the residual gap this hypothesis was meant to explain is now attributed to a measurement difference in Cocquet's *unavailable* solver (H-D / beyond-the-five), not to a pressure-gauge global mode. Untestable without their `.edp`. | PARKED  | — (external)   |
| ~~O7~~ | ~~Newton stalls short of machine precision in some runs~~ | **FALSIFIED 2026-05-26.** [test/extended/CocquetTubeTest/results/freefem_meshes/run_freefem_uniform.log](../../test/extended/CocquetTubeTest/results/freefem_meshes/run_freefem_uniform.log) shows every Galerkin solve at N∈{10,20,30,…,100} reaches iter-4 residual inf-norm between 1.7×10⁻¹⁵ and 1.0×10⁻¹³ (machine precision). The N=200 reference solve goes 7 iterations, stops at 6.0×10⁻¹² — still well below the configured `ftol=1e-11`. | CLOSED | — |
| ~~O8~~ | ~~Anderson acceleration interferes with the Galerkin runs~~ | **FALSIFIED 2026-05-26.** [galerkin_driver.jl:48-49](../../test/extended/CocquetTubeTest/galerkin_driver.jl#L48-L49) constructs `SafeNewtonSolver` for both Picard and Newton modes — neither takes the Anderson accelerator. The Anderson config block in [paper_comparison.json:84-88](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L84-L88) only enters `solve_system`'s ASGS/OSGS pathway (used for the P1/P1 and P2/P2 ASGS rows), not the Galerkin pathway. Newton from a zero initial guess; line-search Armijo with full step size α=1.0 throughout. | CLOSED | — |
| **S1** | **Re-run the configs that previously inherited physical_epsilon=1e‑8 against the now-explicit 1e‑7** | **structured canonical: DONE 2026-05-26** — ASGS P1/P1 numbers bit-identical to 4 sig figs (Δ at N=80 is < 1e‑10 — empirically invisible). Still pre-retrofit: ModifiedCorner, Alpha1, Deviatoric, LinearReaction, AllDirichlet. Freefem-divs / FreeFem-mesh / LiteralPicard always had 1e‑7. | PARTIALLY CLOSED | Low — the canonical confirms the change is invisible. The 5 remaining sweeps are bookkeeping only. |
| ~~S2~~ | ~~Cocquet uses BAMG-adaptive corner-refined mesh, not uniform Delaunay~~                    | **CLOSED 2026-07-04.** Superseded by the mesh-topology settlement (§1): Phase 11 showed a *plain* unstructured Delaunay mesh (no adaptivity) already recovers $O(h^2)$, matching the paper's slope. Whether Cocquet additionally used BAMG adaptivity is no longer load-bearing for the settled *slope* conclusion; the residual *magnitude* gap is a measurement question (§1a), not a mesh-adaptivity one. | CLOSED  | —              |
| ~~S3~~ | ~~Low-frequency / global mode contaminates L²(u) but not H¹(u). Project u_h − u_ref onto P0~~ | **CLOSED 2026-07-04.** The global-mode hypothesis was ruled out for the load-bearing cases: S3a's domain-mean share χ_Ω is ≲5% for Galerkin P2/P1 and ASGS P2/P2 (§5.4a / the S3a detail below). With the slope explained by mesh-topology (§1) and the magnitude gap attributed to a measurement difference (§1a), there is no residual global mode left to diagnose. | CLOSED  | —              |
| ~~**S5**~~ | ~~Audit untouched Cocquet siblings~~ | **CLOSED 2026-05-26.** See §5.7. Three siblings (Alpha1 / Deviatoric / LinearReaction) exonerate all porous-physics knobs as cap sources. AllDirichlet implicates the **mixed-BC outlet corner**: P2/P2 slope jumps from ~1.3 to 2.46 when the corner is eliminated. LiteralPicard exonerates linearization choice. | CLOSED  | — |

~~**S4** (Galerkin-mode MMS to confirm unstabilized pathway is correct on smooth fields)~~ — **CLOSED**. The CocquetFormMMS test already includes a Galerkin P2/P1 leg as part of the three-way comparison. Result: L²(u) = 2.85×10⁻⁷ at h=1/320, slope ‑2.87 (optimal P2). See §5.6.

---

## 7 — What we have NOT yet verified (the frontier)

> **Mostly settled (2026-07-04).** With the slope explained as a structured-mesh artifact (§1) and the magnitude gap attributed to a measurement difference in Cocquet's unavailable solver (§1a), the only genuinely-open item is the external one (H-D, below). S2 and S3 are now CLOSED (see §6); their entries are kept here struck-through for provenance.

Organized by likely diagnostic value:

### The one genuinely-open item — requires external action
- **H-D — Cocquet's solver may iterate on the lifting `w = u − V` and plot the error in `w`.** Unfalsifiable from inside the repo (only the mesh-generation `.edp` is available). This — or "something beyond the original five" (a different reported functional, mislabelled figure axis, etc.) — is the residual explanation for the ~30–300× *magnitude* gap. Resolution needs Cocquet's actual solver source (corresponding-author email at hal-02561058 / supplementary-materials hunt). See §1a.

### CLOSED — folded into the mesh-topology settlement (2026-07-04)
- ~~**S3 — Spatial L²(u) decomposition of u_h − u_ref.**~~ CLOSED. The global-mode hypothesis was ruled out (S3a χ_Ω ≲5% on the load-bearing cases); no residual global mode remains once the slope is mesh-topology (§1) and the magnitude gap is a measurement question (§1a).
- ~~**S2 — Cocquet's mesh is BAMG-adaptive / corner-refined.**~~ CLOSED. A *plain* unstructured Delaunay mesh already recovers $O(h^2)$ (Phase 11); adaptivity is not load-bearing for the settled slope conclusion.

### Cheapest to settle — ALL CLOSED 2026-05-26
- ~~**O4**~~ — Cocquet plots absolute L²(u) (paper p. 32 `Err_tot` definition explicit).
- ~~**O7**~~ — All Galerkin Newton solves reach iter-4 residual ≤ 1.0×10⁻¹³, well past `ftol=1e-11`.
- ~~**O8**~~ — Galerkin path uses `SafeNewtonSolver` only; Anderson never fires on the Galerkin row.

### Bookkeeping (after the canonical re-run confirmed Δ is invisible)
- **S1 (partially closed) — Re-sweep the 5 remaining pre-retrofit siblings** (ModifiedCorner, Alpha1, Deviatoric, LinearReaction, AllDirichlet) for audit-trail consistency. The canonical structured variant was re-run 2026-05-26 with `physical_epsilon=1e-7` explicit and gave numbers bit-identical to 4 significant figures vs the pre-retrofit `1e-8` run — so the empirical effect is below 1e-10 on L²(u). The 5 remaining sweeps are pure bookkeeping; defer until the user wants AC-powered overnight compute.

### Hypotheses suggested by the audit but probably not worth testing
- **Inlet quadratic profile + P2 representation.** With c_in=0.5 the inlet is *exactly* representable by P2; no projection error.
- **Newton damping × Forchheimer interaction.** Already fixed by [lessons_learned.md](../lessons_learned.md) row 25 (τ-floor leakage); `h_floor_weight=0` is set in [paper_comparison.json:27](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L27).

### Suggested but speculative
- **Wall boundary layer at y=0** (porosity minimum, Forchheimer drag maximum). Could need anisotropic refinement. Would manifest as concentration of L²(u) error near y=0; testable by re-deriving the `localize_err.jl` decomposition with a different region partition (wall vs bulk vs corner). (`localize_err.jl` was removed in the CocquetTubeTest unification and is recoverable from git history — its recorded findings are preserved above in §4.3 / §5.3.) The S5 audit's LinearReaction sibling (which zeroes β·|u|·u while keeping the linear Darcy α(ε) drag) keeps the same slope as baseline, weakly suggesting the *nonlinear* part of the wall-layer is not the source — but the linear-Darcy wall layer itself is untouched and could still contribute.

---

## 8 — Document relationships and update protocol

- **This file** — **CANONICAL Cocquet reference; start here.** The single canonical doc: the settled conclusion (§1), the merged lateral-hypothesis verdicts (§1a), the supporting synthesis (§1b onward), and the folded-in hypothesis ledger (Appendix). Update it in place when a hypothesis crosses the open→closed boundary. (The earlier framings of this file as an "append-only ledger" and as a "replaceable point-in-time snapshot" are retired — those were two conflicting self-descriptions; the resolved role is the one canonical reference.)
- [docs/cocquet/convergence-analysis.md](convergence-analysis.md) — **historical phased-slope diary** (Phases 1–12; some early verdicts withdrawn — read via this synthesis).
- [docs/cocquet/replicating-cocquet-transcript.md](archive/replicating-cocquet-transcript.md) / [docs/cocquet/corner-singularity-transcript.md](archive/corner-singularity-transcript.md) — **archived raw transcripts** (provenance only; conclusions merged here).

**Open action items:** only H-D remains, and it is external (obtain Cocquet's solver `.edp`; see §1a / §7). All in-repo threads (O1–O8, S1–S5, S3a/S3b, H-A…H-E) are closed, parked, or confirmed paper-faithful.


---

## Appendix — hypothesis ledger (archived)

_Folded in from the former `cocquet_magnitude_investigation.md` chronological ledger; the conclusions are summarised above, this is the per-hypothesis growth log._

Started 2026-05-24 (this conversation). Persistent record of every hypothesis
checked while investigating why our absolute L²(u) errors against the Cocquet
benchmark are ~10²–10³× larger than the paper reports — despite slopes that
bend in the right direction.

**Bottom-line state (last updated 2026-05-24):**
- Slopes: our finest-segment Galerkin P2/P1 L²(u) reaches **2.05** (freefem-divs
  unstructured) — matches the paper's ~2.18.
- Magnitudes: paper reports L²(u) ~ **3×10⁻⁵** at N=10, **2×10⁻⁷** at N=100;
  we observe **9.68×10⁻³** at N=10 and **1.69×10⁻⁴** at N=100.
- The ratio is not a constant: it grows from ~320× at N=10 to ~850× at N=100,
  i.e. our convergence rate at coarse N is markedly lower than the paper's
  (we recover near-optimal only in the finest segment).
- This is consistent with either (a) our N=200 reference being unconverged
  relative to truth by ~1e-4, or (b) our solution being a different, less
  smooth flow than Cocquet's.

### Hypotheses ruled out (no further work needed)

> The paper-faithfulness hypotheses **H3–H12** below are the detailed growth-log prose; the same claims are summarised in compact table form in **§4.1**. The two are not redundant (table = quick reference, this = per-hypothesis provenance). H1 (FALSIFIED — ModifiedCorner made structured convergence ~3× worse) is the load-bearing experimental record and is kept in full here.

#### H1: Outlet-corner Dirichlet pin causes the convergence cap
- **Status:** FALSIFIED.
- **How tested:** `test/extended/CocquetTubeTest/data/modified_corner/` — entities 2,4
  removed from "walls" tag, releasing the outlet-corner DOFs.
- **Result:** Made absolute error 3-8× WORSE and slopes lower at every N. The
  released DOF takes an algebraically-driven value that pollutes the bulk field.
- **Recorded:** `memory/cocquet-modified-corner-experiment.md`.

#### H2: Structured Cartesian mesh's corner-aligned diagonals lock the singularity
- **Status:** PARTIALLY CONFIRMED — explains the slope but not the magnitude.
- **How tested:** `test/extended/CocquetTubeTest/data/unstructured_gmsh/` (gmsh Delaunay
  uniform 1/N divisions) and `test/extended/CocquetTubeTest/data/freefem_divisions/`
  (literal `buildmesh(a(N)+b(N)+c(N)+d(N))` N-per-side).
- **Result:** Unstructured Delaunay recovers near-optimal slopes (P2/P2 → 2.39 final
  segment uniform, 2.13 freefem-divs; Galerkin P2/P1 → 1.80 uniform, **2.05 freefem-divs**).
  But the absolute error magnitudes are still ~10³× higher than the paper.
- **Recorded:** `memory/cocquet-mesh-topology-controls-convergence.md`.

#### H3: ν = 1/Re mapping (Reynolds-number convention)
- **Status:** CONFIRMED — convention matches the paper.
- **Paper (Eq. 1, p. 3, also p. 4 line 1):** dimensionless DBF reads
  `−div(2 Re⁻¹ ε S(u) − ε u⊗u) + ε ∇p + α(ε)u + β(ε)|u|u = ε f` and "Re is the
  Reynolds number". The paper does NOT give an explicit `Re = ρUL/μ` formula;
  it uses Re only as the dimensionless coefficient of the viscous term.
- **Our code:** `weak_viscous_operator(u, v, α, ν) = 2 * α * ν * (ε(u) ⊙ ε(v))` with
  `nu = 1/Re = 0.002` (matches paper's `2 Re⁻¹ ε S(u):S(v)`).
- **Conclusion:** Both papers solve the SAME dimensionless equation with
  Re=500 ⇒ ν=0.002. No conversion / scaling discrepancy.

#### H4: Forchheimer closure law (Eq. 49)
- **Status:** CONFIRMED — porosity weighting is applied correctly.
- **Paper:** `α(ε) = (150/Re)·((1-ε)/ε)²`, `β(ε) = 1.75·(1-ε)/ε`.
- **Our code:** `src/models/reaction.jl::sigma` computes
  `a_term = sigma_linear * ((1-α_porosity)/α_porosity)²` with
  `sigma_linear = 0.30 = 150/500` and
  `b_term = sigma_nonlinear * (1-α_porosity)/α_porosity` with
  `sigma_nonlinear = 1.75`. Variable-name collision: our code's `α` is the porosity
  field (paper's `ε`); paper's `α(ε)`, `β(ε)` are the porosity-dependent drag
  coefficient *functions*.
- **Spot check:** at ε=0.45 both formulas yield linear coefficient 0.448 and
  nonlinear coefficient 2.139. At ε=1 both yield 0 (recover NS).

#### H5: Outlet boundary condition (ε-weighted traction-free)
- **Status:** CONFIRMED.
- **Paper (Eq. 2):** `ε(2 Re⁻¹ S(u) − p)·n = 0` on Γ_out.
- **Our code:** weak form integrates `∫ε σ:∇v dx` by parts ⇒ natural BC
  `εσ·n = 0` automatically on Γ_out. No explicit boundary integral added.

#### H6: Convective form
- **Status:** CONFIRMED.
- **Paper:** `c(ε; u, v, w) = ∫ ε (u·∇)v · w dx + ∫ β(ε) |u| v·w dx`.
- **Our code:** `conv_term = v ⋅ (α_porosity * (∇(u)' ⋅ u))` is `v·(ε(u·∇)u)`.

#### H7: Mass equation / pressure-divergence form
- **Status:** CONFIRMED.
- **Paper:** `b(ε; v, q) = -∫ q · div(εv) dx`, and mass eq `div(εu) = 0`.
- **Our code:**
  - `pres_term = -p * (α*(∇·v) + ∇(α)·v) = -p·div(εv)`
  - `mass_term = q * (physical_epsilon*p + α*(∇·u) + u·∇(α)) = q · (physical_epsilon·p + div(εu))`
  - `physical_epsilon=1e-7 = paper's η`.

#### H8: Pressure penalty η
- **Status:** CONFIRMED.
- **Paper (p. 30):** `η = 1e-7`. Our config: `physical_epsilon = 1e-7`.

#### H9: P₁ porosity interpolant for Galerkin
- **Status:** CONFIRMED for the Galerkin comparison row.
- **Paper:** ε_h is the P₁ FE interpolant of ε regardless of velocity order.
- **Our code:** `comparison_runs` for Galerkin sets `porosity_order=1` explicitly.

#### H10: Domain, inlet profile, porosity profile
- **Status:** CONFIRMED.
- **Paper:** Ω=(0,2)×(0,1), Γ_in={0}×[0,1], Γ_out={2}×[0,1],
  Γ_w=[0,2]×({0}∪{1}); inlet `u_in(y) = c_in·y(1-y) e_x` with c_in=0.5 for Fig 2;
  porosity `ε(y) = 0.45 + 0.55·exp(y-1)`.
- **Our config:** identical (`bounding_box: [0,2,0,1]`, `alpha_func(x) = 0.45 + 0.55*exp(x[2]-1.0)`).

#### H11: Self-reference via N=200 finest mesh
- **Status:** CONFIRMED methodology-equivalent.
- **Paper (p. 32):** "We note (u_ex, p_ex) the solution obtained with N=200 and we
  compute the error between the discrete solution for N ≤ 100 and (u_ex, p_ex)."
- **Our code:** `build_solver(N_ref, ...)` with N_ref=200 in `paper_comparison*.json`.

#### H12: Taylor-Hood P2/P1 unstabilized
- **Status:** CONFIRMED for the Galerkin row.
- **Paper:** uses Taylor-Hood P2/P1, no stabilization.
- **Our code:** `comparison_runs` Galerkin row sets `k_velocity=2, k_pressure=1`;
  `galerkin_driver.jl` invokes the assembly with `mult_mom=mult_mass=0` ⇒
  no τ-stabilization terms.

### Open hypotheses to test next

#### O1: ~~Our N=200 reference itself is under-converged against the truth solution~~ — INVALIDATED
- **Status:** RULED OUT BY ARGUMENT (Guillermo, 2026-05-24).
- **Why ruled out:** Self-reference at N=200 is the SAME methodology Cocquet uses.
  If our reference at N=200 were under-converged against truth, Cocquet's reference
  at N=200 would be under-converged by the same amount, and their reported coarse
  errors would plateau at the same level as ours. They don't (they hit 2e-7 at N=100).
  So the gap is not about reference noise floor — it's that our DISCRETE SOLVER
  produces a less accurate solution at every N (including at N=200), not that the
  self-reference methodology is biased.
- **Reframed as:** "What in our discrete formulation/implementation makes our
  solution at any N less accurate against the continuous truth than Cocquet's?"
  (This is now an umbrella for hypotheses O3, O5, and any new ones found.)

#### O2: `u_base_floor_ref = 1e-4` regularization adds a noise floor — RULED OUT
- **Status:** RULED OUT 2026-05-24 (probe `/tmp/probe_with_floor_off.jl`,
  log `/tmp/probe_floor_off.log`). Re-ran a single Galerkin P2/P1 N=80 + N=200
  reference solve with `u_base_floor_ref=0` and `epsilon_floor=0` overrides.
  L²(u) consistent at N=80 was **2.6685e-4** — identical to 4 significant figures
  to the existing HDF5 value 2.669e-4 (with the inherited 1e-4 default).
  Newton convergence trace was bit-identical at iter-0 residual `0.001164580518`,
  confirming the floor never actually fires on this flow's discrete state
  (true |u| > 1e-4 everywhere except at wall vertices where u=0 exactly).
- **Conclusion:** The `u_base_floor_ref=1e-4` default is not the source of the
  magnitude gap.

#### O3: Solution magnitude / spatial profile may differ from Cocquet's — PARTIALLY RULED OUT
- **Status:** Probe ran 2026-05-24 (`/tmp/probe_cocquet_reference.jl`, log
  `/tmp/probe_cocquet_ref.log`). Flow magnitudes look correct and physically
  reasonable:
  - `‖u‖_L²(Ω) = 0.125` (consistent with c_in/4 inflow scale).
  - Inlet `u(0, 0.5) = (0.125, 0)` — exact.
  - Centerline u_x decreases 0.125 → 0.102 along the channel (Darcy drag).
  - Cross-stream profile at x=1 is asymmetric about y=0.5 (peak at y=0.75)
    because the porosity ramps from 0.45 at y=0 to 1.0 at y=1, so flow
    diverts to the high-porosity top half. Physically correct.
  - Newton converged to residual 3.7e-15 in 4 iterations.
- **Conclusion:** Our discrete equation IS being solved exactly, and we ARE
  solving the right PDE. The flow Cocquet solves is the same flow we solve.
  The magnitude gap is **pure discretization error in our specific
  implementation** — not solver failure, not wrong PDE, not wrong scale.
- **Confirmed:** The ~3.4e-4 L²(u) error at N=80 is **1700× worse than the
  P1-porosity O(h²) cap** predicts. Cocquet's 1e-7 at N=100 is **3000× better
  than the cap**, suggesting they're nearly at the unrestricted optimal P2 rate.
- **Still to find:** what extra error source our implementation has that
  Cocquet's doesn't.

#### O4: Cocquet may report a RELATIVE L²(u) error, not absolute
- **Why suspect:** Standard convention in some FEM papers is to plot
  `‖u_h - u_ref‖_L² / ‖u_ref‖_L²`. If our absolute L²(u) at N=100 is 1.69e-4
  and our `‖u_ref‖_L² ≈ 0.06`, our relative error is 2.8e-3 — still 14000×
  the paper's 2e-7, so this alone doesn't close the gap. But worth ruling
  out the convention.
- **Test:** Same as O3 — the probe script also reports `‖u_ref‖_L²` so we can
  divide our errors by it and see.

#### O5: Quadrature degree may be insufficient for the Forchheimer non-polynomial term — RULED OUT
- **Status:** RULED OUT 2026-05-24 (probe `/tmp/probe_quad_plus12.jl`,
  log `/tmp/probe_quad12.log`). The default quadrature degree for k_v=2
  with ForchheimerErgunLaw is **9** (from `min_quadrature_degree = 4*k_v + k_v÷2`).
  Re-ran the N=200 reference with `Measure(Ω, 21)` (bump +12) and compared
  the iter-0 PDE residual against the baseline default quadrature:
  - baseline iter-0: `9.588812700822735e-4`
  - quad+12 iter-0: `9.588812745812068e-4`
  - delta: ~5e-9, i.e. quadrature error of the Forchheimer term is at the
    1e-9 level — five orders of magnitude below our 1e-4 L²(u) error.
- **Conclusion:** Quadrature is already adequate. Bumping the degree could
  not change the final L²(u) by more than ~5e-9, vastly below the 845× gap
  to the paper. The Forchheimer non-polynomial integrand is not the source.
- **Probe was stopped after this diagnosis** to free CPU; no need to wait
  for the full solve.

#### O6: Pressure mean-zero gauge / hydrostatic constant — PARKED (2026-07-04)
> **PARKED** by the mesh-topology settlement (§1) + the ~30–300× reading (§1a): the residual gap O6 was meant to explain is now attributed to a measurement difference in Cocquet's unavailable solver (H-D / beyond-the-five), and the global-mode pathway a gauge offset would drive was ruled out by S3a (χ_Ω ≲5%). Untestable without their `.edp`.
- **Why suspect:** Pressure is only defined up to a constant under the natural-BC
  outlet. We add `η p` penalty to fix it. Cocquet may use a different gauge
  (Lagrange multiplier, zero-mean projection, fix-a-point). This affects p
  by an O(1) constant offset but not its gradient, so should not affect L²(u).
  Worth confirming we're not double-fixing or mis-fixing.
- **Test:** Inspect ‖p_h - p_ref‖_L² behavior with and without explicit zero-mean
  projection of (p_h - p_ref) before taking the norm.

#### O7: Newton convergence may stall short of machine precision
- **Why suspect:** `xtol=1e-11, ftol=1e-11` in the config. Should be tight enough,
  but worth confirming Newton actually reached these tolerances and the residual
  is at machine precision at every solve. A Newton stagnation at 1e-5 would
  obviously add ~1e-5 to absolute error.
- **Test:** Inspect solver iteration logs from the recent full sweep — confirm
  final residuals are < 1e-10 for all solves.

#### O8: Anderson acceleration mode interaction
- **Why suspect:** Config sets `accelerator.type = "Anderson", m = 5`. We invoke
  the Newton solver explicitly for Galerkin (not Picard), so Anderson should not
  fire. But verify the cascade isn't introducing artifacts.
- **Test:** Confirm in solver logs that all Galerkin runs use Newton (mode), not
  Picard.

#### S3a: Domain-mean share χ_Ω — RULED OUT for the load-bearing Galerkin P2/P1 case (2026-05-29)
- **Outcome:** For the Galerkin P2/P1 row (the actual paper-comparison case),
  `χ_Ω(u)` is **0.017, 0.034, 0.046, 0.049** across `N = 10, 20, 40, 80`
  (`results/structured/convergence.h5`, group `Galerkin/P2P1`). The L²(u)
  cross-mesh error is essentially zero-mean (≲ 5 %); the 845 × magnitude gap to
  the paper is *not* explained by a rigid domain-wide offset surviving
  self-reference. The ASGS P2/P2 row is the same picture: `χ_Ω(u) ≤ 0.028`
  for every N. Both are P2 velocity spaces, which represent the parabolic
  inlet Dirichlet `c_in · y · (1 − y)` exactly, so there is no
  Dirichlet-interpolation surplus to leak into the mean.
- **Where χ_Ω did fire — ASGS P1/P1 inlet-interpolation error:** the same
  probe reports `χ_Ω(u) = 0.752, 0.797, 0.772, 0.605` for ASGS P1/P1. The
  P1 velocity space cannot represent the parabolic inlet exactly; the
  Dirichlet-interpolation error scales with `h²` per cell, so the coarse and
  reference meshes inject *different* mean fluxes through `Γ_in` and the
  difference carries a non-trivial domain mean. χ_Ω drops at `N = 80`
  because the residual interpolation error is small enough there that it no
  longer dominates the (already small) total error.
- **Methodological takeaway, and a correction:** the cell-average share
  `‖P₀ e_h‖² / ‖e_h‖²` is *not* a rigid-offset diagnostic on its own — it
  tends to 1 generically for any smooth converging FE error because piecewise
  constants approximate smooth fields well on small cells. The right
  discriminator is the domain-mean share `χ_Ω`, which is zero for any
  zero-mean error and only saturates when the error is a true rigid offset.
  A prior reading of the cell-average signal as "global mode confirmed" was
  overconfident; χ_Ω disagrees on the load-bearing case.
- **Status:** open hypothesis closed for Galerkin P2/P1 and ASGS P2/P2.
  Re-route the magnitude-gap investigation to non-rigid bulk-distributed
  modes — the corner-excluded L²(u) (S3b) accounts for ~44 % of the squared
  error at N=80 for Galerkin P2/P1, leaving roughly half the gap in a
  zero-mean, non-corner-localised "bulk" component that neither S3a nor
  S3b explain.

#### S3a (legacy header for context — superseded by the result above)
- **Why suspect:** The cross-norm comparison above noted that L²(u) is
  disproportionately off relative to H¹(u) for the same solution pair —
  consistent with something specific to L²(u). A *globally-applied* BC,
  gauge, or boundary-flux bias is invisible to self-reference (it appears
  identically in u_h and u_ref, so cancels in their difference); but a
  finite-h dependent rigid offset that scales differently between the
  coarse and reference mesh can survive. The probe is whether e_h carries
  a non-trivial domain-wide constant component.
- **Probe (self-reference-invariant):**
  `χ_Ω := |Ω| · ‖ē_Ω‖² / ‖e_h‖²` with `ē_Ω = (∫_Ω e)/|Ω|`.
  χ_Ω → 1 only when e_h is a rigid domain-wide offset, χ_Ω → 0 for any
  zero-mean error. Computed by
  `compute_mode_decomposition(f_h, if_ref, V_free, dΩ_h)` in
  [src/metrics.jl](../../src/metrics.jl); persisted to HDF5 as
  `chi_Omega_u`, `chi_Omega_p` by
  [run_convergence.jl](../../test/extended/CocquetTubeTest/run_convergence.jl).
- **Companion diagnostic (does NOT discriminate S3 on its own):**
  the cell-average share
  `fraction_cellavg := ‖P₀ e_h‖² / ‖e_h‖²` is computed and stored in the
  same call (`cellavg_frac_u`, `cellavg_frac_p`). It tends to 1 generically
  as `h → 0` for any C¹ error because piecewise constants approximate
  smooth functions on small cells, so a high value is the expected
  signature of a converging FE difference, not of a pathological global
  mode. The within-cell-share signal (1 − fraction_cellavg) is useful for
  contrasting P₁ and P₂ velocity spaces but should not be read as evidence
  for a gauge/BC bias.
- **Interpretation:**
  - χ_Ω ≳ 0.5, stable across N: a finite-h rigid offset survives
    self-reference → pivot to **O6** (pressure-gauge probe) and to whatever
    couples gauge to velocity in this discretisation.
  - χ_Ω ≲ 0.05 across N: error has effectively zero mean → S3 ruled out
    as the dominant source of the magnitude gap; weight shifts to
    discretisation-level mechanisms that don't average down (corner-localised
    content in S3b, or non-constant low-frequency modes not captured by χ_Ω).
  - Intermediate (0.05–0.5): partial contribution; report and pair with the
    corner-excluded decomposition in S3b.

#### S3b: Corner-localised error decomposition (corner-excluded L² norm) — CLOSED (2026-07-04)
> **CLOSED / parked** by the mesh-topology settlement (§1): the corner-localisation is the *structured-mesh* manifestation of the cap; on an unstructured Delaunay mesh the corner content drops to ~1% and the slope recovers to $O(h^2)$. The residual magnitude gap is a measurement question (§1a), not a corner-decomposition one. Probe definition kept for provenance.
- **Why suspect:** The earlier localised analysis (item 3 in the queue, line 240)
  found that on the structured Cartesian-simplexified mesh **71 %** of the
  squared error is concentrated within 0.1 of the two outlet–wall corners
  `(L_max, 0)` and `(L_max, y_max)`. On unstructured Delaunay the same
  fraction drops to ~1 %, but the total ‖e‖ grows. A systematic R-sweep on
  the structured mesh quantifies *how much* of the magnitude gap is corner-
  bound and at what radius it saturates.
- **Probe:** `compute_corner_excluded_norm(...; corners, R)` (added to
  [src/metrics.jl](../../src/metrics.jl)). Thin wrapper around
  `compute_reference_errors` that AND's the existing `bounding_rule` with an
  exclusion mask `x ↦ ⋂_c ‖x−c‖ > R`. Run for R ∈ {0.05, 0.1, 0.2}, with
  outlet corners derived from the bounding box (not hard-coded), and persist
  as `l2_eu_corner_excl[N_idx, R_idx]` and the same for H¹ / p. R=0.1 is
  plotted by default; the other radii are stored for analysis.
- **Interpretation:**
  - If excluding R=0.1 collapses the magnitude gap to ≲ 10×: corner pollution
    confirmed at the discrete level → pivot to **S2** (BAMG / adaptmesh near
    the corner) or to a corner-graded mesh experiment.
  - If the gap persists at R=0.2: error is diffuse — not corner-localised —
    weakening the corner-pollution reading and pointing back at S3a's
    global-mode result.

Both probes operate on the existing solver outputs (no new sweeps); the
reference solution at N=200 is already computed in-memory per run
([run_convergence.jl:257–266](../../test/extended/CocquetTubeTest/run_convergence.jl#L257-L266)).
Output is appended to the existing HDF5 file per group; the plotter
[plot_convergence.py](../../test/extended/CocquetTubeTest/plot_convergence.py)
emits a third subplot showing the cell-average fraction (linear, 0..1) on
the primary axis and the R=0.1 corner-excluded L²(u) (log) on a twin axis,
for each (method × element pair). Older HDF5 files without these keys are
detected at plot time and the diagnostic subplot is hidden — the existing
L²/H¹ subplots are unchanged.

### Cross-norm comparison reveals the L²(u) ratio is anomalously large

At N=100 Galerkin P2/P1 freefem-divs:
- our L²(u) = 1.69e-4 vs paper ~2e-7 → **845×** gap
- our H¹(u) seminorm = 2.13e-2 vs paper Err_tot ~3e-3 → **7×** gap
- our L²(p) = 8.6e-5 vs paper L²(p) component (subset of 3e-3) → likely small

The L²(u) discrepancy is **120× worse than the H¹(u) discrepancy** for the same
solution pair. This is a striking pattern: if our solution were uniformly less
accurate, we'd expect comparable ratios across norms. The fact that L²(u) is
disproportionately off suggests something specific to the L²(u) measurement
(e.g., a global mode in u_h - u_truth that doesn't show up in gradients), not
a uniformly worse discrete solution.

### What we've NOT yet looked at (queue, in priority order)

1. ~~The CocquetFormMMS test~~ — DONE 2026-05-24. `cocquet_form_mms.h5`,
   config_1_ASGS (P2/P1) at h=1/80 (corresponds to N=80 in Cocquet sweep):
   - L²(u) = **8.54e-7**, slope ~3.3 (optimal P2)
   - Cocquet's reported L²(u) at N=100 is ~2e-7 — **same order of magnitude
     as our MMS at N=80**.
   - **OUR CODE CAN ACHIEVE THE PAPER'S MAGNITUDES**. The implementation is
     not bugged. The Cocquet benchmark specifically inflates our error by
     ~300× relative to what the code can deliver on smooth problems.
2. ~~Literal FreeFem `.msh` files via paper_comparison_freefem.json~~ — DONE
   2026-05-24. With Cocquet's exact mesh recipe (generated by
   create_paper_meshes.edp via FreeFem on a machine that has the binary),
   Galerkin P2/P1 L²(u) at N=100 is **4.59e-5** — closer than our gmsh
   Delaunay (1.69e-4 freefem-divs, 8.10e-5 uniform-divs), but still **230×**
   the paper's reported 2e-7. So even with their EXACT mesh, we don't match
   their magnitudes.
3. ~~Localized error analysis via `localize_err.jl`~~ — DONE 2026-05-24
   (log `/tmp/localize_err.log`; `localize_err.jl` was removed in the
   CocquetTubeTest unification and is recoverable from git history — the
   finding below stays recorded). Comparing structured Cartesian-simplexified
   vs gmsh Delaunay uniform-divs at N_coarse=10 vs N_ref=80, Galerkin P2/P1:
   ```
                          outlet_corner  inlet_corner  wall_strip  bulk    ‖e‖
   STRUCTURED          :  71.0%          0.10%         3.6%        25.3%   3.19e-4
   UNSTRUCTURED        :   1.08%         29.4%         7.7%        61.8%   5.74e-3
   ```
   - **Structured mesh's error is dominated by the outlet corner** (71% of ‖e‖²).
   - **Unstructured mesh diffuses the outlet corner singularity** (only 1.08%
     of ‖e‖² there) — confirms the Gemini hypothesis from earlier in the
     conversation that BAMG/Delaunay node placement near the corner avoids
     locking in the singularity.
   - BUT: total ‖e‖ on unstructured is **18× LARGER** than on structured at
     N=10. The unstructured mesh's bulk error is large because coarse
     Delaunay triangulations have worse triangle quality / less alignment
     with the parabolic channel flow.
   - Reaches Cocquet's reported 3e-5 at N=10: NEITHER mesh reaches it. The
     bulk error contribution (1.72e-4 even on structured after truncating
     outlet corners) is still ~6× the paper's reported value.
   - Even excluding the outlet corner entirely from the error norm doesn't
     close the gap. Cocquet's discretization apparently produces a fundamentally
     more accurate bulk solution at every N — even at our most favorable
     mesh (structured-with-truncated-corners).
4. Try `outlet_truncation_delta = 0.1` (excludes the strip x > 1.9 from the
   error norm). If the bulk-only L²(u) drops to ~paper-level, the gap is
   entirely from corner contamination — and Cocquet must somehow be
   excluding the corner from their error norm too (or their corner region
   produces a much smaller error).
5. Try quadrature `degree + 12` (all_dirichlet diagnostic style). If
   absolute error drops, our default quadrature is under-integrating the
   non-polynomial Forchheimer term.
6. ~~Re-verify Cocquet's Figure 2 right-panel y-axis labels~~ — DONE
   (Guillermo, 2026-05-24). Y-axis range on Figure 2 right panel
   (L² error of velocity, Re=500, c_in=0.5) is confirmed
   **10⁻⁴ (top tick) → 10⁻⁷ (bottom tick)**. The curve goes from ~3e-5 at
   N=10 down to ~2e-7 at N=100. Figure 3 (Re=1000, c_in=1) right panel
   y-axis goes 10⁰ (top) → 10⁻⁵ (bottom labelled tick, actual data points
   extend down to ~10⁻⁸). So Cocquet's reported magnitudes ARE 10⁻⁴ to 10⁻⁸
   range — orders of magnitude below ours.

### How to use this ledger

When investigating the Cocquet absolute-magnitude issue, **start by reading
this file** to avoid re-testing hypotheses already ruled out. Update it
in-place when:
- A hypothesis is ruled out (move from "open" to "ruled out").
- A new hypothesis is suspected (add to "open").
- A diagnostic experiment finishes — record the result inline.
