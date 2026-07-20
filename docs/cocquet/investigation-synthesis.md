# Cocquet absolute-magnitude investigation — synthesis snapshot (2026-05-26)

> **CANONICAL Cocquet reference — start here.** This is the single canonical doc for the Cocquet tube-flow benchmark. Companions in this folder: the phased slope investigation [convergence-analysis.md](../archive/cocquet-convergence-analysis.md), the (unsent) [email-questions.md](../archive/cocquet-email-questions.md), and the archived raw transcripts [replicating-cocquet-transcript.md](../archive/cocquet-replicating-transcript.md) / [corner-singularity-transcript.md](../archive/cocquet-corner-singularity-transcript.md). (`findings.md` §6 is the *different* CocquetFormMMS sibling; this file is the sole canonical home for the tube-flow verdict.)

## 1 — Executive summary — SETTLED conclusion

**The convergence cap is MESH-TOPOLOGY.** The sub-optimal Cocquet convergence rate on our runs is a **structured-mesh artifact**: a structured Cartesian-simplexified mesh with edges aligned to the 90° no-slip/traction outlet corner **locks in the corner singularity**, capping the velocity-L² slope. An **unstructured Delaunay mesh (FreeFem `buildmesh` paradigm) recovers $O(h^2)$**, matching Cocquet's Fig. 2. This settles the *slope* question and **withdraws** the earlier Phase 9 "proven mesh- and method-independent corner singularity" and Phase 10 "the paper's $H^3{\times}H^2$ claim is incorrect" verdicts — both rested on the structured mesh (see [convergence-analysis.md](../archive/cocquet-convergence-analysis.md) Phase 11).

**The residual is a MAGNITUDE gap of ~30× (coarse) to ~300× (fine).** After Phase 11–12's literal-FreeFem-mesh reproduction and the corrected figure reading (the [replicating-cocquet](../archive/cocquet-replicating-transcript.md) session), the honest gap on the Cocquet Galerkin P2/P1 row is **~30× at N=10 and ~300× at N=100** — i.e. our slope is right but the curve sits on a different baseline. The magnitude gap is most plausibly a *measurement* difference in what Cocquet plots (see §1a), not a bug in our code.

**Cocquet's reported numbers** (Fig 2 right, Re=500, c_in=0.5, N=200 self-reference): L²(u) ≈ 3×10⁻⁵ at N=10, ≈ 2×10⁻⁷ at N=100. Our best Galerkin P2/P1 (freefem-divs unstructured mesh) gives 9.68×10⁻³ at N=10 and 1.69×10⁻⁴ at N=100. Finest-segment slope matches: ours ‑2.05 (N=40→80), paper ‑2.18.

---

## 1a — Lateral-hypothesis verdicts (merged from the 2026-05-24 replicating-cocquet session)

Five lateral hypotheses were raised to explain the ~30–300× magnitude gap on the freefem-divs mesh. Four were tested and discarded *from inside the repo*; one is parked pending external code. See the archived [replicating-cocquet-transcript.md](../archive/cocquet-replicating-transcript.md) for the full working.

- **H-A (paper plots SQUARED L² norms).** DISCARDED. Plotting `‖e_u‖²` crosses the paper curve between N=20 and N=40 (above at coarse N, below at fine N) and gives slope −3.52 vs the paper's −2.30 — neither the ratios nor the slope match. Squaring is not the explanation.
- **H-B (Cocquet's Picard stopped short of the discrete fixed point at Re=500, c_in=0.5).** DISCARDED. A literal pure-Picard sibling (`CocquetTubeTest/data/literal_picard`, itmax=10, tol=1e-10, no Newton) lands on the true discrete fixed point in 7–10 iterations; its L²(u) is bit-identical (≤0.018%) to the converged-Galerkin solution at every N. Cocquet's stopping rule applied literally closes none of the gap. (Also corrects the earlier "squared-norm" reading of the paper's `‖·‖_{L²(Ω)²}` stopping-rule notation: it is the vector-valued L², not a squared norm.)
- **H-C (they measure `‖I_h u_ref − u_h‖`, a trial-space best-approximation projection, not `‖u_ref − u_h‖`).** DISCARDED. A trial-space-projection metric agrees with the free-DOF metric to ≤1.4% on the load-bearing Galerkin P2/P1 case (the P2-exact inlet + zero walls bound the difference to cross-mesh boundary-node noise). It cannot close even 1% of the gap.
- **H-D (their FreeFem solver iterates on the lifting `w = u − V` and reports the error in `w`, not `u`).** PARKED — unfalsifiable from inside the repo (only the mesh-generation `.edp` is available, not Cocquet's solver `.edp`). Requires external action (corresponding-author email / supplementary-code hunt). Note `V` is P2-exact for the quadratic inlet, so a *divergence-free* lifting would leave the error unchanged; only an h-dependent approximate lifting could matter.
- **H-E (a factor-of-2 in `S(u)` ⇒ effective ν doubled ⇒ smoother solution).** DISCARDED by code reading. `viscous_operators.jl` uses `2·α·ν·(ε(u)⊙ε(v))` with `ε(u)=½(∇u+∇uᵀ)`, matching the paper's `2Re⁻¹ S(u)` with `S(u)=½(∇u+∇uᵀ)`. Effective ν is genuinely `1/Re`, not `2/Re`. No mismatch.

**Net:** nothing we can change in our code, solver, metric, or viscosity convention closes any meaningful part of the ~30–300× magnitude gap. The remaining viable explanation is a *measurement* difference in what Cocquet's (unavailable) solver plots (H-D or something beyond the original five), not a defect in our discretization.

---

## 1b — Supporting findings (pre-settlement synthesis)

> These paragraphs pre-date the mesh-topology settlement (§1) and the ~30–300× correction (§1a). They remain accurate as **sub-findings** — the S5 sibling audit and the CocquetFormMMS exculpation are real — but note that "Dominant cap mechanism" below is now understood as *the structured-mesh manifestation* of the corner singularity: on an unstructured mesh the slope cap disappears (see §1).

**What we know.** The code is a term-by-term faithful transcription of Cocquet's weak form (verified — §4 below). The Newton solve converges to machine-precision residual (3.7×10⁻¹⁵). On a manufactured solution with the **same** Forchheimer/porosity operators **and the same unstabilized Taylor-Hood P2/P1 Galerkin formulation** ([CocquetFormMMS three-way comparison](../../test/extended/CocquetFormMMS/)) we hit L²(u) = 2.85×10⁻⁷ at h=1/320 with slope 2.87 (optimal P2) — i.e., the code **can** reach paper magnitudes on smooth problems, **on the exact same Galerkin pathway used by Cocquet**. The remaining Cocquet-benchmark gap is therefore a **non-smoothness × discretization interaction**, not a solver bug or a stabilization-vs-Galerkin artifact.

**Dominant cap mechanism identified (S5 audit, §5.7).** A side-by-side sweep of five one-knob siblings shows:

- Removing the **porous drag entirely** (Alpha1), switching to a **deviatoric viscous operator** (Deviatoric), and **zeroing the Forchheimer β·\|u\|·u nonlinearity** (LinearReaction) all leave the P2/P2 slope stuck at ~1.26–1.38 — three independent exonerations of porous-physics knobs.
- Replacing the natural traction-free outlet with **all-Dirichlet BCs** (AllDirichlet) jumps P2/P2 slope from ~1.27 to **2.46**.
- Picard vs Newton linearization is irrelevant (LiteralPicard).

→ The convergence cap is driven by the **mixed-BC outlet corner singularity** at `(2,0)` and `(2,1)`, where Dirichlet walls meet the Neumann outlet. This also explains the localize_err finding (71% of structured-mesh L²(u)² lives at the outlet corner).

**The remaining gap.** Even AllDirichlet's best run (P2/P2 N=80, L²(u) = 4.7×10⁻⁶) is still **~12× the paper's ~4×10⁻⁷** at N=80. So eliminating the corner singularity peels off the slope component of the gap, but a residual magnitude factor persists. All currently-tested specific mechanisms (H1–H12, O1–O8, S2–S5) and the five lateral hypotheses (H-A…H-E, §1a) have been falsified, confirmed paper-faithful, or closed by the mesh-topology settlement. **The only genuinely-open thread is external:** the ~30–300× magnitude gap is now attributed to a *measurement* difference in Cocquet's unavailable solver (H-D / beyond-the-five, §1a), not to any knob in our code. The earlier "L²(u)/H¹(u) low-frequency global mode" candidate (§5.4) was ruled out by S3a (χ_Ω ≲5%).

---

## 1c — 2026-07-19 update: Frontal-Delaunay mesh + always-on interpolation-error floor

Two additions this date, both of which **confirm and quantify** the §1 settlement rather than change it.

**(1) Best-quality unstructured mesh (`unstructured_frontal`, `mesh_algorithm=6`).** A mesh-quality screen
showed gmsh Frontal-Delaunay (alg 6) produces near-equilateral triangles (gamma quality mean 0.998,
min 0.87, sliver tail gone) vs plain Delaunay (alg 5) mean 0.949 / min 0.65, with *fewer* cells; BAMG
(alg 7) is unusable here (482 huge triangles at N=40 — reproduces the Phase-12 "half the domain in one
triangle"). On the Frontal mesh **all three methods recover the optimal O(h²) L²(u) slope** at the finest
segment (Re=500, c_in=0.5, N∈{10,20,40,80,100}, ref N=200): Cocquet Galerkin P2/P1 → **2.20**, ASGS
P2/P2 → **2.27**, ASGS P1/P1 → **2.33** (vs the structured cap of ~1.1–1.5). The Taylor-Hood and our
VMS P2/P2 are neck-and-neck in magnitude (L²(u) @N=100 = 7.94e-5 vs 9.24e-5).

**(2) The interpolation-error floor is now computed on every run** (`interp_{l2,h1}_{u,p}`, the MMS
practice: `‖u_ref − I_h u_ref‖`, same consistent metric as the FE error). It sharpens two prior
qualitative claims into numbers:

- **The structured L² cap is corner POLLUTION through the solve, NOT a representation limit — the
  Frontal mesh lifts it.** The floor separates the two mechanisms per norm (numbers = Galerkin P2/P1):
  - **L²**: the *structured* best-approximation floor is near-optimal (interp slope **1.7–1.9**), yet the
    structured FE L² slope is only ~1.5 with efficiency FE/interp **GROWING 1.2→3.8** — the corner
    singularity, locked by the corner-aligned diagonals, pollutes L² through the solve. On *Frontal* the
    FE L² slope recovers to ~2.2 with efficiency **SHRINKING 30→9** (the isotropic mesh does not lock the
    corner). The two meshes *represent* the solution comparably at coarse N (floor 2.56e-4 structured vs
    1.82e-4 Frontal at N=10); the ~18× magnitude difference there is almost entirely a solve-efficiency
    gap (eff 1.2 vs 30), not a representation gap.
  - **H¹ (energy)**: on the *structured* mesh the FE sits on its floor (eff **≈1.15, constant**) and BOTH
    are capped at slope **~0.78** — the corner singularity genuinely limits the H¹ best-approximation on a
    quasi-uniform mesh; only corner-graded refinement would move it. *Frontal* FE H¹ reaches ~1.30
    (eff 7→1.7; its H¹ floor is pre-asymptotic too, slope 0.4–0.8). Pressure efficiency ≈1 on both.
  - **Mesh QUALITY is not the magnitude lever**: Frontal (near-perfect quality) gave essentially the same
    magnitude as plain Delaunay (5.52e-3 vs Phase-11 5.86e-3 at N=10). (Correction to a first reading that
    said "structured eff ≈1": that holds only for H¹ and only at coarse N in L² — the L² efficiency grows.)
- **The magnitude gap is reinforced as a measurement difference in Cocquet's solver.** Our Frontal
  interpolation floor is itself **5–56× above Cocquet's reported L²(u)** (1.82e-4 vs ~3.7e-5 at N=10;
  8.48e-6 vs ~1.5e-7 at N=100). A Galerkin solution cannot beat its own best interpolant, so Cocquet's
  numbers sitting *below our achievable best-approximation floor* means the residual ~150–530× gap cannot
  be a defect in our discretization — it is a representation/measurement difference (H-D, §1a/§7).

Canonical result files: `test/extended/CocquetTubeTest/results/unstructured_frontal/` (config +
`convergence.h5` + `convergence.png`). Paper Fig-2 magnitudes re-verified from 400-DPI PDF crops:
L²(u) ≈ 3.7e-5 (N=10) → ~1.5e-7 (N=100, floor); Err_tot ≈ 4.5e-4 → ~3.5e-6, slope ≈ −2.

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
| **Structured Cartesian — Galerkin P2/P1** (canonical) | **2.31×10⁻⁵ at N=80** (58× paper @ N=80 ~4×10⁻⁷) | **‑1.22** | [test/extended/CocquetTubeTest/results/structured/convergence.h5](../../test/extended/CocquetTubeTest/results/structured/convergence.h5) |
| Structured Cartesian — ASGS P2/P2 | 9.19×10⁻⁶ at N=80 (23× paper) | ‑1.64 | same h5 |
| Structured Cartesian — ASGS P1/P1 | 8.84×10⁻⁵ at N=80 | ‑1.82 | same h5 |
| AllDirichlet — ASGS P2/P2 (no mixed-BC corner) | 4.70×10⁻⁶ at N=80 (12× paper) | ‑2.46 | [test/extended/CocquetTubeTest/results/all_dirichlet/](../../test/extended/CocquetTubeTest/results/all_dirichlet/) |
| **MMS — Galerkin P2/P1 unstabilized (same pathway as Cocquet)** | **2.85×10⁻⁷ at h=1/320** | **‑2.87 (optimal P2)** | Galerkin leg in [test/extended/CocquetFormMMS/results/cocquet_form_mms_taylorhood.h5](../../test/extended/CocquetFormMMS/) |
| MMS — Stabilized P2/P2 ASGS (our method) | 3.42×10⁻⁷ at h=1/320 | ‑2.95 (optimal P2) | `cocquet_form_mms_vms_k2.h5` |
| MMS — Stabilized P1/P1 ASGS (our method) | 1.34×10⁻⁴ at h=1/320 | ‑1.64 | `cocquet_form_mms_vms.h5` |

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

The code is a **term-by-term faithful transcription** of Cocquet's weak form — verified by code inspection (not by running experiments): ν=1/Re mapping, the Forchheimer α(ε)/β(ε) closure, the ε-weighted natural traction-free outlet, the convective / mass / pressure-divergence forms, the `physical_epsilon` pressure penalty, the P1 porosity interpolant, domain/inlet/porosity profiles, the N=200 self-reference, and the unstabilized `mult_mom=mult_mass=0` Galerkin codepath all match the paper exactly. File:line map:

| ID  | Claim                                                                 | Path that exercises the claim                                                                                                                                       |
|-----|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| H3  | ν = 1/Re mapping (Re=500 ⇒ ν=0.002)                                   | [src/formulations/viscous_operators.jl:101-103](../../src/formulations/viscous_operators.jl#L101-L103) + [paper_comparison.json:24](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L24) |
| H4  | Forchheimer α(ε), β(ε) closure (a=0.30·((1‑ε)/ε)², b=1.75·(1‑ε)/ε)    | [src/models/reaction.jl:57-62](../../src/models/reaction.jl#L57-L62) + [paper_comparison.json:25-26](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L25-L26) |
| H5  | Outlet BC ε-weighted traction-free (natural, `εσ·n = 0`)             | [src/formulations/continuous_problem.jl:241](../../src/formulations/continuous_problem.jl#L241) |
| H6  | Convective form ε(u·∇)u                                               | [src/formulations/continuous_problem.jl:239](../../src/formulations/continuous_problem.jl#L239) |
| H7  | Mass / pressure-divergence form `q·(physical_epsilon·p + div(εu))`   | [src/formulations/continuous_problem.jl:241,244-245](../../src/formulations/continuous_problem.jl#L241) |
| H8  | Pressure penalty η = 1e-7 matches paper (`physical_epsilon`)         | [paper_comparison.json:25](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L25) sets `"physical_epsilon": 1e-7`; the field now defaults to 0.0 in [base_config.json](../../config/base_config.json) so it cannot silently inherit. |
| H9  | P1 porosity interpolant for the Galerkin row                          | [paper_comparison.json:11](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L11) (`"porosity_order": 1`) |
| H10 | Domain, inlet profile, porosity profile                               | [run_convergence.jl:9-10](../../test/extended/CocquetTubeTest/run_convergence.jl#L9-L10) + [paper_comparison.json:31-41](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L31-L41) |
| H11 | N_ref=200 self-reference (same methodology as paper)                  | [paper_comparison.json:62-67](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L62-L67) (convergence_partitions) + driver |
| H12 | Galerkin Taylor-Hood unstabilized (mult_mom=mult_mass=0)              | [galerkin_driver.jl:26-27,62-64](../../test/extended/CocquetTubeTest/galerkin_driver.jl#L26-L27) |

### 4.2 — Experimental hypotheses (closed)

| ID | Claim | Status | Test directory / probe | Result |
|----|-------|--------|------------------------|--------|
| H1 | Outlet-corner Dirichlet pin caps convergence | FALSIFIED | [test/extended/CocquetTubeTest/data/modified_corner/](../../test/extended/CocquetTubeTest/data/modified_corner/) — entities 2,4 removed from "walls" tag, freeing outlet-corner DOFs | Made absolute L²(u) **3.1× worse at N=10** (9.36e‑4 vs 3.01e‑4) and **3.0× worse at N=80** (6.89e‑5 vs 2.31e‑5). The released DOF takes an algebraically-driven value that pollutes the bulk. Memory: [cocquet-modified-corner-experiment.md](/Users/guillermocasasgonzalez/.claude/projects/-Users-guillermocasasgonzalez-repos-porous-NS-with-Gridap/memory/cocquet-modified-corner-experiment.md) |
| H2 | Structured-mesh corner-aligned diagonals lock the corner singularity | PARTIALLY CONFIRMED (slope only) | [test/extended/CocquetTubeTest/data/unstructured_gmsh/](../../test/extended/CocquetTubeTest/data/unstructured_gmsh/) (uniform 1/N gmsh Delaunay) + [test/extended/CocquetTubeTest/data/freefem_divisions/](../../test/extended/CocquetTubeTest/data/freefem_divisions/) (literal `buildmesh(a(N)+b(N)+c(N)+d(N))` N-per-side) | Unstructured Delaunay recovers near-optimal slopes (Galerkin P2/P1 freefem-divs final segment **‑2.05**, uniform-divs ‑1.80; P2/P2 ASGS 2.39 uniform, 2.13 freefem-divs). But absolute magnitudes are still 320–845× the paper. Memory: [cocquet-mesh-topology-controls-convergence.md](/Users/guillermocasasgonzalez/.claude/projects/-Users-guillermocasasgonzalez-repos-porous-NS-with-Gridap/memory/cocquet-mesh-topology-controls-convergence.md) |
| O1 | Our N=200 reference is itself under-converged vs the truth solution | RULED OUT BY ARGUMENT | (No experimental test — argument from symmetry of methodology.) | Paper uses the same self-reference methodology. If our reference were under-converged by ~1e‑4, paper's would be too and would also plateau at that floor. It doesn't (paper reaches 2e‑7 at N=100). Therefore the gap is not in the reference. Strengthened in §5.2. |
| O2 | `u_base_floor_ref = 1e-4` velocity-magnitude regularization adds a noise floor | RULED OUT BY PROBE | `/tmp/probe_with_floor_off.jl`, log `/tmp/probe_floor_off.log` — single Galerkin P2/P1 N=80 + N_ref=200 solve with `u_base_floor_ref=0` and `epsilon_floor=0` | L²(u) at N=80 = **2.6685e‑4** — identical to 4 sig figs to baseline (2.669e‑4). Newton iter-0 residual bit-identical at 0.001164580518. The floor never fires on this flow (true \|u\| > 1e‑4 everywhere except wall vertices where u=0). |
| O3 | Solution magnitude / spatial profile differs from Cocquet's | PARTIALLY RULED OUT | `/tmp/probe_cocquet_reference.jl`, log `/tmp/probe_cocquet_ref.log` | Flow magnitudes physically reasonable: ‖u‖_L²(Ω) = 0.125 (consistent with c_in/4 inflow scale); inlet u(0,0.5) = (0.125, 0) exact; centerline u_x decreases 0.125 → 0.102 (Darcy drag); asymmetric profile (peak at y=0.75) because porosity ramps. Newton converged to 3.7e‑15 in 4 iterations. We're solving the right PDE. Caveat: a smooth, global error mode would slip past these checks — see §5.5. |
| O4/O7/O8 | Relative-not-absolute norm / Newton stall / Anderson interference | CLOSED (all FALSIFIED 2026-05-26) | see §6 | O4: paper p.32 defines `Err_tot` absolutely (no relative normalization). O7: every Galerkin solve reaches iter-4 residual ≤ 1.0×10⁻¹³ (past `ftol=1e-11`). O8: Galerkin path builds `SafeNewtonSolver` only; Anderson never fires on the Galerkin row. |
| O5 | Quadrature degree insufficient for non-polynomial Forchheimer term | RULED OUT BY PROBE | `/tmp/probe_quad_plus12.jl`, log `/tmp/probe_quad12.log` — N=200 reference solve with `Measure(Ω, 21)` (default for k_v=2 is 9; bump +12) | Iter-0 PDE residual: baseline `9.588812700822735e‑4`, quad+12 `9.588812745812068e‑4`. Delta ~5e‑9 — five orders of magnitude below our 1e‑4 error level. Quadrature is adequate. |

### 4.3 — Diagnostic experiments referenced in the ledger

| Diagnostic | What it measured | Where to find it | Headline |
|------------|------------------|------------------|----------|
| **CocquetFormMMS** (smooth-solution MMS with same operators) | Best-case attainable L²(u) with same Forchheimer/porosity operators | [test/extended/CocquetFormMMS/](../../test/extended/CocquetFormMMS/) | **L²(u) = 8.54×10⁻⁷ at h=1/80** (ASGS P2/P1). Same order as paper's N=100 magnitude. Confirms the code is not bugged. |
| **Literal FreeFem .msh comparison** | Solve on Cocquet's exact mesh-recipe `.msh` files (generated by FreeFem) | [test/extended/CocquetTubeTest/results/freefem_meshes/convergence.h5](../../test/extended/CocquetTubeTest/results/freefem_meshes/convergence.h5), driver [paper_comparison_freefem.json](../../test/extended/CocquetTubeTest/data/freefem_meshes/) | **L²(u) = 4.59×10⁻⁵ at N=100** — closer than gmsh Delaunay variants, but still 230× the paper's 2×10⁻⁷. |
| **localize_err.jl** (spatial decomposition of L²(u)² over regions) — *script removed in the CocquetTubeTest unification; recoverable from git history* | Where in the domain does the error live? | `/tmp/localize_err.log` (committed analysis output) | Structured: 71.0% outlet_corner, 25.3% bulk. Unstructured Delaunay: 1.08% outlet_corner, 61.8% bulk. Unstructured *total* ‖e‖ is 18× *larger* than structured at N=10 (5.74e‑3 vs 3.19e‑4). |
| **Figure 2 y-axis re-verification** | Confirm paper's reported magnitudes | Direct inspection of the paper PDF | Y-axis: 10⁻⁴ (top tick) → 10⁻⁷ (bottom). Curve from ~3×10⁻⁵ at N=10 to ~2×10⁻⁷ at N=100. Magnitudes are real. |

---

## 5 — Logical soundness audit (second pass)

This section walks every previously-recorded conclusion and tests it against the **code as it actually exists today**, the math in [theory/cocquet/cocquet_formulation.tex](../../theory/cocquet/cocquet_formulation.tex) and [theory/paper/article.tex](../../theory/paper/article.tex), and the result files on disk. Each subsection ends with the concrete next action, if any.

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

> **CLOSED 2026-07-04.** The global-mode reading below was **ruled out**: S3a's domain-mean share χ_Ω is ≲5% for the load-bearing Galerkin P2/P1 and ASGS P2/P2 rows. The magnitude gap is *not* a rigid/global mode surviving self-reference. Kept for provenance and because the cross-norm numbers themselves are still valid observations.

Numbers from the ledger:

- L²(u) ratio = 845× (ours / paper at N=100).
- H¹(u) seminorm ratio = 7×.

For a discrete solution with uniform error, Aubin–Nitsche gives `‖u − u_h‖_L² ≤ C·h·‖u − u_h‖_H¹`. The L² ratio should shrink relative to H¹ as h decreases — *not grow by 120×*. The opposite scaling implies **a non-uniform error structure**: most likely a low-frequency / global mode that contributes a lot to L² but little to H¹.

**S3a outcome (2026-05-29).** For the Galerkin P2/P1 row, the domain-mean share `χ_Ω(u)` is **0.017, 0.034, 0.046, 0.049** across `N = 10, 20, 40, 80` (`results/structured/convergence.h5`, group `Galerkin/P2P1`) — the cross-mesh error is essentially zero-mean (≲5%). The ASGS P2/P2 row is the same picture (`χ_Ω(u) ≤ 0.028` for every N). Both are P2 velocity spaces, which represent the parabolic inlet Dirichlet `c_in·y·(1−y)` exactly, so there is no Dirichlet-interpolation surplus to leak into the mean. (Where χ_Ω *did* fire: ASGS P1/P1 gives `χ_Ω(u) = 0.752, 0.797, 0.772, 0.605` — the P1 velocity space cannot represent the parabolic inlet exactly, so the Dirichlet-interpolation error injects different mean fluxes on coarse vs reference mesh.)

**Methodological correction.** The cell-average share `‖P₀ e_h‖² / ‖e_h‖²` is *not* a rigid-offset diagnostic on its own — it tends to 1 generically for any smooth converging FE error because piecewise constants approximate smooth fields well on small cells. The right discriminator is the domain-mean share `χ_Ω`, which is zero for any zero-mean error and only saturates when the error is a true rigid offset. A prior reading of the cell-average signal as "global mode confirmed" was overconfident; χ_Ω disagrees on the load-bearing case.

**Action (S3): DONE — global mode ruled out.** χ_Ω ≲5% on the load-bearing cases, so the gap is *not* a global mode. The residual magnitude gap is a measurement question (§1a).

### §5.5 — O3 "we're solving the same PDE" relies on a single coarse-grained probe

The probe checked: ‖u‖_L²=0.125, inlet point value, centerline drag, cross-stream asymmetry, Newton residual. All consistent. **But** a smooth global error mode of magnitude 1×10⁻⁴ would not be visible in any of these checks — it's smaller than the precision of the coarse-grained diagnostics.

**Action:** see §5.4 — the P0-projection diagnostic also strengthens this conclusion. Additionally, comparing against Cocquet's Fig 3 (Re=1000, c_in=1) — where they DO show field plots — would give a direct visual check.

### §5.6 — CocquetFormMMS is the strongest exculpatory evidence, AND it directly exercises the unstabilized Galerkin pathway

The [CocquetFormMMS](../../test/extended/CocquetFormMMS/) test is a **deliberate three-way comparison** of the same MMS problem under all three formulations: the user's stabilized P1/P1 ASGS, the user's stabilized P2/P2 ASGS, and Cocquet's literal unstabilized Galerkin P2/P1 — its designed modes `cocquet_form_mms_{vms,vms_k2,taylorhood}.json`. The Galerkin (Taylor–Hood) leg uses a dedicated path — [`execute_solver_galerkin_inline!`](../../test/extended/CocquetFormMMS/run_test.jl#L160-L207) — that calls `build_stabilized_weak_form_residual` / `build_stabilized_weak_form_jacobian` with `mult_mom=mult_mass=0`, i.e., the **exact** assembly used by the production Cocquet comparison_runs Galerkin row.

Results at h=1/320:

| Method | k_v / k_p | L²(u) | Slope |
|---|---|---|---|
| ASGS (user's stabilized) | P1/P1 | 1.34×10⁻⁴ | ‑1.64 |
| ASGS (user's stabilized) | P2/P2 | 3.42×10⁻⁷ | ‑2.95 (optimal P2) |
| **Galerkin (Cocquet's literal)** | **P2/P1** | **2.85×10⁻⁷** | **‑2.87 (optimal P2)** |

**What this proves.** Assembly, integration, Newton, the Forchheimer reaction operator, the viscous + convective + pressure-divergence terms, AND the unstabilized `mult_mom=mult_mass=0` Galerkin codepath itself are all correct on smooth fields, including at the optimal P2 rate. The Cocquet-benchmark gap is therefore unambiguously a **non-smoothness × discretization interaction**, not specific to stabilization vs. Galerkin and not a hidden assembly defect in the unstabilized path.

### §5.7 — S5 sibling audit (CLOSED 2026-05-26): mixed-BC outlet corner is the dominant cap mechanism

Each sibling is a deliberate one-knob flip of the canonical [paper_comparison.json](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json). All results below were produced 2026-05-22 with `h_floor_weight=0` (post τ-floor regression fix).

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

For the Galerkin P2/P1 row (the actual Cocquet pathway) on the structured mesh, the canonical numbers are: L²(u) = 2.31×10⁻⁵ at N=80 → **58× paper**, slope ‑1.22. The freefem-divs Delaunay mesh gives 2.67×10⁻⁴ at N=80 (670× paper) with the better slope ‑2.05 — the structured mesh is in fact 12× *more accurate* in absolute terms at N=80 despite the worse slope (the structured mesh's lower coarse-N bulk error wins until h is small enough for the slope advantage to flip). This crossover is why the §2 headline numbers (freefem-divs) overstate the magnitude gap at finite N — they reflect the SLOPE-aligned mesh, not the absolute-magnitude-best one.

**Status:** S5 is CLOSED. Findings folded into §6 and into the executive summary.

### §5.8 — Paper–code divergences ledger does not apply to Cocquet's Galerkin pathway

[docs/theory-code-map.md §2](../theory-code-map.md) lists 9 documented divergences. All are stabilization-specific (τ parameters, projection policies, adjoint signs, OSGS iterative loop, etc.). With `mult_mom=mult_mass=0`, **none** fire on the Galerkin row. The Cocquet comparison is therefore not affected by any known divergence.

---

## 6 — Open hypotheses + closed rows

| ID  | Claim                                                                                          | Status  |
|-----|------------------------------------------------------------------------------------------------|---------|
| ~~O4 / O7 / O8~~ | Relative-not-absolute L²(u) / Newton stall / Anderson interference | **CLOSED (all FALSIFIED 2026-05-26).** O4: Cocquet p.32 defines `Err_tot := ‖ũ_h − ũ‖_X + ‖p_h − p‖_L²(Ω)` (absolute); no relative normalization. O7: every Galerkin solve reaches iter-4 residual between 1.7×10⁻¹⁵ and 1.0×10⁻¹³ (past `ftol=1e-11`; the N=200 reference stops at 6.0×10⁻¹²). O8: [galerkin_driver.jl:48-49](../../test/extended/CocquetTubeTest/galerkin_driver.jl#L48-L49) builds `SafeNewtonSolver` for both Picard and Newton; Anderson never fires on the Galerkin row. |
| ~~O6~~  | ~~Cocquet uses Lagrange-multiplier or zero-mean pressure gauge instead of η-penalty~~ | **PARKED 2026-07-04.** Subsumed by the mesh-topology settlement (§1) + the ~30–300× reading (§1a): the residual gap this hypothesis was meant to explain is now attributed to a measurement difference in Cocquet's *unavailable* solver, and the global-mode pathway a gauge offset would drive was ruled out by S3a (χ_Ω ≲5%). Untestable without their `.edp`. |
| ~~S2~~ | ~~Cocquet uses BAMG-adaptive corner-refined mesh, not uniform Delaunay~~ | **CLOSED 2026-07-04.** A *plain* unstructured Delaunay mesh (no adaptivity) already recovers $O(h^2)$ (Phase 11). Whether Cocquet additionally used BAMG adaptivity is no longer load-bearing for the settled *slope* conclusion; the residual *magnitude* gap is a measurement question (§1a). |
| ~~S3 / S4~~ | ~~Global-mode L²(u) contamination / Galerkin-mode MMS confirmation~~ | **CLOSED.** S3: S3a's domain-mean share χ_Ω ≲5% for Galerkin P2/P1 and ASGS P2/P2 (§5.4) rules out a global mode. S4: the CocquetFormMMS test already includes a Galerkin P2/P1 leg — L²(u) = 2.85×10⁻⁷ at h=1/320, slope ‑2.87 (optimal P2); see §5.6. |
| ~~S5~~ | ~~Audit untouched Cocquet siblings~~ | **CLOSED 2026-05-26.** See §5.7. Three siblings (Alpha1 / Deviatoric / LinearReaction) exonerate all porous-physics knobs; AllDirichlet implicates the **mixed-BC outlet corner** (P2/P2 slope 1.3 → 2.46); LiteralPicard exonerates linearization choice. |

---

## 7 — What we have NOT yet verified (the frontier)

> **Mostly settled (2026-07-04).** With the slope explained as a structured-mesh artifact (§1) and the magnitude gap attributed to a measurement difference in Cocquet's unavailable solver (§1a), the only genuinely-open item is the external one (H-D, below). All in-repo threads (O1–O8, S1–S5, S3a/S3b, H-A…H-E) are closed, parked, or confirmed paper-faithful.

### The one genuinely-open item — requires external action
- **H-D — Cocquet's solver may iterate on the lifting `w = u − V` and plot the error in `w`.** Unfalsifiable from inside the repo (only the mesh-generation `.edp` is available). This — or "something beyond the original five" (a different reported functional, mislabelled figure axis, etc.) — is the residual explanation for the ~30–300× *magnitude* gap. Resolution needs Cocquet's actual solver source (corresponding-author email at hal-02561058 / supplementary-materials hunt). See §1a.

### Hypotheses suggested by the audit but probably not worth testing
- **Inlet quadratic profile + P2 representation.** With c_in=0.5 the inlet is *exactly* representable by P2; no projection error.
- **Newton damping × Forchheimer interaction.** Already fixed by [lessons_learned.md](../lessons_learned.md) row 25 (τ-floor leakage); `h_floor_weight=0` is set in [paper_comparison.json:27](../../test/extended/CocquetTubeTest/data/structured/paper_comparison.json#L27).
- **Wall boundary layer at y=0** (porosity minimum, Forchheimer drag maximum). Would manifest as concentration of L²(u) error near y=0. The S5 audit's LinearReaction sibling (which zeroes β·|u|·u while keeping the linear Darcy α(ε) drag) keeps the same slope as baseline, weakly suggesting the *nonlinear* part of the wall-layer is not the source — but the linear-Darcy wall layer itself is untouched and could still contribute.

---

## 8 — Document relationships and update protocol

- **This file** — **CANONICAL Cocquet reference; start here.** The single canonical doc: the settled conclusion (§1), the merged lateral-hypothesis verdicts (§1a), and the supporting synthesis (§1b onward). Update it in place when a hypothesis crosses the open→closed boundary.
- [docs/cocquet/convergence-analysis.md](../archive/cocquet-convergence-analysis.md) — **historical phased-slope diary** (Phases 1–12; some early verdicts withdrawn — read via this synthesis).
- [docs/cocquet/replicating-cocquet-transcript.md](../archive/cocquet-replicating-transcript.md) / [docs/cocquet/corner-singularity-transcript.md](../archive/cocquet-corner-singularity-transcript.md) — **archived raw transcripts** (provenance only; conclusions merged here).

**Open action items:** only H-D remains, and it is external (obtain Cocquet's solver `.edp`; see §1a / §7). All in-repo threads (O1–O8, S1–S5, S3a/S3b, H-A…H-E) are closed, parked, or confirmed paper-faithful.
