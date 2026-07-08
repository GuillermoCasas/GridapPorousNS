# Open questions — unresolved theoretical & numerical items

**Purpose.** This is the living index of the porous-Navier–Stokes VMS solver's **OPEN** questions — the ones that survive current investigation. Each entry states the phenomenon, the leading hypothesis, what has been ruled out, and **what would settle it**. The practical deliverables built on top of each question are secure; these are theory-completeness / calibration questions, not blockers.

For the full experiment-by-experiment evidence behind any entry, follow the links into the preserved dossiers under `docs/` and the LaTeX sources under `theory/`. Resolved siblings live in `docs/mms/`, `docs/cocquet/`, and `docs/solver/` (indexed in [docs/README.md](README.md)).

---

## 1. [OPEN] CocquetFormMMS α=0.1 × Re=1e5 fold — the exact mechanism

**Phenomenon.** In `test/extended/CocquetFormMMS` (equal-order stabilized VMS vs Galerkin Taylor–Hood, full Forchheimer–Ergun reaction `σ=a(α)+b(α)|u|`), the α=0.1 × Re=1e5 corner exhibits a **genuine coarse-mesh solution-branch fold**: no discrete root with `‖R‖≤tol` exists for N≤80. This part is **RESOLVED** — a true, FE-optimal root appears at N≥160 (ASGS H¹u≈1.07 / OSGS 1.10, L²u≈3.0 both; N=160→320 ladder), matching the sister harness's α=0.05 corner (`docs/mms/fold-recovery.md`). See `docs/cocquet/cocquet-form-mms-status.md` §4.1.

**What stays OPEN:** *why* the coarse-mesh branch folds — the exact mechanism.

**Established (empirical):**
- The fold is **reaction-magnitude driven**, not nonlinearity-driven. An α-sweep at fixed Re=1e5 converges at α=0.9/0.5 and folds at α=0.2/0.1; a **linear-reaction control** (`σ_nl=0`, matched `Da_eff`) folds *identically* → the `b(α)|u|` Forchheimer nonlinearity (`∂σ/∂u` basin-shrink) is **not** the cause.
- Fold cells are genuinely **pre-asymptotic** (err_u ~0.12–0.20, ~50–100× the converged cells), not "optimal-but-gate-floored".
- The fold **recedes as N↑ and deepens as α↓**.

**Ruled out (2026-07-08 direct tests, `docs/cocquet/cocquet-form-mms-status.md` §4.3):**
- **Jacobian / linearization bug — NO.** Exact-Newton J vs centered-FD: `‖J−J_fd‖/‖J_fd‖ ≈ 1e-8..1e-11` for k=1 *and* k=2 at corner parameters; Newton converges quadratically (‖R‖: 2.9e-4→1.3e-5→2.7e-8→2.3e-13). Now guarded permanently by `test/extended/cocquet_jacobian_consistency_extended_test.jl` (closed a real gap: the velocity-dependent `∂σ/∂u` Exact-Newton tangent for SymmetricGradient+Forchheimer was previously unguarded).
- **c₁ (coercivity constant) — NOT the lever.** At high Re, `τ_NS⁻¹` is convection-dominated (`c₂|u|/h ≫ c₁ν/h²`), so raising c₁ barely moves the stabilization. c₁×4 gave only marginal help (α=0.1: converges *only* N=10, still folds N=20/40; α=0.2: N=40 error ~5× smaller but same convergence pattern). This is expected and UNLIKE 3D-P2, because CocquetFormMMS folds are **k=1**, where the viscous 2nd-derivative subscale is identically zero, so c₁ acts only through `τ_NS`. A c₁×64 confirmation was launched but killed under CPU contention.
- **Correction: "Taylor–Hood converges everywhere" is FALSE at the corner.** TH (Galerkin P2/P1) at α=0.1×Re=1e5 does **not** reach a root: its residual stalls at O(1) (`‖R‖` = 650→190→50→13→**3.1** at N=10→160) and velocity error is flat at 0.40 (rate 0). TH is convectively unstable here (LBB pressure but no SUPG for convection): pressure converges optimally (L²p rate 2.0) while velocity is garbage. The real contrast is **VMS folds hard at coarse mesh (NaN) but converges to an accurate root at N≥160, where TH still cannot** — VMS is the *better* method at the corner. (At Re=1, TH converges cleanly, rate 2.94 → the corner failure is the high-Re convective instability.)

**Leading hypothesis: the paper's `σ̃_α` reaction-in-stabilization coercivity weakening.** In the *ASGS* stability estimate (`article.tex` §`sec:StabilityASGS`, `eq:SigmaAlpha`) the reaction's presence in the stabilization replaces the full velocity control `σ` by the weaker `σ̃_α = τ_{1,NS}⁻¹σ/(τ_{1,NS}⁻¹+σ/α_K)`, which collapses as α→0; Taylor–Hood, having no stabilization, keeps the full `σ`. It is paper-grounded and consistent with every observation (reaction-magnitude driven, low-α specific, fold-recedes-with-mesh). **Caveat:** the paper deems `σ̃_α` *benign for the convergence rate* — the link "weaker coercivity ⇒ nonlinear-solver fold" is an unproven extension.

⚠️ **ASGS vs OSGS carry the reaction differently — do not conflate.** `σ̃_α` is the *ASGS* estimate. OSGS removes the reaction from the orthogonal projection (for constant σ the orthogonal subscale is exactly zero, `article.tex` ~line 619) — a *documented* OSGS convergence concern that the paper says "facilitates the convergence of the nonlinear iterations." The code trim `ProjectResidualWithoutReactionWhenConstantSigma` is **OSGS-only** ([`src/stabilization/projection.jl:76-80`](../src/stabilization/projection.jl#L76)) and fires only for `Constant_Sigma`. Under the Forchheimer law the sweep uses, the reaction stays in the stabilization for **both** methods.

**What would settle it (the clean isolation):** strip σ from `𝓛U`/`𝓛*V` **only, holding τ₁ physical**. The 2026-07-08 strip test (`STRIP_REACTION_FROM_STAB`, since **REVERTED** — a stabilization-reaction strip is not paper-faithful) was **inconclusive and confounded**: it cleared only 1 of 3 folding meshes (N=40, to an inaccurate root L²u=0.092, ~3× the accurate value), N=20/N=80 still failed from the exact guess. The confound: stripping σ *also enlarges* τ₁ (removes σ from its denominator, `1/(α/τ_NS+σ)→τ_NS/α`), and σ is genuinely entangled in the stabilization scale (σ̃_α itself contains τ₁). So the strip test **neither confirmed nor cleanly refuted** σ̃_α — but it did show the fold is **not reducible to a single removable term**; reaction-in-stab is at most a partial contributor. The natural next probe (strip `𝓛U`/`𝓛*V` only, hold τ₁) is **deferred** — the practical deliverable (convergence above the fold, §4.1) does not depend on it.

**Alternative next probes:** finish the **OSGS** trim-vs-full A/B at low α (started but killed — OSGS low-α fold cells thrash); the **k=2 analog** where the viscous 2nd-derivative subscale exists (the faithful test of whether the 3D-P2 c₁ mechanism transfers — the `C1_MULT` hook is committed, default-off).

**Evidence:** `docs/cocquet/cocquet-form-mms-status.md` §4.2/§4.3; `theory/tau_saturation_note/tau_saturation_note.tex` (Layer-1 τ₁-saturation, dormant at Re=1e5).

---

## 2. [OPEN] Gridap-vs-Kratos MMS magnitude offset (~3–12×)

**Phenomenon (open since 2026-06-17).** When the high-Re/low-α corner cells are reproduced in Gridap, the normalized Free-Mesh-Error (FME) come out **~3–12× larger** than the paper's Kratos values, in a **norm-dependent** way: velocity L² ~7×, pressure L² ~5×, H¹ ~2.4×.

**This is a code-vs-code calibration question, NOT a convergence failure.** Convergence *rates* agree (≈2–3), and the Gridap TRI numbers match the Gridap QUAD continuation to ~2%, so the discretization is **internally consistent**. It affects how literally `results/paper_tables.tex` can be read against the paper.

**Candidate causes:**
- Characteristic-scale `U_c`/`P_c` normalization.
- Porosity-field definition.
- MMS amplitude.
- **Element-size `h` convention** (min-edge vs diameter). Note the new **`stabilization.element_size` config knob** — `StabilizationConfig.element_size`, implemented in [`src/geometry.jl`](../src/geometry.jl) (`element_size_field` / `element_size_convention`, valid conventions in `ELEMENT_SIZE_CONVENTIONS`; schema at `config/porous_ns.schema.json:222`) — which makes the h-convention a controllable variable and lets an A/B quantify how much of the offset it explains. (Related: in 3D-P2 the h-convention was tested and found to be a strong lever on the *margin* but **not** the root cause of that separate defect — see §3.)

**What would settle it:** an A/B sweeping the candidate knobs (especially `stabilization.element_size`) against the paper's Kratos calibration, ideally with Kratos's actual normalization constants and mesh statistics.

**Evidence:** `docs/known_issues.md` (Confirmed, "Gridap-vs-Kratos MMS magnitude offset"); canonical detail `docs/mms/fold-recovery.md` ("Current status & remaining work").

---

## 3. [OPEN caveats] 3D-P2 "converged-but-wrong" — residual weaknesses on an otherwise-RESOLVED verdict

**Status.** The headline 3D-P2 question is **RESOLVED** (2026-07-06): the P2/structured-Kuhn-tet instability is **not a code bug and not a Gridap↔paper discrepancy** — `c₁ = 4k⁴` is **under-margined for high-`C_inv` structured tets**. The viscous 2nd-derivative subscale is anti-coercive by construction (self-adjoint op ⇒ `B_S` has `−τ‖𝓛_visc V‖²`), dominated by `c₁ > 2ξ·C_inv²`; `4k⁴` has ~zero margin for 3D tets (`C_inv²` Kuhn **214** vs quad **60**, mesh-independent; Kuhn shape-regularity `h/ρ=8.36` worst). Proven by Céa (50× error at `4k⁴` ⇒ coercivity≈0, refuting "definite-but-buggy") + the `c1_mult=4` **monotone convergent ladder**. Remedy = element-aware `c₁` (`article.tex` line 910), **not** a code change. The 2026-07-05 "Gridap↔paper discrepancy / c₁-masks-a-bug" framing is WITHDRAWN. Canonical: [`docs/mms/3d-p2-instability-investigation.md`](mms/3d-p2-instability-investigation.md); full for-scrutiny record: [`docs/mms/3d-p2-coercivity-resolution-dossier.md`](mms/3d-p2-coercivity-resolution-dossier.md).

**But three honest caveats remain OPEN — what would overturn or sharpen the verdict** (dossier §0 items + §6 weaknesses):

1. **The absolute threshold constant `ξ`.** The derived condition `c₁ ≥ 2·C_inv²` is ~2× conservative: the Q2 quad has `C_inv²=60 ⇒ 2C_inv²=120 > c₁=64` by the formula, yet the quad *works*. The true condition is closer to `c₁ ≳ C_inv²` (paper's `ξ ≈ ½`), because the dropped convective/reaction terms add coercivity. So the **relative** ordering (Kuhn 214 ≫ quad 60) is robust, but the **absolute** "just barely enough for 2D" claim rests on `ξ`, which was **not pinned independently**. — *What would sharpen it:* directly assemble the stabilized Jacobian and compute the spectrum of `(J+Jᵀ)/2` at `4k⁴` vs `c₁×4` (deferred — Céa already forces the qualitative answer; this would only quantify severity).

2. **The un-demonstrated well-shaped-tet positive test.** A genuinely well-shaped, quasi-uniform 3D tet family (uniform aspect `h/ρ ≲ 6`, no boundary slivers) that **converges at plain `4k⁴`** would confirm "well-shaped tets are inside the margin, Kuhn is the pathology." This **could not be built cleanly**: gmsh Delaunay+optimize leaves slivers *worse* than Kuhn (aspect 11–21, erratic); the hand-built **BCC lattice** has good bulk (5.66) but its boundary caps (8.83 ≈ Kuhn) reintroduce a Kuhn-level defect at the boundary (erratic, L²u 0.126→0.097→0.011). So "well-shaped 3D tets converge at `4k⁴`" is **inferred** (from the `C_inv` ordering + line 910 + the paper's unstructured §5.2 result), **not directly demonstrated**. — *What would overturn it:* if a clean well-shaped family *also* failed at `4k⁴`, the correct statement is **stronger** — `4k⁴` is under-margined for *all* 3D tets and the paper's optimal §5.2 leans on its short 3-mesh ladder (rate from the two finest) rather than element quality. Conversely, Kratos's *actual* §5.2 mesh statistics: worst-aspect `>8` yet optimal at `4k⁴` over a *long* ladder → would refute the `C_inv`/margin explanation and reopen a code-difference hunt.

3. **A deviatoric-specific sharper root.** The paper (line 255) flags that Korn-type arguments "may be nontrivial for [the deviatoric operator]." The **deviatoric** `C_inv` may be genuinely worse than the pure symmetric-gradient's — i.e. the deviatoric operator may be *intrinsically* more fragile. This was characterized (`TAU_VISC_MULT=4/3`, the deviatoric Fourier spectral-radius correction, gives a 7.5× improvement) but **not fully separated** from the general `C_inv` story.

**Refuted alternatives (for the record, all with a discriminating test — dossier §4):** tolerance/under-converged (byte-identical wrong at ε_M=1e-12), bad Jacobian (Taylor-verified, ‖R‖→1e-14), grad-div operator + grad-div-in-subscale, quadrature, MMS-oracle mismatch, mesh quality, inf-sup/equal-order pressure (TH P2-P1 also wrong), Newton-vs-Picard (identical wrong root to 8 sig figs ⇒ defect in residual `F`, not linearization — but this is now moot given the coercivity verdict), a sign error in `L*_visc`, h-convention bug (diameter makes it *worse*; margin is h-independent), a single code factor, simplex-specific (2D-P2-on-triangles is **optimal**, rate→2.97/1.92 ⇒ genuinely 3D-only), and anisotropy (isotropic cube equally erratic).

**Hooks (committed, default-off, byte-identical, Blitz 243/243):** `smoke3d.jl` `solve_one(...; c1_mult, h_conv∈{regular_tet,d_fact,diameter,shortest_edge}, ablation="picard_only")`; env `VISC_ADJ_MULT` (`src/formulations/continuous_problem.jl`); env `TAU_VISC_MULT` (`src/stabilization/tau.jl`); the 2D-TRI-P2 discriminator `data/phase1_tri_k2.json`.

---

## 4. [OPEN] OSGS-P2-3D ∂π/∂u coupling — accurate solution, `ok=false`

**Phenomenon.** In the official 3D §5.2 structured sweep (2026-06-30), **OSGS P1 is fully SOLVED** (robust `eps_used=1` all 4 meshes AND optimal: L²u→2.0, H¹u→1.0, L²p→1.8; ~2–4× more accurate than ASGS). **OSGS P2** produces an **accurate, near-optimal solution** (L²u rate ~2.4→2.9, H¹u ~1.6→1.7) but the solver reports **`ok=false`** — the convergence *gate* is not met. This is a solver / convergence-detection problem, **not** a discretization problem.

**Key evidence the discretization is right:** JFNK+boot-skip+penalty on (12,12,3) reaches **L²u=0.0012187** (H¹u=0.059, L²p=0.0029) — *exactly the c₁×4 value* — so the OSGS root is correct and c₁ is genuinely not needed here.

**The blocker (mechanism):** the OSGS coupled tangent drops the dense `∂π/∂u` coupling (frozen-π inexact Newton) — benign-ish in 2D, genuinely worse in 3D:
- **JFNK's GMRES doesn't fully converge** — the frozen-π preconditioner is a weak 3D saddle-point preconditioner; from far guesses (`eps_pert`=1/0.1/0.01) it gets no traction at all, so only `eps_pert=0` reaches the root.
- **The merit-based line search depletes near the root** — the re-projecting merit (`Φ=½Σ(bᵢ/wᵢ)²`, block-equilibrated `_update_merit_weights!`) jumps in 3D where it doesn't in 2D. The merit is **not broken** (works in 2D); near the 3D OSGS root it correctly backtracks a genuinely bad `∂π/∂u` step that the looser frozen-π fallback can't improve either.
- The **staggered π-update between outer iterations diverges** in 3D (outer 2: ‖R‖→2.06, merit 1.6e6) — the dropped ∂π/∂u makes the π-iteration non-contractive (Anderson over-extrapolation likely worsens it; a plain Picard staggered may be more stable — a manual test converged outer-1, outer-2 never observed).
- (Note: the **ASGS Stage-I boot is harmful for OSGS** — it converges to the *different* ASGS fixed point, from which OSGS overshoots. `osgs_skip_asgs_boot` lets OSGS run from the guess; the eps_pert homotopy supplies the globalization. With boot-skip, the OSGS *first* staggered inner solve converges even from `eps_pert=1` — it's robust *to the start*; the π-update is the remaining issue.)

**What would settle it (solver-engineering only — ranked):**
1. A real **saddle-point / MG preconditioner** for the OSGS coupled tangent so JFNK's GMRES converges from any guess — the principled, Kratos-matching fix.
2. A **stabilized (damped, plain-Picard, no-Anderson-extrapolation, relaxation<1) staggering** — cheaper than JFNK; test whether it makes the π-update contractive in 3D (the manual plain-staggered inner solve converged).
3. Fix the **merit / gate near the OSGS root** for the re-projecting residual.
4. Confirm the paper-c₁ OSGS converged root is optimal once a robust solver reaches it (it should be — ASGS at paper c₁ is optimal and the discretization is shared).

Not a quick knob. The discretization, c₁, the iterative penalty, and the operators are all confirmed correct. (2D k=2 OSGS needs JFNK for the same ∂π/∂u reason.)

**Related fix that is NOT this (context):** the **iterative penalty** (`ε_num·(pⁿ−pⁿ⁻¹)` in the mass residual) fixed 3D all-Dirichlet **well-posedness** (ε=0 is ill-posed) — a *separate*, RESOLVED fix, orthogonal to both this OSGS-P2 gate issue and the §3 P2 accuracy/coercivity story.

**Evidence:** [`docs/mms/3d-iterative-penalty-fix-and-osgs-coupling.md`](mms/3d-iterative-penalty-fix-and-osgs-coupling.md) §3.5 (sweep map) + §4 (the coupling problem). Config flags `iterative_penalty_enabled`, `iterative_penalty_max_iters`, `osgs_skip_asgs_boot` (all default-off). JFNK landing: [`docs/solver/jfnk-phase0-preconditioner-gate.md`](solver/jfnk-phase0-preconditioner-gate.md).

---

## 5. [OPEN] Paper editorial / theory items needing author judgment

Items from the 2026-06-04 documentation audit that need an **author decision**, not a mechanical fix. Full list: [`docs/paper/errata.md`](paper/errata.md). (The paper compiles cleanly — `latexmk` exit 0, 0 undefined refs, 43 pages; the one unambiguous typo, the duplicate Fourier label `eq:728`, is already corrected.)

- **"Kratos Multiphysics" implementation claim** ([`article.tex` line 1015](../theory/paper/article.tex#L1015)). This repository is a **Gridap.jl** solver. Confirm whether the paper's numerical experiments were run in Kratos (historical) or should now read Gridap.jl. **Do not silently flip — author call.** (This is the same Kratos-vs-Gridap boundary that underlies the §2 magnitude offset and the §3 first-author reconciliation.)
- **Results-section figures** ([`article.tex` line 1436](../theory/paper/article.tex#L1436), `\Guillermo{Add figures}`). Convergence results are currently tables; figure environments not yet added. (Staged convergence-plot PDFs were removed at the author's request as unreferenced.)
- **`supplement.tex` is SIAM template boilerplate** (`\input{ex_shared}` / `\lipsum` / `thm:bigthm`). Either replace with the real supplement or drop the `\externaldocument{supplement}` line.
- **Merge `centered_encoding.tex` into `article.tex`** (self-describes as "not yet merged"; no label clashes; strip its standalone preamble + trailing `\end{document}`, drop its `\Reyn`/`\Damk`/`\code` `\newcommand`s). Placement is the author's choice.
- **Softer math to verify** (not corrected): the reaction matrix `S(w)` mass-equation-row porosity-gradient entries vs `∇·(αu) = α∇·u + ∇α·u`; and whether the `CocquetExperiment` Galerkin run uses the operator the paper claims to compare against (`cocquet_formulation.tex` uses the full symmetric gradient `S(u)`, while `article.tex` uses the deviatoric-symmetric operator).

---

## Cross-references

- **Detailed evidence dossiers:** [`docs/mms/3d-p2-coercivity-resolution-dossier.md`](mms/3d-p2-coercivity-resolution-dossier.md) (3D-P2 full record + caveats), [`docs/cocquet/cocquet-form-mms-status.md`](cocquet/cocquet-form-mms-status.md) (Cocquet fold), [`docs/mms/3d-iterative-penalty-fix-and-osgs-coupling.md`](mms/3d-iterative-penalty-fix-and-osgs-coupling.md) (OSGS-P2 coupling), `docs/formulation-audit-2026-06-24.md` (theory↔code + results audit), `docs/known_issues.md` (open code-correctness issues incl. the schema/loader method-enum drift and the CocquetFormMMS hardcoded-ASGS dispatch).
- **Theory (LaTeX):** [`theory/paper/article.tex`](../theory/paper/article.tex) (authoritative — `sec:StabilityASGS`/`eq:SigmaAlpha`, lines 508/619/910/1015/1375/1383), `theory/tau_saturation_note/tau_saturation_note.tex`, `theory/osgs_reaction_note/osgs_reaction_note.tex`, `theory/cocquet/cocquet_formulation.tex`, `theory/centered_encoding/centered_encoding.tex`.
- **Code anchors:** [`src/stabilization/projection.jl:76-80`](../src/stabilization/projection.jl#L76) (OSGS-only reaction trim), [`src/geometry.jl`](../src/geometry.jl) (`element_size` convention knob), [`src/formulations/continuous_problem.jl`](../src/formulations/continuous_problem.jl) (`p_prev` iterative penalty; `VISC_ADJ_MULT`), [`src/stabilization/tau.jl`](../src/stabilization/tau.jl) (`TAU_VISC_MULT`).
