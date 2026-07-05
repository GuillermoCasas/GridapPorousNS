# 3D P2 MMS "converged-but-wrong": root cause isolated

> **Status — external clean-room diagnosis (2026-07-02); its central VERDICT is REFUTED (2026-07-05).**
> This is a standalone NumPy/SciPy reimplementation, committed as an evidentiary bundle (report + reproducer
> `files/{pns3d,assemble,verify,ckpt_run}.py`, see §8). Its central claim below — the failure is an
> **element-family c₁ coercivity deficit** (paper `4k⁴` sub-critical on Kuhn tets; raise c₁) — is
> **REFUTED by the paper's first author**: Kratos assembles the **full subscale** (no terms removed) and
> solves this exact 3D-P2 case **optimally at paper c₁ = 4k⁴ on tetrahedra**. So paper c₁ is correct and
> `C_inv`-exceeds-`4k⁴` is not the mechanism. **This report's own reasoning is circular**: its weak form was
> transcribed *term-by-term from the Gridap code* (`continuous_problem.jl`, per §1), so it **inherited** the
> Gridap↔paper implementation discrepancy that actually causes the failure, then attributed the resulting
> c₁-sensitivity to element-family coercivity. The real root cause is a **code↔paper discrepancy** (that c₁
> masks), still **OPEN** — see the canonical
> [../../mms/3d-p2-instability-investigation.md](../../mms/3d-p2-instability-investigation.md). Kept
> verbatim for provenance and because its careful term-by-term transcription is a useful map of what the
> Gridap assembly computes (i.e. where to hunt the discrepancy). Read the verdict below as REFUTED.

**Verdict.** The failure is *formulation-level, not a Gridap or code defect*: at the paper's
c₁ = 4k⁴ = 64, the ASGS coupling between the **viscous second-derivative terms in the subscale
residual R(u_h)** and the viscous terms in the adjoint L*(v) is *sub-critical in the coercivity
sense* on P2 Kuhn tetrahedra. The discrete problem is well-posed enough to have a unique,
solver-reachable root — but that root is not quasi-optimal. The critical constant for this
element family sits between **1.5× and 2× the paper value** (i.e. c₁* ∈ (96, 128) at k = 2);
at c₁ ≥ 128 the solve lands on the interpolant (ratio 1.02–1.03×) at every mesh tested.

This was established with a **clean-room reimplementation** (pure NumPy/SciPy, no Gridap, no
Julia): P2/P1 Lagrange on the same Kuhn tetrahedralisation of (0,1)²×(0,0.3), *analytic* basis
Hessians (exact on affine simplices), the §5.2 oracle re-derived independently and verified
against finite differences, and the weak form transcribed term-by-term from
`continuous_problem.jl` (ASGS branch, iterative penalty, τ per `eq:Tau1/Tau2`, deviatoric
viscosity, c₂ = 2k² scaled together with c₁ exactly as `smoke3d.jl`'s `c1_mult` hook does).

---

## 1. Why the clean room is trustworthy

Every layer was verified independently before any conclusion was drawn:

- **Oracle**: ∇u_ex, Δu_ex, ∇(∇·u_ex), ∇α, ∇²α all match central finite differences to
  ≤ 4·10⁻⁹; the forcing f matches an independent FD-divergence of the flux
  2Aν∇ᵈu to 2·10⁻⁸; ∇·(αu_ex) = 0 to 8·10⁻¹⁶ (200 random interior points).
- **Basis**: P2 shape functions satisfy partition of unity; gradients and *Hessians* match FD
  to 10⁻⁷ at random reference points. On affine tets the physical Hessians
  J⁻ᵀ∇²φ̂J⁻¹ are exact — this is the quantity a hypothetical Gridap ∇∇ bug would corrupt,
  and here it provably cannot be wrong.
- **Assembler**: at the state x = 0 the residual is affine up to terms that cancel under
  central differences, so the assembled Picard matrix must equal the FD Jacobian *exactly*.
  It does (relative 10⁻¹²) for Galerkin blocks and all three viscous operators; with
  stabilisation on, the ~5·10⁻⁵ gap is precisely the legitimately Picard-dropped
  dL*·R_old term.
- **Quadrature**: collapsed Gauss–Legendre q = 7 (343 pts/tet); the flagship healthy result
  was re-run at q = 9 (729 pts) and agrees to 5 significant figures
  (L²u 2.0454·10⁻³ vs 2.0455·10⁻³), so the sharp α-transition is not under-integrated in any
  way that affects the verdict.
- **Nonlinear solve**: Picard (matching `build_picard_jacobian`) + a JFNK stage
  (exact Newton on the frozen-p_prev residual, Picard-LU preconditioned), outer
  penalty passes with p_prev frozen per pass as in `solve_system`. All reported roots have
  ‖R‖ ≤ 8·10⁻¹¹, most ≤ 5·10⁻¹².

The root of a nonlinear system is determined by the residual alone; the residual was verified
against analytic and FD oracles end-to-end. So when this implementation reproduces the repo's
pathology, the pathology belongs to the *discretisation*, not to either codebase.

## 2. Results

§5.2 problem exactly as in `smoke3d.jl`: (α₀, Re, Da) = (0.5, 1, 1), ν = 1, ConstantSigma
σ = 1, all-Dirichlet u = u_ex, ε_phys = 0, iterative penalty ε_num ≈ 1.667·10⁻⁵, structured
Kuhn tets, h = (6√2·V)^{1/3}, errors normalised as in `calc_errors3d` (L²p mean-removed).
"ratio" = L²u(solve)/L²u(nodal P2 interpolant on the same mesh).

| case | mesh | L²u | interp | **ratio** | H¹u | L²p | ‖R‖_end |
|---|---|---|---|---|---|---|---|
| **ASGS-P2, paper c₁** | (8,8,2) | 6.905e-2 | 4.386e-3 | **15.7×** | 1.69 | 0.402 | 1e-12 |
| **ASGS-P2, paper c₁** | (12,12,3) | 2.177e-2 | 2.006e-3 | **10.8×** | 0.845 | 0.144 | 8e-11 |
| c₁×1.5 (=96) | (8,8,2) | 3.516e-2 | 4.386e-3 | **8.0×** | 0.992 | 0.126 | 3e-11 |
| c₁×2 (=128) | (8,8,2) | 4.875e-3 | 4.386e-3 | **1.11×** | 0.142 | 0.039 | 2e-12 |
| c₁×4 (=256) | (8,8,2) | 4.519e-3 | 4.386e-3 | **1.03×** | 0.144 | 0.089 | 5e-12 |
| c₁×4 | (12,12,3) | 2.045e-3 | 2.006e-3 | **1.02×** | 0.095 | 0.047 | 9e-13 |
| c₁×4 | (16,16,4) | 1.101e-3 | 1.069e-3 | **1.03×** | 0.067 | 0.031 | 3e-13 |
| c₁×4, q=9 check | (12,12,3) | 2.0455e-3 | 2.0064e-3 | 1.02× | 0.095 | 0.047 | 9e-13 |
| drop ∇²-viscous from R_u only, paper c₁ | (8,8,2) | 7.304e-3 | 4.386e-3 | **1.67×** | 0.169 | 0.052 | 2e-12 |
| drop ∇²-viscous from R_u only, paper c₁ | (12,12,3) | 3.808e-3 | 2.006e-3 | **1.90×** | 0.133 | 0.030 | 7e-13 |
| drop from R_u **and** L*, paper c₁ | (8,8,2) | 6.572e-3 | 4.386e-3 | **1.50×** | 0.143 | 0.081 | 6e-12 |
| bare-Galerkin Taylor–Hood P2/P1 (no stab) | (8,8,2) | 4.759e-3 | 4.386e-3 | **1.09×** | 0.139 | 0.041 | 2e-12 |
| bare-Galerkin Taylor–Hood P2/P1 | (12,12,3) | 2.053e-3 | 2.006e-3 | **1.02×** | 0.091 | 0.020 | 4e-13 |
| control: ASGS-P1, paper c₁ | (8,8,2) | 2.832e-2 | 2.233e-2 | 1.27× | 0.363 | 0.314 | 9e-12 |
| control: ASGS-P1, paper c₁ | (12,12,3) | 1.737e-2 | 1.082e-2 | 1.60× | 0.267 | 0.222 | 9e-12 |

Rates (h: 0.149 → 0.0994 → 0.0745), c₁×4: L²u 1.96 → 2.15 against interpolant
1.93 → 2.19; H¹u 1.04 → 1.23. The discrete solution *tracks the interpolant to 2–3 %
per mesh* — the observed sub-3 rates are the interpolant's own preasymptotics (the α-transition
annulus, same effect as the repo's OSGS-JFNK rates 2.39 → 2.92 and the paper's k=1
preasymptotics at h ≈ 0.05), not a method deficiency.

## 3. Mechanism, in the paper's own terms

The paper's Remark after `eq:conditions_on_num_param` (quoted in `smoke3d.jl` at the
`c1_mult` hook) states the coercivity bound needs **c₁ > 2ξ·C_inv²**, where C_inv is the
inverse-inequality constant for ‖∇·(2νε^d(v_h))‖ on the element family, and notes the optimal
c₁ is element-dependent. The destabilising product is exactly
τ₁·⟨viscous part of L*(v), viscous ∇²-part of R(u_h)⟩, which is negative-definite-ish
and scales as C_inv²/c₁ relative to the Galerkin viscous term — *h-independent*, which is why
the wrongness neither converges away nor blows up, and why refinement looks erratic rather
than divergent. C_inv² for P2 on right-angled Kuhn tetrahedra evidently exceeds the 4k⁴
budget by a factor of ~1.5–2; the observed knee at c₁* ∈ (96, 128) *is* the coercivity
threshold. Everything in the table is the textbook signature: below c₁* a converged-but-wrong
root with inflated H¹ error; above c₁*, quasi-optimality with the constant essentially
independent of c₁ (×2 vs ×4 differ by 8 % in L²u).

Three corroborating observations:

1. **P1 immunity** — the ∇²-terms vanish identically at k = 1. ✓ (control rows.)
2. **Iteration pathology from the same term** — at paper c₁ the (8,8,2) Picard iteration
   oscillates non-monotonically (ρ ≈ 0.85); with the term dropped or c₁ raised it converges
   monotonically in ≤ 3 iterations. The dropped dL*·R_old Newton correction is
   Hessian-bearing; sub-critical coercivity amplifies it. This also matches the repo's §4
   OSGS finding verbatim: "π = Π(R(u)) with R ∋ ∇²u; each cycle amplifies high-frequency
   content ~1/h²". Prediction: after either fix, the OSGS ω-scan becomes contractive.
3. **OSGS root goodness** — the penalty-fix doc's own JFNK+boot-skip datum
   (L²u = 0.0012187 ≈ interpolant at (12,12,3)) fits: orthogonal projection removes the
   FE-resolvable part of the residual, which is where the destabilising coupling lives.

## 4. Reconciling "the author + Kratos say paper c₁ is correct"

Both statements can be true simultaneously; they are about different discretisations:

- The **theory** is correct and *predicts* this behaviour: c₁ must exceed 2ξC_inv², and
  C_inv is element-family-dependent. 4k⁴ is a calibration, not a theorem.
- **Kratos** very plausibly does not assemble the full ∇²-viscous terms in the subscale at
  high order — the repo's own investigation doc (§6, step 2) hypothesises exactly this
  ("many production VMS codes omit it at high order — plausibly what Kratos does"). My
  drop-from-R_u run *is* that variant, and it is healthy at paper c₁ (1.67–1.90×). If Kratos
  additionally runs hexahedra, C_inv is smaller again.
- The repo's **2D k=2 validation was on QUADs** (per the doc); quad C_inv < tet C_inv, so
  paper c₁ suffices there. The CocquetFormMMS α = 0.5 case that converged 5/5 at k=2 on
  *triangles* is also consistent: its large reaction σ enters τ₁'s denominator and damps the
  viscous-subscale product.

So Hypothesis A in `3d-p2-instability-investigation.md` ("c₁×4 only shrinks the error
constant, masks — refuted, author-steered") needs revision. The discriminator between
"masking" and "fixing" is whether the ratio-to-interpolant is pinned at ~1 across a mesh
family — it is (1.03, 1.02, 1.03 over three meshes, with L²u rates tracking the
interpolant's own). A masked defect would leave a large or drifting ratio.

## 5. Recommended actions in the repo (cheap, decisive)

1. **Gridap-side confirmation with zero new code**: `smoke3d.jl` already has the `c1_mult`
   hook. Run ASGS-P2 (12,12,3) at `c1_mult ∈ {1, 1.5, 2, 4}`. Expected: the same knee
   between 1.5 and 2, L²u collapsing from 0.0494 to ≈ 0.0012 (≈ interpolant). This
   single sweep confirms or falsifies the whole diagnosis inside the real stack in minutes.
2. **Implement §6 step 2 as planned**: gate `subscale_drop_viscous` (zero `div_visc_u` in
   `eval_strong_residual_u` and correspondingly in `R_du`; optionally also the viscous part of
   the adjoint). Expected at paper c₁: ratio ~1.7–1.9 and monotone Picard. Then re-run the
   OSGS ω-scan — predicted contractive.
3. **Decide the canonical fix.** Options, in decreasing paper-faithfulness:
   (a) element-family-aware c₁ (keep full residual; e.g. c₁ = 4k⁴ on quads/hexes, 2·4k⁴ or a
   measured C_inv²-based value on simplices — the knee gives the constant to cite);
   (b) drop the ∇²-viscous subscale terms at k ≥ 2 (production-VMS practice, matches Kratos
   behaviour, costs ~1.6× in the constant vs (a));
   (c) prefer OSGS for k ≥ 2 once its solver is fixed. For the paper's §5.2 table the
   honest statement is that the 3D-P2 columns need element-appropriate c₁ (or the
   Kratos-style reduced subscale, if that is what generated them — worth one email to the
   author asking precisely which subscale terms Kratos assembles at k=2).
4. **Doc updates** to `3d-p2-instability-investigation.md`: revise Hypothesis A; fix the
   "method-INDEPENDENT" TL;DR (the OSGS entry in the §2 table is a frozen-π one-shot, not the
   OSGS fixed point — the companion doc's JFNK root at 0.0012187 shows the OSGS *root* is
   good); note that Experiment D checked `EvalDivDevSymOp` on an analytic field only, which
   never probed FE-basis Hessians (the hole is now moot, but the audit logic should be
   recorded); mark §6 steps 1–4: step 4 (bare-Galerkin TH) is done here and is clean,
   step 2 is done here and heals, step 1 (2D-TRI) is now predicted to *fail at k=2 for
   sufficiently small σ/large ν-dominance and heal with c₁×2* — still worth running as a
   cheap confirmation of the element-family story.

## 6. Secondary findings (independent of the main diagnosis)

- **Pressure-mean drift under the outer penalty loop.** With all-Dirichlet velocity data the
  discrete boundary flux ∮αu_h·n is a fixed nonzero ρ (interpolation/quadrature level;
  ≈ −1.1·10⁻³ at P1 (8,8,2)). Each outer pass with refreshed p_prev shifts the pressure mean
  by −ρ/(ε_num|Ω|) ≈ 200 per pass, forever. Mean-removed L²p is immune, but after a few
  passes |p̄| ≫ |p| and (i) relative precision in p degrades, (ii) the constant leaks into
  the momentum equation at quadrature-error level. Cheap hardening: re-centre p between
  passes (constant mode is arbitrary), or cap passes. Worth checking how many passes
  `solve_system` actually executes on the 3D case and whether reported p carries a large mean.
- **τ₂ regularisation**: paper `eq:Tau2` carries +εh² in the denominator; the code uses
  `tau_reg_lim` instead. Negligible here, but a doc-noteworthy paper/code divergence.
- Already-known items that remain open in `known_issues.md`: `base_config.json` missing
  `eps_val` (canonical example unloadable), schema method-enum drift, CocquetFormMMS
  hard-coded ASGS dispatch. The CocquetFormMMS α = 0.1 solver folds are a separate,
  reaction-magnitude-driven phenomenon (solver folds, not converged-but-wrong) and are not
  explained away by this diagnosis.

## 7. Honest limitations

- The clean room is method-identical but not byte-identical to the Gridap stack: collapsed-GL
  quadrature (cross-checked q7 vs q9) vs Gridap's degree-12 simplex rule; velocity-floor
  constants at their defaults (irrelevant at |u| ~ 1). Step 1 above closes this gap in
  minutes.
- My sick pair (15.7× → 10.8× over two meshes) reproduces "converged-but-wrong at
  O(10×)" but not the full erratic/non-monotone zoo of the repo's finer sweeps; that zoo is
  expected from an h-independent sub-critical term with mesh-dependent constants, but I did
  not chase it.
- Kratos internals are inferred, not inspected. The reconciliation in §4 rests on the repo's
  own hypothesis plus the drop-R experiment; the author can confirm in one line.

## 8. Reproducibility

`cleanroom_mms3d/`: `pns3d.py` (mesh/basis/quadrature/oracle), `assemble.py`
(residual + Picard Jacobian + solver + errors), `verify.py` (the verification chain of §1),
`ckpt_run.py` (all cases; resumable), `run_ladder.py`, and the `result_*.json` files behind
the table. Python ≥ 3.11, NumPy ≥ 2, SciPy ≥ 1.14. Each case: `python3 ckpt_run.py P2_12_c1x4`
(re-invoke until DONE; state checkpoints to `state_*.npz`).
