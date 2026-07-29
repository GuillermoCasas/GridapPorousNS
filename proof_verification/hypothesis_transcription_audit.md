# Hypothesis-transcription audit (the trusted-base gate)

**What this is.** The Coq development (`coq-formal/`) proves the a-priori chain as a *conditional*:
"IF the ~50 named hypotheses of `coq_coverage.tex` Table `tab:inventory` hold, THEN stability /
continuity / interpolation-continuity / convergence hold." It machine-checks the *deductive assembly*
(triangle inequalities, Cauchy–Schwarz, ring regroupings, Céa / inf–sup composition) but **never
unfolds the functional-analytic content of the hypotheses** and **never proves the paper's concrete
finite-element objects satisfy them** (the abstract→concrete bridge, `verification-gap-coverage.md`
Layer 2, is unbuilt). The SymPy suite checks concrete *algebra* only.

So one whole class of error is invisible to both machine layers **by construction**: a hypothesis fed
into the trusted base whose *stated proof in the paper does not actually establish it*. The
2026-07-23 external audit found exactly this class (`ω_χ→0`, patch equivalence, advection regularity).
Coq even certified the *one-sided* `abstract_continuity` (`BS <= Ctot*(...)`, no `Rabs`,
`AbstractContinuity.v:46`) — a true-but-weaker statement — which is why the missing-abs-values defect
also slipped through.

**This register is the defense.** It enumerates every trusted-base obligation whose justification is a
*non-algebraic* lemma, ties each to (i) the paper locus that proves/assumes it and (ii) an explicit
"does the stated proof actually establish it?" verdict. It is meant to be re-read on every revision
that touches `sec:StabilityASGS`, `asgs_convergence.tex`, `osgs_appendix.tex`, or the standing
assumptions — the human/LLM pass that the machine layers cannot perform.

The *statement-decoration* half of this class (abs-value bars on continuity LHS, inf–sup domains
excluding 0) is now machine-guarded by `sympy/theorem_statement_verification.py` (in the `run_all.py`
gate). The *hypothesis-content* half below has no symbolic proxy and must be audited by reading.

## Legend
- **Visibility** — `algebraic` (a SymPy script can/does check it) · `trusted` (abstract Coq hypothesis,
  content invisible by design) · `statement-lint` (guarded by `theorem_statement_verification.py`).
- **Verdict** — `OK` (stated proof establishes it) · `assumed` (legitimately taken as hypothesis, not
  claimed proved) · `repaired-2026-07-23` (was an over-claim; fixed this pass).

## Register

| # | Obligation | Paper locus | What must establish it | Visibility | Verdict |
|---|---|---|---|---|---|
| 1 | Element-size surrogate oscillation `ω_χ(h)→0` | `H:patch` / `eq:SmoothGrading`; used in `oa:lem:smoothing` | A **smooth-grading hypothesis** — bounded patch equivalence (`eq:PatchQuasiUniformity`) does NOT imply it (fixed-ratio graded mesh keeps `ω_χ=O(1)`). | trusted | **repaired-2026-07-23**: `eq:SmoothGrading` added as an explicit hypothesis; smoothing proof reworded to invoke it, not "by construction". |
| 2 | Patch **multiplicative** equivalence of `α_K`, `τ_i` | `oa:lem:patch` | Shared-vertex chain `α_K ≤ δ_α α(x_a) ≤ δ_α² α_{K'}` from the resolved-porosity condition; constant `δ_α²=(1+C_∇α)²` independent of the contrast `α_∞/α_0`. An *additive* Lipschitz bound does not give it. | trusted | **repaired-2026-07-23**: additive→multiplicative leap replaced by the resolved-porosity chain; `|a|` handled by comparing the full positive τ expressions over the viscous floor (not multiplicative equivalence of `|a|` near its zeros). |
| 3 | Global regularity of `α a·∇u` on patches for the Scott–Zhang best approximation | `oa:lem:bestapprox` → `oa:lem:consistency`; `H:advectionsmooth` | `a ∈ W^{kᵤ,∞}(Ω)` **globally**. Elementwise-`W^{kᵤ,∞}` + global-`C⁰` is not in `H^{kᵤ}(S_K)` across faces for `kᵤ≥2`. | trusted | **repaired-2026-07-23**: `H:advectionsmooth` strengthened from elementwise `‖Dʲa‖_{L^∞(K)}` to global `a∈W^{kᵤ,∞}(Ω)`, symmetric to the already-global `α∈W^{m,∞}` of `H:porositysmooth`. |
| 4 | Working norm is a **norm** (definiteness) | `H:projector`/`eq:PiKorn` (A3); `lem:projfamily` (§2.1 `sec:ViscousProjector`), `lem:definiteness` (App C), `oa:lem:definiteness` (App D) | Korn compatibility `‖∇v‖ ≤ C_K‖Π∇v‖` on `H¹₀`, now an explicit standing assumption. For the family it follows from the elementary identity `‖dev sym ∇v‖² = ½‖∇v‖² + (½−1/d)‖∇·v‖²` chained with the pointwise Pythagoras `eq:projmonotone`, with the uniform `C_K=√2`. No Korn/conformal-Killing machinery needed. | trusted → **algebraic** | **repaired-2026-07-23** (identity added, App D re-pointed); **generalized 2026-07-29**: promoted to the standing assumption (A3), so the obligation is now *printed* rather than implicit, and both halves are machine-covered — the chaining in `ViscousProjector.v` (`korn_sqrt2`, `nested_pythagoras`), the concrete identity and the sharp constants in `sympy/projector_algebra_verification.py`. |
| 5 | Interpolant admissibility in the zero-mean pressure space (`ε=0`) | interpolant setup, App C; `oa:rem:meanshift`, App D | Mean-correct `p̂_h := I_h p − |Ω|⁻¹∫I_h p`; the shift is bounded by `‖p−I_h p‖` and invisible in gradient / zero-mean pairings. | trusted | **repaired-2026-07-23** (App C): mean-corrected interpolant now stated (App D already had it). |
| 6 | Absolute-value bars on continuity-lemma LHS | `eq:continuity`, `eq:sharpcont` (App C); `oa:eq:ConsistencyBound` (App D) | A continuity estimate bounds `|B(·,·)|`, not the signed `B`. | **statement-lint** | **repaired-2026-07-23** (App C bars added); now guarded by `theorem_statement_verification.py`. |
| 7 | inf–sup sup/inf domains exclude 0 | `oa:eq:InfSup` and its proof quotient | `\sup_{V_h∈Xhz\setminus\{0\}}` (Rayleigh quotient undefined at 0). | **statement-lint** | **repaired-2026-07-23** (proof-body quotient); guarded going forward. |
| 8 | Projection stability (condition (35)) | `H:projection`; `codina2008analysis` | A Codina–Blasco property of the element **pair**, to be verified per family; not implied by the common mesh-and-data setting. Route-A sufficiency (`c₁,c₂` large, non-sharp) is stated as such. | trusted | **assumed** (correctly flagged non-automatic and non-sharp). A generalized-eigenvalue computation of `β₀` over representative regimes would upgrade this to `algebraic`. |
| 9 | Coercivity threshold `c₁ > 2ξ C_inv²` | `lem:coercivity`, `H:coercivity` | `C_inv` is element-family dependent (`c1_dimension_note`, validated by `element_c1.jl`); `4k⁴` is under-margined for high-`C_inv` structured tets (documented). | algebraic (partly) | **assumed** with the element-dependence documented; `coverage_coercivity_numeric_verification.py` covers the algebra. |
| 10 | Weighted inverse estimates | `lem:winv` | Reference-element scaling + porosity-resolution factors. | algebraic | **OK** — `coverage_weighted_inverse_verification.py`. |
| 11 | Jump condition `H:jump` (ASGS) | `H:jump`; derived in `oa:lem:patch` restricted to face-adjacent elements | Facewise `φ₁` comparability with `σ` cancelling in `[τ₁⁻¹]`. | trusted | **assumed / dispensable** — implied by the standing hypotheses; dispensing with it in the OSGS route is presentation economy (no `[τ₁]` face term exists there). |

| 12 | Viscous projector is a pointwise, constant-coefficient **orthogonal projection** (idempotent + Frobenius self-adjoint) | `H:projector` (A3), first half | Used four ways, all previously *unstated*: the symmetry of the viscous form `2ν(∇v,ανΠ∇u)=2ν(Π∇v,ανΠ∇u)` (`eq:GalerkinCoercivityIdentity`, `oa:eq:CoercivityIdentity`, `eq:Bstab` T₁); the contraction `|ΠT|≤|T|` (`eq:winv-grad`, `eq:interpgrad`, OSGS Step 3 and (I1)); preservation of the elementwise polynomial structure of `Π∇w_h`, so `lem:winv` applies with the same, projector-independent `C_inv`; and the major symmetry `P_{aibj}=P_{bjai}` behind `K_ji^⊤=K_ij` in `eq:AdjointFlux`/`eq:AdjointDifferentialOperator`. | algebraic | **repaired-2026-07-29**: was an unstated reading obligation (`coq_coverage.tex` §stratum1 (I4), piece (iv)); now printed as (A3). Contraction and nesting are kernel-checked (`ViscousProjector.proj_nonexpansive`, `nested_pythagoras`); idempotence, self-adjointness and the major symmetry of the four instances are checked exactly for `d=2,3` in `sympy/projector_algebra_verification.py` (A1, A5), which also verifies that the displayed `K_ij` is the dev-sym instance of `[K_ij]_{ab}=2να P_{aibj}`. |

## How to run the machine half
```
cd proof_verification/sympy && python3 run_all.py      # includes theorem_statement_verification.py
cd proof_verification/coq-formal && ./run_all.sh        # compiles + coqchk + Print Assumptions
```
A green suite certifies the *assembly* and the *statement decorations*. It says nothing about rows
marked `trusted` above — those are this document's job, and must be re-read by a human on every
revision of the analysis. See `coq_coverage.tex` for the full paper↔Coq map and the `tab:inventory`
trusted-base inventory this register audits.
