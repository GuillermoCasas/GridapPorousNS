# 3D P2 MMS: the converged-but-wrong instability — investigation log (UNRESOLVED)

> **Status: OPEN / UNRESOLVED as of 2026-07-01.** Canonical record of the 2026-06-30 → 07-01 investigation
> into why the 3D §5.2 MMS **fails at k=2 (P2)** while **k=1 (P1) works** and the **2D harness works at k=2**.
> This doc catalogs every hypothesis tried and its verdict (with numbers), so future sessions do not re-run
> refuted leads. Companion: [3d-iterative-penalty-fix-and-osgs-coupling.md](3d-iterative-penalty-fix-and-osgs-coupling.md)
> (the penalty fix + the OSGS ∂π/∂u solver problem). Harness: `test/extended/ManufacturedSolutions3D/`
> (`smoke3d.jl`). Every experiment + its numbers are in the §3 table; the ad-hoc probe scripts were cleaned
> after the investigation (see §7 to reproduce).
>
> **⚠ Contested (2026-07-03).** An independent clean-room reimplementation (pure NumPy/SciPy, no Gridap)
> now argues the failure **IS** an element-family **c₁ coercivity deficit** — paper `4k⁴` sub-critical on
> Kuhn tets, knee at `c₁×1.5–2`, reconciling the "Kratos runs paper c₁" fact via reduced high-order
> subscale assembly — **directly contradicting TL;DR #2's "not c₁" refutation** below. See
> [../convergence_problems_audit/files/p2_3d_diagnosis_report.md](../convergence_problems_audit/files/p2_3d_diagnosis_report.md).
> It is **UNconfirmed in this stack**: the deciding test is its §5.1 (`smoke3d.jl c1_mult ∈ {1,1.5,2,4}`
> at ASGS-P2 (12,12,3) — expect L²u collapse ≈0.049→0.0012 with the knee between 1.5 and 2, ratio-to-
> interpolant pinned ~1 across meshes). **This doc's verdict stands until that sweep runs**; if it
> reproduces, revise TL;DR #2 and Hypothesis A (§3) accordingly.

## TL;DR

1. **The P2-3D discrete solution is CONVERGED-BUT-WRONG.** ASGS-P2 converges (residual → 1e-8…1e-14) but to a
   solution **20–95× the interpolant error**, and the error is **erratic / non-monotone** under refinement
   (SymmetricGradient even *diverges*). Tightening the gate to ε_M=1e-12 gives a **byte-identical** wrong answer
   ⇒ it is a genuine property of the discrete solution, not under-convergence.
2. **Config-INDEPENDENT.** The failure persists across **all** of: viscous operator (Deviatoric / SymmetricGradient
   / Laplacian), mesh (structured Kuhn *and* Frontal-optimized unstructured), pressure space (equal-order P2-P2
   *and* Taylor-Hood P2-P1), and method (ASGS *and* OSGS). It is **not** c₁ (Kratos runs paper c₁), **not** mesh
   quality, **not** grad-div, **not** inf-sup, **not** tolerance, **not** a bad Jacobian. All refuted — see §3.
3. **Root cause NOT identified.** The surviving fact: a config-independent P2-3D discrete solution that is wrong at
   a deep residual. The strongest *untested* lead is whether this is **P2-on-simplices** (reproducible in cheap
   2D-TRI) vs genuinely 3D — §6 step 1.
4. **OSGS-P2 additionally can't be solved** (a *separate* solver problem on top of the discretization one): the
   staggered π fixed-point is **violently non-contractive** (ρ≈8–65, damping fails) and **JFNK is not
   budget-fixable** (noisy matrix-free Jᵥ from the re-projecting residual). See §4.

## 1. Problem statement

3D §5.2 MMS: z-extruded field on (0,1)×(0,1)×(0,0.3), (α₀,Re,Da)=(0.5,1,1), ConstantSigma, structured Kuhn tets.
**P1 works** (ASGS/OSGS optimal-to-sub-optimal; see the official sweep in the companion doc). **P2 fails.** The
paper author confirms Kratos solves this exact 3D-P2 case optimally at paper c₁ for both ASGS and OSGS — so this
is very likely a code/formulation issue, not a physics or constant one.

Two intertwined symptoms:
- **ASGS-P2: converged-but-wrong** (this doc's focus).
- **OSGS-P2: won't converge** (§4; the ∂π/∂u solver problem, also in the companion doc).

## 2. The unifying reframe: solves ≫ interpolant

The P2 **interpolant** of u_ex on the structured Kuhn mesh is optimal (L²u ≈ 0.0011 at h=0.099, converges
monotonically). Every stabilized **solve** lands far from it:

| (12,12,3) | L²u | ×interp |
|---|---|---|
| interpolant | 0.00112 | 1× |
| ASGS Deviatoric | 0.0494 | 44× |
| ASGS SymmetricGradient | 0.0298 | 27× |
| ASGS Laplacian | 0.0130 | 12× |
| OSGS (frozen-π one-shot) | 0.0157 | 14× |

So the mesh *represents* u_ex fine; the **stabilized solve** produces a solution 12–44× worse. The stabilization
is degrading P2-3D, and the same mechanism destabilizes the OSGS π-iteration (§4).

## 3. Hypotheses tried — all REFUTED (do not re-run)

| # | Hypothesis | Experiment | Result | Verdict |
|---|---|---|---|---|
| A | **c₁ needs k/dimension adaptation** | (prior) c₁×4 | only shrinks the error *constant*, masks; author + Kratos say paper c₁ is correct | ❌ refuted (author-steered) |
| B | **Tolerance / under-converged** | ASGS-P2 gate ε_M 1e-9 vs 1e-12 | **byte-identical** wrong answer (L²u=0.049370 both) | ❌ converged-but-wrong |
| C | **Bad Jacobian / NR floor** | same as B + Newton trace | residual descends cleanly to 1e-8…1e-14; P2-P1 hits 1.5e-14 | ❌ not the Jacobian |
| D | **grad-div term in the viscous op** | numeric check of `EvalDivDevSymOp` on an off-diagonal-Hessian field `u=(xy,yz,xz)` | matches analytic `(1/6,1/6,1/6)` to **4.8e-17** | ❌ operator is correct |
| E | **grad-div in the SUBSCALE destabilizes** | operator study: Deviatoric (1/6 grad-div) vs Symmetric (1/2) vs **Laplacian (no grad-div)** | **all three erratic**; Laplacian also non-monotone | ❌ not grad-div-specific |
| F | **Quadrature under-integration** | measure degree audit | degree = `get_quadrature_degree+4` = 12 for P2 (exact to P11); projection reuses it | ❌ sufficient |
| G | **MMS oracle ≠ formulation (viscous)** | term-by-term compare `mms3d.jl` visc vs `strong_viscous_operator` | deviatoric forcing matches exactly (`νAΔu + (νA/3)∇(∇·u)` + porosity-grad term) | ❌ consistent |
| H | **Mesh quality (Kuhn tets low-quality)** | ASGS-P2 on **Frontal+optimized** unstructured meshes vs interpolant | 9.5–25× interp, **also erratic** (finest worse) | ❌ mesh-independent (magnitude helped ~2–4×, a minor constant factor) |
| I | **Equal-order inf-sup / pressure instability** | **Taylor-Hood** P2-P1 (inf-sup stable) vs P2-P2 | P2-P1 **also wrong** (L²u 57–95×, absolute L²p ~0.4 = same/worse); converges to 1.5e-14 (well-conditioned) | ❌ not inf-sup |

Notes:
- The huge L²p **ratio** for equal-order (700–2900×) is a red herring: it's large because the P2-pressure
  *interpolant* is tiny (1e-4). The *absolute* pressure error (~0.4) is the same under inf-sup-stable Taylor-Hood
  (H-row), so it is **not** spurious pressure modes — the whole solution is simply wrong.
- "Erratic/non-monotone" on a **regular** mesh with **constant** element quality cannot be a mesh-quality
  (constant-factor) effect — it is a formulation instability. (User's own observation; correct.)

## 4. The OSGS-P2 solver problem (separate from §3, also open)

Even granting the (wrong) discretization, OSGS-P2 cannot be *solved*:
- **Damped staggered π-iteration is violently non-contractive.** ω-scan on (12,12,3): the π-drift ratio is
  **8–65** for ω=1.0 down to 0.1; every ω diverges (L²u explodes to O(1)). Mechanism: π = Π(R(u)) with R∋∇²u
  (P2 viscous 2nd-derivative); each cycle amplifies high-frequency content ~1/h². ρ≫1, not the "≈1" one might
  hope — Anderson can't rescue it. **Idea "damped staggering" is dead.**
- **JFNK is not budget-fixable.** Inner GMRES stalls at rel-res 0.01–0.16 regardless of maxiter (30/100/300),
  **non-monotone across budgets** ⇒ the matrix-free Jᵥ = [R(u+εv)−R(u)]/ε is *noisy* because R re-projects π
  inside every FD probe. The frozen-π preconditioner can't cluster that operator. JFNK takes no accepted step
  (stays at the interpolant). A real saddle-point/MG preconditioner would be needed — but see §3: the *root*
  is reachable-but-wrong, so a better OSGS solver alone would still land on a sub-optimal P2-3D solution.
- **k=1 datum:** a *single* frozen-π solve from the interpolant gives L²u=0.0157 (better than ASGS) — the
  fixed point exists and is reasonable; only the *iteration to it* is unstable.

## 5. Current diagnosis (honest)

The P2-3D stabilized discretization produces a **wrong** solution (20–95× interpolant, erratic) that is
**independent of** viscous operator, mesh, pressure space, and method, and is reached at a **deep residual**
(so it is the genuine discrete-weak-form solution, not a solver artifact). Since the MMS forcing is shared with
P1 (which works), the forcing is not grossly wrong. What is left is a **consistency/stability defect specific to
the P2 discrete solution in 3D** — most plausibly in how the VMS **stabilization** interacts with the P2 fields
(the ∇²u terms in R(u_h) and the adjoint L*(v), which are zero for P1 and, on QUADs in 2D, structured
differently than on simplices). **Not yet isolated.**

## 6. Ranked next steps (for a future session)

1. **2D-P2 on TRIANGLES (cheap, decisive).** The "2D-P2 works" control was on **QUADs**; 3D is **tets**. Run the
   2D MMS at kv=2 on **TRI** vs QUAD. If TRI-P2 is *also* converged-but-wrong ⇒ the defect is **P2-on-simplices**
   (reproducible + debuggable in fast 2D, not 3D-specific). If TRI-P2 is clean ⇒ genuinely 3D. This single test
   most narrows the search and unlocks cheap debugging.
2. **Drop the FULL viscous 2nd-derivative from the SUBSCALE** (`div_visc_u = 0` in `eval_strong_residual_u`,
   distinct from the Laplacian operator which keeps Δu in the subscale — see hook at continuous_problem.jl:289,
   used at :354). If ASGS-P2 recovers optimality ⇒ the viscous-in-subscale is the destabilizer (and many
   production VMS codes omit it at high order — plausibly what Kratos does). Gate it (`subscale_drop_viscous`,
   default OFF) and re-test ASGS accuracy + the OSGS ω-scan contraction.
3. **Re-audit the MMS oracle's HAND-CODED analytical 2nd derivatives** (`lap_u_ex3d`, `grad_div_u_ex3d` in
   `mms3d.jl`) against autodiff/FD of u_ex — a bug there would be P2-visible, P1-invisible only if it entered a
   P2-only term; verify it is genuinely correct beyond the viscous-term structural match already done.
4. **Bare-Galerkin (unstabilized) P2-3D on a Taylor-Hood pair** (mult_mom=mult_mass=0) — if *that* is clean, the
   defect is unambiguously in the **stabilization**, not the Galerkin discretization or the oracle.
5. **A real saddle-point/MG preconditioner for the OSGS coupled tangent** — only worthwhile *after* the
   discretization is fixed (else it converges to a wrong root).

## 7. Reproducing (scripts removed; method is here)

The ad-hoc probe scripts and their temporary harness hooks were **removed after the investigation** to keep the
harness clean — the method and results above (§3 table + §4) are the durable record. To re-run a given probe,
re-implement it from the §3/§4 description against `smoke3d.jl`'s `solve_one` (mesh via `structured_kuhn_model`
or `build_box_tet_model(...; algorithm=4)` for Frontal). Two small `solve_one`/`build_config` hooks used during
the investigation and then reverted (re-add if reproducing):
- `return_setup=true` — early-return the assembled `(setup, vmsform, config, X, Y, V_free, Q_free, dΩ, c_1, c_2,
  h_cf, f_cf, g_cf, alpha_cf, u_ex, p_ex, U_c, P_c)` so an external driver (ω-scan) can run its own iteration
  on the exact operators.
- `kp` (pressure order; `kp=kv-1` = Taylor-Hood) — threaded into `element_spaces.k_pressure` and `refe_p`.
