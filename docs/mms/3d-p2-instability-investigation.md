# 3D P2 MMS: the converged-but-wrong instability — OPEN (a Gridap↔paper discrepancy, MASKED by c₁)

> **Status: OPEN — root cause is a Gridap↔paper IMPLEMENTATION discrepancy in the P2-3D case; c₁ MASKS it
> (updated 2026-07-05, authoritative).** The paper's **first author** confirms that **Kratos assembles the
> FULL subscale (no terms removed) and solves this exact 3D §5.2 P2 case OPTIMALLY at paper `c₁ = 4k⁴` on
> tetrahedra, for both ASGS and OSGS.** Therefore **paper c₁ is CORRECT**, and the "element-family c₁
> coercivity deficit" reading below is **REFUTED**. The Gridap `c₁×4`-fixes behaviour (40× L²u collapse,
> ratio-to-interpolant → 1, optimal asymptotic rates) is a **symptom**: the Gridap implementation
> **under-stabilizes the P2-3D case relative to the paper**, and c₁×4 merely **compensates/masks** the gap —
> it is NOT a real coercivity requirement and NOT the fix. The NumPy clean-room does **not** exonerate Gridap:
> it was transcribed **term-by-term from `continuous_problem.jl`**, so it **inherited** the same discrepancy
> and reproduced the same c₁ symptom. **Root cause remains OPEN** — a term-level discrepancy between the code
> and the paper, most likely (but not certainly) in the **P2-3D viscous 2nd-derivative subscale**, since the
> bug is constrained to **P2-only** (P1 works) ∩ **3D-only** (2D-QUAD k=2 works) = the 3D viscous
> 2nd-derivative subscale / grad-div coupling `(½−1/d)∇(∇·u)` (0 in 2D, 1/6 in 3D). It may be broader — the
> author notes there may be other overlooked differences. Canonical record of the 2026-06-30 → 07-05
> investigation. Companion: [3d-iterative-penalty-fix-and-osgs-coupling.md](3d-iterative-penalty-fix-and-osgs-coupling.md)
> (the penalty fix + OSGS ∂π/∂u). Harness: `test/extended/ManufacturedSolutions3D/` (`smoke3d.jl`). The
> clean-room diagnosis at [../convergence_problems_audit/](../convergence_problems_audit/) has its verdict
> refuted (it argued element-family c₁); kept for provenance + the code-transcription it encodes.

## TL;DR

1. **The P2-3D discrete solution is CONVERGED-BUT-WRONG.** ASGS-P2 converges (residual → 1e-8…1e-14) but to a
   solution **20–95× the interpolant error**, and the error is **erratic / non-monotone** under refinement
   (SymmetricGradient even *diverges*). Tightening the gate to ε_M=1e-12 gives a **byte-identical** wrong answer
   ⇒ it is a genuine property of the discrete solution, not under-convergence.
2. **Config-INDEPENDENT symptom; c₁ CONTROLS it but is NOT the cause (2026-07-05).** The failure persists across
   viscous operator, mesh (structured Kuhn *and* Frontal), pressure space (P2-P2 *and* Taylor-Hood P2-P1), and
   method (ASGS *and* OSGS). It is **not** mesh quality, grad-div, inf-sup, tolerance, or a bad Jacobian (refuted
   — §3). In Gridap, **c₁ controls it** (c₁×4 collapses L²u ~40×, ratio-to-interpolant → 1; c₁×2 masks
   partially) — but this is a **symptom, not the mechanism**: the first author confirms **Kratos, with the full
   subscale at paper c₁ on tets, is optimal**, so paper c₁ is correct and the "element-family c₁" reading is
   **refuted**. c₁×4 masks a **Gridap↔paper implementation discrepancy**.
3. **Root cause: OPEN = a code↔paper discrepancy in the P2-3D case** (masked by c₁, NOT fixed by it). Most likely
   the P2-3D **viscous 2nd-derivative subscale** (P2-only ∩ 3D-only), possibly broader. The c₁ multiplier is a
   MASK, not a fix; the correct fix is to find and correct the discrepancy so paper c₁=4k⁴ works, as it does in
   Kratos. (The clean-room's element-family-C_inv argument is refuted; it inherited the discrepancy by
   transcribing from `continuous_problem.jl`.)
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
| A | **c₁ under-budgets coercivity for the element family** | c1_mult ∈ {1,1.5,2,4} across the Kuhn mesh family, ratio-to-interpolant (§5.1) | c₁×4 **pins** ratio→interp ~1 (ASGS & OSGS), even optimal rates on the 4-mesh ladder; c₁×2 masks | ❌ **REFUTED (2026-07-05, authoritative).** The mesh-family data (c₁ controls the error, ×4 pins) is real, but the *interpretation* is wrong: the **first author** confirms **Kratos runs the full subscale at paper c₁ on tets, optimally** — so `4k⁴` is NOT under-budgeted and `C_inv` on tets is NOT the issue. c₁ merely **masks a Gridap↔paper implementation discrepancy** (the clean-room, transcribed from `continuous_problem.jl`, inherited it — so its "C_inv exceeds 4k⁴" argument is circular). c₁ is a symptom-mover, not the cause. |
| B | **Tolerance / under-converged** | ASGS-P2 gate ε_M 1e-9 vs 1e-12 | **byte-identical** wrong answer (L²u=0.049370 both) | ❌ converged-but-wrong |
| C | **Bad Jacobian / NR floor** | same as B + Newton trace | residual descends cleanly to 1e-8…1e-14; P2-P1 hits 1.5e-14 | ❌ not the Jacobian |
| D | **grad-div term in the viscous op** | numeric check of `EvalDivDevSymOp` on an off-diagonal-Hessian field `u=(xy,yz,xz)` | matches analytic `(1/6,1/6,1/6)` to **4.8e-17** | ❌ operator is correct |
| E | **grad-div in the SUBSCALE destabilizes** | operator study: Deviatoric (1/6 grad-div) vs Symmetric (1/2) vs **Laplacian (no grad-div)** | **all three erratic**; Laplacian also non-monotone | ❌ not grad-div-specific |
| F | **Quadrature under-integration** | measure degree audit | degree = `get_quadrature_degree+4` = 12 for P2 (exact to P11); projection reuses it | ❌ sufficient |
| G | **MMS oracle ≠ formulation (viscous)** | term-by-term compare `mms3d.jl` visc vs `strong_viscous_operator` | deviatoric forcing matches exactly (`νAΔu + (νA/3)∇(∇·u)` + porosity-grad term) | ❌ consistent |
| H | **Mesh quality (Kuhn tets low-quality)** | ASGS-P2 on **Frontal+optimized** unstructured meshes vs interpolant | 9.5–25× interp, **also erratic** (finest worse) | ❌ mesh-independent (magnitude helped ~2–4×, a minor constant factor) |
| I | **Equal-order inf-sup / pressure instability** | **Taylor-Hood** P2-P1 (inf-sup stable) vs P2-P2 | P2-P1 **also wrong** (L²u 57–95×, absolute L²p ~0.4 = same/worse); converges to 1.5e-14 (well-conditioned) | ❌ not inf-sup |
| J | **Newton vs Picard linearization** (paper uses plain Picard; Gridap defaults to Exact-Newton) | `ablation_mode` A/B at (12,12,3), ASGS-P2, paper c₁, exact guess: Exact-Newton (`full`) vs Picard (`picard_only`) — same shared residual `F`, different tangent | **Identical wrong root to 8 sig figs** (L²u 0.049370051 vs 0.049370052; H¹u/L²p match). Genuinely different paths (Newton 3 iters quadratic → 2.8e-13; Picard 5 iters → 8.4e-11) → same destination | ❌ **RULED OUT (2026-07-05).** Newton and Picard share the residual `F` in the code (only the Jacobian differs), so both solve the same `F=0`; at Re=1/ConstantSigma the nearby root is unique → same solution. This also **empirically closes** the "spurious root adjacent to the exact solution" loophole (Picard's different path landed on the same root). **The defect is provably in the assembled residual `F`** — not the Jacobian, not the solver, not the linearization. "Paper used Picard" is not the resolution; Kratos assembles a *different* `F`. |

Notes:
- The huge L²p **ratio** for equal-order (700–2900×) is a red herring: it's large because the P2-pressure
  *interpolant* is tiny (1e-4). The *absolute* pressure error (~0.4) is the same under inf-sup-stable Taylor-Hood
  (H-row), so it is **not** spurious pressure modes — the whole solution is simply wrong.
- "Erratic/non-monotone" on a **regular** mesh with **constant** element quality cannot be a mesh-quality
  (constant-factor) effect — it is a formulation instability. (User's own observation; correct.)

**Where this leaves the hunt (2026-07-05).** Row J proves the defect is in the assembled **residual `F`**, and
paper c₁ is correct (banner), so it is a **Gridap↔paper discrepancy in `F`**, constrained to **P2-only** (P1
has no 2nd derivatives) ∩ **3D-only** (2D-QUAD k2 works) = the **viscous 2nd-derivative subscale**. But rows D
and G already verified the **strong viscous operator itself** — `∇·(2νεᵈu) = νΔu + (ν/3)∇(∇·u)` + the porosity-
gradient term — matches the analytic/oracle value to ~1e-17. So the discrepancy is **not** in computing
`R(u_h)`'s viscous term; the remaining P2∩3D-specific suspects in `F` are (i) the **viscous part of the adjoint**
`L*(v) = …±∇·(2νεᵈv)` in the stabilization pairing `τ₁(R(u_h), L*(v))` — is that 2nd-derivative adjoint term
present, and with the sign/factor the paper/Kratos uses? — and (ii) the **τ₁ weight** on the viscous subscale.
This is the target of the residual audit (diff vs `article.tex` `eq:StrongMomentumEquation` + the `B_S` adjoint
under `eq:OSGSProblem`).

**Discriminator — 2D-P2 on TRIANGLES is OPTIMAL ⇒ the bug is genuinely 3D-specific, NOT simplex-specific
(2026-07-05).** The "2D-P2 works" evidence had all been on **quads**; the standing open lead was whether the
instability is a **P2∩simplex** effect (which would reproduce cheaply in 2D-TRI). Tested directly:
`data/phase1_tri_k2.json` (2D-TRI, ASGS-P2, the **same** cell as 3D: Re=1, Da=1, α=0.5, same harness/MMS as the
known-good QUAD-P2 control), N=10→80:

| N | 10 | 20 | 40 | 80 | rate → |
|---|---|---|---|---|---|
| TRI-P2 L²u | 5.77e-3 | 1.10e-3 | 1.48e-4 | 1.90e-5 | 2.39 → 2.90 → **2.97** (opt 3) |
| TRI-P2 H¹u | 0.373 | 0.142 | 0.0400 | 0.0106 | 1.39 → 1.83 → **1.92** (opt 2) |
| QUAD-P2 L²u (control) | 4.07e-3 | 8.31e-4 | 1.04e-4 | 1.32e-5 | 2.29 → 2.99 → 2.99 |

TRI-P2 **converges optimally**, monotone, tracking QUAD-P2 to a ~1.3× constant (identical preasymptotic first
step + same recovery to O(h³)/O(h²)) — **nothing like** 3D-TET-P2 (erratic, non-monotone, L²u~0.049, 20–95×
interpolant). So the **P2 simplex basis, 2nd-derivative subscale, and the `√2` simplex-h convention all behave
correctly in 2D**; whatever breaks P2 requires the **third dimension**. Remaining 3D-specific suspects:
1. **The `h` convention for tets** — `smoke3d.jl:222` uses `h=(6√2·V)^(1/3)` (regular-tet edge length), while the
   2D harness uses `(d!·V)^(1/d)` = `√(2·Area)` (→ h = grid spacing). The 3D formula carries an **extra `√2`**
   (h ≈ 1.122× the 2D-consistent `(6V)^(1/3)`), mis-scaling `c₁ν/h²` in the same direction c₁×4 compensates —
   but only ~1.26×, a contributing factor, not the whole ×2–4. **Cheap test:** 3D-P2 at paper c₁ with
   `h=(6V)^(1/3)`.
2. **Anisotropy** — the 3D domain is a thin slab `(0,1)×(0,1)×(0,0.3)` with 3 elements in z (h_z≈0.1 vs
   h_xy≈0.083); 2D is isotropic. A scalar volume-based `h` smears that. **Cheap discriminator:** 3D-P2 on an
   **isotropic cube `(0,1)³`** with equal spacing.
3. The **z-component of the viscous 2nd-derivative** subscale on tets (exact field is z-extruded, but the
   discrete P2 tet field carries genuine `∂²/∂z²`).

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
