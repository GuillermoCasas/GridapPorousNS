# MMS sweep — NEXT NECESSARY ACTIONS (post-restart)

> **STATUS — point-in-time action snapshot (2026-06-02).** The authority on the *resolved* parts of
> this story is [`lessons_learned.md`](../lessons_learned.md) (2026-06-02 entries). Two cautions for any
> reader: (1) the §3a/§3b "cure" (lift the inner cap + scale-invariant gates) is **SUPERSEDED** — it was
> tested 2026-06-02 and did **not** cure the OSGS defect; the remaining defect is in the discrete
> staggered map, not the gates. (2) The sweep-completion state below (the partial N=320 tail, k=2) is
> as of 2026-06-02 and has not been re-verified since. The encoding-policy and sweep-finishing actions
> still stand; the gate/cap fixes do not.

> **UPDATE 2026-06-02 (results of testing §3a/§3b below).** Two things were resolved by experiment:
> 1. **`eps_val` covariance bug — FOUND & FIXED.** The continuity pressure penalty `eps_val·p` was
>    injected as a fixed `1e-8` for all encodings; it is dimensional (`[eps_val]=(U/L)/P_c`). Scaling it
>    per-cell (`build_mms_formulation`) restored **ASGS** scale-covariance from 3.6e-6 → ~1e-10. Guarded
>    by `encoding_invariance_test.jl`. (See `lessons_learned.md` 2026-06-02.)
> 2. **§3a/§3b (gate scale-invariance + lift inner cap) were TESTED and did NOT cure OSGS.** With a
>    unit-invariant gate and inner cap 1→10, minmax's OSGS outer staggered loop still oscillated
>    (rate 0.89, 11× worse than centered). So the **remaining OSGS defect is in the discrete staggered
>    map** `Π(R(U(π)))` / stabilised assembly — NOT the gates, NOT `eps_val`. Still open; needs a
>    direct look at the OSGS projection/assembly for any other non-covariant or non-contractive term.
> The encoding-policy / sweep-completion actions below still stand.


**Read this first.** This is the executable action list left for an agent picking up after a machine
restart. Context: the latest finding (full record in [`lessons_learned.md`](../lessons_learned.md),
2026-06-02 entries) is that with the **`minmax` encoding default the OSGS staggered loop fails to
converge at low-Re/high-Da (Re=1,Da=1e6) — velocity rate stagnates**, whereas the old `centered`
encoding converges to the fixed point and is optimal on the same cell.

**Mechanism (pinned by trace instrumentation — supersedes ALL earlier hypotheses: not the τ-balance,
not "absolute gate vs dimensionless tol", not a floating-point noise floor).** It is a **scale-coupled
inner-Newton gate + a 1-iteration inner cap**. Evidence (Re=1,Da=1e6,α=1,OSGS, N=160 traces):
- The per-field inner gate `eff_ftol = rel·‖x‖ ∝ ‖x‖ ∝ U` is encoding-coupled — measured centered
  3.9e-10 vs minmax 1.3e-8 (ratio ≈33 ≈ U-ratio 31.6).
- `osgs_inner_newton_iters` caps the inner Newton at ~1 step.
- Consequence in the stop-reasons: **centered** inner solves fire `initial_ftol`/`ftol_reached` (≤1
  step, residual → machine 6.5e-15) — the OSGS projection correction is *below* the gate, so OSGS
  **silently degenerates to ASGS → optimal** (i.e. "centered OSGS optimal" is partly an artifact, it
  isn't really doing OSGS). **minmax** inner solves fire `max_iters_stagnation` — the residual is
  *above* the gate, the single allowed Newton step lands ~17000× above the gate (`res_out_norm`≈1.7e4),
  so every outer iteration feeds an UNDER-CONVERGED state into the projection → outer drift floors at
  ~1e-4 and oscillates → stagnant. The ~1e-4 floor is a single-step-residual/under-convergence level,
  NOT roundoff (it sits ~12 orders above machine ε). The encoding flips inert-ASGS vs engaged-but-capped
  purely through the `‖x‖∝U` normalisation in the gate.

> ⚠ **Machine must stay awake / on AC power.** The sweep was killed three times mid-`N=320` because
> the machine slept. Disable sleep before launching anything long. The `N=320` cells are the slow ones.

## Current state (as of 2026-06-02)
- **k=1 sweep**: 253/288 runs done — **every cell at N=10,20,40,80,160 complete (both ASGS & OSGS)**;
  only the **N=320 row is partial (13/48)**. DB: `test/extended/ManufacturedSolutions/results/phase1_quad_k1.h5`. k=2 not started.
- All completed runs carry `f_norm` trajectory traces (`results/traces/`).
- A/B evidence: `data/abtest_centered.json` (a `centered`-encoding copy of `phase1_quad_k1.json`) and
  scratch DBs `results/abtest_centered.h5`, `results/abtest_minmax.h5` (one cell, OSGS, N≤160).

## Actions, in order

### 1. Map the encoding-regression SCOPE  *(decisive; do first)*
Run the `centered`-vs-`minmax` A/B across more OSGS cells to learn exactly which cells `minmax`
degrades. Focus on the Da=1e6 column (all α, both Re=1 and Re=1e-6) and a couple of Da=1 cells.
From `test/extended/ManufacturedSolutions/`:
```bash
# example: one cell, both encodings, N<=160 (fast). Vary the --filter to sweep the column.
F="Re=1,Da=1e6,alpha0=0.5,etype=QUAD,kv=1,method=OSGS"
julia --project=../../.. run_test.jl abtest_centered.json   --filter "$F" --max-N 160 --h5 ab_c.h5
julia --project=../../.. run_test.jl phase1_quad_k1.json     --filter "$F" --max-N 160 --h5 ab_m.h5
# then compare err_u_l2 / err_u_h1 rates between results/ab_c.h5 and results/ab_m.h5 (h5py).
```
**Acceptance:** a table {cell × encoding → final L²/H¹ rate} showing the boundary of where `minmax`
stagnates and `centered` is optimal. (Known so far: Re=1,Da=1e6,α=1 → centered OPTIMAL, minmax STAGNANT.)

### 2. Check whether `centered` is actually needed at high Re
`minmax` was adopted because `centered`'s `U_amp ~ 1e9` at Re=1e6 was feared to wreck conditioning.
Verify empirically — run `centered` on the Re=1e6 OSGS cells:
```bash
julia --project=../../.. run_test.jl abtest_centered.json --filter "Re=1e6,etype=QUAD,kv=1,method=OSGS" --max-N 160 --h5 ab_c_hiRe.h5
```
**Acceptance:** do the Re=1e6 OSGS cells converge under `centered`, or do they fold/diverge? This
decides between a global revert and a per-regime policy.

### 3. Fix it (two distinct fixes + a practical workaround)
- **(3a) [SUPERSEDED — tested 2026-06-02, did NOT cure; see status header] lift the 1-iteration
  inner-Newton cap.** The stagnation is that minmax's OSGS inner solve does a single Newton step and stops
  at `max_iters_stagnation` ~17000× above its gate, feeding under-converged states to the projection.
  Raise `osgs_inner_newton_iters` (config / `run_test.jl` OSGS stab block ~L967) so each outer
  iteration's inner solve actually converges. Then re-run Re=1,Da=1e6,α=1 OSGS under BOTH encodings:
  - if both now converge optimally → confirmed it was the inner cap × scale-coupled gate;
  - if minmax still stalls with converged inner solves → the OSGS *staggered* loop genuinely fails to
    contract at this regime (the real OSGS limitation), now exposed honestly rather than hidden by
    centered's inert/ASGS degeneration. Either way you learn the true OSGS behaviour, which centered
    currently HIDES (its inner solves fire `initial_ftol` → projection below gate → OSGS ≡ ASGS).
- **(3b) [SUPERSEDED — tested 2026-06-02, did NOT cure; see status header] make the gates scale-invariant
  via TERM-BY-TERM residual normalisation.** The bug: the per-field gate `‖R_k‖ ≤ rel·‖x_k‖` normalises the residual by
  the SOLUTION norm `‖x‖∝U`, but the residual's natural size at high Da is the REACTION term `∝σ·u`, so
  the achieved relative residual is `rel·U/(σU)=rel/σ` — σ-coupled, hence encoding-coupled (`eff_ftol`
  measured 33× between encodings). **Fix: normalise the residual by the sum (or max) of its own term
  norms**, per equation:
  `normalised R_mom = ‖R_mom‖ / (‖conv‖+‖press‖+‖visc‖+‖reaction‖+‖f‖)`  (and the analogous continuity
  denominator), with a small floor `max(Σ‖term‖, ε)`. Numerator and denominator share units → the ratio
  is DIMENSIONLESS and unit/encoding-invariant; the σ that poisoned `rel/σ` cancels (→ `rel` cleanly).
  Keep the target = the truncation floor: `normalised R ≤ c_sf·h^{k+1}` (i.e. reuse `dynamic_ftol`).
  Implementation: swap the per-field denominator in `_initialize_effective_thresholds`/the per-field
  gate (`src/solvers/nonlinear.jl`) from `‖x_k‖` to `Σ‖term‖`; the OSGS strong-residual path already
  assembles terms in `inner_projection_u` (`porous_solver.jl:31`) so the term norms are nearly free
  there (the inner Newton's algebraic residual needs the extra per-term assembly). This is standard CFD
  practice (Ferziger–Perić / OpenFOAM-style residual normalisation) and the natural MMS choice.
  Payoff: the convergence VERDICT becomes encoding-independent, so centered and minmax AGREE — unmasking
  whether OSGS genuinely converges here (gate artifact) or genuinely stalls (real OSGS limitation).
  For the OSGS OUTER state-drift gate (`x_diff ≤ osgs_tol`, `porous_solver.jl:221`/`299`) the right
  scale-invariant analog is NOT term-by-term but the **relative iterate change** `‖ΔU‖/‖U‖` (it's an
  iterate change, not a residual). It does NOT add inner iterations — with an honest gate, if the inner
  solve still reports "not converged in 1 step" (now identically in both encodings) that is the signal
  to do §3a (lift the cap). So: fix the gate first, let it tell you about the cap.
  NOTE: the `τ₁` formula is fine — already unit-invariant (`σh²/(c₁ν)=Da·α∞/(c₁N²)`; the early
  "balance flips" idea was an `h=1/N` vs physical `h=L/N` slip). The leak is entirely in the gates.
- **Encoding policy (from §1+§2):** if `centered` is globally adequate for OSGS → revert the default
  (`run_test.jl` `encoding_strategy` ~L481 + `data/*.json`); if `centered` fails at Re=1e6 → per-regime
  encoding selection in `compute_L_and_U`. This is a practical workaround; 3a is the principled fix.

### 4. Finish the sweep data (under the chosen encoding)
Leave `results/phase1_quad_k1.h5` in place — the harness is resume-aware and skips the 253 done cells.
From `test/extended/ManufacturedSolutions/`:
```bash
# finish k=1 N=320 tail (~35 cells)
for k in 1 2 3 4; do julia --project=../../.. run_test.jl phase1_quad_k1.json --shard $k/4 & done; wait
# then k=2 (P2/P2)
for k in 1 2 3 4; do julia --project=../../.. run_test.jl phase1_quad_k2.json --shard $k/4 & done; wait
```
If the encoding policy changed in step 3, the OSGS cells affected by the change must be recomputed
(delete just those groups, or use a fresh `--h5`).

### 5. (Enrichment) make future OSGS diagnosis self-contained
The 2026-06-02 diagnosis had to *reconstruct* τ/σ because they aren't stored. Add to the HDF5 group
attrs + JSON trace, per mesh: **`tau1`/`tau2` (min/max/representative over Ω), `sigma`, `|u|_max`,
`encoding_strategy` + the L/U scale factors**. Optionally surface τ₁ / σ-share-of-`1/τ₁` on the
trajectory plot. (Solver already has these values at each mesh; it's only plumbing into
`run_test.jl`'s HDF5 write + the trace sidecar.) This makes the encoding/τ-balance story readable
straight from the data next time.

## Cleanup note
`data/abtest_centered.json` and `results/abtest_*.h5` are A/B scratch; keep them until step 1–3 are
done (they're the reusable centered config + evidence), then remove `data/abtest_centered.json` if a
permanent per-regime mechanism replaces it.
