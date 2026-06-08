# Known code issues (report-only)

Genuine CODE-correctness issues surfaced during the 2026-06-04 documentation audit. They are
recorded here **for triage, not fixed in the audit** (the audit's scope was docs/comments/theory).
Each is verified against the working tree. Severity is the author's call.

## Confirmed

- ~~**`cfg.phys.f_x` / `cfg.phys.f_y` would throw** — [run_simulation.jl:336-337](../src/run_simulation.jl#L336-L337)
  reads `cfg.phys.f_x`, but the config field is `physical_properties`.~~ **FIXED 2026-06-04 (P6):** corrected to
  `cfg.physical_properties.f_x` / `.f_y`; the production path is now exercised by
  `test/quick/production_schedule_smoke_quick_test.jl`.

- **`config/base_config.json` is missing the required `eps_val`** — `PhysicalProperties` declares
  `eps_val::Float64` with **no default** ([config.jl:10](../src/config.jl#L10)) and asserts `eps_val > 0`
  ([config.jl:125](../src/config.jl#L125)); `config/base_config.json` carries only `epsilon_floor` (a *different*
  field). So `load_frozen_config("config/base_config.json")` — and the `run_simulation("config/base_config.json")`
  example in the README/CLAUDE.md — fails at struct construction. (Note: `eps_val` was deliberately removed
  from `config/base_config.json` so it can't be silently inherited; see `lessons_learned.md` 2026-05-26. The
  consequence is that the "canonical example" config is no longer directly loadable.) Fix: add an explicit
  `eps_val` to `config/base_config.json`, or update the docs to point at a complete example config.

- **Schema `method` enum vs loader mismatch** — `porous_ns.schema.json` allows
  `method ∈ {ASGS, OSGS, VMS, Galerkin}` (lines ~195-199), but [config.jl:148](../src/config.jl#L148) asserts
  `method ∈ {ASGS, OSGS}`. A `VMS`/`Galerkin` config passes schema validation then dies at the assertion.
  Fix: trim the schema enum to `{ASGS, OSGS}` (or implement the others).

- **Latent hardcoded-`ASGS` dispatch** — [CocquetFormMMS/run_test.jl:466](../test/extended/CocquetFormMMS/run_test.jl#L466)
  hardcodes `"method" => "ASGS"` in the per-cell config dict (the same pattern fixed in the main MMS harness on
  2026-05-26, `lessons_learned.md`). Not triggered today (that test only iterates `{ASGS, Galerkin}`), but it
  would mislabel runs if `OSGS` is ever added.

## Minor / cleanup

- ~~**Dead helper after the covariance fix** — `_resolve_solution_scale_per_field` + the `x_per_field_raw`
  buffer were no longer called (the gate uses the frozen initial-residual scale `‖R₀‖` directly).~~
  **RESOLVED 2026-06-04 (P5):** both removed; see [`solver/normalization-audit.md`](solver/normalization-audit.md) gate #1.

- **Single-run path uses a fixed `eps_val`** — `run_simulation` injects a fixed dimensional `eps_val` rather than
  the per-encoding covariant value the MMS harness now derives (`lessons_learned.md` 2026-06-02). Harmless for a
  single run (no encoding sweep), but inconsistent with the harness.

- **Schema `additionalProperties: true` on `physical_properties`** (`porous_ns.schema.json` ~line 111) admits
  unknown keys, against the strict-config intent. (The loader does `@warn` on unknown keys,
  [config.jl:168](../src/config.jl#L168), so it is soft-guarded.)

## Open numerical defect (not a code "bug")

- **OSGS rate-stagnation in the reaction-dominated corner (high Da, low/moderate Re).** A convergence-*rate*
  issue, distinct from the (now-fixed) scale-covariance bug. **Confirmed with complete data 2026-06-04** — the
  full k=1 sweep (288/288, N=10→320) on fully-covariant code (covariant inner gate `‖R₀‖` + relative warmup +
  covariant `eps_val`; snapshot in `test/extended/ManufacturedSolutions/previous_results/_archive_postFix_covariant_complete/`).
  Findings, now definitive:
  - **Scope:** OSGS velocity rates collapse only at **Da=1e6 with Re ∈ {1e-6, 1}** — H¹ rate **0.62–0.83**
    (vs ASGS ~1.1–2.1), L² **1.59–1.78** (vs ASGS ~1.9–2.65) in this pre-leaning snapshot. At **Re=1e6,
    same Da=1e6 OSGS is healthy** (2.0/1.0). For Da ∈ {1e-6, 1} OSGS matches or beats ASGS everywhere.
    *(These magnitudes are the pre-leaning snapshot; for the current post-leaning numbers — final-segment
    H¹ ~0.71–0.74 at α₀ ∈ {0.5, 1}, recovering to 0.95 at α₀=0.05 — see the canonical
    [`mms/convergence-baseline.md`](mms/convergence-baseline.md). The verdict is unchanged.)*
  - **It is reaction-dominance, not the porosity fold:** the stagnation is just as severe at **α₀=1**
    (1.59/0.62, no porosity layer) as at α₀=0.05 — i.e. independent of α₀, a different axis from the
    high-Re/low-α₀ coarse-mesh fold.
  - **It is not the gates/encoding:** scale-covariance is fixed and verified, yet the stagnation persists
    exactly where predicted.
  - **It is the fixed point itself (CORRECTED 2026-06-05; updated 2026-06-08 for the coupled leaning).**
    The earliest account here — "discrete staggered map oscillating ~1e-4, budget exhausted" — was
    **superseded** by a controlled A/B (then varying only the projection coupling, `staggered` vs
    `coupled`) that gave bit-identical errors to ~4 sig figs across 5+ meshes, with the coupled solve
    converging *gracefully* to `ftol`. Since the 2026-06-08 leaning the **`coupled` solve is the sole
    production OSGS path** — one inexact-Newton solve recomputing `π=Π(R(u))` at every residual
    evaluation; the `staggered` map (and its outer relaxation loop) no longer exists in code. The
    conclusion stands and is now cleaner to state: the coupled solve **does** reach the discrete fixed
    point (it iterates to `ftol_reached`), so the suboptimality is a property of the **OSGS discrete
    fixed point itself**, matching the coercivity gap of
    [`../theory/osgs_reaction_note/osgs_reaction_note.tex`](../theory/osgs_reaction_note/osgs_reaction_note.tex) Prop. 1 (degrades with Da_h,
    recovers with Re_h). The reaction-projection trim is correct and *exonerated*. The old
    "budget exhausted" flag was the MMS *plateau-rate* verifier giving up, not a failed solve — and note
    that the coupled solve is **exempt from the Stage-I stall sensor** (`stall_window=0`): its slow,
    monotone, linear-rate convergence is *not* a stall. Mis-enabling that sensor on the coupled path is a
    distinct, newer failure mode — it bails the solve after ~2 steps and lets `OSGS_INNER_POLICY` accept
    the `no_progress_stall` as success, so OSGS silently degenerates into ASGS and reports ASGS's optimal
    H¹ rate (~1.0) under the OSGS label (regression found and fixed 2026-06-08; see
    [`../docs/lessons_learned.md`](lessons_learned.md) 2026-06-08).
  - **RESOLVED by deep audit (2026-06-06): genuine, pre-asymptotic — not a bug.** A 22-agent code/theory
    audit + three independent empirical checks settled it. (i) **Annihilation probe**: `‖(I−Π)(σu_h)‖/‖σu_h‖
    = 3e-16` (machine-exact, with the Dirichlet lift) ⇒ the trim is bit-identical to full-residual OSGS at
    the fixed point — trim **exonerated**. (ii) **Da-sweep** (fixed mesh): OSGS is bit-optimal at Da≤10²,
    *super-convergent recovery* at Da=10⁴ as the mesh-Damköhler `Da_h = σh²/(c₁ν) = Da·α∞/(c₁N²)` crosses 1,
    degraded only at Da=10⁶ — Da-gated, so **pre-asymptotic** (gap closes ∝1/N²), not a bug. (iii)
    **Full-vs-trim A/B**: full-residual OSGS is **solver-unstable** at high σ (stalls/diverges → ASGS
    fallback); the trim's `−σ·du` Jacobian term is load-bearing for stability — *why the paper trims*.
    NOTE: the audit's auto-synthesis wrongly called it a "code-bug" by misreading the ASGS-fallback as
    "full-OSGS optimal"; refuted by the probe. **Root cause = the OSGS coercivity gap (Prop. 1), pre-asymptotic.**
  - **Fix** (for optimal rate at practical meshes) = formulation-level **split/term-by-term OSGS** (keep the
    reactive term with ASGS identity-projection; convection+pressure orthogonal). Full detail + evidence
    tables: canonical doc [`solver/osgs-reaction-dominated-rate.md`](solver/osgs-reaction-dominated-rate.md) §8.
