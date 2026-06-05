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
    (vs ASGS ~1.1–2.1), L² **1.59–1.78** (vs ASGS ~1.9–2.65). At **Re=1e6, same Da=1e6 OSGS is healthy**
    (2.0/1.0). For Da ∈ {1e-6, 1} OSGS matches or beats ASGS everywhere.
  - **It is reaction-dominance, not the porosity fold:** the stagnation is just as severe at **α₀=1**
    (1.59/0.62, no porosity layer) as at α₀=0.05 — i.e. independent of α₀, a different axis from the
    high-Re/low-α₀ coarse-mesh fold.
  - **It is not the gates/encoding:** scale-covariance is fixed and verified, yet the stagnation persists
    exactly where predicted.
  - **It is the fixed point, not the staggered map (CORRECTED 2026-06-05).** The earlier account here —
    "discrete staggered map oscillating ~1e-4, budget exhausted" — is **superseded**. A controlled A/B
    varying only the projection coupling (`staggered` vs `coupled`, the latter recomputing `π=Π(R(u))`
    every Newton iteration with no staggering lag) gives **bit-identical errors to ~4 sig figs across 5+
    meshes**, with `coupled` converging *gracefully* to `ftol`. So the staggered loop *does* reach the
    fixed point; the "budget exhausted" flag is the MMS *plateau-rate* verifier giving up, not a failed
    solve. The suboptimality is a property of the **OSGS discrete fixed point itself**, matching the
    coercivity gap of [`../theory/osgs_reaction_note.tex`](../theory/osgs_reaction_note.tex) Prop. 1
    (degrades with Da_h, recovers with Re_h). The reaction-projection trim is correct and *exonerated*.
  - **Open:** the realized H¹ rate creeps upward (0.57→0.74 over N=10→320), so *slow pre-asymptotic
    climb* vs *asymptotic reduction* is unresolved; a fine ladder (N=640, 1280) settles it.
  - **Full write-up, evidence tables, and options** (incl. the `freeze_after_k` recommendation and the
    formulation-level "split-OSGS" lever): canonical doc
    [`solver/osgs-reaction-dominated-rate.md`](solver/osgs-reaction-dominated-rate.md).
