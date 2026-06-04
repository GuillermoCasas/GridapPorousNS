# Known code issues (report-only)

Genuine CODE-correctness issues surfaced during the 2026-06-04 documentation audit. They are
recorded here **for triage, not fixed in the audit** (the audit's scope was docs/comments/theory).
Each is verified against the working tree. Severity is the author's call.

## Confirmed

- **`cfg.phys.f_x` / `cfg.phys.f_y` would throw** — [run_simulation.jl:336-337](../src/run_simulation.jl#L336-L337)
  reads `cfg.phys.f_x`, but the config field is `physical_properties` (used correctly at
  [line 317](../src/run_simulation.jl#L317)). `cfg.phys` does not exist, so this errors at runtime when the
  `run_simulation` path reaches it. Fix: `cfg.physical_properties.f_x` / `.f_y`.

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

- **Dead helper after the covariance fix** — `_resolve_solution_scale_per_field`
  ([nonlinear.jl](../src/solvers/nonlinear.jl)) is no longer called (the gate now uses the frozen initial-residual
  scale `‖R₀‖` directly). The `x_per_field_raw` buffer allocated just before the gate is likewise unused.
  Safe to remove.

- **Single-run path uses a fixed `eps_val`** — `run_simulation` injects a fixed dimensional `eps_val` rather than
  the per-encoding covariant value the MMS harness now derives (`lessons_learned.md` 2026-06-02). Harmless for a
  single run (no encoding sweep), but inconsistent with the harness.

- **Schema `additionalProperties: true` on `physical_properties`** (`porous_ns.schema.json` ~line 111) admits
  unknown keys, against the strict-config intent. (The loader does `@warn` on unknown keys,
  [config.jl:168](../src/config.jl#L168), so it is soft-guarded.)

## Open numerical defect (not a code "bug")

- **OSGS rate-stagnation at high Da** (Re=1, Da=1e6): a convergence-*rate* issue, distinct from the (now-fixed)
  scale-covariance bug. The "lift inner cap + scale-invariant gates" cure was tested 2026-06-02 and did **not**
  fix it; the remaining defect is in the discrete staggered map. See `lessons_learned.md` 2026-06-02 and
  `mms_convergence_status.md` caveat #4.
