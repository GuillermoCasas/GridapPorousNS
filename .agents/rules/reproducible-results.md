---
name: porousns-reproducible-results
description: Every result must be reconstructable to the exact parameters and configuration that produced it. Never sever the result↔config provenance link — do not unify, rename, or delete configs or results in a way that loses which parameters produced which output.
glob: "**/*.{json,jl,h5,md}"
---
# Mindset: parameters → results must always be reconstructable

The guiding philosophy of this codebase: **we must always be able to reconstruct which
parameters produced which results.** A number — an error, a convergence rate, a `fold` flag, an
iteration count — is scientifically meaningless unless it can be traced back to the *exact*
configuration that produced it (Re, Da, α₀, mesh, element order, method, encoding, solver
tolerances, skip-list, …). The entire mode of reasoning in this project is *comparing results
across parameter regimes to understand the method*; that reasoning breaks permanently the moment a
result can no longer be tied to its inputs.

Treat any action that severs the **result ↔ configuration** link as a correctness violation — on
the same footing as introducing a magic number or a silent default.

---

## 1. Never sever the result↔config provenance link

- **Do not unify, merge, rename, or delete a configuration file** if existing results (committed,
  archived under `previous_results/`, or live on disk) were produced by it and would thereby lose
  their parameter provenance. A config that is "of the same kind" as another is **not** redundant
  when it is the *record of how a specific result set was obtained*. "Consolidate the scrap configs"
  is the wrong instinct precisely because each one documents a past run.
- **Do not consolidate result files** in a way that drops the per-run parameters.
- Sharded / targeted re-runs (per-`(etype, k)`, high-Re-only, tight-tolerance patches, an
  N=640 asymptotic extension) are obtained with `run_test.jl --filter / --shard`, **not** by
  collapsing the configs that document earlier runs.

## 2. Every result must carry its own provenance

When writing or modifying result-emitting code, ensure each result records enough to reconstruct its
parameters **without** needing the original config in hand:

- the per-cell sweep parameters stored as result/group attributes — the MMS HDF5 already does this
  (`attributes(grp)["Re"]`, `["Da"]`, `["alpha_0"]`, `["k_velocity"]`, `["method"]`, …);
- trajectory / sidecar JSONs whose **filename and body** both encode the full cell identity
  (`traj_Re…_Da…_a…_kv…_kp…_<etype>_<method>_N….json` + a `cell` key);
- archived snapshots in `previous_results/…` kept **together** with the config and the
  convergence report that produced them.

If you add a new result artifact, add its provenance in the same change.

## 3. When in doubt, preserve — or make provenance explicit first

If a cleanup, refactor, or "unification" would make it harder to answer *"what parameters gave this
number?"*, **stop and keep the link.** If consolidation is genuinely wanted, make the provenance
explicit *before* collapsing anything — e.g. embed a config snapshot into the result, or write a
manifest mapping each result file → its config + parameters — and only then proceed.

Reversible is not an excuse: the parameters that produced an archived `.h5`/`.json` are not recoverable
from git history alone if the producing config was deleted in the same era.
