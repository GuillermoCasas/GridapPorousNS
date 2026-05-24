# src/metrics.jl

"""
    compute_reference_errors(f_h, f_ref, if_ref, V_free, dΩ_h, dΩ_ref; filter_func=x->true, search_method=nothing)

Computes exactly the cross-mesh topological errors for a single generic field (velocity, pressure, etc).
An optional `filter_func(x) -> Bool` can geometrically isolate the evaluation metrics (e.g. `x -> x[1] < 1.5`).

`search_method` controls the cross-mesh point-location used to build the *coarse* `Interpolable`
evaluated at the reference quadrature points (the "consistent" metric). When `nothing` (default), Gridap's
default `KDTreeSearch` is used — this is exact and robust for STRUCTURED, nested meshes (the benchmark
path is byte-for-byte unchanged). For NON-nested UNSTRUCTURED meshes, reference quadrature points near the
boundary can otherwise fall outside every coarse cell and the default search throws; pass a tolerant
`KDTreeSearch(num_nearest_vertices=k)` (k≈10) to widen the candidate-cell set. The caller should build
`if_ref` with the same tolerant search so the nested branch is robust too.

Returns:
`l2_nested, h1_nested, l2_cons, h1_cons, e_nested, e_consistent`
"""
function compute_reference_errors(f_h, f_ref, if_ref, V_free, dΩ_h, dΩ_ref; filter_func=(x)->true, search_method=nothing)
    # ---------------------------------------------------------------------
    # 1. EXACT SUBSPACE PROJECTION (Nested Coarse Formulation)
    # ---------------------------------------------------------------------
    f_ref_proj = interpolate(if_ref, V_free)

    e_nested = f_ref_proj - f_h
    grad_e_nested = ∇(f_ref_proj) - ∇(f_h)
    mask_h = CellField(x -> filter_func(x) ? 1.0 : 0.0, Gridap.FESpaces.get_triangulation(f_h))

    # ⊙ (inner product) acts agnostically on scalars, vectors, and tensors in Gridap
    l2_nested = sqrt(sum(∫( mask_h * (e_nested ⊙ e_nested) ) * dΩ_h))
    h1_nested = sqrt(sum(∫( mask_h * (grad_e_nested ⊙ grad_e_nested) ) * dΩ_h))

    # ---------------------------------------------------------------------
    # 2. EXACT CONTINUOUS METRIC (Consistent Fine-Mesh Formulation)
    # ---------------------------------------------------------------------
    # Default (structured/nested): exact KDTree search, unchanged. Unstructured/non-nested: caller
    # supplies a tolerant KDTreeSearch so near-boundary reference quadrature points still locate a coarse cell.
    if_coarse      = search_method === nothing ? Gridap.FESpaces.Interpolable(f_h)    : Gridap.FESpaces.Interpolable(f_h;    searchmethod=search_method)
    if_coarse_grad = search_method === nothing ? Gridap.FESpaces.Interpolable(∇(f_h)) : Gridap.FESpaces.Interpolable(∇(f_h); searchmethod=search_method)

    e_cons = f_ref - if_coarse
    grad_e_cons = ∇(f_ref) - if_coarse_grad
    mask_ref = CellField(x -> filter_func(x) ? 1.0 : 0.0, Gridap.FESpaces.get_triangulation(f_ref))

    l2_cons = sqrt(sum(∫( mask_ref * (e_cons ⊙ e_cons) ) * dΩ_ref))
    h1_cons = sqrt(sum(∫( mask_ref * (grad_e_cons ⊙ grad_e_cons) ) * dΩ_ref))

    return (l2_nested, h1_nested, l2_cons, h1_cons, e_nested, e_cons)
end


"""
    compute_trial_projection_errors(f_h, if_ref, U_trial, dΩ_h; filter_func=x->true)

H-C diagnostic metric: project the reference field onto the coarse-mesh **trial** space
`U_trial` (with Dirichlet conditions baked in) rather than the free-DOF test space used by
`compute_reference_errors`'s `l2_nested`. Returns `(l2_trial, h1_trial)`.

For a problem with P2-exact Dirichlet data (Cocquet's inlet `c_in y(1-y)` is exactly P2) this
metric should agree with `l2_nested` to floating-point noise — any meaningful gap signals that
the boundary-trace evaluation of the cross-mesh interpolant is contributing measurable error
on unstructured meshes, which is the H-C hypothesis. See:
  - `theory/Replicating Cocquet et al. convergence results.md` (H-C)
  - `/Users/guillermocasasgonzalez/.claude/plans/i-have-added-replicating-tidy-stonebraker.md`

Standalone (does not change `compute_reference_errors`'s signature or behaviour).
"""
function compute_trial_projection_errors(f_h, if_ref, U_trial, dΩ_h; filter_func=(x)->true)
    f_ref_in_trial = interpolate(if_ref, U_trial)
    e_trial = f_ref_in_trial - f_h
    grad_e_trial = ∇(f_ref_in_trial) - ∇(f_h)
    mask_h = CellField(x -> filter_func(x) ? 1.0 : 0.0, Gridap.FESpaces.get_triangulation(f_h))
    l2_trial = sqrt(sum(∫( mask_h * (e_trial ⊙ e_trial) ) * dΩ_h))
    h1_trial = sqrt(sum(∫( mask_h * (grad_e_trial ⊙ grad_e_trial) ) * dΩ_h))
    return (l2_trial, h1_trial)
end
