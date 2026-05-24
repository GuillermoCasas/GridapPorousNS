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
