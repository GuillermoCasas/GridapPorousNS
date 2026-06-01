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
    compute_mode_decomposition(f_h, if_ref, V_free, dΩ_h)

Decomposes the nested cross-mesh error `e_h := f_ref_proj − f_h` along two
spectral cuts and reports each cut's share of the L² norm-squared.

The output named tuple is

    (l2_e, l2_cellavg, l2_domainmean, fraction_cellavg, chi_Omega)

* `l2_e = ‖e_h‖_{L²(Ω)}` — the same nested-projection norm
  `compute_reference_errors` returns as `l2_nested`.
* `l2_cellavg = ‖P₀ e_h‖_{L²(Ω)}`, where `P₀` is the L²-projection onto
  piecewise constants on Ω_h. Per cell `ē_K = (∫_K e) / |K|`, so
  `‖P₀ e‖² = ∑_K (∫_K e)⋅(∫_K e) / |K|`.
* `l2_domainmean = ‖ē_Ω · 1_Ω‖_{L²(Ω)} = √|Ω| · ‖ē_Ω‖`,
  with `ē_Ω = (∫_Ω e dΩ)/|Ω|` the single domain-wide mean.
* `fraction_cellavg = ‖P₀ e‖² / ‖e‖²` — share of L² norm-squared captured
  by the piecewise-constant projection.
* `chi_Omega = |Ω| · ‖ē_Ω‖² / ‖e‖² = (∫_Ω e)⋅(∫_Ω e) / (|Ω|·‖e‖²)` —
  share captured by the single domain-wide constant.

What these numbers actually probe (relevant to
`docs/cocquet_magnitude_investigation.md` S3):

* `fraction_cellavg` measures how smooth `e_h` is on the coarse-cell scale.
  For any C¹ error it tends to 1 as `h → 0` because piecewise constants
  approximate smooth functions on small cells; it is a generic feature of
  converging FE error, not a signature of a pathological global mode. It
  does discriminate within-cell oscillation (e.g. P₂ vs P₁ velocity
  spaces), but it is not the right test for a globally-applied gauge or
  boundary-flux bias.
* `chi_Omega` is self-reference-invariant in a sharper sense:
  `χ_Ω → 1` only when `e_h` carries a rigid, domain-wide constant offset,
  and `χ_Ω → 0` for any zero-mean error. This is the actual discriminator
  for a "true global mode" — anything that survives the self-reference
  cancellation must have a non-trivial domain mean.

Both numbers are computed from the same `e_h` so the cost is one extra
pass over the assembled mesh integrals.
"""
function compute_mode_decomposition(f_h, if_ref, V_free, dΩ_h)
    f_ref_proj = interpolate(if_ref, V_free)
    e = f_ref_proj - f_h

    Ω = Gridap.FESpaces.get_triangulation(f_h)

    # Full nested L² norm (matches the `l2_nested` returned by compute_reference_errors).
    l2_e_sq = sum(∫(e ⊙ e) * dΩ_h)

    # Per-cell integrals of e (vector-valued for velocity, scalar for pressure) and per-cell measures.
    cell_int_e = (∫(e) * dΩ_h)[Ω]
    cell_meas  = get_array(∫(1.0) * dΩ_h)

    # ‖P₀ e‖² = Σ_K (∫_K e)⋅(∫_K e) / |K|.  `iv ⋅ iv` reduces to multiplication for scalars
    # and to the inner product for VectorValue, so this works for u and p alike.
    l2_cellavg_sq = sum((iv ⋅ iv) / m for (iv, m) in zip(cell_int_e, cell_meas))

    # Domain-wide integral and domain measure, recovered without re-traversing the mesh.
    total_int_e = sum(cell_int_e)
    total_meas  = sum(cell_meas)
    # ‖ē_Ω · 1_Ω‖² = |Ω| · (ē_Ω ⋅ ē_Ω) = (∫_Ω e) ⋅ (∫_Ω e) / |Ω|
    l2_domainmean_sq = (total_int_e ⋅ total_int_e) / total_meas

    fraction_cellavg = l2_e_sq > 0 ? l2_cellavg_sq    / l2_e_sq : 0.0
    chi_Omega        = l2_e_sq > 0 ? l2_domainmean_sq / l2_e_sq : 0.0

    return (l2_e             = sqrt(l2_e_sq),
            l2_cellavg       = sqrt(max(l2_cellavg_sq,    0.0)),
            l2_domainmean    = sqrt(max(l2_domainmean_sq, 0.0)),
            fraction_cellavg = fraction_cellavg,
            chi_Omega        = chi_Omega)
end


"""
    compute_reference_errors_multimask(f_h, f_ref, if_ref, V_free, dΩ_h, dΩ_ref;
                                          masks::AbstractVector{<:Function},
                                          search_method=nothing)

Batched variant of `compute_reference_errors` that builds the (expensive) cross-mesh
interpolation data exactly once and then evaluates the L² and H¹ norms (nested and
consistent) under each `filter_func` in `masks`. Returns a named tuple
`(per_mask, e_nested, e_cons)` where `per_mask` is a Vector of named tuples
`(l2_nested, h1_nested, l2_cons, h1_cons)`, one per mask in the input order, and
`e_nested`, `e_cons` are the unmasked cross-mesh error CellFields (for VTU export).

For the Cocquet diagnostic this collapses what otherwise would be
`1 + |radii|` independent calls (one for the full norm, one per corner-exclusion
radius) into a single call sharing the heavy `Interpolable` build and the cross-mesh
search. The wall-time of the per-N error step drops from O(|radii|·cross_mesh)
to O(cross_mesh + |radii|·simple_integral).
"""
function compute_reference_errors_multimask(f_h, f_ref, if_ref, V_free, dΩ_h, dΩ_ref;
                                              masks::AbstractVector,
                                              search_method=nothing)
    # Nested-projection branch: build cross-mesh interpolation onto the coarse FE space.
    f_ref_proj = interpolate(if_ref, V_free)
    e_nested = f_ref_proj - f_h
    grad_e_nested = ∇(f_ref_proj) - ∇(f_h)

    # Consistent branch: build the cross-mesh interpolation onto the reference quadrature.
    if_coarse      = search_method === nothing ? Gridap.FESpaces.Interpolable(f_h)    : Gridap.FESpaces.Interpolable(f_h;    searchmethod=search_method)
    if_coarse_grad = search_method === nothing ? Gridap.FESpaces.Interpolable(∇(f_h)) : Gridap.FESpaces.Interpolable(∇(f_h); searchmethod=search_method)
    e_cons = f_ref - if_coarse
    grad_e_cons = ∇(f_ref) - if_coarse_grad

    Ω_h   = Gridap.FESpaces.get_triangulation(f_h)
    Ω_ref = Gridap.FESpaces.get_triangulation(f_ref)

    per_mask = map(masks) do filter_func
        mask_h   = CellField(x -> filter_func(x) ? 1.0 : 0.0, Ω_h)
        mask_ref = CellField(x -> filter_func(x) ? 1.0 : 0.0, Ω_ref)
        l2_n = sqrt(sum(∫(mask_h   * (e_nested     ⊙ e_nested))     * dΩ_h))
        h1_n = sqrt(sum(∫(mask_h   * (grad_e_nested ⊙ grad_e_nested)) * dΩ_h))
        l2_c = sqrt(sum(∫(mask_ref * (e_cons       ⊙ e_cons))       * dΩ_ref))
        h1_c = sqrt(sum(∫(mask_ref * (grad_e_cons  ⊙ grad_e_cons))  * dΩ_ref))
        (l2_nested=l2_n, h1_nested=h1_n, l2_cons=l2_c, h1_cons=h1_c)
    end
    return (per_mask=per_mask, e_nested=e_nested, e_cons=e_cons)
end


"""
    compute_corner_excluded_norm(f_h, f_ref, if_ref, V_free, dΩ_h, dΩ_ref;
                                  corners, R, base_filter=x->true,
                                  search_method=nothing)

Thin wrapper around `compute_reference_errors` that excludes open balls of
radius `R` around each point in `corners` from both the nested (coarse) and
consistent (reference) error integrands. `corners` is an iterable of length-D
tuples / `VectorValue`s (D = ambient dimension). `base_filter` lets the caller
keep an outer mask (e.g. the existing `bounding_rule`) and AND it with the
corner exclusion.

Returns the same six-tuple as `compute_reference_errors`:
`(l2_nested, h1_nested, l2_cons, h1_cons, e_nested, e_consistent)`, evaluated
on the corner-excluded subdomain.

Designed for the Cocquet-benchmark outlet-corner localisation hypothesis: if
the magnitude gap collapses to ≲ 10× when balls of radius ≲ 0.1 are excluded
around the two outlet–wall corners, structured-mesh corner pollution is the
dominant contribution; if the gap persists, the error is more diffuse.
"""
function compute_corner_excluded_norm(f_h, f_ref, if_ref, V_free, dΩ_h, dΩ_ref;
                                       corners, R::Float64,
                                       base_filter=(x)->true,
                                       search_method=nothing)
    R2 = R * R
    filter_func = function(x)
        base_filter(x) || return false
        @inbounds for c in corners
            d2 = 0.0
            @inbounds for i in 1:length(c)
                Δ = x[i] - c[i]
                d2 += Δ * Δ
            end
            d2 ≤ R2 && return false
        end
        return true
    end
    return compute_reference_errors(f_h, f_ref, if_ref, V_free, dΩ_h, dΩ_ref;
                                     filter_func=filter_func,
                                     search_method=search_method)
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
