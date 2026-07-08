# src/geometry.jl
#
# Configurable element characteristic size h(K) for the stabilization parameters τ₁, τ₂
# (src/stabilization/tau.jl: τ_{1,NS} = (c₁ν/h² + c₂|u|/h)⁻¹, τ₂ = h²/(c₁ α τ_{1,NS})).
#
# The convention is a STABILIZATION-CALIBRATION choice (StabilizationConfig.element_size) — the paper's
# c₁=4k⁴, c₂=2k² and the inverse-inequality constant C_inv are stated for a specific h_K, so which length
# feeds τ rescales the effective stabilization by an O(1) factor (e.g. on a right-isoceles triangle the
# diameter is √2× the min edge). Making it explicit lets us match Codina/the paper exactly instead of
# wondering whether the h-convention is "that other thing". [known-fragility] changing the default here
# changes τ on non-structured meshes (on the structured simplex mesh :shortest_edge ≡ :volume ≡ grid
# spacing, so it is byte-identical there).
#
# h is built ONCE per mesh into a CellField (cached), so the convention adds ZERO per-quadrature-point /
# per-solve cost — it only touches one-time mesh setup.
using Gridap
using Gridap.Geometry: get_grid_topology, get_polytopes, num_cell_dims
using Gridap.ReferenceFEs: get_faces

# The valid conventions (kept in one place so config validation and the dispatch cannot drift).
const ELEMENT_SIZE_CONVENTIONS = (:volume, :shortest_edge, :average_edge, :diameter)

@inline _vertex_dist(a, b) = sqrt((a - b) ⋅ (a - b))

"""
    element_size_field(Ω, model, convention::Symbol) -> CellField

Per-cell characteristic length `h(K)` for the VMS stabilization, one of `ELEMENT_SIZE_CONVENTIONS`:

- `:volume`        — `(d!·|K|)^{1/d}` for simplices, `|K|^{1/d}` for tensor-product cells: the grid-spacing
                     proxy (2D → `√(2·A)` on triangles, `√A` on quads; 3D → `(6·V)^{1/3}` on tets). The
                     legacy convention that was hard-coded in the harnesses.
- `:shortest_edge` — the minimum EDGE length `min_e ‖x_i − x_j‖` (true edges only — excludes quad/hex
                     diagonals). Codina's convention; equals the grid spacing on the structured simplex
                     mesh. **The default (StabilizationConfig.element_size).**
- `:average_edge`  — the mean edge length over the cell's edges.
- `:diameter`      — the strict mathematical element diameter `max_{i<j} ‖x_i − x_j‖` over ALL vertex
                     pairs (for a simplex this is the longest edge; the paper reports `h = √2/N`, i.e. this).

Edges are taken from the cell polytope's dim-1↔dim-0 incidence (`get_faces(p,1,0)`), so `:shortest_edge`
and `:average_edge` are correct on quads/hexes (not fooled by diagonals). Assumes a single-polytope mesh.
"""
function element_size_field(Ω, model, convention::Symbol)
    cc = get_cell_coordinates(Ω)
    if convention === :volume
        d = num_cell_dims(model)
        simplex = length(first(cc)) == d + 1          # d+1 vertices ⇒ simplex; 2^d ⇒ tensor-product
        fac = simplex ? factorial(d) : 1              # simplex volume = edge^d / d! ⇒ recover the grid edge
        ha = collect(lazy_map(v -> (fac * abs(v))^(1.0 / d), get_cell_measure(Ω)))
        return CellField(ha, Ω)
    elseif convention === :diameter
        ha = [maximum(_vertex_dist(v[i], v[j]) for i in 1:length(v) for j in (i + 1):length(v)) for v in cc]
        return CellField(ha, Ω)
    elseif convention === :shortest_edge || convention === :average_edge
        e2v = get_faces(only(get_polytopes(get_grid_topology(model))), 1, 0)   # local edge → local vertex ids
        reduce_edges = convention === :shortest_edge ? minimum : (es -> sum(es) / length(es))
        ha = [reduce_edges([_vertex_dist(v[a], v[b]) for (a, b) in e2v]) for v in cc]
        return CellField(ha, Ω)
    else
        error("Unknown element_size convention :$convention; valid: $(ELEMENT_SIZE_CONVENTIONS)")
    end
end

"""
    element_size_convention(s::AbstractString) -> Symbol

Parse (and validate) the `StabilizationConfig.element_size` string into a convention Symbol.
"""
function element_size_convention(s::AbstractString)
    sym = Symbol(s)
    sym in ELEMENT_SIZE_CONVENTIONS ||
        error("Invalid element_size '$s'; valid: $(ELEMENT_SIZE_CONVENTIONS)")
    return sym
end
