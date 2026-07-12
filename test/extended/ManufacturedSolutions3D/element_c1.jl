# test/extended/ManufacturedSolutions3D/element_c1.jl
# =============================================================================================
# Element-type-aware viscous stabilization constant c₁*(K).
#
# A faithful numerical transcription of theory/numerical_constants/c1_dimension_note.tex
# ("A note on the dimension and shape dependence of the viscous stabilization constant c₁ for
#  quadratic simplicial elements in ASGS/VMS methods", G. Casas).
#
# THE LESSON (note §2–§3, condensed):
#   The Galerkin+ASGS elemental VISCOUS energy ledger (diffusion-dominated limit, a=σ=0,
#   α const) is, after all physical parameters factor out (note eq:ledger2),
#       Q_K/(αν) = 2‖Π∇u‖²_K − (4h_K²/c₁)‖∇·(Π∇u)‖²_K + ‖∇·u‖²_K ,
#   with the DEVIATORIC-symmetric projector  Π∇u = symΞ∇u − (1/d)(∇·u)I .
#   Q_K ≥ 0 for every elemental P_k velocity  ⟺  c₁ ≥ c₁*(K) := 2 ĉ²(K) (note Prop.1, eq:c1star):
#       c₁*(K) = max_{u∈[P_k(K)]^d, Π∇u≠0}
#                  4 h_K² ‖∇·(Π∇u)‖²_K / ( 2‖Π∇u‖²_K + ‖∇·u‖²_K ).
#   c₁*(K) is a PURE SHAPE constant (similarity-invariant; note Prop.1 proof) with h_K = diam K.
#
#   • P1: ∇·(Π∇u) ≡ 0 identically  ⇒  c₁*(K) = 0 in ANY dimension (note Prop.2). Any positive c₁
#     keeps the elemental ledger nonnegative; c₁ is accuracy-governed, not coercivity-governed.
#     THIS is why the (mostly linear-element) Codina/ASGS literature never needed a 2D→3D rescaling.
#   • P2: strongly shape- and dimension-dependent. Note Table 1 (h_K = diam K):
#         2D right triangle (quad split) : c₁* = 96          [ĉ² = 48 ]
#         2D equilateral triangle        : c₁* = 48          [ĉ² = 24 ]
#         3D right tetrahedron           : c₁* = 880/3 ≈293.3 [ĉ² =440/3]
#         3D Kuhn tet (cube split)       : c₁* = 400+20√2 ≈428.3 [ĉ²=200+10√2]
#         3D regular tetrahedron         : c₁* = 400/3 ≈133.3 [ĉ² = 200/3]
#     The binding element of a mesh is the WORST-shaped one; the practical (global, conforming)
#     threshold sits somewhat BELOW c₁*(K) (note Remark "elementwise versus global"), so c₁*(K) is a
#     conservative sufficient floor — use it with a safety factor s ≲ 1.
#
#   h-CONVENTION. c₁*(K) above is stated with h_K = diam K. If the code feeds τ a DIFFERENT element
#   length h_code(K) (:regular_tet, :shortest_edge, …), the ledger's h_K IS that h_code, so the floor
#   in code units is  c₁*_code(K) = 4 h_code(K)² λ_max(K) = c₁*_diam(K)·(h_code/diam)²  (note Rem.
#   "h-convention sensitivity"). This file computes it DIRECTLY in whichever convention by passing the
#   element's h_code as `h`.
#
# This module computes c₁*(K) for an ARBITRARY simplex from its vertices, via the SAME generalized
# eigenvalue problem the note solves (note eq:c1star). Everything is exact up to the (over-resolved)
# quadrature: the integrands are degree ≤ 2 for P2, and a 5-pt Gauss/Duffy rule is exact to degree 9.
# Validated against Table 1 at the bottom (`element_c1_selftest()`), and cross-checked against the
# note's SymPy supplement values in theory/numerical_constants/c1_note_supplementary/constants.json.
# =============================================================================================
using LinearAlgebra

# ---- 1D Gauss–Legendre on [0,1] (5 nodes, exact to degree 9) -------------------------------
# Nodes/weights on [-1,1] (standard 5-pt GL), mapped to [0,1] by ξ↦(ξ+1)/2, w↦w/2.
const _GL5_X = [-0.906179845938664, -0.5384693101056831, 0.0,
                 0.5384693101056831,  0.906179845938664]
const _GL5_W = [0.23692688505618908, 0.47862867049936647, 0.5688888888888889,
                0.47862867049936647, 0.23692688505618908]
const _GL01_X = (_GL5_X .+ 1.0) ./ 2.0
const _GL01_W = _GL5_W ./ 2.0

# ---- Quadrature on the UNIT reference simplex (Duffy/collapsed-coordinate tensor rule) ------
# Reference d-simplex: vertices 0, e₁,…,e_d (volume 1/d!). Exact for polynomials up to degree 9.
# 2D:  x=u, y=v(1−u),          Jac=(1−u)
# 3D:  x=u, y=v(1−u), z=w(1−u)(1−v),  Jac=(1−u)²(1−v)
function _ref_simplex_quadrature(d::Int)
    pts = Vector{Vector{Float64}}()
    wts = Vector{Float64}()
    if d == 2
        for (u, wu) in zip(_GL01_X, _GL01_W), (v, wv) in zip(_GL01_X, _GL01_W)
            x = u; y = v * (1 - u)
            push!(pts, [x, y]); push!(wts, wu * wv * (1 - u))
        end
    elseif d == 3
        for (u, wu) in zip(_GL01_X, _GL01_W), (v, wv) in zip(_GL01_X, _GL01_W), (w, ww) in zip(_GL01_X, _GL01_W)
            x = u; y = v * (1 - u); z = w * (1 - u) * (1 - v)
            push!(pts, [x, y, z]); push!(wts, wu * wv * ww * (1 - u)^2 * (1 - v))
        end
    else
        error("_ref_simplex_quadrature: d must be 2 or 3 (got $d)")
    end
    return pts, wts
end

# ---- Physical monomial exponents for [P_k]^scalar ------------------------------------------
function _monomial_exponents(d::Int, k::Int)
    exps = Vector{NTuple{3,Int}}()
    if d == 2
        for a in 0:k, b in 0:(k - a)
            push!(exps, (a, b, 0))
        end
    else
        for a in 0:k, b in 0:(k - a), c in 0:(k - a - b)
            push!(exps, (a, b, c))
        end
    end
    return exps
end

# value, gradient (d-vector), Hessian (d×d) of a monomial x^e1 y^e2 z^e3 at point x (length d)
@inline _pow(base, p) = p == 0 ? 1.0 : base^p
function _monomial_derivs(e::NTuple{3,Int}, x::AbstractVector{Float64}, d::Int)
    e1, e2, e3 = e
    xs = (d == 2 ? (x[1], x[2], 0.0) : (x[1], x[2], x[3]))
    val = _pow(xs[1], e1) * _pow(xs[2], e2) * _pow(xs[3], e3)
    es = (e1, e2, e3)
    g = zeros(Float64, d)
    H = zeros(Float64, d, d)
    for a in 1:d
        ea = es[a]
        if ea == 0
            g[a] = 0.0
        else
            ee = collect(es); ee[a] -= 1
            g[a] = ea * _pow(xs[1], ee[1]) * _pow(xs[2], ee[2]) * _pow(xs[3], ee[3])
        end
        for b in 1:d
            # ∂²/∂x_a∂x_b
            ee = collect(es)
            fac = 1.0
            ok = true
            for c in (a, b)
                if ee[c] == 0; ok = false; break; end
                fac *= ee[c]; ee[c] -= 1
            end
            H[a, b] = ok ? fac * _pow(xs[1], ee[1]) * _pow(xs[2], ee[2]) * _pow(xs[3], ee[3]) : 0.0
        end
    end
    return val, g, H
end

"""
    element_gram_matrices(vertices, k) -> (A, B)

Assemble the two Gram matrices of note eq:c1star on the physical simplex `vertices`
(a Vector of d+1 points, each a length-d vector or Gridap VectorValue), for [P_k]^d velocities:

  A = Gram of  u ↦ ∇·(Π∇u)          (the anti-coercive viscous 2nd-derivative operator)
  B = Gram of  (√2 Π∇u, ∇·u)  i.e.  B = 2·Gram(Π∇u) + Gram(∇·u)   (the coercive denominator)

with Π∇u = sym∇u − (1/d)(∇·u)I the deviatoric-symmetric projector. The largest generalized
eigenvalue λ_max(A,B) then gives c₁*(K) = 4 h² λ_max (see `element_c1_star`).
"""
function element_gram_matrices(vertices, k::Int)
    d = length(vertices) - 1
    (d == 2 || d == 3) || error("element_gram_matrices: need a triangle or tet (got $(length(vertices)) vertices)")
    v0 = Float64[vertices[1][i] for i in 1:d]
    M = zeros(Float64, d, d)                 # affine ref→phys map columns v_{i}-v_0
    for j in 1:d, i in 1:d
        M[i, j] = Float64(vertices[j + 1][i]) - v0[i]
    end
    detM = abs(det(M))
    refpts, refwts = _ref_simplex_quadrature(d)

    exps = _monomial_exponents(d, k)
    nm = length(exps)
    N = d * nm                                # coeff DOFs: (component l, monomial i)

    A = zeros(Float64, N, N)
    B = zeros(Float64, N, N)
    # per-quad-point evaluation of the three linear maps on every basis field u^{(α)} = m_i e_l
    Dq = zeros(Float64, N, d)                 # ∇·(Π∇u^{(α)})  (d-vector)
    Pq = zeros(Float64, N, d, d)              # Π∇u^{(α)}       (d×d matrix)
    Vq = zeros(Float64, N)                    # ∇·u^{(α)}       (scalar)
    for (qp, wq0) in zip(refpts, refwts)
        xq = v0 .+ M * qp                     # physical quad point
        wq = wq0 * detM
        # precompute grad/Hess of each monomial at xq
        gv = Vector{Vector{Float64}}(undef, nm)
        hv = Vector{Matrix{Float64}}(undef, nm)
        for i in 1:nm
            _, g, H = _monomial_derivs(exps[i], xq, d)
            gv[i] = g; hv[i] = H
        end
        for l in 1:d, i in 1:nm
            α = (l - 1) * nm + i
            g = gv[i]; H = hv[i]
            # grad u = e_l ⊗ ∇m_i ;  (∇u)_{ab} = δ_{al} g[b]
            divu = g[l]                                   # ∇·u = ∂_l m_i
            Vq[α] = divu
            # E_{ab} = ½(δ_{al} g[b] + δ_{bl} g[a]) ; Π_{ab} = E_{ab} − (1/d) divu δ_{ab}
            for a in 1:d, b in 1:d
                E = 0.5 * ((a == l ? g[b] : 0.0) + (b == l ? g[a] : 0.0))
                Pq[α, a, b] = E - (a == b ? divu / d : 0.0)
            end
            # ∇·(Π∇u)_a = ½ δ_{al} Δm_i + (½ − 1/d) ∂_a∂_l m_i
            lap = 0.0; for b in 1:d; lap += H[b, b]; end
            for a in 1:d
                Dq[α, a] = 0.5 * (a == l ? lap : 0.0) + (0.5 - 1.0 / d) * H[a, l]
            end
        end
        # accumulate Gram contributions
        for α in 1:N, β in α:N
            # A: D_α · D_β
            aαβ = 0.0
            for a in 1:d; aαβ += Dq[α, a] * Dq[β, a]; end
            A[α, β] += wq * aαβ
            # B: 2 (P_α : P_β) + V_α V_β
            pαβ = 0.0
            for a in 1:d, b in 1:d; pαβ += Pq[α, a, b] * Pq[β, a, b]; end
            B[α, β] += wq * (2.0 * pαβ + Vq[α] * Vq[β])
        end
    end
    # symmetrize
    for α in 1:N, β in α+1:N
        A[β, α] = A[α, β]; B[β, α] = B[α, β]
    end
    return A, B
end

# largest generalized eigenvalue λ of A x = λ B x on the positive part of B (both PSD).
function _gen_eig_max(A::Matrix{Float64}, B::Matrix{Float64}; tol::Float64=1e-10)
    As = Symmetric(0.5 * (A + A'))
    Bs = Symmetric(0.5 * (B + B'))
    w, U = eigen(Bs)
    wmax = maximum(w)
    keep = findall(>(tol * wmax), w)
    isempty(keep) && return 0.0
    P = U[:, keep] ./ sqrt.(w[keep])'         # whitening P: PᵀBP = I on kept subspace
    Ar = Symmetric(P' * (As * P))
    λ = eigvals(Ar)
    return maximum(λ)
end

"""
    element_lambda_max(vertices, k) -> Float64

The shape Rayleigh maximum λ_max(A,B) of note eq:c1star (size-independent; scales as length⁻²).
Zero for k = 1 (Π∇u is constant ⇒ ∇·(Π∇u) ≡ 0). Multiply by 4·h² to get c₁*(K) in the h-convention.
"""
function element_lambda_max(vertices, k::Int)
    k == 1 && return 0.0
    A, B = element_gram_matrices(vertices, k)
    return _gen_eig_max(A, B)
end

"""
    element_c1_star(vertices, k; h) -> Float64

Elementwise coercivity floor c₁*(K) = 4 h² λ_max(K) for the simplex `vertices` and velocity
order `k`, in the h-CONVENTION supplied via `h` (pass the SAME element length the code feeds τ —
:regular_tet, :shortest_edge, or diam — so the floor is directly comparable to the code's c₁).
If `h` is omitted, the element DIAMETER is used (note Table 1 convention).
"""
function element_c1_star(vertices, k::Int; h::Union{Nothing,Float64}=nothing)
    λ = element_lambda_max(vertices, k)
    hh = h === nothing ? _element_diameter(vertices) : h
    return 4.0 * hh * hh * λ
end

function _element_diameter(vertices)
    d = length(vertices[1])
    dm = 0.0
    n = length(vertices)
    for i in 1:n, j in (i + 1):n
        s = 0.0
        for c in 1:d; s += (Float64(vertices[i][c]) - Float64(vertices[j][c]))^2; end
        dm = max(dm, sqrt(s))
    end
    return dm
end

# ---- Mesh calibration: per-element c₁*_code distribution → robust scalar --------------------
#
# The binding element is the worst-shaped one (note Remark "elementwise versus global"), so a mesh
# is calibrated by the HIGH tail of {c₁*_code(K)}. We compute c₁*_code(K) = 4 h_code(K)² λ_max(K)
# per cell, using the SAME per-cell h the code feeds τ (`h_array`), and reduce with a percentile
# (default 100 = strict worst element) and a safety factor s (default 1.0).
#
# λ_max(K) is a pure SHAPE constant, so it is cached by a similarity-invariant signature (sorted,
# max-normalized squared edge lengths — a tet is fixed up to reflection by its 6 edge lengths, and
# λ_max is reflection-invariant). Red refinement yields few congruence classes, so the cache makes
# even ~10⁵-cell meshes a ~ms one-time setup cost.

function _edge_signature(vertices)
    d = length(vertices[1]); n = length(vertices)
    e2 = Float64[]
    for i in 1:n, j in (i + 1):n
        s = 0.0
        for c in 1:d; s += (Float64(vertices[i][c]) - Float64(vertices[j][c]))^2; end
        push!(e2, s)
    end
    m = maximum(e2)
    sort!(e2)
    return Tuple(round.(e2 ./ m, digits=9))       # scale-free, order-free shape key
end

"""
    calibrate_c1_over_cells(cell_coords, h_array, k; percentile=100.0, safety=1.0)
        -> (c1_star_scalar, percell, stats)

Per-cell elementwise coercivity floor c₁*_code(K) = 4 h_array[c]² λ_max(K) over a mesh, and the
percentile/safety reduction to a single mesh scalar. `cell_coords` = `collect(get_cell_coordinates(Ω))`
(each entry the cell's vertex points); `h_array` = the same per-cell τ length the solver uses.
Returns the calibrated scalar, the per-cell floor array, and a stats NamedTuple (distribution
percentiles + λ_max shape stats + the multiplier vs the paper baseline 4k⁴).
"""
function calibrate_c1_over_cells(cell_coords, h_array, k::Int; percentile::Float64=100.0, safety::Float64=1.0)
    ncell = length(cell_coords)
    length(h_array) == ncell || error("calibrate_c1_over_cells: h_array/cell mismatch ($(length(h_array)) vs $ncell)")
    cache = Dict{Any,Float64}()
    percell = Vector{Float64}(undef, ncell)
    lam = Vector{Float64}(undef, ncell)
    for c in 1:ncell
        V = cell_coords[c]
        λ = if k == 1
            0.0
        else
            key = _edge_signature(V)
            get!(cache, key) do; element_lambda_max(V, k); end
        end
        lam[c] = λ
        percell[c] = 4.0 * h_array[c]^2 * λ
    end
    c1_paper = 4.0 * k^4
    q(p) = _percentile(percell, p)
    c1_reduced = safety * q(percentile)
    stats = (ncell=ncell, nshapes=length(cache), k=k, percentile=percentile, safety=safety,
             c1_paper=c1_paper, c1_star_reduced=c1_reduced,
             mult_reduced=c1_reduced / c1_paper,
             c1_min=minimum(percell), c1_p50=q(50.0), c1_p90=q(90.0), c1_p99=q(99.0), c1_max=maximum(percell),
             lam_min=minimum(lam), lam_p50=_percentile(lam, 50.0), lam_max=maximum(lam))
    return c1_reduced, percell, stats
end

function _percentile(v::AbstractVector{Float64}, p::Float64)
    isempty(v) && return 0.0
    s = sort(v)
    p <= 0 && return s[1]
    p >= 100 && return s[end]
    r = (p / 100) * (length(s) - 1) + 1
    lo = floor(Int, r); hi = ceil(Int, r); frac = r - lo
    return s[lo] * (1 - frac) + s[hi] * frac
end

"""
    calibrated_c1_mult(cell_coords, h_array, k; percentile, safety, floor_at_paper=true) -> Float64

The element-aware c₁ MULTIPLIER (relative to the paper baseline 4k⁴) for a mesh: reduces the per-cell
floor to a scalar and divides by 4k⁴. With `floor_at_paper=true` (default) the multiplier is at least
1 (never weaken the validated paper constant — for P1 λ_max≡0 so this pins it at exactly 1). The
solver then uses c₁ = mult · 4k⁴ (c₁-ONLY; c₂ stays at the paper value — see the note, only the
viscous constant gates coercivity).
"""
function calibrated_c1_mult(cell_coords, h_array, k::Int; percentile::Float64=100.0, safety::Float64=1.0,
                            floor_at_paper::Bool=true)
    c1_reduced, _, stats = calibrate_c1_over_cells(cell_coords, h_array, k; percentile=percentile, safety=safety)
    m = stats.mult_reduced
    return floor_at_paper ? max(1.0, m) : m
end

# ---- Self-test against note Table 1 (h_K = diam K) -----------------------------------------
function element_c1_selftest(; verbose::Bool=true)
    cases = [
        ("2D right triangle (quad split)", [[0.0,0.0],[1.0,0.0],[0.0,1.0]],                       96.0),
        ("2D equilateral triangle",        [[0.0,0.0],[1.0,0.0],[0.5, sqrt(3)/2]],                48.0),
        ("3D right tetrahedron",           [[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], 880/3),
        ("3D Kuhn tet (cube split)",       [[0.0,0.0,0.0],[1.0,0.0,0.0],[1.0,1.0,0.0],[1.0,1.0,1.0]], 400+20*sqrt(2)),
        ("3D regular tetrahedron",         [[0.0,0.0,0.0],[1.0,0.0,0.0],[0.5,sqrt(3)/2,0.0],[0.5,sqrt(3)/6,sqrt(6)/3]], 400/3),
    ]
    ok = true
    verbose && println("=== element_c1.jl self-test vs note Table 1 (c₁* = 2ĉ², h_K = diam K) ===")
    for (name, V, expect) in cases
        got = element_c1_star(V, 2)                       # diameter convention
        rel = abs(got - expect) / expect
        pass = rel < 1e-6
        ok &= pass
        verbose && println(rpad(name, 34), " c₁*=", round(got, digits=4),
                           "  expect ", round(expect, digits=4), "  rel=", round(rel, sigdigits=3),
                           pass ? "  ✓" : "  ✗ FAIL")
    end
    # P1 must be identically zero (note Prop.2)
    for (name, V, _) in cases
        z = element_c1_star(V, 1)
        p1ok = abs(z) < 1e-10
        ok &= p1ok
        verbose && p1ok || println("  P1 ", name, " c₁*=", z, p1ok ? "" : "  ✗ FAIL (should be 0)")
    end
    verbose && println(ok ? "ALL PASS ✓" : "SELF-TEST FAILED ✗")
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    element_c1_selftest()
end
