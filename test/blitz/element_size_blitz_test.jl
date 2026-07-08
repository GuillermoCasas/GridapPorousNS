# ==============================================================================================
# Nature & Intent:
# Unit-correctness of the configurable element characteristic size h(K) (src/geometry.jl,
# `element_size_field`) that feeds the stabilization τ₁/τ₂. Every convention is checked against a
# CLOSED-FORM expected value on shapes whose geometry is known exactly, across BOTH dimensions and
# BOTH element families:
#   2D: QUAD (square + anisotropic rectangle), TRI (right-isoceles from simplexify)
#   3D: HEX  (cube  + anisotropic box),        TET (Kuhn from simplexify — 6 congruent tets, edges
#                                                    {s,s,s, s√2,s√2, s√3})
# The anisotropic QUAD/HEX cases are the load-bearing checks: they pin that `:shortest_edge` and
# `:average_edge` use TRUE EDGES (get_faces(polytope,1,0)) and are NOT fooled by the longer face/body
# DIAGONALS, and that `:diameter` IS the (excluded) diagonal — the exact distinction that separates the
# four conventions and that a naive "all vertex pairs" implementation would get wrong.
#
# Conventions (src/geometry.jl):
#   :volume        (d!·|K|)^{1/d} simplex / |K|^{1/d} tensor   (grid-spacing proxy; legacy default)
#   :shortest_edge min edge length                             (Codina; the current default)
#   :average_edge  mean edge length
#   :diameter      max‖x_i − x_j‖ over ALL vertex pairs        (strict mathematical diameter)
#
# Associated: src/geometry.jl, src/config.jl (StabilizationConfig.element_size), config/base_config.json.
# ==============================================================================================

using Test
using PorousNSSolver
using Gridap
const _P = PorousNSSolver

# The (uniform) per-cell h for a convention, as a scalar: measure-weighted mean over cells. For all
# meshes below every cell is congruent, so this equals the single-cell h.
function _cell_h(model, convention::Symbol)
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2)
    h = _P.element_size_field(Ω, model, convention)
    return sum(∫(h)dΩ) / sum(∫(1.0)dΩ)
end

@testset "fast: element_size conventions — shapes, dims, options" begin
    r2 = sqrt(2.0)
    r3 = sqrt(3.0)

    @testset "2D QUAD — square (side s)" begin
        s = 0.5
        m = CartesianDiscreteModel((0.0, s, 0.0, s), (1, 1))
        @test _cell_h(m, :volume)        ≈ s          # √(Area) = s
        @test _cell_h(m, :shortest_edge) ≈ s          # all 4 edges = s
        @test _cell_h(m, :average_edge)  ≈ s
        @test _cell_h(m, :diameter)      ≈ s * r2      # the diagonal
    end

    @testset "2D QUAD — anisotropic rectangle (a×b): edges NOT diagonals" begin
        a, b = 0.5, 2.0
        m = CartesianDiscreteModel((0.0, a, 0.0, b), (1, 1))
        @test _cell_h(m, :volume)        ≈ sqrt(a * b)
        @test _cell_h(m, :shortest_edge) ≈ min(a, b)                # = a, NOT the √(a²+b²) diagonal
        @test _cell_h(m, :average_edge)  ≈ (a + b) / 2             # 2 edges of a + 2 of b
        @test _cell_h(m, :diameter)      ≈ sqrt(a^2 + b^2)         # the diagonal (excluded from edges)
    end

    @testset "2D TRI — right-isoceles (legs s), from simplexify" begin
        s = 0.5
        m = simplexify(CartesianDiscreteModel((0.0, s, 0.0, s), (1, 1)))
        @test _cell_h(m, :volume)        ≈ s                       # √(2·Area) = √(2·s²/2) = s
        @test _cell_h(m, :shortest_edge) ≈ s                       # the two legs
        @test _cell_h(m, :average_edge)  ≈ s * (2 + r2) / 3        # {s, s, s√2}
        @test _cell_h(m, :diameter)      ≈ s * r2                  # the hypotenuse
    end

    @testset "3D HEX — cube (side s)" begin
        s = 0.5
        m = CartesianDiscreteModel((0.0, s, 0.0, s, 0.0, s), (1, 1, 1))
        @test _cell_h(m, :volume)        ≈ s                       # V^{1/3} = s
        @test _cell_h(m, :shortest_edge) ≈ s                       # all 12 edges = s
        @test _cell_h(m, :average_edge)  ≈ s
        @test _cell_h(m, :diameter)      ≈ s * r3                  # the body diagonal
    end

    @testset "3D HEX — anisotropic box (a×b×c): edges NOT face/body diagonals" begin
        a, b, c = 0.5, 1.0, 2.0
        m = CartesianDiscreteModel((0.0, a, 0.0, b, 0.0, c), (1, 1, 1))
        @test _cell_h(m, :volume)        ≈ (a * b * c)^(1 / 3)
        @test _cell_h(m, :shortest_edge) ≈ min(a, b, c)            # = a; never a diagonal
        @test _cell_h(m, :average_edge)  ≈ (a + b + c) / 3         # 4 edges each of a, b, c
        @test _cell_h(m, :diameter)      ≈ sqrt(a^2 + b^2 + c^2)   # body diagonal (excluded from edges)
    end

    @testset "3D TET — Kuhn (side s), from simplexify: edges {s,s,s,s√2,s√2,s√3}" begin
        s = 0.5
        m = simplexify(CartesianDiscreteModel((0.0, s, 0.0, s, 0.0, s), (1, 1, 1)))
        @test _cell_h(m, :volume)        ≈ s                       # (6·V)^{1/3} = (6·s³/6)^{1/3} = s
        @test _cell_h(m, :shortest_edge) ≈ s                       # the 3 axis edges
        @test _cell_h(m, :average_edge)  ≈ s * (3 + 2r2 + r3) / 6  # mean of {1,1,1,√2,√2,√3}·s
        @test _cell_h(m, :diameter)      ≈ s * r3                  # the body-diagonal edge
    end

    @testset "invalid convention / parser" begin
        m = CartesianDiscreteModel((0.0, 1.0, 0.0, 1.0), (1, 1))
        Ω = Triangulation(m)
        @test_throws ErrorException _P.element_size_field(Ω, m, :not_a_convention)
        @test_throws ErrorException _P.element_size_convention("bogus")
        @test _P.element_size_convention("shortest_edge") === :shortest_edge
        @test _P.element_size_convention("diameter") === :diameter
        @test Set(_P.ELEMENT_SIZE_CONVENTIONS) == Set((:volume, :shortest_edge, :average_edge, :diameter))
    end
end
