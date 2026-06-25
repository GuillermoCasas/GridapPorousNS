# ==============================================================================================
# Nature & Intent:
# Guards FORM-01: `build_picard_jacobian` must be byte-for-byte identical to the general
# `build_stabilized_weak_form_jacobian(..., freeze_cusp, PicardMode())`. In PicardMode the general
# builder's dσ/du, dτ/du and dL*/du terms collapse to structural zeros (×0.0), so the two assembled
# Jacobian matrices must coincide exactly. This is the safety net that lets `build_picard_jacobian` be a
# one-line wrapper over the general builder (docs/formulation-audit-2026-06-24.md §D, FORM-01 / F2):
# any future divergence between the two Picard paths fails here immediately.
#
# Associated Files / Functions:
# - `src/formulations/continuous_problem.jl` (`build_picard_jacobian`, `build_stabilized_weak_form_jacobian`)
# ==============================================================================================

using Test
using PorousNSSolver
using Gridap
using LinearAlgebra

@testset "fast: Picard Jacobian equals general builder in PicardMode (FORM-01)" begin
    # Tiny non-trivial problem: 2x2 QUAD, equal-order P1/P1, a velocity-dependent (Forchheimer)
    # reaction so the frozen σ at the iterate is non-trivial, and a non-zero iterate so |u| > 0.
    model = tiny_model_2d(n=(2, 2))
    X, Y, V, Q = tiny_spaces_2d(model; kv=1, kp=1)
    V_free = TestFESpace(model, ReferenceFE(lagrangian, VectorValue{2,Float64}, 1), conformity=:H1)
    Q_free = TestFESpace(model, ReferenceFE(lagrangian, Float64, 1), conformity=:H1)
    Ω, dΩ, h_cf = tiny_measure(model; degree=6)

    visc = PorousNSSolver.DeviatoricSymmetricViscosity()
    rxn  = PorousNSSolver.ForchheimerErgunLaw(2.0, 4.0)
    proj = PorousNSSolver.ProjectFullResidual()
    reg  = PorousNSSolver.SmoothVelocityFloor(1e-3, 0.5, 1e-8, PorousNSSolver.VELOCITY_MAGNITUDE_DERIVATIVE_FLOOR)
    form = PorousNSSolver.PaperGeneralFormulation(visc, rxn, proj, reg, 1e-2, 1e-3)

    c_1, c_2 = PorousNSSolver.get_c1_c2(PorousNSSolver.PaperGeneralFormulation, 1)
    alpha_cf = CellField(alpha_lin, Ω)
    f_cf = CellField(f_zero, Ω)
    g_cf = CellField(g_zero, Ω)

    setup = PorousNSSolver.FETopology(X, Y, model, Ω, dΩ, V_free, Q_free, h_cf, f_cf, alpha_cf, g_cf)
    formulation = PorousNSSolver.VMSFormulation(form, c_1, c_2)
    phys_cfg = (; tau_regularization_limit = 1e-8)

    # Non-trivial linearization point (|u| > 0 everywhere so the derivatives are well-defined).
    xh = interpolate_everywhere([u_poly, p_poly], X)

    # OSGS projections π_h: any fields work — both builders consume them identically, so equality must hold.
    pi_u = CellField(u_poly, Ω)
    pi_p = CellField(p_poly, Ω)

    for (label, pu, pp) in (("ASGS", nothing, nothing), ("OSGS", pi_u, pi_p))
        A_picard = assemble_matrix(
            (dx, y) -> PorousNSSolver.build_picard_jacobian(xh, dx, y, setup, formulation, phys_cfg; pi_u=pu, pi_p=pp),
            X, Y)
        A_general = assemble_matrix(
            (dx, y) -> PorousNSSolver.build_stabilized_weak_form_jacobian(xh, dx, y, setup, formulation, phys_cfg, false, PorousNSSolver.PicardMode(); pi_u=pu, pi_p=pp),
            X, Y)
        # Byte-for-byte equality (exact ==, not isapprox) of the dense matrices.
        @test Array(A_picard) == Array(A_general)
    end
end
