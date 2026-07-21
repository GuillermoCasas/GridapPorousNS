# test/blitz/mms_genuine3d_exactness_blitz_test.jl
# ==============================================================================================
# Nature & Intent:
# Exactness guard for the GENUINELY three-dimensional manufactured solution (audit response
# R6/N19), implemented as UExFuncGenuine3D / PExFuncGenuine3D in src/problems/mms_paper.jl.
# Unlike the default z-extruded field, this one has all three velocity components depending on
# x,y,z with u_z != 0, built so that the weighted flux alpha*u = curl(A) is exactly
# divergence-free. The oracle builds the forcing from the hand-written analytic derivatives
# grad_u_ex / lap_u_ex / grad_div_u_ex; a sign or factor slip there would silently corrupt the
# forcing and spoil MMS convergence. This test pins those derivatives against central finite
# differences and asserts div(alpha*u)=0 to machine precision, at points in the porosity
# transition layer, the constant-alpha plateau and the outer constant region.
#
# The companion symbolic proof (field div-freeness, _v3/_grad_v3 closed forms, Delta v=-3k^2 v)
# lives in proof_verification/sympy/genuine3d_mms_verification.py.
# ==============================================================================================
using Test
using PorousNSSolver
const PNS = PorousNSSolver
using Gridap.TensorValues

@testset "Genuinely-3D MMS exactness (R6/N19)" begin
    af = PNS.SmoothRadialPorosity(0.5, 1.0, 0.2, 0.4)
    U, L = 1.3, 1.0
    f = PNS.UExFuncGenuine3D(U, af.alpha_0, af, L)

    pt(a) = VectorValue(a[1], a[2], a[3])
    uu(a) = f(pt(a))
    h = 1e-6

    fd_grad(a) = begin
        G = zeros(3,3)
        for i in 1:3
            ap = copy(a); ap[i]+=h; am = copy(a); am[i]-=h
            du = (uu(ap) - uu(am)) / (2h)
            for j in 1:3; G[i,j] = du[j]; end
        end
        G
    end
    fd_lap(a) = begin
        Lp = zeros(3); u0 = uu(a)
        for i in 1:3
            ap = copy(a); ap[i]+=h; am = copy(a); am[i]-=h
            d2 = (uu(ap) - 2.0*u0 + uu(am)) / h^2
            for j in 1:3; Lp[j] += d2[j]; end
        end
        Lp
    end
    divu(a) = begin s=0.0; for i in 1:3; ap=copy(a);ap[i]+=h; am=copy(a);am[i]-=h; d=(uu(ap)-uu(am))/(2h); s+=d[i]; end; s end
    fd_graddiv(a) = begin
        g = zeros(3)
        for j in 1:3; ap=copy(a);ap[j]+=h; am=copy(a);am[j]-=h; g[j]=(divu(ap)-divu(am))/(2h); end
        g
    end
    divau(a) = begin s=0.0; for i in 1:3; ap=copy(a);ap[i]+=h; am=copy(a);am[i]-=h;
        aup = af(pt(ap))*uu(ap); aum = af(pt(am))*uu(am); s += (aup[i]-aum[i])/(2h); end; s end

    test_points = [
        [0.22, 0.20, 0.15],   # transition layer (alpha varies): exercises grad_alpha/hess_alpha terms
        [0.25, 0.18, 0.22],
        [0.30, 0.10, 0.05],
        [0.05, 0.05, 0.10],   # plateau r<r1: alpha constant (grad_alpha = 0)
        [0.45, 0.45, 0.20],   # outer r>r2: alpha constant
    ]

    for a in test_points
        Ga = PNS.grad_u_ex(f, pt(a)); Gf = fd_grad(a)
        @test maximum(abs(Ga[i,j]-Gf[i,j]) for i in 1:3, j in 1:3) < 1e-6   # 1st deriv: tight
        La = PNS.lap_u_ex(f, pt(a)); Lf = fd_lap(a)
        @test maximum(abs(La[j]-Lf[j]) for j in 1:3) < 1e-2                 # 2nd deriv: FD-noise loose
        Da = PNS.grad_div_u_ex(f, pt(a)); Df = fd_graddiv(a)
        @test maximum(abs(Da[j]-Df[j]) for j in 1:3) < 1e-3                 # nested 2nd deriv
        @test abs(divau(a)) < 1e-8                                          # div(alpha*u) = 0 (the point)
    end

    # genuinely 3D: nonzero z-velocity, all components z-dependent.
    @test abs(uu([0.25,0.18,0.22])[3]) > 1e-3
    @test abs(uu([0.25,0.18,0.22])[3] - uu([0.25,0.18,0.10])[3]) > 1e-3      # u_z varies with z
end
