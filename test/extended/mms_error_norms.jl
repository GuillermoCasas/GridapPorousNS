# ==============================================================================================
# Shared MMS error functional — the SINGLE source of truth for the dimensionless error norms that
# every manufactured-solution harness reports, and that the paper's convergence tables print.
#
# Extracted from `ManufacturedSolutions/run_test.jl` (2026-07-17) so that the solver-error harnesses
# and the *interpolation-reference* harness (`run_interpolation_reference.jl`) measure with the SAME
# functional. That matters more than it looks: the reference and the data rows are printed side by side
# in the same table and divided into one another as an efficiency, so any mismatch in normalisation,
# pressure gauge, or quadrature silently corrupts the comparison rather than failing. Three details
# below are easy to get wrong in a hand-rolled reimplementation and would each be invisible:
#   (i)   the L² errors divide by U_c·√(|Ω|) (not just U_c) — and √(|Ω|) ≠ 1 whenever L ≠ 1;
#   (ii)  the pressure error is MEAN-CENTRED before its L² norm (all-Dirichlet ⇒ p is fixed only up to
#         a constant, so the gauge mode must not be charged as error);
#   (iii) the H¹ semi-norms divide by U_c / P_c with NO L factor (in 2D the L from integration cancels
#         the 1/L from the gradient — see the derivation below).
# This mirrors the precedent set by `harness_dynamic_budget.jl`: one definition, read by every harness.
#
# Consumers: ManufacturedSolutions/run_test.jl, ManufacturedSolutions/run_interpolation_reference.jl,
#            CocquetFormMMS/run_test.jl.
# ==============================================================================================

using Gridap

# Error evaluator returning genuinely DIMENSIONLESS norms (encoding-invariant under L-rescaling).
#
# Derivation (writing the dimensional error as e(x) = U_c · ê(x/L), with x̂ = x/L the
# dimensionless coordinate and Ω the L-scaled physical domain):
#
#   ‖e‖²_{L²(Ω)}     = ∫_Ω |e|² dx = U_c² · ∫_Ω |ê(x/L)|² dx
#                    = U_c² · L^d · ‖ê‖²_{L²(Ω̂)}
#   ⇒  ‖e‖_{L²(Ω)}   = U_c · √(|Ω|) · ‖ê‖_{L²(Ω̂)}
#
#   ‖∇e‖²_{L²(Ω)}    = ∫_Ω (U_c/L)² |∇_x̂ ê|² dx = (U_c/L)² · L^d · ‖∇ê‖²_{L²(Ω̂)}
#   ⇒  ‖∇e‖_{L²(Ω)}  = U_c · L^{d/2 - 1} · ‖∇ê‖_{L²(Ω̂)}   (independent of L in 2D where d=2)
#
# So the dimensionless quantities are:
#
#     el2_u_dimless   = ‖e_u‖_{L²(Ω)}  / (U_c · √(|Ω|))   = ‖ê_u‖_{L²(Ω̂)}
#     eh1_u_dimless   = ‖∇e_u‖_{L²(Ω)} / U_c              = ‖∇ê_u‖_{L²(Ω̂)}        (2D)
#
# For L=1 with a unit-area baseline bounding box `[-0.5, 0.5]²` this collapses to the
# legacy form `el2_u = ‖e‖/U_c`, `eh1_u = ‖∇e‖/(U_c/L)` — so existing centered-encoding
# K=1 sweeps reproduce bit-identically. For L≠1 (balanced / minmax encodings) the new form
# removes the L-inflation that the legacy normalisation introduced.
function calculate_normalized_errors(u_h, p_h, u_final, p_final, U_c, P_c, L, dΩ)
    e_u = u_final - u_h
    e_p = p_final - p_h

    # Domain measure |Ω| (this is the *physical* area of the L-scaled domain).
    area = sum(∫(1.0)dΩ)
    sqrt_area = sqrt(abs(area))     # √(|Ω|) = L · √(|Ω̂|); for L=1 with [-0.5, 0.5]² this is 1.0

    # 1. Velocity L² error, dimensionless via ‖e_u‖/(U_c · √(|Ω|)).
    el2_u = sqrt(sum(∫(e_u ⋅ e_u)dΩ)) / (U_c * sqrt_area)

    # Pressure null-space alignment (centre the error so the gauge-mode is not penalised).
    mean_e_p = sum(∫(e_p)dΩ) / area
    e_p_centered = e_p - mean_e_p

    # 2. Pressure L² error, dimensionless via ‖e_p‖/(P_c · √(|Ω|)).
    el2_p = sqrt(sum(∫(e_p_centered * e_p_centered)dΩ)) / (P_c * sqrt_area)

    # 3. Semi-H¹ errors. In 2D the L from integration cancels the 1/L from the gradient, so
    # the dimensionless H¹ semi-norm is simply ‖∇e‖/U_c (no L factor in the divisor).
    eh1_semi_u = sqrt(sum(∫(∇(e_u) ⊙ ∇(e_u))dΩ)) / U_c
    eh1_semi_p = sqrt(sum(∫(∇(e_p) ⋅ ∇(e_p))dΩ)) / P_c

    return el2_u, el2_p, eh1_semi_u, eh1_semi_p
end
